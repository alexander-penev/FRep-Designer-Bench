// tools/dump_artifacts.cpp
//
// frep_dump — emit analysis artifacts for a scene, for system study and
// for writing about the compiler/retargeting pipeline. Given a scene file
// it writes, alongside a chosen output basename:
//
//   <base>.pre.ll     LLVM IR straight from codegen (before optimisation)
//   <base>.post.ll    LLVM IR after the O3 pipeline (to gauge how much the
//                     generated code optimises — instruction counts in the
//                     stats quantify this)
//   <base>.glsl       generated GLSL compute shader (GPU retargeting path)
//   <base>.spv        SPIR-V bytecode (if glslangValidator is on PATH)
//   <base>.scene.json normalised dump of the input scene
//   <base>.stats.json structured metrics: timings, memory, parallelism,
//                     the explicitly-chosen modes, and IR instruction counts
//
// Pure CLI, no GUI. Side-by-side / pixel-diff analysis is left to external
// tools working on these dumps. The point is reproducible, inspectable
// artifacts for each retargeting path on one scene.

#include "core/io/scene_io.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/node.hpp"
#include "core/compiler/codegen.hpp"
#include "core/compiler/jit_engine.hpp"
#include "core/compiler/incremental.hpp"
#include "core/compiler/llvm_compat.hpp"
#include "core/accel/guard_calibration.hpp"
#include "core/tracer/tile_scheduler.hpp"
#include "core/gpu/glsl_emitter.hpp"
#include "core/gpu/glsl_compile.hpp"
#include "core/gpu/vulkan_ctx.hpp"
#include "core/gpu/shader_push_builder.hpp"

#include <llvm/IR/Module.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Object/ObjectFile.h>
#include <llvm/Support/InitLLVM.h>
#include <llvm/Support/TargetSelect.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Support/SmallVectorMemoryBuffer.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Host.h>
#include <llvm/Transforms/Utils/Cloning.h>
#include <llvm/Passes/PassBuilder.h>

#include <chrono>
#include <cstdio>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <thread>
#include <vector>

using namespace frep;
using clk = std::chrono::steady_clock;

namespace {

std::size_t read_proc_kb(const char* field) {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind(field, 0) == 0)
            return std::strtoul(line.c_str() + std::strlen(field), nullptr, 10);
    return 0;
}
std::size_t current_rss_kb() { return read_proc_kb("VmRSS:"); }
std::size_t peak_rss_kb()    { return read_proc_kb("VmHWM:"); }

double ms_since(clk::time_point t) {
    return std::chrono::duration<double, std::milli>(clk::now() - t).count();
}

// Count IR instructions across all functions — a direct measure of code
// size, useful pre vs post opt.
std::size_t count_instructions(const llvm::Module& m) {
    std::size_t n = 0;
    for (const auto& f : m)
        for (const auto& bb : f)
            n += bb.size();
    return n;
}
std::size_t count_functions(const llvm::Module& m) {
    std::size_t n = 0;
    for (const auto& f : m) if (!f.isDeclaration()) ++n;
    return n;
}

void write_text(const std::string& path, const std::string& text) {
    std::ofstream f(path, std::ios::trunc);
    f << text;
}

std::string ir_to_string(const llvm::Module& m) {
    std::string s;
    llvm::raw_string_ostream os(s);
    m.print(os, nullptr);
    os.flush();
    return s;
}

// Run the O3 pipeline on a module in place (mirrors JitEngine::optimize at
// O3, which is the default the renderer uses).
void optimize_o3(llvm::Module& mod) {
    llvm::PassBuilder pb;
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3).run(mod, mam);
}

// Compile a module to a native object file in memory and return the total
// size of its executable (.text) sections in bytes — the actual machine
// code footprint the CPU executes, the analogue of SPIR-V size for the
// GPU. Returns {text_bytes, total_object_bytes}; {0,0} on failure.
std::pair<std::size_t, std::size_t> native_code_size(const llvm::Module& src) {
    // Clone so we can give the object emitter its own module with a target.
    auto mod = llvm::CloneModule(src);

    std::string triple = llvm::sys::getDefaultTargetTriple();
    frep::llvm_compat::set_target_triple(*mod, triple);
    std::string err;
    // LLVM 22 changed lookupTarget / createTargetMachine to take a Triple
    // object rather than a triple string (PR #130940), matching the
    // setTargetTriple change shimmed in llvm_compat. Branch on the version
    // so this compiles on both the project's LLVM 22 target and the LLVM 20
    // fallback used in CI.
    [[maybe_unused]] llvm::Triple triple_obj(triple);
#if LLVM_VERSION_MAJOR >= 22
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple_obj, err);
#else
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(triple, err);
#endif
    if (!target) return {0, 0};

    llvm::TargetOptions opt;
    auto rm = std::optional<llvm::Reloc::Model>();
#if LLVM_VERSION_MAJOR >= 22
    std::unique_ptr<llvm::TargetMachine> tm(
        target->createTargetMachine(triple_obj, "generic", "", opt, rm));
#else
    std::unique_ptr<llvm::TargetMachine> tm(
        target->createTargetMachine(triple, "generic", "", opt, rm));
#endif
    if (!tm) return {0, 0};
    mod->setDataLayout(tm->createDataLayout());

    llvm::SmallVector<char, 0> buf;
    llvm::raw_svector_ostream os(buf);
    llvm::legacy::PassManager pm;
    if (tm->addPassesToEmitFile(pm, os, nullptr,
                                llvm::CodeGenFileType::ObjectFile))
        return {0, 0};
    pm.run(*mod);

    std::size_t total = buf.size();
    std::size_t text = 0;
    // Parse the emitted object and sum executable section sizes.
    auto mb = llvm::MemoryBuffer::getMemBuffer(
        llvm::StringRef(buf.data(), buf.size()), "obj", false);
    auto obj_or = llvm::object::ObjectFile::createObjectFile(mb->getMemBufferRef());
    if (obj_or) {
        for (const auto& sec : (*obj_or)->sections())
            if (sec.isText()) text += sec.getSize();
    }
    return {text, total};
}

const char* sdf_mode_name(SceneCodegen::SceneSdfMode m) {
    switch (m) {
        case SceneCodegen::SceneSdfMode::Guarded: return "Guarded";
        case SceneCodegen::SceneSdfMode::Split:   return "Split";
        default:                                  return "Inlined";
    }
}

// Decide the SDF mode the real renderer would pick for this scene, so the
// dumped IR matches what actually runs. Mirrors IncrementalCompiler's
// pre-screen + calibration logic.
SceneCodegen::SceneSdfMode choose_mode(const SceneGraph& scene, bool guards,
                                       std::string& why) {
    if (!guards) { why = "guards disabled"; return SceneCodegen::SceneSdfMode::Inlined; }
    long total = 0; int n = 0, maxc = 0;
    for (const auto& [id, obj] : scene.objects()) {
        if (!obj.visible || !obj.geometry) continue;
        int c = node_count(*obj.geometry);
        total += c; ++n; if (c > maxc) maxc = c;
    }
    if (n < 8 || maxc < 3) { why = "scene too simple (pre-screen)"; return SceneCodegen::SceneSdfMode::Inlined; }
    auto cal = accel::get_or_calibrate(false);
    double avg = n ? (double)total / n : 0.0;
    if (accel::should_guard(cal, n, avg)) {
        why = "calibrated: avg " + std::to_string(avg) + " >= threshold "
            + std::to_string(cal.node_threshold);
        return SceneCodegen::SceneSdfMode::Guarded;
    }
    why = "calibrated: below threshold " + std::to_string(cal.node_threshold);
    return SceneCodegen::SceneSdfMode::Inlined;
}

std::string json_escape(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') { o += '\\'; o += c; }
        else if (c == '\n') o += "\\n";
        else o += c;
    }
    return o;
}

} // namespace

int main(int argc, char** argv) {
    llvm::InitLLVM init_llvm(argc, argv);
    llvm::InitializeNativeTarget();
    llvm::InitializeNativeTargetAsmPrinter();

    if (argc < 2 || std::string(argv[1]) == "--help" || std::string(argv[1]) == "-h") {
        std::printf(
            "usage: %s <scene.json> [out-basename] [options]\n"
            "  Writes analysis artifacts for a scene:\n"
            "    <base>.pre.ll / .post.ll  LLVM IR before / after O3\n"
            "    <base>.glsl / .spv        GPU shader source / SPIR-V\n"
            "    <base>.scene.json         normalised scene\n"
            "    <base>.stats.json         timings, memory, parallelism, modes\n"
            "  options:\n"
            "    --width N / --height N    render size for timing (default 800x600)\n"
            "    --paths LIST              which paths' artifacts to emit:\n"
            "                              cpu_ir/gpu_ir → .ll, gpu_glsl → .glsl/.spv\n"
            "                              (default cpu_ir,gpu_glsl)\n"
            "    --no-guards               force Inlined SDF (disable spatial guards)\n"
            "    --no-render               skip the render-timing pass\n",
            argv[0]);
        return argc < 2 ? 1 : 0;
    }

    std::string scene_path = argv[1];
    std::string base;
    int W = 800, H = 600;
    bool guards = true, do_render = true;
    std::string paths = "cpu_ir,gpu_glsl";  // which paths' artifacts to emit
    int argi = 2;
    if (argc > 2 && argv[2][0] != '-') { base = argv[2]; argi = 3; }
    for (int i = argi; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--width"  && i+1 < argc) W = std::atoi(argv[++i]);
        else if (a == "--height" && i+1 < argc) H = std::atoi(argv[++i]);
        else if (a == "--paths"  && i+1 < argc) paths = argv[++i];
        else if (a == "--no-guards") guards = false;
        else if (a == "--no-render") do_render = false;
    }
    // Path → artifact mapping: an IR path (cpu_ir / gpu_ir) emits the .ll
    // dumps; the GLSL path (gpu_glsl) emits the .glsl shader and .spv. Select
    // with --paths so the tool speaks the same vocabulary as the others.
    const bool want_ir   = paths.find("cpu_ir") != std::string::npos ||
                           paths.find("gpu_ir") != std::string::npos;
    const bool do_spirv  = paths.find("gpu_glsl") != std::string::npos;
    if (base.empty()) {
        // Derive from scene path: strip directory + extension.
        auto slash = scene_path.find_last_of("/\\");
        std::string name = (slash == std::string::npos) ? scene_path : scene_path.substr(slash+1);
        auto dot = name.find_last_of('.');
        base = (dot == std::string::npos) ? name : name.substr(0, dot);
    }

    // ── Load scene ──────────────────────────────────────────────────────────
    SceneGraph scene;
    try {
        scene = io::load_scene(scene_path, nullptr);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "Failed to load %s: %s\n", scene_path.c_str(), e.what());
        return 1;
    }

    int obj_count = 0; long node_total = 0; int node_max = 0;
    for (const auto& [id, obj] : scene.objects()) {
        if (!obj.geometry) continue;
        int c = node_count(*obj.geometry);
        ++obj_count; node_total += c; if (c > node_max) node_max = c;
    }
    double avg_nodes = obj_count ? (double)node_total / obj_count : 0.0;

    std::printf("frep_dump: %s (%d objects, avg %.1f nodes, max %d)\n",
                scene_path.c_str(), obj_count, avg_nodes, node_max);

    std::string mode_why;
    SceneCodegen::SceneSdfMode mode = choose_mode(scene, guards, mode_why);
    std::printf("  SDF mode: %s (%s)\n", sdf_mode_name(mode), mode_why.c_str());

    // ── Scene dump ──────────────────────────────────────────────────────────
    io::save_scene(scene, base + ".scene.json");

    // ── CPU path: codegen → pre-opt IR → optimise → post-opt IR ─────────────
    std::size_t rss_before = current_rss_kb();
    auto t_cg = clk::now();
    auto ctx = std::make_unique<llvm::LLVMContext>();
    TracerConfig cfg;
    SceneCodegen cg(*ctx, cfg);
    cg.emit_render_tile(scene, mode);
    double codegen_ms = ms_since(t_cg);

    llvm::Module* mod = cg.module();
    std::size_t pre_insts = count_instructions(*mod);
    std::size_t pre_funcs = count_functions(*mod);
    if (want_ir) write_text(base + ".pre.ll", ir_to_string(*mod));

    // Clone for post-opt so the pre-opt module stays intact for reference.
    auto cloned = llvm::CloneModule(*mod);
    auto t_opt = clk::now();
    optimize_o3(*cloned);
    double opt_ms = ms_since(t_opt);
    std::size_t post_insts = count_instructions(*cloned);
    std::size_t post_funcs = count_functions(*cloned);
    if (want_ir) write_text(base + ".post.ll", ir_to_string(*cloned));

    // Native (CPU machine code) size: emit the optimised module to an
    // object file and measure its .text. This is the real executable
    // footprint, the CPU analogue of SPIR-V size.
    auto [text_bytes, obj_bytes] = native_code_size(*cloned);

    std::printf("  IR: %zu insts / %zu fns  →  (O3 %.1f ms)  →  %zu insts / %zu fns  (%.1f%% of pre)\n",
                pre_insts, pre_funcs, opt_ms, post_insts, post_funcs,
                pre_insts ? 100.0 * post_insts / pre_insts : 0.0);
    std::printf("  Native code: %zu bytes .text (%zu bytes object)\n",
                text_bytes, obj_bytes);

    // ── CPU render timing + parallelism scaling ─────────────────────────────
    double jit_ms = -1, render_ms = -1;
    int threads_used = 0;
    std::vector<std::pair<int,double>> scaling;  // (threads, ms)
    {
        auto jctx = std::make_unique<llvm::LLVMContext>();
        SceneCodegen jcg(*jctx, cfg);
        jcg.emit_render_tile(scene, mode);
        auto jmod = jcg.take_module();
        JitEngine jit;
        auto t_jit = clk::now();
        auto fn = jit.load(std::move(jmod), std::move(jctx));
        jit_ms = ms_since(t_jit);
        if (fn && do_render) {
            std::vector<float> px((std::size_t)W * H * 4);
            int hw = (int)std::thread::hardware_concurrency();
            // Default render (all hardware threads) for the headline number
            // and the PNG.
            {
                RenderParams rp; rp.width = W; rp.height = H;
                threads_used = hw > 0 ? hw : 1;
                auto t_r = clk::now();
                TileScheduler::render(*fn, px.data(), scene.camera(), rp);
                render_ms = ms_since(t_r);
            }
            // Parallelism scaling sweep: 1, 2, 4, … up to hardware threads.
            // Shows how the tiled renderer scales across cores — a core
            // metric for the heterogeneous-execution work to come.
            for (int t = 1; t <= hw; t *= 2) {
                RenderParams rp; rp.width = W; rp.height = H; rp.num_threads = t;
                double best = 1e30;
                for (int r = 0; r < 2; ++r) {
                    auto tr = clk::now();
                    TileScheduler::render(*fn, px.data(), scene.camera(), rp);
                    double m = ms_since(tr); best = m < best ? m : best;
                }
                scaling.emplace_back(t, best);
            }
            if (hw > 1 && (scaling.empty() || scaling.back().first != hw)) {
                RenderParams rp; rp.width = W; rp.height = H; rp.num_threads = hw;
                auto tr = clk::now();
                TileScheduler::render(*fn, px.data(), scene.camera(), rp);
                scaling.emplace_back(hw, ms_since(tr));
            }
            // Final rendered image as a visual reference next to the code
            // artifacts. PPM (binary P6) — the format the other render
            // tools use, no extra deps. Convert float RGBA → 8-bit RGB.
            {
                std::ofstream f(base + ".render.ppm", std::ios::binary | std::ios::trunc);
                f << "P6\n" << W << " " << H << "\n255\n";
                for (int i = 0; i < W * H; ++i) {
                    for (int c = 0; c < 3; ++c) {
                        float v = px[(std::size_t)i * 4 + c];
                        v = v < 0 ? 0 : (v > 1 ? 1 : v);
                        unsigned char b = (unsigned char)(v * 255.0f + 0.5f);
                        f.put((char)b);
                    }
                }
            }
        }
    }
    std::size_t rss_after = current_rss_kb();
    std::size_t rss_delta = rss_after > rss_before ? rss_after - rss_before : 0;

    // ── GPU path: GLSL → SPIR-V ─────────────────────────────────────────────
    double glsl_ms = -1, spv_ms = -1;
    std::size_t glsl_len = 0, spv_bytes = 0;
    bool spirv_ok = false;
    std::string spirv_err;
    {
        auto t_g = clk::now();
        auto glsl_res = gpu::GlslEmitter::emit(scene, cfg);
        glsl_ms = ms_since(t_g);
        if (!glsl_res) {
            std::fprintf(stderr, "GLSL emit failed: %s\n", glsl_res.error().c_str());
            // Still write the stats we have; GLSL/SPIR-V left empty.
        } else {
            const std::string& glsl = glsl_res->source;
            glsl_len = glsl.size();
            if (do_spirv) write_text(base + ".glsl", glsl);

            if (do_spirv) {
                auto t_s = clk::now();
                auto spv_path_or = gpu::compile_glsl_to_spv(glsl);
                spv_ms = ms_since(t_s);
                if (spv_path_or) {
                    // compile_glsl_to_spv returns the PATH to the .spv file;
                    // read its bytes and copy to our output basename.
                    std::ifstream in(*spv_path_or, std::ios::binary);
                    std::string bytes((std::istreambuf_iterator<char>(in)),
                                       std::istreambuf_iterator<char>());
                    if (!bytes.empty()) {
                        spirv_ok = true;
                        spv_bytes = bytes.size();
                        std::ofstream f(base + ".spv", std::ios::binary | std::ios::trunc);
                        f.write(bytes.data(), (std::streamsize)bytes.size());
                    } else {
                        spirv_err = "compiled .spv was empty";
                    }
                    std::error_code ec;
                    std::filesystem::remove(*spv_path_or, ec);  // clean temp
                } else {
                    spirv_err = spv_path_or.error();
                }
            }
        }
    }

    std::size_t peak = peak_rss_kb();

    // ── Optional: real GPU init breakdown (if Vulkan is available) ──────────
    // Mirrors the benchmark's per-phase init timing. Skipped silently in
    // environments without a real Vulkan device (e.g. CI/software-only).
    bool gpu_ran = false;
    double g_device = 0, g_shader = 0, g_pipeline = 0, g_buffers = 0, g_misc = 0;
    double g_init_total = 0, g_render = 0;
    std::size_t g_device_kb = 0;
    if (do_render && do_spirv && gpu::VulkanCtx::available()) {
        auto e = gpu::GlslEmitter::emit(scene, cfg);
        if (e) {
            auto spvm = gpu::compile_glsl_to_spv_managed(e->source);
            if (spvm) {
                auto ctx_or = gpu::VulkanCtx::create(
                    spvm->path(), e->mesh_voxels, e->texture_pixels);
                if (ctx_or) {
                    auto& gctx = **ctx_or;
                    auto pushc = gpu::build_push_from_scene(scene, W, H);
                    std::vector<std::uint8_t> gpx;
                    auto tg = clk::now();
                    gctx.render(pushc, gpx);
                    g_render = ms_since(tg);
                    const auto& st = gctx.stats();
                    g_init_total = st.init_ms;
                    g_device = st.init_device_ms;
                    g_shader = st.init_shader_ms;
                    g_pipeline = st.init_pipeline_ms;
                    g_buffers = st.init_buffers_ms;
                    g_misc = st.init_misc_ms;
                    std::size_t dev = (std::size_t)W * H * 4
                        + e->mesh_voxels.size() * sizeof(float)
                        + e->texture_pixels.size();
                    g_device_kb = dev / 1024;
                    gpu_ran = true;
                }
            }
        }
    }

    // ── stats.json ──────────────────────────────────────────────────────────
    {
        std::ofstream s(base + ".stats.json", std::ios::trunc);
        s << "{\n";
        s << "  \"scene\": \"" << json_escape(scene_path) << "\",\n";
        s << "  \"object_count\": " << obj_count << ",\n";
        s << "  \"avg_node_count\": " << avg_nodes << ",\n";
        s << "  \"max_node_count\": " << node_max << ",\n";
        s << "  \"sdf_mode\": \"" << sdf_mode_name(mode) << "\",\n";
        s << "  \"sdf_mode_reason\": \"" << json_escape(mode_why) << "\",\n";
        s << "  \"render_size\": [" << W << ", " << H << "],\n";
        s << "  \"cpu\": {\n";
        s << "    \"codegen_ms\": " << codegen_ms << ",\n";
        s << "    \"optimize_ms\": " << opt_ms << ",\n";
        s << "    \"jit_total_ms\": " << jit_ms << ",\n";
        s << "    \"render_ms\": " << render_ms << ",\n";
        s << "    \"threads_used\": " << threads_used << ",\n";
        s << "    \"native_text_bytes\": " << text_bytes << ",\n";
        s << "    \"native_object_bytes\": " << obj_bytes << ",\n";
        s << "    \"ir_pre_insts\": " << pre_insts << ",\n";
        s << "    \"ir_post_insts\": " << post_insts << ",\n";
        s << "    \"ir_pre_funcs\": " << pre_funcs << ",\n";
        s << "    \"ir_post_funcs\": " << post_funcs << ",\n";
        s << "    \"ir_post_pct_of_pre\": "
          << (pre_insts ? 100.0 * post_insts / pre_insts : 0.0) << ",\n";
        s << "    \"thread_scaling\": [";
        for (std::size_t i = 0; i < scaling.size(); ++i) {
            if (i) s << ", ";
            s << "[" << scaling[i].first << ", " << scaling[i].second << "]";
        }
        s << "]\n";
        s << "  },\n";
        s << "  \"gpu\": {\n";
        s << "    \"glsl_emit_ms\": " << glsl_ms << ",\n";
        s << "    \"glsl_bytes\": " << glsl_len << ",\n";
        s << "    \"spirv_compile_ms\": " << spv_ms << ",\n";
        s << "    \"spirv_bytes\": " << spv_bytes << ",\n";
        s << "    \"spirv_ok\": " << (spirv_ok ? "true" : "false") << ",\n";
        s << "    \"ran_on_device\": " << (gpu_ran ? "true" : "false");
        if (gpu_ran) {
            s << ",\n";
            s << "    \"init_total_ms\": " << g_init_total << ",\n";
            s << "    \"init_device_ms\": " << g_device << ",\n";
            s << "    \"init_shader_ms\": " << g_shader << ",\n";
            s << "    \"init_pipeline_ms\": " << g_pipeline << ",\n";
            s << "    \"init_buffers_ms\": " << g_buffers << ",\n";
            s << "    \"init_misc_ms\": " << g_misc << ",\n";
            s << "    \"render_ms\": " << g_render << ",\n";
            s << "    \"device_kb\": " << g_device_kb;
        }
        if (!spirv_ok && !spirv_err.empty())
            s << ",\n    \"spirv_error\": \"" << json_escape(spirv_err) << "\"";
        s << "\n  },\n";
        s << "  \"memory\": {\n";
        s << "    \"rss_delta_kb\": " << rss_delta << ",\n";
        s << "    \"peak_rss_kb\": " << peak << ",\n";
        s << "    \"hardware_threads\": " << std::thread::hardware_concurrency() << "\n";
        s << "  }\n";
        s << "}\n";
    }

    std::printf("  GPU: GLSL %zu bytes (%.1f ms)%s\n", glsl_len, glsl_ms,
                spirv_ok ? "" : (do_spirv ? "  [SPIR-V failed]" : "  [SPIR-V skipped]"));
    if (spirv_ok)
        std::printf("       SPIR-V %zu bytes (%.1f ms)\n", spv_bytes, spv_ms);
    if (do_render)
        std::printf("  CPU render %.1f ms on %d threads;  peak RSS %zu KB\n",
                    render_ms, threads_used, peak);
    std::printf("  wrote: %s.{%s%sscene.json, stats.json}\n", base.c_str(),
                want_ir ? "pre.ll, post.ll, " : "",
                do_spirv ? (spirv_ok ? "glsl, spv, " : "glsl, ") : "");
    return 0;
}
