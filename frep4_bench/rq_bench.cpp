// rq_bench — the ray-query test. For a single-primitive scene it renders the
// SAME frame two ways against the SAME acceleration structure and compares the
// per-frame GPU trace time:
//
//   SBT   : VK_KHR_ray_tracing_pipeline — the sphere-march runs in an
//           intersection shader (rtx_shaders/rtx_pipeline).
//   query : VK_KHR_ray_query — the sphere-march runs inline in a plain compute
//           shader; rayQueryEXT does only the hardware BVH broad phase.
//
// Hypothesis under test: hardware ray tracing shouldn't be slower than a compute
// march. The SBT path measured ~5x the GLSL compute march on a single primitive
// because the long march sits in the intersection stage. Ray query moves the
// march back to compute while keeping the hardware broad phase, so its trace
// time should land near the GLSL march (~1x), not the SBT path (~5x).
//
// It also reports max per-pixel RGB difference between the two frames, so a
// speed win that broke the image is caught.
#include "core/io/scene_io.hpp"
#include "core/frep/operations.hpp"
#include "core/gpu/rtx_ctx.hpp"
#include "core/gpu/rtx_accel.hpp"
#include "core/gpu/rtx_shaders.hpp"
#include "core/gpu/rtx_pipeline.hpp"
#include "core/gpu/rtx_query.hpp"
#include "core/gpu/glsl_compile.hpp"
#include "core/gpu/shader_push_builder.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

using namespace frep;

// Compile an RT/compute stage source to SPIR-V words (mirrors rtx_executor).
static std::string compile_stage(const std::string& src, const char* stage,
                                 std::vector<std::uint32_t>& out) {
    auto spv = gpu::compile_rt_stage_to_spv(src, stage);
    if (!spv) return spv.error();
    std::FILE* f = std::fopen(spv->c_str(), "rb");
    if (!f) return "cannot read " + *spv;
    std::fseek(f, 0, SEEK_END); long n = std::ftell(f); std::fseek(f, 0, SEEK_SET);
    out.resize((std::size_t)n / sizeof(std::uint32_t));
    if (std::fread(out.data(), 1, (size_t)n, f) != (size_t)n) { std::fclose(f); return "short read"; }
    std::fclose(f);
    std::remove(spv->c_str());
    return {};
}

static double median(std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v.empty() ? 0.0 : v[v.size() / 2];
}

static double max_rgb_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0;
    std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) if ((i & 3) != 3)  // skip alpha
        m = std::max(m, (double)std::fabs(a[i] - b[i]));
    return m;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <scene.json> [res ...]\n", argv[0]);
        return 2;
    }
    std::string scene_path = argv[1];
    std::vector<int> res;
    for (int i = 2; i < argc; ++i) res.push_back(std::atoi(argv[i]));
    if (res.empty()) res = {128, 256, 512, 1024};

    SceneGraph scene;
    try { scene = io::load_scene(scene_path); }
    catch (const std::exception& e) { std::fprintf(stderr, "load: %s\n", e.what()); return 1; }

    auto ctxr = gpu::RtxCtx::create();
    if (!ctxr) { std::fprintf(stderr, "rtx ctx: %s\n", ctxr.error().c_str()); return 1; }
    gpu::RtxCtx ctx = std::move(*ctxr);
    std::printf("device: %s   ray_query=%s\n", ctx.device_name().c_str(),
                ctx.has_ray_query() ? "yes" : "NO");
    if (!ctx.has_ray_query()) {
        std::fprintf(stderr, "device lacks VK_KHR_ray_query — cannot run the test\n");
        return 1;
    }

    TracerConfig cfg;

    // Whole-scene single box → one BLAS. Both paths trace this SAME TLAS.
    FRepNode::AABB box{-1, -1, -1, 1, 1, 1};
    bool first = true;
    for (auto& [id, o] : scene.objects()) {
        if (!o.geometry) continue;
        auto b = o.geometry->aabb();
        if (first) { box = b; first = false; } else box = FRepNode::AABB::merge(box, b);
    }
    const float m = 0.05f;
    box.min_x -= m; box.min_y -= m; box.min_z -= m;
    box.max_x += m; box.max_y += m; box.max_z += m;
    auto accelr = gpu::RtAccel::build_whole_scene(ctx, box);
    if (!accelr) { std::fprintf(stderr, "accel: %s\n", accelr.error().c_str()); return 1; }
    gpu::RtAccel accel = std::move(*accelr);

    // Compile both shader sets once.
    auto sbt = gpu::emit_rt_shaders(scene, cfg);
    if (!sbt) { std::fprintf(stderr, "sbt emit: %s\n", sbt.error().c_str()); return 1; }
    std::vector<std::uint32_t> rgen, rint, rchit, rmiss;
    std::string e = compile_stage(sbt->rgen, "rgen", rgen);
    if (e.empty()) e = compile_stage(sbt->rint,  "rint",  rint);
    if (e.empty()) e = compile_stage(sbt->rchit, "rchit", rchit);
    if (e.empty()) e = compile_stage(sbt->rmiss, "rmiss", rmiss);
    if (!e.empty()) { std::fprintf(stderr, "sbt compile: %s\n", e.c_str()); return 1; }

    auto qsrc = gpu::emit_ray_query_compute(scene, cfg);
    if (!qsrc) { std::fprintf(stderr, "rq emit: %s\n", qsrc.error().c_str()); return 1; }
    std::vector<std::uint32_t> comp;
    e = compile_stage(*qsrc, "comp", comp);
    if (!e.empty()) { std::fprintf(stderr, "rq compile: %s\n", e.c_str()); return 1; }

    std::printf("\nscene: %s\n", scene_path.c_str());
    std::printf("  (b = per-frame pipeline/SBT BUILD ms; trace = submit->fence ms / ns per ray)\n");
    std::printf("%6s | %-21s | %-21s | %6s | %s\n",
                "res", "SBT  ms/ns  build", "query ms/ns  build",
                "ratio", "max|dRGB|");
    std::printf("-------+-----------------------+-----------------------+--------+---------\n");

    const int WARM = 3, ITERS = 9;
    for (int R : res) {
        gpu::ShaderPush sp = gpu::build_push_from_scene(scene, R, R);
        gpu::RtPushConstants pc;
        std::memcpy(&pc, &sp, sizeof(pc));

        std::vector<float> sbt_img, q_img;
        std::vector<double> sbt_t, q_t, sbt_b, q_b;

        for (int i = 0; i < WARM + ITERS; ++i) {
            auto rs = gpu::rtx_trace(ctx, accel, rgen, rint, rchit, rmiss, pc, R, R);
            if (!rs) { std::fprintf(stderr, "sbt trace: %s\n", rs.error().c_str()); return 1; }
            if (i >= WARM) { sbt_t.push_back(rs->trace_ms); sbt_b.push_back(rs->pipeline_ms); }
            if (i == WARM) sbt_img = rs->rgba;
        }
        for (int i = 0; i < WARM + ITERS; ++i) {
            auto rq = gpu::rtx_query_trace(ctx, accel, comp, pc, R, R);
            if (!rq) { std::fprintf(stderr, "rq trace: %s\n", rq.error().c_str()); return 1; }
            if (i >= WARM) { q_t.push_back(rq->trace_ms); q_b.push_back(rq->pipeline_ms); }
            if (i == WARM) q_img = rq->rgba;
        }

        double sms = median(sbt_t), qms = median(q_t);
        double sb = median(sbt_b), qb = median(q_b);
        double rays = (double)R * R;
        double s_ns = sms * 1e6 / rays, q_ns = qms * 1e6 / rays;
        double diff = max_rgb_diff(sbt_img, q_img);
        std::printf("%5d² | %6.3f/%6.1f b%6.2f | %6.3f/%6.1f b%6.2f | %5.2fx | %.5f\n",
                    R, sms, s_ns, sb, qms, q_ns, qb, qms > 0 ? sms / qms : 0.0, diff);
        std::fflush(stdout);
    }
    return 0;
}
