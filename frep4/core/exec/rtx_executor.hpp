// core/exec/rtx_executor.hpp
//
// IExecutor for the GpuRtx path.
//
// The RTX path renders the *same* implicit F-Rep scene as the other three. A
// hardware acceleration structure does broad-phase — which object's AABB does a
// ray enter — and a custom intersection shader sphere-traces the exact SDF
// inside that AABB. The surface stays implicit, so the output is bit-comparable
// with cpu_ir / gpu_ir / gpu_glsl at the shared-IR floor.
//
// Independent objects are grouped CSG-aware and each gets its own BLAS, so the
// RT cores cull groups a ray misses (real broad-phase) rather than running the
// full O(N) scene_sdf for every ray. Full feature parity (17/17 scenes) is
// hardware-confirmed on an RTX 2080.
//
// Hardware: real RT cores from Turing (RTX 2080). On a card without them
// (Pascal / GTX 1050 Ti) or a host with no Vulkan device, available() still
// reports true when a software fallback is selected, and render() uses a
// software BVH walk instead — same intersection semantics, no RT cores.

#pragma once

#include "core/exec/multipath.hpp"
#include "core/frep/scene.hpp"
#include "core/gpu/rtx_caps.hpp"
#include "core/gpu/rtx_ctx.hpp"
#include "core/gpu/rtx_accel.hpp"
#include "core/gpu/rtx_shaders.hpp"
#include "core/gpu/rtx_csg_groups.hpp"
#include "core/gpu/rtx_pipeline.hpp"
#include "core/gpu/glsl_compile.hpp"
#include "core/frep/operations.hpp"
#include "core/gpu/shader_push_builder.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

namespace frep::exec {

class RtxExecutor : public IExecutor {
public:
    // allow_software: when true (default), a machine without RT cores still
    // runs the path via the software BVH-walk fallback (with a warning the
    // caller can surface). When false, the path is only available on real RT
    // hardware — useful for benchmarks that must measure RT cores specifically.
    explicit RtxExecutor(const TracerConfig& cfg = {}, bool allow_software = true)
        : cfg_(cfg), allow_software_(allow_software),
          caps_(gpu::detect_rtx_caps()) {}

    PathKind path() const override { return PathKind::GpuRtx; }

    bool available() const override {
        if (caps_.backend == gpu::RtxBackend::Hardware) return true;
        if (caps_.backend == gpu::RtxBackend::Software) return allow_software_;
        return false;  // None
    }

    // True only when the path is running on real RT cores (not the fallback).
    bool using_hardware() const { return caps_.hardware(); }

    const gpu::RtxCaps& caps() const { return caps_; }

    TileResult render(const SceneGraph& scene, int W, int H,
                      const Tile& tile) override {
        TileResult r;
        r.ok = false;
        if (!available()) {
            r.error = "gpu_rtx: " + caps_.describe();
            return r;
        }
        // Env-gated wall breakdown (FREP_RTX_PROFILE=1): where does render()'s
        // wall-clock go on a warm call? rebuild = shader/AS rebuild block,
        // trace = the rtx_trace_*_cached call, crop = full-frame → tile copy.
        const bool prof = [] { const char* v = std::getenv("FREP_RTX_PROFILE");
                               return v && v[0] == '1'; }();
        using pclk = std::chrono::steady_clock;
        auto pt0 = pclk::now();

        // 1. RT device — created once and kept, so the pipeline cache's GPU
        //    objects (which belong to this device) stay valid across frames.
        if (!ctx_) {
            auto ctx = gpu::RtxCtx::create();
            if (!ctx) { r.error = "gpu_rtx: " + ctx.error(); return r; }
            ctx_ = std::make_unique<gpu::RtxCtx>(std::move(*ctx));
        }

        // 2–4. Shaders + SPIR-V + acceleration structure depend only on the
        //    scene, not the camera. Rebuild them only when the scene changes;
        //    a camera move (the common interactive case) reuses everything and
        //    the pipeline cache then makes the trace a warm-cache hit.
        std::uint64_t skey = scene_key(scene, W, H);
        if (skey != scene_key_ || !accel_) {
            // Scene (or frame size) changed → rebuild shaders, SPIR-V, AS, and
            // drop the stale pipeline cache (different shaders).
            cache_.release(*ctx_);

            auto compile = [](const std::string& src, const char* stage,
                              std::vector<std::uint32_t>& out) -> std::string {
                auto spv = gpu::compile_rt_stage_to_spv(src, stage);
                if (!spv) return spv.error();
                std::ifstream f(*spv, std::ios::binary | std::ios::ate);
                if (!f) return std::string("cannot read ") + *spv;
                auto n = f.tellg(); f.seekg(0);
                out.resize((std::size_t)n / sizeof(std::uint32_t));
                f.read(reinterpret_cast<char*>(out.data()), n);
                std::remove(spv->c_str());
                return {};
            };

            rgen_.clear(); rint_.clear(); rchit_.clear(); rmiss_.clear();
            rints_group_.clear();
            group_mode_ = false;

            // Partition the scene at hard unions into CSG-independent groups.
            // With >1 group we build a multi-BLAS TLAS so the RT cores do real
            // broad-phase (culling groups a ray misses) instead of running the
            // whole scene_sdf per ray — the case where RTX beats the compute
            // paths on many-object scenes. A single group falls through to the
            // one-AABB path below.
            std::vector<FRepNode::Ptr> geoms;
            for (auto& [id, obj] : scene.objects())
                if (obj.visible && obj.geometry) geoms.push_back(obj.geometry);
            FRepNode::Ptr root = geoms.empty() ? nullptr : union_all(geoms);
            auto groups = root ? gpu::partition_csg_groups(root)
                               : std::vector<gpu::CsgGroup>{};

            if (groups.size() > 1) {
                std::vector<SceneGraph> gscenes;
                for (auto& g : groups) {
                    SceneGraph s;
                    s.add_object(g.root);
                    s.templates() = scene.templates();  // template defs for the group shader
                    gscenes.push_back(std::move(s));
                }
                auto gs = gpu::emit_rt_group_shaders(scene, gscenes, cfg_);
                if (gs) {   // mesh/texture scenes are rejected -> single-BLAS fallback
                    std::string err = compile(gs->rgen, "rgen", rgen_);
                    if (err.empty()) err = compile(gs->rchit, "rchit", rchit_);
                    if (err.empty()) err = compile(gs->rmiss, "rmiss", rmiss_);
                    rints_group_.resize(gs->rint_per_group.size());
                    for (std::size_t i = 0; err.empty() && i < rints_group_.size(); ++i)
                        err = compile(gs->rint_per_group[i], "rint", rints_group_[i]);
                    if (err.empty()) {
                        std::vector<gpu::RtAabb> boxes;
                        const float m = 0.05f;
                        for (auto& g : groups)
                            boxes.push_back(gpu::RtAabb{
                                {g.box.min_x - m, g.box.min_y - m, g.box.min_z - m},
                                {g.box.max_x + m, g.box.max_y + m, g.box.max_z + m}});
                        auto accel = gpu::RtAccel::build_groups(*ctx_, boxes);
                        if (accel) {
                            accel_ = std::make_unique<gpu::RtAccel>(std::move(*accel));
                            tex_pixels_.clear(); mesh_voxels_.clear();
                            group_mode_ = true;
                        }
                    }
                }
                if (!group_mode_) {  // group path failed → clean up, use single
                    rgen_.clear(); rchit_.clear(); rmiss_.clear(); rints_group_.clear();
                }
            }

            if (!group_mode_) {
                auto shaders = gpu::emit_rt_shaders(scene, cfg_);
                if (!shaders) { r.error = "gpu_rtx: " + shaders.error(); return r; }
                std::string err = compile(shaders->rgen, "rgen", rgen_);
                if (err.empty()) err = compile(shaders->rint,  "rint",  rint_);
                if (err.empty()) err = compile(shaders->rchit, "rchit", rchit_);
                if (err.empty()) err = compile(shaders->rmiss, "rmiss", rmiss_);
                if (!err.empty()) { r.error = "gpu_rtx: compile: " + err; return r; }
                tex_pixels_ = shaders->texture_pixels;
                mesh_voxels_ = shaders->mesh_voxels;

                FRepNode::AABB box = scene_bounds(scene);
                const float m = 0.05f;
                box.min_x -= m; box.min_y -= m; box.min_z -= m;
                box.max_x += m; box.max_y += m; box.max_z += m;
                auto accel = gpu::RtAccel::build_whole_scene(*ctx_, box);
                if (!accel) { r.error = "gpu_rtx: " + accel.error(); return r; }
                accel_ = std::make_unique<gpu::RtAccel>(std::move(*accel));
            }

            scene_key_ = skey;
        }
        auto pt_rebuild = pclk::now();

        // 5. Push constants — rebuilt every frame (camera/lights change).
        gpu::ShaderPush sp = gpu::build_push_from_scene(scene, W, H);
        gpu::RtPushConstants pc;
        static_assert(sizeof(pc) == sizeof(sp),
                      "RtPushConstants must match ShaderPush layout");
        std::memcpy(&pc, &sp, sizeof(pc));

        // 6. Trace full frame. Both paths reuse the cached pipeline + SBT via
        //    cache_ (warm → pipeline_ms ≈ 0, only trace + readback recur), so an
        //    interactive camera move pays only the trace.
        auto img = group_mode_
            ? gpu::rtx_trace_groups_cached(*ctx_, *accel_, cache_,
                                           rgen_, rints_group_, rchit_, rmiss_,
                                           pc, W, H)
            : gpu::rtx_trace_cached(*ctx_, *accel_, cache_,
                                    rgen_, rint_, rchit_, rmiss_,
                                    pc, W, H, tex_pixels_, mesh_voxels_);
        if (!img) { r.error = "gpu_rtx: " + img.error(); return r; }
        auto pt_trace = pclk::now();

        // 7. Crop to the requested tile (full-frame render for now). When the
        //    tile covers the whole frame — the common case — move the buffer
        //    instead of copying it (a 512² frame is a 4 MB memcpy otherwise, ~1 ms).
        int tw = tile.x1 - tile.x0, th = tile.y1 - tile.y0;
        if (tile.x0 == 0 && tile.y0 == 0 && tw == W && th == H) {
            r.rgba = std::move(img->rgba);
        } else {
            r.rgba.resize((std::size_t)tw * th * 4);
            for (int y = 0; y < th; ++y)
                for (int x = 0; x < tw; ++x) {
                    int sx = tile.x0 + x, sy = tile.y0 + y;
                    const float* s = &img->rgba[((std::size_t)sy * W + sx) * 4];
                    float* d = &r.rgba[((std::size_t)y * tw + x) * 4];
                    d[0] = s[0]; d[1] = s[1]; d[2] = s[2]; d[3] = s[3];
                }
        }
        // Setup (pipeline + SBT) is amortizable across frames; only trace is the
        // recurring per-frame GPU cost. Expose both so the paper can separate
        // them rather than reporting one conflated number.
        r.compile_ms = img->pipeline_ms;
        r.render_ms  = img->trace_ms;
        r.ok = true;
        if (prof) {
            auto pt_crop = pclk::now();
            auto d = [](pclk::time_point a, pclk::time_point b) {
                return std::chrono::duration<double, std::milli>(b - a).count();
            };
            std::fprintf(stderr,
                "[rtx-prof %s] wall=%.2f  rebuild=%.2f  trace_call=%.2f  crop=%.2f"
                "   | inside trace: setup(pipeline_ms)=%.2f trace(trace_ms)=%.2f "
                "readback=%.2f\n",
                group_mode_ ? "groups" : "single",
                d(pt0, pt_crop), d(pt0, pt_rebuild), d(pt_rebuild, pt_trace),
                d(pt_trace, pt_crop),
                img->pipeline_ms, img->trace_ms, img->readback_ms);
        }
        return r;
    }

private:
    // Whole-scene bounding box (union of every object's AABB).
    static FRepNode::AABB scene_bounds(const SceneGraph& scene) {
        bool first = true;
        FRepNode::AABB box{-1, -1, -1, 1, 1, 1};
        for (const auto& kv : scene.objects()) {
            if (!kv.second.geometry) continue;
            auto b = kv.second.geometry->aabb();
            if (first) { box = b; first = false; }
            else box = FRepNode::AABB::merge(box, b);
        }
        return box;
    }

    // A cheap fingerprint of the scene + frame size, to detect when shaders/AS
    // must be rebuilt (vs a camera-only change, which reuses them). Hashes each
    // object's geometry pointer + material + the W/H; a structural edit (add/
    // remove/retype a node) or a resize changes it, a camera move doesn't.
    static std::uint64_t scene_key(const SceneGraph& scene, int W, int H) {
        std::uint64_t h = 1469598103934665603ull;
        auto mix = [&](std::uint64_t v) { h ^= v; h *= 1099511628211ull; };
        mix((std::uint64_t)W); mix((std::uint64_t)H);
        for (const auto& kv : scene.objects()) {
            mix(std::hash<const void*>{}(kv.second.geometry.get()));
            mix(kv.second.visible ? 1u : 0u);
            // Material affects the emitted albedo/shade, so fold it in.
            const auto& m = kv.second.material;
            mix(std::hash<float>{}(m.albedo[0] + m.albedo[1] + m.albedo[2]));
            mix(std::hash<float>{}(m.roughness));
            mix(std::hash<float>{}(m.metallic));
        }
        return h;
    }

private:
    TracerConfig  cfg_;
    bool          allow_software_;
    gpu::RtxCaps  caps_;

    // Persistent GPU state, reused across frames. Declared so the cache is torn
    // down BEFORE the device (destructor runs members in reverse order, but the
    // cache needs the device to free its objects → explicit dtor below).
    std::unique_ptr<gpu::RtxCtx>   ctx_;
    std::unique_ptr<gpu::RtAccel>  accel_;
    gpu::RtxPipelineCache          cache_;
    std::vector<std::uint32_t>     rgen_, rint_, rchit_, rmiss_;
    std::vector<std::vector<std::uint32_t>> rints_group_;  // per-group rint SPIR-V
    bool                           group_mode_ = false;    // multi-BLAS this scene
    std::vector<std::uint32_t>     tex_pixels_;
    std::vector<float>             mesh_voxels_;
    std::uint64_t                  scene_key_ = 0;

public:
    ~RtxExecutor() override {
        // Free cached pipeline/SBT/etc. while the device is still alive.
        if (ctx_) cache_.release(*ctx_);
    }
};

}  // namespace frep::exec
