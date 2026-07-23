// tools/validate_runtime_params.cpp
//
// Validates the opt-in unified-placement fast path on GPU_GLSL:
//   1. With an interactive policy, editing a runtime parameter leaves the
//      emitted shader BYTE-IDENTICAL  -> no re-emit / no SPIR-V recompile.
//   2. The shader actually reads the runtime buffer (binding 3 / P.v[]).
//   3. Baseline contrast: with NO policy the same edit changes the baked
//      shader  -> would force a full recompile.
//   4. GPU correctness (needs a Vulkan device): the buffer-path image equals
//      the full-recompile image, and a runtime edit visibly changes the frame.
//
// Run with the software rasterizer:
//   VK_DRIVER_FILES=/usr/share/vulkan/icd.d/lvp_icd.json ./build/frep_validate_runtime_params

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/gpu/glsl_emitter.hpp"
#include "core/gpu/vulkan_ctx.hpp"
#include "core/compiler/scene_bindings.hpp"
#include "core/compiler/compile_policy.hpp"
#include "core/exec/gpu_executor.hpp"
#include "core/exec/multipath.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <memory>

using namespace frep;

static double max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    double m = 0.0;
    std::size_t n = std::min(a.size(), b.size());
    for (std::size_t i = 0; i < n; ++i) m = std::max(m, std::fabs((double)a[i] - (double)b[i]));
    return m;
}

int main() {
    int fails = 0;
    auto check = [&](bool ok, const char* msg) {
        std::printf("[%s] %s\n", ok ? "PASS" : "FAIL", msg);
        fails += ok ? 0 : 1;
    };

    ByParamClassPolicy pol = ByParamClassPolicy::interactive();

    auto make_scene = [](float r) {
        SceneGraph s;
        s.add_object(std::make_shared<SphereNode>(r, "ball"));
        s.add_object(std::make_shared<PlaneNode>(0.0f, 1.0f, 0.0f, 1.0f, "floor"));
        return s;
    };

    // ── 1 & 2: shader invariance + buffer usage (CPU only) ───────────────────
    SceneGraph s1 = make_scene(1.0f);
    SceneGraph s2 = make_scene(0.6f);
    auto bt1 = build_bindings(s1, pol);
    auto bt2 = build_bindings(s2, pol);
    auto e1  = gpu::GlslEmitter::emit(s1, {}, &bt1);
    auto e2  = gpu::GlslEmitter::emit(s2, {}, &bt2);

    bool emitted = e1.has_value() && e2.has_value();
    check(emitted, "emit succeeds with an interactive policy");
    if (emitted) {
        check(e1->source == e2->source,
              "runtime-param edit leaves the GPU-GLSL shader byte-identical (no recompile)");
        bool uses_buf = e1->source.find("binding = 3") != std::string::npos &&
                        e1->source.find("P.v[")        != std::string::npos;
        check(uses_buf, "shader reads the runtime buffer (binding 3 / P.v[])");
        std::printf("       runtime parameter count = %zu, placement_hash = %zu\n",
                    bt1.runtime_count(), (std::size_t)e1->placement_hash);
    }

    // ── 3: baseline contrast — no policy bakes the value ─────────────────────
    auto n1 = gpu::GlslEmitter::emit(s1);
    auto n2 = gpu::GlslEmitter::emit(s2);
    check(n1 && n2 && n1->source != n2->source,
          "baseline (no policy): the same edit changes the baked shader (would recompile)");

    // ── 4: GPU correctness via the executor (uses the real camera) ───────────
    if (gpu::VulkanCtx::available()) {
        const int W = 128, H = 96;
        exec::Tile full{0, 0, W, H};
        TracerConfig cfg{};

        // Shared mutable scene so a parameter edit is picked up on re-render.
        auto ball = std::make_shared<SphereNode>(1.0f, "ball");
        SceneGraph sd;
        sd.add_object(ball);
        sd.add_object(std::make_shared<PlaneNode>(0.0f, 1.0f, 0.0f, 1.0f, "floor"));

        exec::GpuGlslExecutor ex(cfg);
        ex.set_compile_policy(&pol);
        auto ra = ex.render(sd, W, H, full);          // 1st frame: emit + SPIR-V build
        ball->params["r"] = 0.6f;                      // runtime parameter edit
        auto rb = ex.render(sd, W, H, full);           // fast path: buffer re-upload only

        exec::GpuGlslExecutor ref(cfg);                // fresh, NO policy
        auto rc = ref.render(sd, W, H, full);          // full-recompile baseline (r = 0.6)

        check(ra.ok && rb.ok && rc.ok, "all three GPU renders succeeded");
        std::printf("       compile_ms: first=%.2f  after-edit=%.2f  (after-edit should be << first)\n",
                    ra.compile_ms, rb.compile_ms);

        double d_bc = max_abs_diff(rb.rgba, rc.rgba);
        check(d_bc < 0.02,
              "buffer-path image equals the full-recompile image");
        std::printf("       max|Δ| (buffer-path vs full-recompile) = %.5f\n", d_bc);

        double d_ab = max_abs_diff(ra.rgba, rb.rgba);
        check(d_ab > 0.01, "the runtime edit visibly changed the rendered frame");
        std::printf("       max|Δ| (before vs after edit)          = %.5f\n", d_ab);
    } else {
        std::printf("[skip] no Vulkan device — GPU correctness checks skipped\n");
    }

    std::printf("\n%s (%d failure%s)\n",
                fails == 0 ? "ALL CHECKS PASSED" : "SOME CHECKS FAILED",
                fails, fails == 1 ? "" : "s");
    return fails ? 1 : 0;
}
