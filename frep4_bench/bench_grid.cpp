// frep4 grid-eval: JIT the whole-scene SDF and evaluate an N^3 grid.
// Uses the SIMD (8-lane) path when the scene is a single CustomExprNode,
// else falls back to the scalar fn. Inner axis (z) is vectorized.
#include "core/compiler/compile_sdf.hpp"
#include "core/io/scene_io.hpp"
#include "../common/timing.hpp"
#include "../common/field_dump.hpp"
#include <cstdio>
#include <cstring>
#include <string>
#include <functional>
#include <vector>
using namespace frep;

static std::string scene_stem(std::string p) {
    auto sl = p.find_last_of('/'); if (sl != std::string::npos) p = p.substr(sl + 1);
    auto d = p.find_last_of('.');  if (d != std::string::npos) p = p.substr(0, d);
    return p;
}

int main(int argc, char** argv) {
    // Cross-system visual parity: dump min-over-Z orthographic SDF fields on the
    // shared grid so frep4 / libfive / hyperfun images are pixel-comparable.
    //   bench_grid --dump-field R Z outdir scene.json...
    if (argc >= 6 && !std::strcmp(argv[1], "--dump-field")) {
        int R = std::atoi(argv[2]), Z = std::atoi(argv[3]); const char* dir = argv[4];
        for (int a = 5; a < argc; ++a) {
            SceneGraph s = io::load_scene(argv[a]);
            auto sc = jit::compile_scene_sdf(s);
            if (!sc) { std::fprintf(stderr, "%s: %s\n", argv[a], sc.error().c_str()); return 2; }
            auto fn = sc->fn;
            std::string nm = scene_stem(argv[a]);
            fdump::dump_field([fn](float x, float y, float z){ return fn(x, y, z); },
                              R, Z, std::string(dir) + "/" + nm + "_frep4");
            std::fprintf(stderr, "dumped %s_frep4\n", nm.c_str());
        }
        return 0;
    }

    if (argc < 3) { std::fprintf(stderr, "usage: %s N scene.json...\n", argv[0]); return 1; }
    const long N = atol(argv[1]);
    for (int a = 2; a < argc; ++a) {
        SceneGraph s = io::load_scene(argv[a]);
        std::string name = argv[a];
        if (auto sl = name.find_last_of('/'); sl != std::string::npos) name = name.substr(sl + 1);
        name = name.substr(0, name.find(".json"));
        const long total = N * N * N;

        // Optional interval-pruning path: FREP4_PRUNE=1 evaluates only octree
        // leaf cells whose sign interval arithmetic cannot resolve.
        const bool prune = getenv("FREP4_PRUNE") && atoi(getenv("FREP4_PRUNE"));
        auto simd = jit::compile_scene_sdf_simd(s);   // width auto-selected (8 AVX2 / 16 AVX-512)
        const char* backend = "cpu_ir-jit";
        bool did_prune = false;
        if (prune && simd) {
            auto ival = jit::compile_scene_sdf_interval(s);
            auto sca0 = jit::compile_scene_sdf(s);
            if (ival && sca0) {
                static jit::CompiledSdfInterval ik = std::move(*ival);
                static jit::CompiledSdfSimd     sk = std::move(*simd);
                static jit::CompiledSdf         tk = std::move(*sca0);
                auto ivfn = ik.fn; auto vfn = sk.fn; unsigned W = sk.width;
                auto sfn = tk.fn;   // scalar fn for the sub-W tail
                auto coordp = [&](long i){ return -1.6f + 3.2f * i / (N - 1); };
                auto leaves = jit::octree_leaves(ivfn, N, -1.6f, 1.6f, 4);
                long leafcells=0; for(auto&b:leaves)leafcells+=(b.x1-b.x0+1)*(b.y1-b.y0+1)*(b.z1-b.z0+1);
                if (leafcells <= (long)N*N*N*9/10) {   // interval tight enough
                // Leaf boxes are small (<= leaf^3, leaf=4), so a per-box z-run is
                // shorter than the vector width and never fills a lane group.
                // Stream all leaf cells into a W-wide buffer instead, flushing
                // whenever it fills; only the final partial buffer goes scalar.
                auto body = [&, ivfn, vfn, sfn, W] {
                    float acc = 0;                     // local; escapes once below
                    float xs[16], ys[16], zs[16], O[16];
                    unsigned n = 0;
                    auto flush = [&]{ if (!n) return;
                        if (n == W) { vfn(xs, ys, zs, O); for (unsigned l=0;l<W;++l) acc += O[l]; }
                        else        { for (unsigned l=0;l<n;++l) acc += sfn(xs[l], ys[l], zs[l]); }
                        n = 0; };
                    for (auto& b : leaves)
                        for (long i=b.x0;i<=b.x1;++i){ float xi=coordp(i);
                        for (long j=b.y0;j<=b.y1;++j){ float yj=coordp(j);
                        for (long k=b.z0;k<=b.z1;++k){
                            xs[n]=xi; ys[n]=yj; zs[n]=coordp(k);
                            if (++n == W) flush();
                        }}}
                    flush();
                    static volatile float sink; sink = acc;
                };
                auto [ms, Jl] = median_ms_energy(body);
                const long total = N*N*N;
                fprintf(stderr,"%s prune: eval %ld/%ld cells (%.1f%%)\n",name.c_str(),
                        leafcells,total,100.0*leafcells/total);
                csv_row("frep4","cpu_ir-prune",name.c_str(),"grid",N,ms,total/ms/1e3,
                        Jl, Jl<0?-1:Jl*1e6/total);
                did_prune = true;
                }
            }
        }
        if (did_prune) continue;
        std::function<void()> body;
        std::vector<float> X(N), Y(N), Z(N);
        auto coord = [&](long i){ return -1.6f + 3.2f * i / (N - 1); };
        if (simd) {
            // Backend name reflects the width actually used (8 on AVX2, 16 on AVX-512).
            static std::string simd_backend = "cpu_ir-simd" + std::to_string(simd->width);
            backend = simd_backend.c_str();
            auto sca1 = jit::compile_scene_sdf(s);
            if (!sca1) { std::fprintf(stderr, "%s: %s\n", argv[a], sca1.error().c_str()); return 2; }
            static jit::CompiledSdf tailk = std::move(*sca1);
            auto fn = simd->fn; unsigned W = simd->width; auto sfn = tailk.fn;
            body = [&, fn, sfn, W] {
                float O[16]; float acc = 0;
                std::vector<float> zx(N);
                for (long k = 0; k < N; ++k) zx[k] = coord(k);
                for (long i = 0; i < N; ++i) { float xi = coord(i);
                for (long j = 0; j < N; ++j) { float yj = coord(j);
                    long k = 0;
                    for (; k + (long)W <= N; k += W) {
                        float xs[16], ys[16];
                        for (unsigned l = 0; l < W; ++l) { xs[l] = xi; ys[l] = yj; }
                        fn(xs, ys, &zx[k], O);
                        for (unsigned l = 0; l < W; ++l) acc += O[l];
                    }
                    for (; k < N; ++k) acc += sfn(xi, yj, zx[k]);   // scalar tail
                }}
                static volatile float sink; sink = acc;
            };
        } else {
            auto sc = jit::compile_scene_sdf(s);
            if (!sc) { std::fprintf(stderr, "%s: %s\n", argv[a], sc.error().c_str()); return 2; }
            static jit::CompiledSdf keep = std::move(*sc);
            auto fn = keep.fn;
            body = [&, fn] {
                float acc = 0;
                for (long i = 0; i < N; ++i) { float xi = coord(i);
                for (long j = 0; j < N; ++j) { float yj = coord(j);
                for (long k = 0; k < N; ++k) acc += fn(xi, yj, coord(k)); }}
                static volatile float sink; sink = acc;
            };
        }
        auto [ms, Jl] = median_ms_energy(body);
        csv_row("frep4", backend, name.c_str(), "grid", N, ms, total / ms / 1e3,
                Jl, Jl < 0 ? -1 : Jl * 1e6 / total);
    }
    return 0;
}
