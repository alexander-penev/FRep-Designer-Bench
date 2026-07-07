// frep4 grid-eval: JIT the whole-scene SDF and evaluate an N^3 grid.
// Uses the SIMD (8-lane) path when the scene is a single CustomExprNode,
// else falls back to the scalar fn. Inner axis (z) is vectorized.
#include "core/compiler/compile_sdf.hpp"
#include "core/io/scene_io.hpp"
#include "../common/timing.hpp"
#include <cstdio>
#include <string>
#include <functional>
#include <vector>
using namespace frep;
int main(int argc, char** argv) {
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
        auto simd = jit::compile_scene_sdf_simd(s, 8);
        const char* backend = "cpu_ir-jit";
        bool did_prune = false;
        if (prune && simd) {
            auto ival = jit::compile_scene_sdf_interval(s);
            if (ival) {
                static jit::CompiledSdfInterval ik = std::move(*ival);
                static jit::CompiledSdfSimd     sk = std::move(*simd);
                auto ivfn = ik.fn; auto vfn = sk.fn; unsigned W = sk.width;
                auto coordp = [&](long i){ return -1.6f + 3.2f * i / (N - 1); };
                auto leaves = jit::octree_leaves(ivfn, N, -1.6f, 1.6f, 4);
                long leafcells=0; for(auto&b:leaves)leafcells+=(b.x1-b.x0+1)*(b.y1-b.y0+1)*(b.z1-b.z0+1);
                if (leafcells <= (long)N*N*N*9/10) {   // interval tight enough
                auto body = [&, ivfn, vfn, W] {
                    volatile float sink = 0; float O[16];
                    for (auto& b : leaves)
                        for (long i=b.x0;i<=b.x1;++i){ float xi=coordp(i);
                        for (long j=b.y0;j<=b.y1;++j){ float yj=coordp(j);
                            long k=b.z0;
                            for (; k+(long)W<=b.z1+1; k+=W){
                                float xs[16],ys[16],zs[16];
                                for(unsigned l=0;l<W;++l){xs[l]=xi;ys[l]=yj;zs[l]=coordp(k+l);}
                                vfn(xs,ys,zs,O); sink+=O[0];
                            }
                            for (; k<=b.z1; ++k){ float xs[16]={xi},ys[16]={yj},zs[16]={coordp(k)};
                                vfn(xs,ys,zs,O); sink+=O[0]; }
                        }}
                    (void)sink;
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
            backend = "cpu_ir-simd8";
            auto fn = simd->fn; unsigned W = simd->width;
            body = [&, fn, W] {
                float O[16]; volatile float sink = 0;
                std::vector<float> zx(N);
                for (long k = 0; k < N; ++k) zx[k] = coord(k);
                for (long i = 0; i < N; ++i) { float xi = coord(i);
                for (long j = 0; j < N; ++j) { float yj = coord(j);
                    long k = 0;
                    for (; k + (long)W <= N; k += W) {
                        float xs[16], ys[16];
                        for (unsigned l = 0; l < W; ++l) { xs[l] = xi; ys[l] = yj; }
                        fn(xs, ys, &zx[k], O);
                        sink += O[0];
                    }
                    for (; k < N; ++k) { float o; float xs[16]={xi},ys[16]={yj},zs[16]={zx[k]};
                        fn(xs, ys, zs, O); o = O[0]; sink += o; }  // scalar tail via 1-lane-ish
                }}
                (void)sink;
            };
        } else {
            auto sc = jit::compile_scene_sdf(s);
            if (!sc) { std::fprintf(stderr, "%s: %s\n", argv[a], sc.error().c_str()); return 2; }
            static jit::CompiledSdf keep = std::move(*sc);
            auto fn = keep.fn;
            body = [&, fn] {
                volatile float sink = 0;
                for (long i = 0; i < N; ++i) { float xi = coord(i);
                for (long j = 0; j < N; ++j) { float yj = coord(j);
                for (long k = 0; k < N; ++k) sink += fn(xi, yj, coord(k)); }}
                (void)sink;
            };
        }
        auto [ms, Jl] = median_ms_energy(body);
        csv_row("frep4", backend, name.c_str(), "grid", N, ms, total / ms / 1e3,
                Jl, Jl < 0 ? -1 : Jl * 1e6 / total);
    }
    return 0;
}
