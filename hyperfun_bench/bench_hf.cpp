// HyperFun grid-eval benchmark via the bundled ANSI-C interpreter.
#include "hfinterpreter.h"
#include "../common/timing.hpp"
#include "../common/field_dump.hpp"
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
static std::string hf_stem(std::string p) {
    auto sl = p.find_last_of('/'); if (sl != std::string::npos) p = p.substr(sl + 1);
    return p.substr(0, p.find(".hf"));
}
int main(int argc, char** argv) {
    // Cross-system field parity: dump the SDF on the shared grid so hyperfun can
    // be numerically checked against frep4/libfive. HyperFun's convention is
    // f>=0 inside, so negate to the f<=0-inside convention the others use.
    //   bench_hf --dump-field R Z outdir scene.hf...
    if (argc >= 6 && !std::strcmp(argv[1], "--dump-field")) {
        int R = std::atoi(argv[2]), Z = std::atoi(argv[3]); const char* dir = argv[4];
        for (int a = 5; a < argc; ++a) {
            std::ifstream in(argv[a]); std::stringstream ss; ss << in.rdbuf();
            HFInterpreter hf;
            try { hf.parse(ss.str(), "my_model"); }
            catch (ParseError& e) { std::fprintf(stderr, "%s: parse l%d: %s\n", argv[a], e.line, e.error.c_str()); return 2; }
            std::vector<double> X(3); double S[8] = {0};
            fdump::dump_field([&](float x, float y, float z){
                X[0] = x; X[1] = y; X[2] = z; return -(float)hf.calc(X, S); },
                R, Z, std::string(dir) + "/" + hf_stem(argv[a]) + "_hyperfun");
            std::fprintf(stderr, "dumped %s_hyperfun\n", hf_stem(argv[a]).c_str());
        }
        return 0;
    }
    if (argc < 3) { std::fprintf(stderr, "usage: %s N scene.hf...\n", argv[0]); return 1; }
    const long N = atol(argv[1]);
    for (int a = 2; a < argc; ++a) {
        std::ifstream in(argv[a]);
        std::stringstream ss; ss << in.rdbuf();
        HFInterpreter hf;
        try { hf.parse(ss.str(), "my_model"); }
        catch (ParseError& e) { std::fprintf(stderr, "%s: parse error l%d: %s\n", argv[a], e.line, e.error.c_str()); return 2; }
        std::vector<double> X(3);
        double S[8] = {0};
        const long total = N * N * N;
        std::string name = argv[a];
        auto sl = name.find_last_of('/'); if (sl != std::string::npos) name = name.substr(sl + 1);
        name = name.substr(0, name.find(".hf"));
        auto [ms,J] = median_ms_energy([&] {
            volatile double sink = 0;
            for (long i = 0; i < N; ++i) { X[0] = -1.6 + 3.2 * i / (N - 1);
            for (long j = 0; j < N; ++j) { X[1] = -1.6 + 3.2 * j / (N - 1);
            for (long k = 0; k < N; ++k) { X[2] = -1.6 + 3.2 * k / (N - 1);
                sink += hf.calc(X, S);
            }}}
            (void)sink;
        }, 3);
        csv_row("hyperfun", "interpreter", name.c_str(), "grid", N, ms, total / ms / 1e3,
                J, J < 0 ? -1 : J * 1e6 / total);
    }
    return 0;
}
// MSVC-era intrinsics referenced by hfbsp.c
extern "C" double min(double a, double b) { return a < b ? a : b; }
extern "C" double max(double a, double b) { return a > b ? a : b; }
