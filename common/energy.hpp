// CPU package energy via Linux RAPL powercap. joules() = delta since begin().
// Silently reports -1 when RAPL is unavailable (containers, non-Intel/AMD).
#pragma once
#include <cstdio>
#include <string>
struct Rapl {
    std::string path;
    long long t0 = -1, maxr = -1;
    static long long rd(const std::string& p) {
        FILE* f = std::fopen(p.c_str(), "r");
        if (!f) return -1;
        long long v = -1; std::fscanf(f, "%lld", &v); std::fclose(f); return v;
    }
    Rapl() {
        for (const char* c : {"/sys/class/powercap/intel-rapl:0",
                              "/sys/class/powercap/intel-rapl/intel-rapl:0"}) {
            if (rd(std::string(c) + "/energy_uj") >= 0) {
                path = c;
                maxr = rd(path + "/max_energy_range_uj");
                break;
            }
        }
    }
    bool ok() const { return !path.empty(); }
    void begin() { t0 = ok() ? rd(path + "/energy_uj") : -1; }
    double joules() const {                      // -1 if unsupported
        if (t0 < 0) return -1;
        long long t1 = rd(path + "/energy_uj");
        long long d = t1 - t0;
        if (d < 0 && maxr > 0) d += maxr;        // counter wrap
        return d / 1e6;
    }
};
