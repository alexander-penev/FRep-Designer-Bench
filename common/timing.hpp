// Shared timing helper: warmup + median-of-5, CSV row output.
#pragma once
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <vector>
#include "energy.hpp"
template <class F> double median_ms(F&& f, int reps = 5) {
    f(); // warmup
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        auto a = std::chrono::steady_clock::now();
        f();
        auto b = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    std::sort(t.begin(), t.end());
    return t[t.size() / 2];
}
// Same, but also averages CPU-package energy per repetition (RAPL; -1 if absent).
template <class F> std::pair<double,double> median_ms_energy(F&& f, int reps = 5) {
    f();
    Rapl r; r.begin();
    std::vector<double> t;
    for (int i = 0; i < reps; ++i) {
        auto a = std::chrono::steady_clock::now();
        f();
        auto b = std::chrono::steady_clock::now();
        t.push_back(std::chrono::duration<double, std::milli>(b - a).count());
    }
    double J = r.joules();
    std::sort(t.begin(), t.end());
    return { t[t.size() / 2], J < 0 ? -1 : J / reps };
}
inline void csv_row(const char* system, const char* backend, const char* scene,
                    const char* metric, long size, double ms, double thr,
                    double joules = -1, double uj_per_unit = -1) {
    std::printf("%s,%s,%s,%s,%ld,%.3f,%.3f,%.3f,%.4f\n",
                system, backend, scene, metric, size, ms, thr, joules, uj_per_unit);
}
