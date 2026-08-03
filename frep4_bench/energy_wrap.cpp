// energy_wrap — measure the energy a command consumes, whichever system it belongs to.
//
// The suite could report energy for its own backends and for mpr (whose wrapper
// sampled nvidia-smi), but not for the CPU-only comparison systems, so the
// energy axis was effectively a comparison of FRep Designer against itself.
// This wraps any executable and reports the CPU package energy (RAPL, through
// the perf_event PMU) and the GPU energy (NVML) drawn while it ran, so libfive,
// HyperFun, mpr and FRep Designer are all measured the same way.
//
// Usage:
//   energy_wrap <system> <backend> <scene> <size> -- <command> [args...]
//
// Writes one CSV row on stdout in the suite's column order:
//   system,backend,scene,metric,size,ms,throughput,joules,uj_per_unit
// with metric=energy, ms=wall time, joules=CPU+GPU, and uj_per_unit=average
// watts. The command's own stdout/stderr pass through untouched.
//
// A counter that is unavailable contributes nothing rather than a fabricated
// value; if neither is available the row reports -1 joules, which the tables
// render as a dash.
#include "core/power/energy_meter.hpp"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

int main(int argc, char** argv) {
    int sep = -1;
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], "--") == 0) { sep = i; break; }
    if (sep < 5 || sep + 1 >= argc) {
        std::fprintf(stderr,
            "usage: %s <system> <backend> <scene> <size> -- <command> [args...]\n",
            argv[0]);
        return 2;
    }
    const std::string system = argv[1], backend = argv[2], scene = argv[3];
    const int size = std::atoi(argv[4]);

    std::vector<char*> cmd(argv + sep + 1, argv + argc);
    cmd.push_back(nullptr);

    auto cpu = frep::power::make_cpu_energy_counter();
    auto gpu = frep::power::make_gpu_energy_counter();
    const bool have_cpu = cpu && cpu->available();
    const bool have_gpu = gpu && gpu->available();

    if (have_cpu) cpu->begin();
    if (have_gpu) gpu->begin();
    auto t0 = std::chrono::steady_clock::now();

    pid_t pid = fork();
    if (pid < 0) { std::perror("fork"); return 1; }
    if (pid == 0) {
        execvp(cmd[0], cmd.data());
        std::perror("execvp");
        _exit(127);
    }
    int status = 0;
    while (waitpid(pid, &status, 0) < 0 && errno == EINTR) {}

    auto t1 = std::chrono::steady_clock::now();
    double joules = 0; bool any = false;
    if (have_cpu) { if (auto v = cpu->end()) { joules += *v; any = true; } }
    if (have_gpu) { if (auto v = gpu->end()) { joules += *v; any = true; } }

    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double watts = (any && ms > 0) ? joules / (ms / 1e3) : -1.0;
    std::printf("%s,%s,%s,energy,%d,%.2f,-1.000,%.3f,%.4f\n",
                system.c_str(), backend.c_str(), scene.c_str(), size,
                ms, any ? joules : -1.0, watts);
    std::fflush(stdout);

    return WIFEXITED(status) ? WEXITSTATUS(status) : 1;
}
