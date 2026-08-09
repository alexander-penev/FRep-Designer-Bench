// CPU package energy, for the comparison systems' own drivers.
//
// Two readers, tried in order, because the obvious one usually is not readable.
//
//   1. The powercap sysfs counter, /sys/class/powercap/intel-rapl:0/energy_uj.
//      Simple, but on a stock distribution it is mode 0400 root-only -- a
//      2020 mitigation for the PLATYPUS side channel, which recovered AES keys
//      from RAPL traces. So on any machine the suite is likely to run on, this
//      reader silently reports "unavailable", which is exactly what happened:
//      every libfive, HyperFun and mpr row in the results carried -1 joules
//      while FRep Designer's own rows carried real numbers, because FRep
//      Designer reads the other interface.
//
//   2. The same counter through the perf_event PMU, which the kernel exposes
//      as the "power" event source and which honours
//      kernel.perf_event_paranoid instead of file permissions. Setting that to
//      0 is the documented way to get RAPL without root, and it is what the
//      suite already asks for elsewhere.
//
// Falling back to (2) is what makes the energy axis a comparison between
// systems rather than a comparison of FRep Designer against itself.
#pragma once
#include <cstdint>
#include <cstdio>
#include <string>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

struct Rapl {
    // ── reader 1: powercap sysfs ────────────────────────────────────────────
    std::string path;
    long long t0 = -1, maxr = -1;

    static long long rd(const std::string& p) {
        FILE* f = std::fopen(p.c_str(), "r");
        if (!f) return -1;
        long long v = -1;
        if (std::fscanf(f, "%lld", &v) != 1) v = -1;
        std::fclose(f);
        return v;
    }

#if defined(__linux__)
    // ── reader 2: the same RAPL domain through the perf_event PMU ───────────
    int      fd_ = -1;
    double   scale_ = 0.0;      // Joules per raw count
    std::uint64_t raw0_ = 0;

    static bool read_text(const char* p, std::string& out) {
        FILE* f = std::fopen(p, "r");
        if (!f) return false;
        char buf[256];
        bool got = std::fgets(buf, sizeof(buf), f) != nullptr;
        std::fclose(f);
        if (got) out = buf;
        return got;
    }
    static std::uint64_t read_raw(int fd) {
        std::uint64_t v = 0;
        if (::read(fd, &v, sizeof(v)) != (ssize_t)sizeof(v)) return 0;
        return v;
    }
    void open_perf() {
        std::string ts;
        if (!read_text("/sys/devices/power/type", ts)) return;
        long type = std::strtol(ts.c_str(), nullptr, 10);
        if (type <= 0) return;

        std::string ev;
        if (!read_text("/sys/devices/power/events/energy-pkg", ev)) return;
        auto eq = ev.find('=');
        if (eq == std::string::npos) return;
        unsigned long long config = std::strtoull(ev.c_str() + eq + 1, nullptr, 0);

        std::string sc;
        if (read_text("/sys/devices/power/events/energy-pkg.scale", sc))
            scale_ = std::strtod(sc.c_str(), nullptr);
        if (scale_ <= 0) return;

        perf_event_attr attr{};
        attr.type   = (std::uint32_t)type;
        attr.size   = sizeof(attr);
        attr.config = config;
        // System-wide package counter: pid = -1, cpu = 0.
        fd_ = (int)::syscall(SYS_perf_event_open, &attr, -1, 0, -1, 0);
    }
    ~Rapl() { if (fd_ >= 0) ::close(fd_); }
#endif

    Rapl() {
        for (const char* c : {"/sys/class/powercap/intel-rapl:0",
                              "/sys/class/powercap/intel-rapl/intel-rapl:0"}) {
            if (rd(std::string(c) + "/energy_uj") >= 0) {
                path = c;
                maxr = rd(path + "/max_energy_range_uj");
                break;
            }
        }
#if defined(__linux__)
        if (path.empty()) open_perf();
#endif
    }

    bool ok() const {
#if defined(__linux__)
        if (fd_ >= 0) return true;
#endif
        return !path.empty();
    }

    void begin() {
#if defined(__linux__)
        if (fd_ >= 0) { raw0_ = read_raw(fd_); return; }
#endif
        t0 = !path.empty() ? rd(path + "/energy_uj") : -1;
    }

    double joules() const {                      // -1 when no reader worked
#if defined(__linux__)
        if (fd_ >= 0) {
            std::uint64_t now = read_raw(fd_);
            if (now < raw0_) return -1;          // counter went backwards
            return (double)(now - raw0_) * scale_;
        }
#endif
        if (t0 < 0) return -1;
        long long t1 = rd(path + "/energy_uj");
        long long d = t1 - t0;
        if (d < 0 && maxr > 0) d += maxr;        // counter wrap
        return d / 1e6;
    }
};
