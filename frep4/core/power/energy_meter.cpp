// core/power/energy_meter.cpp
#include "core/power/energy_meter.hpp"

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>

#if defined(__linux__)
#  include <dlfcn.h>
#  include <unistd.h>
#  include <sys/syscall.h>
#  include <linux/perf_event.h>
#endif

namespace frep::power {

namespace {

// ── Null counter: always unavailable (sandbox / unsupported hardware) ───────
class NullCounter final : public EnergyCounter {
    std::string dom_ = "none";
public:
    explicit NullCounter(std::string why) : dom_(std::move(why)) {}
    bool available() const override { return false; }
    const std::string& domain() const override { return dom_; }
    void begin() override {}
    std::optional<double> end() override { return std::nullopt; }
};

// ── perf_event RAPL: the package energy PMU (preferred over sysfs) ──────────
// The powercap sysfs file (intel-rapl:*/energy_uj) is root-only on most kernels
// since ~2020 (the PLATYPUS side-channel mitigation), so a normal user gets
// "unavailable". The perf_event RAPL PMU exposes the same package energy and is
// governed by perf_event_paranoid (commonly <= 1, i.e. readable without root).
// We open the "energy-pkg" event and read its accumulated value; the per-event
// .scale file gives Joules per raw unit.
#if defined(__linux__)
class PerfRaplCounter final : public EnergyCounter {
    std::string dom_ = "rapl:energy-pkg(perf)";
    int    fd_ = -1;
    double scale_ = 0.0;          // Joules per raw count
    std::uint64_t start_raw_ = 0;
    bool   ok_ = false;

    static bool read_text(const std::string& p, std::string& out) {
        std::ifstream f(p);
        if (!f) return false;
        std::getline(f, out);
        return true;
    }
    static std::uint64_t read_raw(int fd) {
        std::uint64_t v = 0;
        if (::read(fd, &v, sizeof(v)) != (ssize_t)sizeof(v)) return 0;
        return v;
    }
public:
    PerfRaplCounter() {
        // PMU type number (dynamic, assigned by the kernel).
        std::string ts;
        if (!read_text("/sys/devices/power/type", ts)) return;
        long type = std::strtol(ts.c_str(), nullptr, 10);
        if (type <= 0) return;

        // energy-pkg event config, e.g. "event=0x02".
        std::string ev;
        if (!read_text("/sys/devices/power/events/energy-pkg", ev)) return;
        auto eq = ev.find('=');
        if (eq == std::string::npos) return;
        unsigned long long config =
            std::strtoull(ev.c_str() + eq + 1, nullptr, 0);

        // Scale: Joules per raw unit (e.g. "2.3283064365386962890625e-10").
        std::string sc;
        if (read_text("/sys/devices/power/events/energy-pkg.scale", sc))
            scale_ = std::strtod(sc.c_str(), nullptr);
        if (scale_ <= 0) return;

        perf_event_attr attr{};
        attr.type = (std::uint32_t)type;
        attr.size = sizeof(attr);
        attr.config = config;
        // System-wide RAPL counter: pid=-1, cpu=0 (package counter is per-pkg,
        // any CPU in the package works).
        fd_ = (int)::syscall(SYS_perf_event_open, &attr, -1, 0, -1, 0);
        if (fd_ >= 0) ok_ = true;
    }
    ~PerfRaplCounter() override { if (fd_ >= 0) ::close(fd_); }

    bool available() const override { return ok_; }
    const std::string& domain() const override { return dom_; }
    void begin() override { if (ok_) start_raw_ = read_raw(fd_); }
    std::optional<double> end() override {
        if (!ok_) return std::nullopt;
        std::uint64_t now = read_raw(fd_);
        if (now < start_raw_) return std::nullopt;  // unexpected; skip
        return (double)(now - start_raw_) * scale_;  // raw → Joules
    }
};
#endif  // __linux__

// ── RAPL: /sys/class/powercap/intel-rapl:0/energy_uj (microjoules) ──────────
// A monotonically increasing counter that wraps at max_energy_range_uj. We
// read the package domain (intel-rapl:0); if absent, the meter is unavailable.
#if defined(__linux__)
class RaplCounter final : public EnergyCounter {
    std::string path_;        // .../energy_uj
    std::string dom_ = "rapl:package-0";
    std::uint64_t range_uj_ = 0;   // wrap modulus (0 = unknown)
    std::uint64_t start_uj_ = 0;
    bool ok_ = false;

    static bool read_u64(const std::string& p, std::uint64_t& out) {
        std::ifstream f(p);
        if (!f) return false;
        f >> out;
        return (bool)f;
    }
public:
    RaplCounter() {
        // Find the first package domain.
        const char* base = "/sys/class/powercap/intel-rapl:0";
        std::string ej = std::string(base) + "/energy_uj";
        std::uint64_t probe = 0;
        if (read_u64(ej, probe)) {
            path_ = ej;
            ok_ = true;
            std::uint64_t r = 0;
            if (read_u64(std::string(base) + "/max_energy_range_uj", r))
                range_uj_ = r;
            // Prefer the human name if present.
            std::ifstream nf(std::string(base) + "/name");
            if (nf) { std::string n; std::getline(nf, n); if (!n.empty()) dom_ = "rapl:" + n; }
        }
    }
    bool available() const override { return ok_; }
    const std::string& domain() const override { return dom_; }
    void begin() override { read_u64(path_, start_uj_); }
    std::optional<double> end() override {
        std::uint64_t now = 0;
        if (!ok_ || !read_u64(path_, now)) return std::nullopt;
        std::uint64_t delta;
        if (now >= start_uj_) delta = now - start_uj_;
        else if (range_uj_)   delta = (range_uj_ - start_uj_) + now;  // wrapped
        else return std::nullopt;
        return (double)delta * 1e-6;  // µJ → J
    }
};
#endif  // __linux__

// ── NVML: dlopen libnvidia-ml, read per-device energy/power ─────────────────
// Prefer the cumulative energy counter (mJ) where the driver provides it; else
// integrate instantaneous power (mW) across the interval using wall time.
#if defined(__linux__)
class NvmlCounter final : public EnergyCounter {
    std::string dom_ = "nvml:gpu";
    void* lib_ = nullptr;
    void* handle_ = nullptr;   // nvmlDevice_t
    bool ok_ = false;
    bool use_energy_ = false;  // cumulative energy counter available
    unsigned long long start_mj_ = 0;

    // NVML function pointers (resolved by name; signatures per NVML headers).
    int (*nvmlInit_)() = nullptr;
    int (*nvmlShutdown_)() = nullptr;
    int (*nvmlDeviceGetHandleByIndex_)(unsigned int, void**) = nullptr;
    int (*nvmlDeviceGetTotalEnergyConsumption_)(void*, unsigned long long*) = nullptr;
    int (*nvmlDeviceGetName_)(void*, char*, unsigned int) = nullptr;

    template <class F> void load(F& fp, const char* name) {
        fp = reinterpret_cast<F>(dlsym(lib_, name));
    }
public:
    explicit NvmlCounter(int device_index) {
        lib_ = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
        if (!lib_) lib_ = dlopen("libnvidia-ml.so", RTLD_NOW | RTLD_LOCAL);
        if (!lib_) return;
        load(nvmlInit_, "nvmlInit_v2");
        if (!nvmlInit_) load(nvmlInit_, "nvmlInit");
        load(nvmlShutdown_, "nvmlShutdown");
        load(nvmlDeviceGetHandleByIndex_, "nvmlDeviceGetHandleByIndex_v2");
        if (!nvmlDeviceGetHandleByIndex_)
            load(nvmlDeviceGetHandleByIndex_, "nvmlDeviceGetHandleByIndex");
        load(nvmlDeviceGetTotalEnergyConsumption_,
             "nvmlDeviceGetTotalEnergyConsumption");
        load(nvmlDeviceGetName_, "nvmlDeviceGetName");
        if (!nvmlInit_ || !nvmlDeviceGetHandleByIndex_) return;
        if (nvmlInit_() != 0) return;
        if (nvmlDeviceGetHandleByIndex_((unsigned)device_index, &handle_) != 0)
            return;
        if (nvmlDeviceGetName_) {
            char nm[96] = {0};
            if (nvmlDeviceGetName_(handle_, nm, sizeof(nm)) == 0 && nm[0])
                dom_ = std::string("nvml:") + nm;
        }
        // Probe the cumulative energy counter.
        if (nvmlDeviceGetTotalEnergyConsumption_) {
            unsigned long long e = 0;
            if (nvmlDeviceGetTotalEnergyConsumption_(handle_, &e) == 0) {
                use_energy_ = true;
                ok_ = true;
            }
        }
    }
    ~NvmlCounter() override {
        if (lib_) {
            if (nvmlShutdown_) nvmlShutdown_();
            dlclose(lib_);
        }
    }
    bool available() const override { return ok_; }
    const std::string& domain() const override { return dom_; }
    void begin() override {
        if (use_energy_)
            nvmlDeviceGetTotalEnergyConsumption_(handle_, &start_mj_);
    }
    std::optional<double> end() override {
        if (!ok_) return std::nullopt;
        if (use_energy_) {
            unsigned long long now = 0;
            if (nvmlDeviceGetTotalEnergyConsumption_(handle_, &now) != 0)
                return std::nullopt;
            if (now < start_mj_) return std::nullopt;  // counter reset
            return (double)(now - start_mj_) * 1e-3;   // mJ → J
        }
        return std::nullopt;
    }
};
#endif  // __linux__

}  // namespace

std::unique_ptr<EnergyCounter> make_cpu_energy_counter() {
#if defined(__linux__)
    // Prefer the perf_event RAPL PMU (readable without root on most kernels);
    // fall back to powercap sysfs (often root-only since the PLATYPUS fix).
    auto pc = std::make_unique<PerfRaplCounter>();
    if (pc->available()) return pc;
    auto c = std::make_unique<RaplCounter>();
    if (c->available()) return c;
    return std::make_unique<NullCounter>("rapl:unavailable");
#else
    return std::make_unique<NullCounter>("cpu-energy:unsupported-os");
#endif
}

std::unique_ptr<EnergyCounter> make_gpu_energy_counter(int device_index) {
#if defined(__linux__)
    auto c = std::make_unique<NvmlCounter>(device_index);
    if (c->available()) return c;
    return std::make_unique<NullCounter>("nvml:unavailable");
#else
    return std::make_unique<NullCounter>("gpu-energy:unsupported-os");
#endif
}

}  // namespace frep::power
