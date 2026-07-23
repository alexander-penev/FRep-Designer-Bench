// core/power/energy_meter.hpp
//
// Real energy measurement via hardware performance counters, so the multipath
// study can report pix/kWh (energy efficiency) alongside pix/s (throughput) —
// the separate axis that informs whether to grow a datacenter with CPUs or
// GPUs, and whether running CPU+GPU together is energy-justified.
//
// Two sources, each optional and probed at runtime (so a build runs anywhere):
//   - CPU: Intel/AMD RAPL via /sys/class/powercap/intel-rapl*/energy_uj.
//     A monotonic microjoule counter; we read it before/after and difference.
//   - GPU: NVIDIA NVML (nvmlDeviceGetTotalEnergyConsumption on newer drivers,
//     else integrate nvmlDeviceGetPowerUsage). libnvidia-ml is dlopen'd, so
//     there's no link-time dependency on the CUDA/driver SDK.
//
// Where a counter isn't available (sandbox, non-Intel CPU, no NVIDIA GPU), the
// meter reports `available() == false` and energy is simply omitted — never
// invented. A reading is Joules over the measured interval.

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <string>

namespace frep::power {

// One energy sample over a begin()..end() interval, in Joules. `domain`
// labels the source (e.g. "rapl:package-0", "nvml:cuda0").
struct EnergyReading {
    double joules = 0.0;
    std::string domain;
};

// Abstract counter. begin() latches the starting counter; end() returns the
// energy consumed since begin() (or nullopt if the read failed / wrapped).
class EnergyCounter {
public:
    virtual ~EnergyCounter() = default;
    virtual bool available() const = 0;
    virtual const std::string& domain() const = 0;
    virtual void begin() = 0;
    virtual std::optional<double> end() = 0;  // Joules since begin()
};

// Probe the system and return the best CPU energy counter (RAPL package), or a
// null counter whose available() is false. Never throws.
std::unique_ptr<EnergyCounter> make_cpu_energy_counter();

// Probe for an NVIDIA GPU energy counter via NVML (device 0 by default), or a
// null counter. dlopen's libnvidia-ml at first use; safe when absent.
std::unique_ptr<EnergyCounter> make_gpu_energy_counter(int device_index = 0);

// Convenience: pixels per kWh from a pixel count and Joules.
// 1 kWh = 3.6e6 J. pix/kWh = pixels / (J / 3.6e6) = pixels * 3.6e6 / J.
inline double pixels_per_kwh(double pixels, double joules) {
    return joules > 0.0 ? pixels * 3.6e6 / joules : 0.0;
}

}  // namespace frep::power
