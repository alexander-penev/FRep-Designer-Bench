// Energy meter: the pix/kWh math and graceful behavior when counters are
// unavailable (the sandbox case — must report nullopt, never invent energy).
#include <gtest/gtest.h>
#include "core/power/energy_meter.hpp"

using namespace frep::power;

TEST(EnergyMeter, PixelsPerKwhMath) {
    // 1e6 pixels at 10 J → 3.6e11 pix/kWh (1 kWh = 3.6e6 J).
    EXPECT_NEAR(pixels_per_kwh(1e6, 10.0), 3.6e11, 1.0);
    // Zero/negative joules → 0 (no divide-by-zero).
    EXPECT_EQ(pixels_per_kwh(1e6, 0.0), 0.0);
}

TEST(EnergyMeter, CountersProbeGracefully) {
    auto cpu = make_cpu_energy_counter();
    auto gpu = make_gpu_energy_counter(0);
    ASSERT_TRUE(cpu != nullptr);
    ASSERT_TRUE(gpu != nullptr);
    // Whatever the host, domain() is non-empty and the unavailable path is safe.
    EXPECT_FALSE(cpu->domain().empty());
    EXPECT_FALSE(gpu->domain().empty());
    if (!cpu->available()) {
        cpu->begin();
        EXPECT_FALSE(cpu->end().has_value());  // never fabricates a reading
    }
    if (!gpu->available()) {
        gpu->begin();
        EXPECT_FALSE(gpu->end().has_value());
    }
}
