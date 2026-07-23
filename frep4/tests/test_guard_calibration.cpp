// tests/test_guard_calibration.cpp
//
// Tests for the spatial-guard runtime calibration: the should_guard()
// heuristic logic, the cache round-trip (save → load), CPU-mismatch
// invalidation, and node_count(). The actual calibrate() measurement is
// hardware/timing dependent, so it's exercised only for "produces a valid
// result" rather than a specific threshold.

#include <gtest/gtest.h>

#include "core/accel/guard_calibration.hpp"
#include "core/frep/node.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/operations.hpp"

#include <cstdio>
#include <fstream>

using namespace frep;
using namespace frep::accel;

TEST(GuardCalibration, NodeCountCountsSubtree) {
    EXPECT_EQ(node_count(SphereNode(1.0f, "s")), 1);
    auto t = std::make_shared<TranslateNode>(
        std::make_shared<SphereNode>(1.0f, "s"), 1, 0, 0, "t");
    EXPECT_EQ(node_count(*t), 2);
    auto u = std::make_shared<UnionNode>(
        std::make_shared<SphereNode>(1.0f, "a"),
        std::make_shared<TranslateNode>(
            std::make_shared<BoxNode>(1, 1, 1, "b"), 1, 0, 0, "tb"),
        "u");
    // union(1) + sphere(1) + translate(1) + box(1) = 4
    EXPECT_EQ(node_count(*u), 4);
}

TEST(GuardCalibration, ShouldGuardRequiresValidCalibration) {
    GuardCalibration cal;          // valid == false
    EXPECT_FALSE(should_guard(cal, 100, 10.0));
}

TEST(GuardCalibration, ShouldGuardRespectsThresholdAndCount) {
    GuardCalibration cal;
    cal.valid = true;
    cal.node_threshold = 4;

    // Below the object-count floor → never guard, however complex.
    EXPECT_FALSE(should_guard(cal, 4, 100.0));
    // Enough objects, but too-cheap average → no guard.
    EXPECT_FALSE(should_guard(cal, 50, 2.0));
    // Enough objects AND complex enough → guard.
    EXPECT_TRUE(should_guard(cal, 50, 4.0));
    EXPECT_TRUE(should_guard(cal, 50, 8.5));
}

TEST(GuardCalibration, NeverGuardSentinelDisables) {
    GuardCalibration cal;
    cal.valid = true;
    cal.node_threshold = kNeverGuard;
    EXPECT_FALSE(should_guard(cal, 1000, 1000.0));
}

TEST(GuardCalibration, CacheRoundTrip) {
    // Save a calibration with the real host CPU id, then load it back.
    GuardCalibration c;
    c.valid = true;
    c.node_threshold = 7;
    c.cpu_id = host_cpu_id();
    ASSERT_TRUE(save_calibration(c));

    auto loaded = load_calibration();
    ASSERT_TRUE(loaded.has_value());
    EXPECT_EQ(loaded->node_threshold, 7);
    EXPECT_EQ(loaded->cpu_id, host_cpu_id());
    EXPECT_TRUE(loaded->valid);
}

TEST(GuardCalibration, CacheRejectsDifferentCpu) {
    // Write a cache file with a bogus CPU id; load must reject it (returns
    // nullopt) so a moved machine recalibrates.
    {
        std::ofstream f(calibration_cache_path(), std::ios::trunc);
        ASSERT_TRUE((bool)f);
        f << "version 1\nthreshold 5\ncpu DefinitelyNotThisCpu_xyz\n";
    }
    auto loaded = load_calibration();
    EXPECT_FALSE(loaded.has_value());
}

TEST(GuardCalibration, CalibrateProducesValidResult) {
    // Don't assert a specific threshold (hardware-dependent), just that it
    // runs, returns valid, and a sensible threshold. Crucially, when a
    // finite threshold is found it must be strictly above a bare
    // translated primitive's node count (~2), so simple scenes are never
    // guarded — the property that the noise-robustness rework guarantees.
    GuardCalibration c = calibrate();
    EXPECT_TRUE(c.valid);
    EXPECT_FALSE(c.cpu_id.empty());
    if (c.node_threshold < kNeverGuard) {
        EXPECT_GE(c.node_threshold, 3)
            << "threshold must be above a bare primitive (sphere=2) so "
               "simple scenes stay inlined";
    }
}
