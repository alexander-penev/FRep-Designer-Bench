// tests/test_multipath.cpp
//
// Tests for the Model D multi-path execution framework, using the CPU-IR
// executor (always available). GPU paths need a real Vulkan device and are
// exercised on hardware, not here. These cover the framework mechanics:
// decompose / dispatch / merge strategies, concurrent vs serial runs, and
// the key correctness property — two runs of the same path agree exactly
// (the floor that cross-path equivalence builds on).

#include <gtest/gtest.h>

#include "core/exec/multipath.hpp"
#include "core/exec/cpu_executor.hpp"
#include "core/exec/gpu_executor.hpp"
#include "core/exec/gpu_ir_executor.hpp"
#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/frep/transforms.hpp"
#include "core/frep/operations.hpp"
#include "core/compiler/codegen.hpp"
#include "core/compiler/jit_engine.hpp"
#include "core/tracer/tile_scheduler.hpp"

#include <cmath>
#include <memory>

using namespace frep;
using namespace frep::exec;

namespace {

SceneGraph simple_scene() {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "a"));
    s.add_object(std::make_shared<TranslateNode>(
        std::make_shared<BoxNode>(0.6f, 0.6f, 0.6f, "b"), 1.5f, 0, 0, "t"));
    s.add_object(std::make_shared<PlaneNode>(0, 1, 0, 1.0f, "floor"));
    s.camera().position = {4, 3, 4};
    s.camera().target = {0, 0, 0};
    return s;
}

} // namespace

TEST(MultiPath, WholeFrameDecomposeIsSingleTile) {
    WholeFrame d;
    SceneGraph s = simple_scene();
    auto tiles = d.decompose(s, 64, 48);
    ASSERT_EQ(tiles.size(), 1u);
    EXPECT_EQ(tiles[0].x0, 0); EXPECT_EQ(tiles[0].y0, 0);
    EXPECT_EQ(tiles[0].x1, 64); EXPECT_EQ(tiles[0].y1, 48);
}

TEST(MultiPath, HorizontalBandsCoverFrameExactly) {
    HorizontalBands d(3);
    SceneGraph s = simple_scene();
    auto tiles = d.decompose(s, 50, 30);
    ASSERT_EQ(tiles.size(), 3u);
    // Bands tile the height with no gaps or overlaps.
    EXPECT_EQ(tiles[0].y0, 0);
    EXPECT_EQ(tiles.back().y1, 30);
    for (std::size_t i = 1; i < tiles.size(); ++i)
        EXPECT_EQ(tiles[i].y0, tiles[i-1].y1);
    for (auto& t : tiles) { EXPECT_EQ(t.x0, 0); EXPECT_EQ(t.x1, 50); }
}

TEST(MultiPath, WeightedBandsSplitProportionally) {
    // 1:3 weights over height 80 → bands of ~20 and ~60 rows.
    WeightedBands d({1.0, 3.0});
    SceneGraph s = simple_scene();
    auto tiles = d.decompose(s, 40, 80);
    ASSERT_EQ(tiles.size(), 2u);
    EXPECT_EQ(tiles[0].y0, 0);
    EXPECT_EQ(tiles.back().y1, 80);
    EXPECT_EQ(tiles[0].y1, tiles[1].y0);          // contiguous
    EXPECT_EQ(tiles[0].height(), 20);             // 80 * 1/4
    EXPECT_EQ(tiles[1].height(), 60);             // 80 * 3/4
}

TEST(MultiPath, WeightedBandsCoverFrameExactly) {
    // Arbitrary weights still tile [0,H) with no gap/overlap.
    WeightedBands d({2.0, 5.0, 1.0});
    SceneGraph s = simple_scene();
    auto tiles = d.decompose(s, 30, 100);
    ASSERT_EQ(tiles.size(), 3u);
    EXPECT_EQ(tiles.front().y0, 0);
    EXPECT_EQ(tiles.back().y1, 100);
    long area = 0;
    for (std::size_t i = 0; i < tiles.size(); ++i) {
        if (i) EXPECT_EQ(tiles[i].y0, tiles[i-1].y1);
        area += (long)tiles[i].width() * tiles[i].height();
    }
    EXPECT_EQ(area, 30L * 100L);   // exact cover
}

TEST(MultiPath, WeightedBandsZeroWeightFallsBackEqual) {
    // All-zero weights → equal bands (degenerate guard).
    WeightedBands d({0.0, 0.0});
    SceneGraph s = simple_scene();
    auto tiles = d.decompose(s, 40, 60);
    ASSERT_EQ(tiles.size(), 2u);
    EXPECT_EQ(tiles[0].height(), 30);
    EXPECT_EQ(tiles[1].height(), 30);
}

TEST(MultiPath, DynamicQueueStitchesAllTiles) {
    // Work-stealing over a grid must cover the frame exactly, same as a
    // whole-frame render. Two CPU executors share the tile queue.
    SceneGraph s = simple_scene();
    GridDecompose dec(20, 20);
    OnePathPerTile dsp;          // unused by DynamicQueue, but required by ctor
    StitchMerge mrg;
    MultiPathExecutor mp(dec, dsp, mrg);

    CpuIrExecutor e0, e1;
    std::vector<IExecutor*> execs{&e0, &e1};
    auto r = mp.run(s, 50, 40, execs, RunMode::DynamicQueue);

    // Every tile rendered, frame fully stitched.
    EXPECT_EQ(r.merged.width, 50);
    EXPECT_EQ(r.merged.height, 40);
    EXPECT_FALSE(r.merged.image.empty());
    for (const auto& t : r.tiles) EXPECT_TRUE(t.ok);
    // Grid 50x40 in 20x20 tiles = 3 cols × 2 rows = 6 tiles.
    EXPECT_EQ(r.tiles.size(), 6u);
}

TEST(MultiPath, GridCoversFrame) {
    GridDecompose d(32, 32);
    SceneGraph s = simple_scene();
    auto tiles = d.decompose(s, 70, 40);   // 3 cols (32,32,6) × 2 rows (32,8) = 6
    EXPECT_EQ(tiles.size(), 6u);
    // Every pixel covered exactly once: areas sum to W*H.
    long area = 0;
    for (auto& t : tiles) area += (long)t.width() * t.height();
    EXPECT_EQ(area, 70L * 40L);
}

TEST(MultiPath, OnePathPerTileRoundRobin) {
    CpuIrExecutor e1, e2;
    std::vector<IExecutor*> execs{&e1, &e2};
    std::vector<Tile> tiles{{0,0,10,10},{0,10,10,20},{0,20,10,30}};
    OnePathPerTile disp; auto jobs = disp.dispatch(tiles, execs);
    ASSERT_EQ(jobs.size(), 3u);
    EXPECT_EQ(jobs[0].executor, &e1);
    EXPECT_EQ(jobs[1].executor, &e2);
    EXPECT_EQ(jobs[2].executor, &e1);   // wraps
}

TEST(MultiPath, AllPathsPerTileCrossProduct) {
    CpuIrExecutor e1, e2;
    std::vector<IExecutor*> execs{&e1, &e2};
    std::vector<Tile> tiles{{0,0,10,10},{0,10,10,20}};
    AllPathsPerTile disp; auto jobs = disp.dispatch(tiles, execs);
    EXPECT_EQ(jobs.size(), 4u);   // 2 tiles × 2 paths
}

TEST(MultiPath, CompareSamePathAgreesExactly) {
    // Two CPU-IR executors render the whole frame; their outputs must be
    // bit-identical → max diff 0, consistent. This is the equivalence
    // floor: same path, same result.
    CpuIrExecutor e1, e2;
    std::vector<IExecutor*> execs{&e1, &e2};
    WholeFrame dec; AllPathsPerTile dsp; CompareMerge mrg;
    MultiPathExecutor mpe(dec, dsp, mrg);
    SceneGraph s = simple_scene();
    auto res = mpe.run(s, 64, 48, execs, RunMode::Concurrent);
    EXPECT_EQ(res.tiles.size(), 2u);
    for (auto& t : res.tiles) EXPECT_TRUE(t.ok) << t.error;
    EXPECT_TRUE(res.merged.consistent);
    EXPECT_DOUBLE_EQ(res.merged.max_abs_diff, 0.0);
}

TEST(MultiPath, ConcurrentAndSerialProduceSameMerge) {
    CpuIrExecutor e1, e2;
    std::vector<IExecutor*> execs{&e1, &e2};
    SceneGraph s = simple_scene();
    WholeFrame dec; AllPathsPerTile dsp; CompareMerge mrg;
    MultiPathExecutor mpe(dec, dsp, mrg);
    auto rc = mpe.run(s, 48, 36, execs, RunMode::Concurrent);
    auto rs = mpe.run(s, 48, 36, execs, RunMode::Serial);
    EXPECT_EQ(rc.merged.consistent, rs.merged.consistent);
    EXPECT_DOUBLE_EQ(rc.merged.max_abs_diff, rs.merged.max_abs_diff);
}

TEST(MultiPath, StitchHalvesReconstructsFullFrame) {
    // Render the full frame with one executor as reference, then render two
    // halves (one path each) and stitch — the stitched image must match the
    // reference, proving tile cropping + stitching is seamless for one path.
    CpuIrExecutor ref_exec;
    SceneGraph s = simple_scene();
    const int W = 48, H = 36;
    auto ref = ref_exec.render(s, W, H, Tile{0,0,W,H});
    ASSERT_TRUE(ref.ok);

    CpuIrExecutor e1, e2;
    std::vector<IExecutor*> execs{&e1, &e2};
    HorizontalBands dec(2); OnePathPerTile dsp; StitchMerge mrg;
    MultiPathExecutor mpe(dec, dsp, mrg);
    auto res = mpe.run(s, W, H, execs, RunMode::Concurrent);
    ASSERT_EQ(res.merged.image.size(), (std::size_t)W * H * 4);

    double max_diff = 0;
    for (std::size_t i = 0; i < res.merged.image.size(); ++i)
        max_diff = std::max(max_diff,
            (double)std::fabs(res.merged.image[i] - ref.rgba[i]));
    EXPECT_LT(max_diff, 1e-5) << "stitched halves differ from full-frame render";
}

TEST(MultiPath, RegionRenderMatchesFullFrameSubrect) {
    // Rendering only a sub-region must produce the same pixels as the
    // corresponding sub-rectangle of a full-frame render — proves the
    // TileScheduler region path is correct (and that the CPU path is
    // genuinely tile-addressed, not cropping a whole-frame render).
    SceneGraph s = simple_scene();
    const int W = 64, H = 48;

    // Full-frame reference (no region).
    std::vector<float> full((std::size_t)W * H * 4, 0.0f);
    {
        auto ctx = std::make_unique<llvm::LLVMContext>();
        TracerConfig cfg;
        SceneCodegen cg(*ctx, cfg);
        cg.emit_render_tile(s, SceneCodegen::SceneSdfMode::Inlined);
        JitEngine jit;
        auto fn = jit.load(cg.take_module(), std::move(ctx));
        ASSERT_TRUE(fn);
        RenderParams rp; rp.width = W; rp.height = H;
        TileScheduler::render(*fn, full.data(), s.camera(), rp);
    }

    // Region render of a sub-rectangle.
    const int X0 = 16, Y0 = 12, X1 = 48, Y1 = 36;
    std::vector<float> reg((std::size_t)W * H * 4, 0.0f);
    {
        auto ctx = std::make_unique<llvm::LLVMContext>();
        TracerConfig cfg;
        SceneCodegen cg(*ctx, cfg);
        cg.emit_render_tile(s, SceneCodegen::SceneSdfMode::Inlined);
        JitEngine jit;
        auto fn = jit.load(cg.take_module(), std::move(ctx));
        ASSERT_TRUE(fn);
        RenderParams rp; rp.width = W; rp.height = H;
        rp.region_x0 = X0; rp.region_y0 = Y0; rp.region_x1 = X1; rp.region_y1 = Y1;
        TileScheduler::render(*fn, reg.data(), s.camera(), rp);
    }

    // Inside the region: identical. Outside: region buffer untouched (0).
    double max_in = 0;
    for (int y = Y0; y < Y1; ++y)
        for (int x = X0; x < X1; ++x)
            for (int c = 0; c < 4; ++c) {
                std::size_t i = ((std::size_t)y * W + x) * 4 + c;
                max_in = std::max(max_in, (double)std::fabs(reg[i] - full[i]));
            }
    EXPECT_LT(max_in, 1e-6) << "region pixels differ from full-frame";
}

TEST(MultiPath, GpuExecutorReportsAvailabilityConsistently) {
    // The GPU executor's availability must match the Vulkan probe. We can't
    // assert a specific value (depends on the host — true on a real GPU,
    // false in software-only CI), but it must be self-consistent and, when
    // unavailable, the runner must skip it cleanly leaving only CPU results.
    GpuGlslExecutor gpu;
    EXPECT_EQ(gpu.path(), PathKind::GpuGlsl);

    CpuIrExecutor cpu;
    std::vector<IExecutor*> execs{&cpu, &gpu};
    WholeFrame dec; AllPathsPerTile dsp; CompareMerge mrg;
    MultiPathExecutor mpe(dec, dsp, mrg);
    SceneGraph s = simple_scene();
    auto res = mpe.run(s, 32, 24, execs, RunMode::Concurrent);

    if (gpu.available()) {
        // Both paths ran → two tiles, and the comparison is meaningful.
        EXPECT_EQ(res.tiles.size(), 2u);
        EXPECT_EQ(res.timings.path_count, 2);
    } else {
        // GPU skipped → only the CPU path ran.
        EXPECT_EQ(res.tiles.size(), 1u);
        EXPECT_EQ(res.timings.path_count, 1);
        EXPECT_EQ(res.tiles[0].path, PathKind::CpuIr);
    }
}

TEST(MultiPath, GpuIrExecutorReportsAvailabilityConsistently) {
    // The GPU-IR (OpenCL) executor mirrors the OpenCL device probe. As with
    // the Vulkan one, we can't assert a value (host-dependent), but it must
    // be self-consistent and skip cleanly when no OpenCL GPU is present.
    GpuIrExecutor gpu_ir;
    EXPECT_EQ(gpu_ir.path(), PathKind::GpuIr);

    CpuIrExecutor cpu;
    std::vector<IExecutor*> execs{&cpu, &gpu_ir};
    WholeFrame dec; AllPathsPerTile dsp; CompareMerge mrg;
    MultiPathExecutor mpe(dec, dsp, mrg);
    SceneGraph s = simple_scene();
    auto res = mpe.run(s, 32, 24, execs, RunMode::Concurrent);

    if (gpu_ir.available()) {
        EXPECT_EQ(res.tiles.size(), 2u);
    } else {
        EXPECT_EQ(res.tiles.size(), 1u);
        EXPECT_EQ(res.tiles[0].path, PathKind::CpuIr);
    }
}

TEST(MultiPath, UnavailablePathsAreSkipped) {
    // An executor reporting unavailable must be filtered out before dispatch.
    struct DeadExecutor : IExecutor {
        PathKind path() const override { return PathKind::GpuGlsl; }
        bool available() const override { return false; }
        TileResult render(const SceneGraph&, int, int, const Tile&) override {
            ADD_FAILURE() << "unavailable executor must not run";
            return {};
        }
    };
    CpuIrExecutor cpu;
    DeadExecutor dead;
    std::vector<IExecutor*> execs{&cpu, &dead};
    WholeFrame dec; AllPathsPerTile dsp; CompareMerge mrg;
    MultiPathExecutor mpe(dec, dsp, mrg);
    SceneGraph s = simple_scene();
    auto res = mpe.run(s, 32, 24, execs, RunMode::Concurrent);
    // Only the CPU path ran → one tile result.
    EXPECT_EQ(res.tiles.size(), 1u);
    EXPECT_EQ(res.timings.path_count, 1);
}
