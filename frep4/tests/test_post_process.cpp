// tests/test_post_process.cpp
//
// Tests for the PostProcessStage abstraction — the first formal path stage.
// Focus on the SSAA box-downsample correctness and pipeline composition.

#include <gtest/gtest.h>
#include "core/postprocess/post_process.hpp"
#include <cmath>

using namespace frep::post;

namespace {
// Build a w×h frame where every channel of pixel (x,y) is `value`.
Frame solid(int w, int h, float value) {
    Frame f;
    f.w = w; f.h = h;
    f.rgba.assign(static_cast<std::size_t>(w) * h * 4, value);
    return f;
}
}  // namespace

TEST(PostProcess, SSAAFactorOneIsPassThrough) {
    BoxDownsampleSSAA s(1);
    Frame in = solid(4, 3, 0.5f);
    Frame out = s.apply(in);
    EXPECT_EQ(out.w, 4);
    EXPECT_EQ(out.h, 3);
    EXPECT_EQ(out.rgba, in.rgba);
}

TEST(PostProcess, SSAADownsamplesResolution) {
    // 8×6 at factor 2 → 4×3.
    BoxDownsampleSSAA s(2);
    Frame in = solid(8, 6, 1.0f);
    Frame out = s.apply(in);
    EXPECT_EQ(out.w, 4);
    EXPECT_EQ(out.h, 3);
    ASSERT_TRUE(out.valid());
}

TEST(PostProcess, SSAAAveragesBlock) {
    // 2×2 source, factor 2 → single output texel = mean of the 4 inputs.
    Frame in;
    in.w = 2; in.h = 2;
    in.rgba = {
        0.0f, 0.0f, 0.0f, 1.0f,   // (0,0)
        1.0f, 0.0f, 0.0f, 1.0f,   // (1,0)
        0.0f, 1.0f, 0.0f, 1.0f,   // (0,1)
        1.0f, 1.0f, 0.0f, 1.0f,   // (1,1)
    };
    BoxDownsampleSSAA s(2);
    Frame out = s.apply(in);
    ASSERT_EQ(out.w, 1);
    ASSERT_EQ(out.h, 1);
    EXPECT_FLOAT_EQ(out.rgba[0], 0.5f);   // R mean (0+1+0+1)/4
    EXPECT_FLOAT_EQ(out.rgba[1], 0.5f);   // G mean (0+0+1+1)/4
    EXPECT_FLOAT_EQ(out.rgba[2], 0.0f);   // B
    EXPECT_FLOAT_EQ(out.rgba[3], 1.0f);   // A
}

TEST(PostProcess, EmptyPipelineIsIdentity) {
    PostProcessPipeline pipe;
    EXPECT_TRUE(pipe.empty());
    Frame in = solid(5, 5, 0.3f);
    Frame out = pipe.apply(in);
    EXPECT_EQ(out.w, 5);
    EXPECT_EQ(out.rgba, in.rgba);
}

TEST(PostProcess, PipelineAppliesStagesInOrder) {
    BoxDownsampleSSAA s(2);
    PostProcessPipeline pipe;
    pipe.add(&s);
    Frame in = solid(8, 8, 1.0f);
    Frame out = pipe.apply(in);
    EXPECT_EQ(out.w, 4);
    EXPECT_EQ(out.h, 4);
}

TEST(PostProcess, ReinhardToneMapsHDR) {
    // Reinhard: x/(1+x). Input 1.0 → 0.5; preserves resolution; alpha intact.
    Frame in = solid(2, 2, 1.0f);
    ToneMap tm(ToneMap::Op::Reinhard);
    Frame out = tm.apply(in);
    EXPECT_EQ(out.w, 2);
    EXPECT_FLOAT_EQ(out.rgba[0], 0.5f);   // R
    EXPECT_FLOAT_EQ(out.rgba[3], 1.0f);   // A untouched
}

TEST(PostProcess, ACESClampsToUnit) {
    // Large HDR input must map into [0,1].
    Frame in = solid(2, 2, 100.0f);
    ToneMap tm(ToneMap::Op::ACES);
    Frame out = tm.apply(in);
    EXPECT_LE(out.rgba[0], 1.0f);
    EXPECT_GE(out.rgba[0], 0.0f);
}

TEST(PostProcess, GammaEncodesMidtone) {
    // gamma 2.0: 0.25 → sqrt(0.25) = 0.5.
    Frame in = solid(2, 2, 0.25f);
    GammaCorrect g(2.0f);
    Frame out = g.apply(in);
    EXPECT_NEAR(out.rgba[0], 0.5f, 1e-5f);
    EXPECT_FLOAT_EQ(out.rgba[3], 0.25f);  // alpha untouched
}

TEST(PostProcess, BilateralDenoisePreservesFlatRegion) {
    // A perfectly flat image must be unchanged by the bilateral filter.
    Frame in = solid(5, 5, 0.4f);
    BilateralDenoise d(1, 1.0f, 0.1f);
    Frame out = d.apply(in);
    for (float v : out.rgba) EXPECT_NEAR(v, 0.4f, 1e-5f);
}

TEST(PostProcess, BilateralDenoiseKeepsResolution) {
    Frame in = solid(10, 8, 0.5f);
    BilateralDenoise d(2, 2.0f, 0.2f);
    Frame out = d.apply(in);
    EXPECT_EQ(out.w, 10);
    EXPECT_EQ(out.h, 8);
}

TEST(PostProcess, MultiStagePipeline) {
    // SSAA downsample → tone-map → gamma, applied in order.
    BoxDownsampleSSAA ss(2);
    ToneMap tm(ToneMap::Op::Reinhard);
    GammaCorrect g(2.2f);
    PostProcessPipeline pipe;
    pipe.add(&ss); pipe.add(&tm); pipe.add(&g);
    Frame in = solid(8, 8, 1.0f);
    Frame out = pipe.apply(in);
    EXPECT_EQ(out.w, 4);          // downsampled
    EXPECT_EQ(out.h, 4);
    // 1.0 → Reinhard 0.5 → gamma 0.5^(1/2.2) ≈ 0.7297
    EXPECT_NEAR(out.rgba[0], std::pow(0.5f, 1.0f/2.2f), 1e-4f);
}
