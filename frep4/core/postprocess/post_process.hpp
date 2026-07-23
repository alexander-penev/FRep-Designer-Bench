#pragma once
// core/postprocess/post_process.hpp
//
// PostProcessStage — the first formal "stage" in the path model
// (docs/ARCHITECTURE_PATHS.md). A path is Model → emit → render → POST-PROCESS
// → present; post-process is just another algorithm on the path, applied to a
// whole rendered frame rather than per tile. Pulling it out of the viewport
// into a reusable stage lets the multipath executor apply it AFTER stitching
// (on the assembled frame), which is the architecturally correct place: a
// supersample/denoise/DOF kernel needs neighbouring pixels, so it must run on
// the whole image, not on independently-rendered tiles.
//
// A stage maps a float RGBA image to a float RGBA image and may change the
// resolution (e.g. SSAA downsample: hi-res in, target-res out). Stages
// compose: the output of one is the input of the next.

#include <cstdint>
#include <cmath>
#include <vector>

namespace frep::post {

// A frame buffer: tightly packed float RGBA, row-major, `w`×`h`.
struct Frame {
    std::vector<float> rgba;   // w*h*4
    int w = 0, h = 0;

    Frame() = default;
    Frame(std::vector<float> px, int width, int height)
        : rgba(std::move(px)), w(width), h(height) {}

    bool valid() const {
        return w > 0 && h > 0 &&
               rgba.size() == static_cast<std::size_t>(w) * h * 4;
    }
};

// Abstract post-process stage: image → image, possibly resizing.
class PostProcessStage {
public:
    virtual ~PostProcessStage() = default;
    // Apply to `in`, returning the processed frame. Implementations must not
    // assume `in` outlives the call. A no-op stage returns `in` unchanged.
    virtual Frame apply(const Frame& in) const = 0;
    // Human-readable name for logging/reporting.
    virtual const char* name() const = 0;
};

// Box-downsample SSAA: the frame was rendered at `factor`× the target
// resolution; average each factor×factor block into one output texel (a true
// box filter). factor == 1 is a pass-through. This is the exact filter the
// viewport applied inline; here it is a reusable stage that runs on the whole
// stitched frame, so a CPU+GPU split supersamples correctly across the seam.
class BoxDownsampleSSAA final : public PostProcessStage {
public:
    explicit BoxDownsampleSSAA(int factor) : factor_(factor < 1 ? 1 : factor) {}

    Frame apply(const Frame& in) const override {
        if (factor_ == 1 || !in.valid()) return in;
        const int ss = factor_;
        const int out_w = in.w / ss;
        const int out_h = in.h / ss;
        if (out_w <= 0 || out_h <= 0) return in;

        Frame out;
        out.w = out_w; out.h = out_h;
        out.rgba.assign(static_cast<std::size_t>(out_w) * out_h * 4, 0.0f);
        const float inv = 1.0f / static_cast<float>(ss * ss);
        for (int y = 0; y < out_h; ++y) {
            for (int x = 0; x < out_w; ++x) {
                float acc[4] = {0, 0, 0, 0};
                for (int sy = 0; sy < ss; ++sy) {
                    const std::size_t srow =
                        static_cast<std::size_t>(y * ss + sy) * in.w;
                    for (int sx = 0; sx < ss; ++sx) {
                        const float* s =
                            &in.rgba[(srow + x * ss + sx) * 4];
                        acc[0] += s[0]; acc[1] += s[1];
                        acc[2] += s[2]; acc[3] += s[3];
                    }
                }
                float* d = &out.rgba[(static_cast<std::size_t>(y) * out_w + x) * 4];
                d[0] = acc[0] * inv; d[1] = acc[1] * inv;
                d[2] = acc[2] * inv; d[3] = acc[3] * inv;
            }
        }
        return out;
    }

    const char* name() const override { return "BoxDownsampleSSAA"; }
    int factor() const { return factor_; }

private:
    int factor_;
};

// Tone-mapping: compress HDR radiance into the [0,1] display range. Two
// operators, both pixel-wise on RGB (alpha untouched), neither changing the
// resolution. Useful when lighting pushes channels above 1.0 — clamping
// alone clips highlights to flat white, whereas tone-mapping rolls them off.
class ToneMap final : public PostProcessStage {
public:
    enum class Op { Reinhard, ACES };
    explicit ToneMap(Op op, float exposure = 1.0f)
        : op_(op), exposure_(exposure) {}

    Frame apply(const Frame& in) const override {
        if (!in.valid()) return in;
        Frame out = in;  // copy; we rewrite RGB in place
        const std::size_t n = static_cast<std::size_t>(out.w) * out.h;
        for (std::size_t i = 0; i < n; ++i) {
            float* p = &out.rgba[i * 4];
            for (int c = 0; c < 3; ++c) {
                float x = p[c] * exposure_;
                p[c] = (op_ == Op::Reinhard) ? reinhard(x) : aces(x);
            }
        }
        return out;
    }
    const char* name() const override {
        return op_ == Op::Reinhard ? "ToneMap(Reinhard)" : "ToneMap(ACES)";
    }

private:
    static float reinhard(float x) { return x / (1.0f + x); }
    static float aces(float x) {
        // Narkowicz 2015 ACES fit.
        const float a = 2.51f, b = 0.03f, c = 2.43f, d = 0.59f, e = 0.14f;
        float v = (x * (a * x + b)) / (x * (c * x + d) + e);
        return v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
    }
    Op    op_;
    float exposure_;
};

// Gamma encode: linear → display (sRGB-ish power curve). The CPU/GPU shading
// paths output linear radiance clamped to [0,1]; a gamma stage applied to the
// assembled frame encodes it for a display, identically across all paths (so
// it does not perturb cross-path equivalence). Pixel-wise on RGB.
class GammaCorrect final : public PostProcessStage {
public:
    explicit GammaCorrect(float gamma = 2.2f)
        : inv_gamma_(gamma > 0 ? 1.0f / gamma : 1.0f) {}

    Frame apply(const Frame& in) const override {
        if (!in.valid()) return in;
        Frame out = in;
        const std::size_t n = static_cast<std::size_t>(out.w) * out.h;
        for (std::size_t i = 0; i < n; ++i) {
            float* p = &out.rgba[i * 4];
            for (int c = 0; c < 3; ++c) {
                float x = p[c] < 0.0f ? 0.0f : p[c];
                p[c] = std::pow(x, inv_gamma_);
            }
        }
        return out;
    }
    const char* name() const override { return "GammaCorrect"; }

private:
    float inv_gamma_;
};

// Edge-preserving denoise: a small bilateral filter (spatial + range weights)
// over a (2r+1)² window. Unlike SSAA this is a genuine neighbour-based stage,
// so it MUST run on the whole assembled frame — across a split it would seam
// if applied per tile. Smooths AO/sampling noise while keeping silhouettes.
class BilateralDenoise final : public PostProcessStage {
public:
    BilateralDenoise(int radius, float sigma_spatial, float sigma_range)
        : r_(radius < 1 ? 1 : radius),
          inv2ss_(1.0f / (2.0f * sigma_spatial * sigma_spatial)),
          inv2sr_(1.0f / (2.0f * sigma_range * sigma_range)) {}

    Frame apply(const Frame& in) const override {
        if (!in.valid()) return in;
        Frame out = in;
        for (int y = 0; y < in.h; ++y)
            for (int x = 0; x < in.w; ++x)
                filter_pixel(in, out, x, y);
        return out;
    }
    const char* name() const override { return "BilateralDenoise"; }

private:
    void filter_pixel(const Frame& in, Frame& out, int x, int y) const {
        const float* ctr = &in.rgba[(static_cast<std::size_t>(y) * in.w + x) * 4];
        float sum[3] = {0, 0, 0};
        float wsum = 0.0f;
        for (int dy = -r_; dy <= r_; ++dy) {
            int yy = y + dy; if (yy < 0 || yy >= in.h) continue;
            for (int dx = -r_; dx <= r_; ++dx) {
                int xx = x + dx; if (xx < 0 || xx >= in.w) continue;
                const float* s = &in.rgba[(static_cast<std::size_t>(yy) * in.w + xx) * 4];
                float spatial = (dx * dx + dy * dy) * inv2ss_;
                float dr = s[0]-ctr[0], dg = s[1]-ctr[1], db = s[2]-ctr[2];
                float range = (dr*dr + dg*dg + db*db) * inv2sr_;
                float w = std::exp(-(spatial + range));
                sum[0] += s[0]*w; sum[1] += s[1]*w; sum[2] += s[2]*w;
                wsum += w;
            }
        }
        float* d = &out.rgba[(static_cast<std::size_t>(y) * in.w + x) * 4];
        if (wsum > 0) { d[0]=sum[0]/wsum; d[1]=sum[1]/wsum; d[2]=sum[2]/wsum; }
        // alpha (d[3]) already copied from `in` via the initial out = in
    }
    int   r_;
    float inv2ss_, inv2sr_;
};

// A pipeline of stages applied in order. The output resolution is whatever
// the last stage produces. An empty pipeline is the identity.
class PostProcessPipeline {
public:
    void add(const PostProcessStage* stage) { stages_.push_back(stage); }
    bool empty() const { return stages_.empty(); }

    Frame apply(Frame frame) const {
        for (const auto* s : stages_)
            if (s) frame = s->apply(frame);
        return frame;
    }

private:
    std::vector<const PostProcessStage*> stages_;  // not owned
};

} // namespace frep::post
