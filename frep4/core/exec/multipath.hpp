// core/exec/multipath.hpp
//
// Model D — a general framework for splitting a render across multiple
// retargeting paths / executors and recombining the results. Three
// orthogonal, pluggable strategies, so the various "models" are just
// configurations of one machine:
//
//   decompose : how the final result is split into rectangular Tiles
//   dispatch  : which path(s) render which Tile, and how (concurrent/serial)
//   merge     : what to do with the returned tiles (stitch / compare / pick)
//
//   whole frame  + all paths  + compare  →  cross-path equivalence check
//   two halves   + one path each + stitch →  CPU/GPU frame split
//   many tiles   + remote paths + stitch  →  distributed render (future)
//
// Visual equivalence across paths is a PREREQUISITE for stitch/split to be
// usable (a seam appears otherwise), so `compare` is both a model in its
// own right and the diagnostic that makes the others sound. This header
// defines the vocabulary; concrete executors and strategies live in their
// own units.

#pragma once

#include "core/frep/scene.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

namespace frep { class CompilePolicy; }   // core/compiler/compile_policy.hpp

namespace frep::exec {

// A rectangular region of the output image: pixels [x0,x1) × [y0,y1).
// A whole frame is {0,0,W,H}. (A future animation dimension would add a
// frame index here.)
struct Tile {
    int x0 = 0, y0 = 0, x1 = 0, y1 = 0;
    int width()  const { return x1 - x0; }
    int height() const { return y1 - y0; }
    bool empty() const { return width() <= 0 || height() <= 0; }
};

// Identifies a retargeting path / executor. The local paths mirror the
// three the system already has; Remote is a placeholder for 2b.
enum class PathKind {
    CpuIr,     // codegen → LLVM IR → JIT → native (TileScheduler)
    GpuGlsl,   // codegen → GLSL → glslang → SPIR-V → Vulkan
    GpuIr,     // codegen → LLVM IR → llvm-spirv → SPIR-V → Vulkan
    GpuRtx,    // codegen → GLSL RT shaders → Vulkan ray tracing (hardware BVH
               //   broad-phase + SDF sphere-trace in a custom intersection
               //   shader). Falls back to a software BVH walk where the
               //   VK_KHR_ray_tracing_pipeline extension is absent.
    Remote,    // executed on another machine (future, 2b)
    Lan,       // not a local executor — orchestrates a distributed LAN render
               //   (starts the master in the LAN tab; results stream back into
               //   the viewport). Selectable like a path; re-renders the frame
               //   on every scene change in an interactive loop.
};

const char* path_kind_name(PathKind k);

// How a TileResult carries its pixels. The GPU paths produce the frame as
// 8 bits per channel and the CPU path as floats; handing each back in its
// own form costs nothing, while widening a 512² RGBA8 frame to float was
// 0.4 ms of a 0.55 ms GPU frame. TracerConfig::output_format chooses; a
// consumer that needs floats whatever the path calls floats() / to_float().
enum class PixelFormat { Float, RGBA8 };

// The result of rendering one Tile on one path: the pixels for exactly
// that tile region (row-major RGBA, width()*height()*4 values, in `rgba`
// as floats or in `rgba8` as bytes according to `format`), plus
// per-execution metrics. `ok` is false if that path failed (e.g. no
// Vulkan device); `error` then explains.
struct TileResult {
    Tile        tile;
    PathKind    path;
    PixelFormat format = PixelFormat::Float;
    std::vector<float>        rgba;    // when format == Float
    std::vector<std::uint8_t> rgba8;   // when format == RGBA8
    bool        ok = false;
    std::string error;
    // Metrics (per this tile/path execution).
    double      compile_ms = 0;  // codegen + JIT / GLSL+SPIR-V build
    double      render_ms  = 0;  // the actual dispatch
    std::size_t peak_rss_kb = 0;
    int         threads_used = 0;

    std::size_t pixel_count() const {
        return format == PixelFormat::Float ? rgba.size() / 4 : rgba8.size() / 4;
    }

    // The pixels as floats whatever the format: `rgba` itself, or `rgba8`
    // widened into `scratch` (which the returned reference then aliases).
    const std::vector<float>& floats(std::vector<float>& scratch) const {
        if (format == PixelFormat::Float) return rgba;
        scratch.resize(rgba8.size());
        constexpr float k = 1.0f / 255.0f;
        for (std::size_t i = 0; i < rgba8.size(); ++i) scratch[i] = rgba8[i] * k;
        return scratch;
    }

    // Convert in place.
    void to_float() {
        if (format == PixelFormat::Float) return;
        rgba.resize(rgba8.size());
        constexpr float k = 1.0f / 255.0f;
        for (std::size_t i = 0; i < rgba8.size(); ++i) rgba[i] = rgba8[i] * k;
        rgba8.clear(); rgba8.shrink_to_fit();
        format = PixelFormat::Float;
    }
    void to_rgba8() {
        if (format == PixelFormat::RGBA8) return;
        rgba8.resize(rgba.size());
        for (std::size_t i = 0; i < rgba.size(); ++i) {
            float v = rgba[i];
            v = v < 0.0f ? 0.0f : (v > 1.0f ? 1.0f : v);
            rgba8[i] = static_cast<std::uint8_t>(v * 255.0f + 0.5f);
        }
        rgba.clear(); rgba.shrink_to_fit();
        format = PixelFormat::RGBA8;
    }
};

// An executor renders a given Tile of a scene on one specific path. The
// implementation owns whatever backend state it needs (JIT engine, Vulkan
// context, …). render() must fill result.rgba for exactly `tile`.
class IExecutor {
public:
    virtual ~IExecutor() = default;
    virtual PathKind path() const = 0;
    // Whether this executor can run in the current environment (e.g. a GPU
    // executor returns false when no Vulkan device is present). Checked
    // before dispatch so unavailable paths are skipped, not errored.
    virtual bool available() const = 0;
    // Render exactly `tile` of `scene` at full-frame dimensions (W,H) — the
    // tile coordinates are in that full-frame space, so a path can compute
    // the correct rays for a sub-region. Returns pixels for the tile only.
    virtual TileResult render(const SceneGraph& scene, int W, int H,
                              const Tile& tile) = 0;

    // Install a compile policy (core/compiler/compile_policy.hpp): parameters
    // the policy places at runtime become buffer loads, so editing their
    // values re-uploads a buffer instead of recompiling the path. Not owned;
    // null restores the all-baked default. Paths that have no runtime
    // parameter mechanism ignore it.
    virtual void set_compile_policy(const CompilePolicy* policy) { (void)policy; }
};

// ── Strategies (abstract base classes) ──────────────────────────────────────
//
// Each of the three strategy axes is an abstract base with a virtual
// method. Built-in strategies are concrete subclasses; users can subclass
// to add custom behaviour. Keeping the three axes separate means new
// behaviours compose — N decompose × M dispatch × K merge — instead of
// needing a concrete class per combination.

// Decompose: full-frame dimensions → the set of tiles to render.
class IDecomposeStrategy {
public:
    virtual ~IDecomposeStrategy() = default;
    virtual std::vector<Tile> decompose(const SceneGraph&, int W, int H) const = 0;
};

// A unit of work: render one tile on one executor.
struct Job {
    Tile      tile;
    IExecutor* executor = nullptr;
};

// Dispatch: given the tiles and available executors, produce the jobs
// (which executor renders which tile — possibly several per tile for
// comparison). Concurrency is handled by the runner, not here.
class IDispatchStrategy {
public:
    virtual ~IDispatchStrategy() = default;
    virtual std::vector<Job> dispatch(const std::vector<Tile>&,
                                      const std::vector<IExecutor*>&) const = 0;
};

// Outcome of a merge: an optional stitched image (W*H*4 RGBA float) plus a
// free-form textual report. Either may be empty depending on the merge.
struct MergeResult {
    std::vector<float> image;     // W*H*4, empty if the merge doesn't stitch
    int                width = 0, height = 0;
    std::string        report;
    bool               consistent = true;  // compare: did paths agree?
    double             max_abs_diff = 0;    // compare: worst channel diff
    double             mean_abs_diff = 0;
    int                worst_x = -1, worst_y = -1;  // pixel of the worst diff
    // Optional per-pixel difference heat map (W*H*4 RGBA float), filled by
    // CompareMerge: brightness ∝ max channel difference at that pixel, so
    // where the paths diverge lights up. Empty for non-compare merges.
    std::vector<float> diff_image;
};

// Merge: combine/inspect the tile results into a final outcome.
class IMergeStrategy {
public:
    virtual ~IMergeStrategy() = default;
    virtual MergeResult merge(const std::vector<TileResult>&, int W, int H) const = 0;
};

// ── Built-in decompose strategies ───────────────────────────────────────────

// The whole frame as a single tile (Model B base; trivial single-path case).
class WholeFrame : public IDecomposeStrategy {
public:
    std::vector<Tile> decompose(const SceneGraph&, int W, int H) const override;
};

// `n` horizontal bands, top-to-bottom. Two bands = classic CPU/GPU split.
class HorizontalBands : public IDecomposeStrategy {
public:
    explicit HorizontalBands(int n) : n_(n) {}
    std::vector<Tile> decompose(const SceneGraph&, int W, int H) const override;
private:
    int n_;
};

// Horizontal bands sized by per-executor weights, so a faster executor gets
// a proportionally taller band and all bands finish at about the same time
// (load balancing for a heterogeneous CPU+GPU split). `weights[i]` is the
// relative throughput of executor i (higher = faster); band i's height is
// H * weights[i] / sum(weights). Weights can be supplied directly or filled
// by a calibration pass (a small trial render per executor). Band order
// matches executor order, so OnePathPerTile assigns band i to executor i.
class WeightedBands : public IDecomposeStrategy {
public:
    explicit WeightedBands(std::vector<double> weights)
        : weights_(std::move(weights)) {}
    std::vector<Tile> decompose(const SceneGraph&, int W, int H) const override;
private:
    std::vector<double> weights_;
};

// Regular grid of tile_w × tile_h tiles (distributed-render shape).
class GridDecompose : public IDecomposeStrategy {
public:
    GridDecompose(int tile_w, int tile_h) : tw_(tile_w), th_(tile_h) {}
    std::vector<Tile> decompose(const SceneGraph&, int W, int H) const override;
private:
    int tw_, th_;
};

// ── Built-in dispatch strategies ─────────────────────────────────────────────

// Assign each tile to exactly one executor, round-robin (split/distribute).
class OnePathPerTile : public IDispatchStrategy {
public:
    std::vector<Job> dispatch(const std::vector<Tile>&,
                              const std::vector<IExecutor*>&) const override;
};

// Assign every tile to EVERY executor (cross-path comparison).
class AllPathsPerTile : public IDispatchStrategy {
public:
    std::vector<Job> dispatch(const std::vector<Tile>&,
                              const std::vector<IExecutor*>&) const override;
};

// ── Built-in merge strategies ─────────────────────────────────────────────────

// Stitch tiles into one image (each pixel covered once; last writer wins).
class StitchMerge : public IMergeStrategy {
public:
    MergeResult merge(const std::vector<TileResult>&, int W, int H) const override;
};

// Compare tile results covering the same region across paths: report
// max/mean per-channel diff and whether they agree within `tolerance`
// (0..1 per channel). Stitches the first path's image for reference. This
// is the visual-equivalence diagnostic that makes stitching sound.
class CompareMerge : public IMergeStrategy {
public:
    explicit CompareMerge(double tolerance = 2.0 / 255.0) : tol_(tolerance) {}
    MergeResult merge(const std::vector<TileResult>&, int W, int H) const override;
private:
    double tol_;
};

// ── Runner ──────────────────────────────────────────────────────────────────

// How jobs are executed.
enum class RunMode {
    Concurrent,   // each job on its own thread (std::async), joined at end
    Serial,       // jobs run one after another (baseline / debugging)
    DynamicQueue, // work-stealing: one worker per executor pulls tiles from a
                  // shared queue as it finishes, so a faster executor takes
                  // proportionally more tiles — self-balancing, no calibration.
};

// How a dynamic tile queue hands out its last tiles.
//   Plain       — every executor pulls until the queue is empty. The slowest
//                 can then start a tile the fastest would have finished many
//                 times over, and the frame waits for it: measured at 800² with
//                 the GPU 5–20× the CPU's speed, the combined run reached only
//                 84–96% of the two paths' summed throughput, and on the
//                 heaviest scenes was slower than the GPU alone.
//   FastestTail — an executor whose tiles take k times the fastest's does not
//                 start one when fewer than k tiles remain; the tail goes to
//                 the fast one. What Auto means today.
enum class QueuePolicy { Auto, Plain, FastestTail };

// The shared cursor of a dynamic tile queue, with the policy applied. One
// per frame; every executor's worker calls take() until it returns npos.
class TileQueueCursor {
public:
    static constexpr std::size_t npos = static_cast<std::size_t>(-1);
    TileQueueCursor(std::size_t n_tiles, std::size_t n_execs, QueuePolicy p);
    // Next tile for executor `e`, or npos when it should stop. `last_ms` is
    // how long its previous tile took (<= 0 when it has none yet); it feeds a
    // per-executor smoothed tile time the tail rule compares.
    std::size_t take(std::size_t e, double last_ms);
    std::size_t taken() const { return next_; }
private:
    std::size_t          n_;
    QueuePolicy          policy_;
    std::mutex           m_;
    std::size_t          next_ = 0;
    std::vector<double>  ema_;   // per-executor tile time, smoothed
};

// The wall-clock and sum-of-parts timings of a run — concurrent execution
// should make wall < sum when paths overlap.
struct RunTimings {
    double wall_ms = 0;       // total elapsed for the whole run
    double sum_render_ms = 0; // Σ of per-job render_ms (serial-equivalent)
    int    job_count = 0;
    int    path_count = 0;
};

struct MultiPathResult {
    MergeResult            merged;
    std::vector<TileResult> tiles;   // every job's result (for inspection)
    RunTimings             timings;
};

// Orchestrates one run: decompose → dispatch → execute (concurrent/serial)
// → merge. Strategies and executors are borrowed (caller owns them), so
// the same strategy objects can be reused across runs.
class MultiPathExecutor {
public:
    MultiPathExecutor(const IDecomposeStrategy& decompose,
                      const IDispatchStrategy&  dispatch,
                      const IMergeStrategy&     merge)
        : decompose_(decompose), dispatch_(dispatch), merge_(merge) {}

    MultiPathResult run(const SceneGraph& scene, int W, int H,
                        const std::vector<IExecutor*>& executors,
                        RunMode mode = RunMode::Concurrent);

    // Tail policy of RunMode::DynamicQueue (see QueuePolicy). Auto by default.
    void set_queue_policy(QueuePolicy p) { queue_policy_ = p; }

private:
    QueuePolicy               queue_policy_ = QueuePolicy::Auto;
    const IDecomposeStrategy& decompose_;
    const IDispatchStrategy&  dispatch_;
    const IMergeStrategy&     merge_;
};

} // namespace frep::exec
