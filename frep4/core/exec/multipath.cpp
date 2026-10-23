// core/exec/multipath.cpp

#include "core/exec/multipath.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <future>
#include <sstream>
#include <cstring>
#include <thread>

namespace frep::exec {

using clk = std::chrono::steady_clock;

const char* path_kind_name(PathKind k) {
    switch (k) {
        case PathKind::CpuIr:   return "cpu_ir";
        case PathKind::GpuGlsl: return "gpu_glsl";
        case PathKind::GpuIr:   return "gpu_ir";
        case PathKind::GpuRtx:  return "gpu_rtx";
        case PathKind::Remote:  return "remote";
        case PathKind::Lan:     return "lan";
    }
    return "unknown";
}

// ── Decompose strategies ────────────────────────────────────────────────────

std::vector<Tile> WholeFrame::decompose(const SceneGraph&, int W, int H) const {
    return std::vector<Tile>{ Tile{0, 0, W, H} };
}

std::vector<Tile> HorizontalBands::decompose(const SceneGraph&, int W, int H) const {
    std::vector<Tile> tiles;
    int bands = std::max(1, n_);
    for (int i = 0; i < bands; ++i) {
        int y0 = (int)((long long)H * i / bands);
        int y1 = (int)((long long)H * (i + 1) / bands);
        if (y1 > y0) tiles.push_back(Tile{0, y0, W, y1});
    }
    return tiles;
}

std::vector<Tile> WeightedBands::decompose(const SceneGraph&, int W, int H) const {
    std::vector<Tile> tiles;
    if (weights_.empty()) return tiles;
    double sum = 0;
    for (double w : weights_) sum += (w > 0 ? w : 0);
    if (sum <= 0) {  // degenerate → equal bands
        int n = (int)weights_.size();
        for (int i = 0; i < n; ++i) {
            int y0 = (int)((long long)H * i / n);
            int y1 = (int)((long long)H * (i + 1) / n);
            if (y1 > y0) tiles.push_back(Tile{0, y0, W, y1});
        }
        return tiles;
    }
    // Band boundaries from the cumulative weight fraction, so band i gets
    // H * weights[i]/sum rows. Rounding accumulates into the boundary (not
    // per-band) so the bands exactly tile [0,H) with no gap or overlap.
    double acc = 0;
    int prev_y = 0;
    for (std::size_t i = 0; i < weights_.size(); ++i) {
        acc += (weights_[i] > 0 ? weights_[i] : 0);
        int y1 = (i + 1 == weights_.size())
                   ? H
                   : (int)std::llround((double)H * acc / sum);
        if (y1 < prev_y) y1 = prev_y;
        if (y1 > H) y1 = H;
        if (y1 > prev_y) tiles.push_back(Tile{0, prev_y, W, y1});
        else tiles.push_back(Tile{0, prev_y, W, prev_y});  // empty band placeholder
        prev_y = y1;
    }
    return tiles;
}

std::vector<Tile> GridDecompose::decompose(const SceneGraph&, int W, int H) const {
    std::vector<Tile> tiles;
    int tw = std::max(1, tw_), th = std::max(1, th_);
    for (int y = 0; y < H; y += th)
        for (int x = 0; x < W; x += tw)
            tiles.push_back(Tile{x, y, std::min(x+tw, W), std::min(y+th, H)});
    return tiles;
}

// ── Dispatch strategies ─────────────────────────────────────────────────────

std::vector<Job> OnePathPerTile::dispatch(
    const std::vector<Tile>& tiles, const std::vector<IExecutor*>& execs) const {
    std::vector<Job> jobs;
    if (execs.empty()) return jobs;
    std::size_t k = 0;
    for (const auto& t : tiles) {
        jobs.push_back(Job{t, execs[k % execs.size()]});
        ++k;
    }
    return jobs;
}

std::vector<Job> AllPathsPerTile::dispatch(
    const std::vector<Tile>& tiles, const std::vector<IExecutor*>& execs) const {
    std::vector<Job> jobs;
    for (const auto& t : tiles)
        for (auto* e : execs)
            jobs.push_back(Job{t, e});
    return jobs;
}

// ── Merge strategies ────────────────────────────────────────────────────────

namespace {

// Blit a tile's pixels into a full-frame float image buffer, whatever
// payload the tile carries (an RGBA8 tile is widened on the way in).
void blit(std::vector<float>& img, int W, int H, const TileResult& r) {
    const Tile& t = r.tile;
    int tw = t.width();
    std::vector<float> scratch;
    const std::vector<float>& px = r.floats(scratch);
    // Fast path: tile fully inside the frame → row-wise memcpy. The scalar
    // per-pixel loop below is only for tiles that overhang the frame.
    if (t.x0 >= 0 && t.y0 >= 0 && t.x1 <= W && t.y1 <= H &&
        px.size() >= (std::size_t)tw * t.height() * 4) {
        for (int y = t.y0; y < t.y1; ++y)
            std::memcpy(&img[((std::size_t)y * W + t.x0) * 4],
                        &px[(std::size_t)(y - t.y0) * tw * 4],
                        (std::size_t)tw * 4 * sizeof(float));
        return;
    }
    for (int y = t.y0; y < t.y1; ++y) {
        if (y < 0 || y >= H) continue;
        for (int x = t.x0; x < t.x1; ++x) {
            if (x < 0 || x >= W) continue;
            std::size_t src = ((std::size_t)(y - t.y0) * tw + (x - t.x0)) * 4;
            std::size_t dst = ((std::size_t)y * W + x) * 4;
            if (src + 3 < px.size())
                for (int c = 0; c < 4; ++c) img[dst + c] = px[src + c];
        }
    }
}

} // namespace

MergeResult StitchMerge::merge(const std::vector<TileResult>& results,
                               int W, int H) const {
    MergeResult m;
    m.width = W; m.height = H;
    m.image.assign((std::size_t)W * H * 4, 0.0f);
    int ok = 0;
    for (const auto& r : results) {
        if (!r.ok) { m.consistent = false; continue; }
        blit(m.image, W, H, r);
        ++ok;
    }
    std::ostringstream os;
    os << "stitched " << ok << "/" << results.size() << " tiles into "
       << W << "x" << H;
    m.report = os.str();
    return m;
}

MergeResult CompareMerge::merge(const std::vector<TileResult>& results,
                                int W, int H) const {
    MergeResult m;
    m.width = W; m.height = H;

    double max_diff = 0, sum_diff = 0;
    std::size_t diff_count = 0;
    int compared_pairs = 0;
    m.image.assign((std::size_t)W * H * 4, 0.0f);

    // Per-pixel divergence histogram: how many pixels diverge by how much.
    // Distinguishes "edge-only" divergence (a few pixels, large) from a
    // "systematic" offset (many pixels, small). Buckets on the per-pixel
    // max-channel difference.
    std::size_t hist_le_tol = 0;   // <= tolerance (effectively equal)
    std::size_t hist_small  = 0;   // (tol, 0.05]
    std::size_t hist_med    = 0;   // (0.05, 0.2]
    std::size_t hist_large  = 0;   // > 0.2
    std::size_t pixel_total = 0;

    std::vector<bool> used(results.size(), false);
    // Per-pixel diff heat map, accumulated over compared path-pairs.
    m.diff_image.assign((std::size_t)W * H * 4, 0.0f);
    for (std::size_t i = 0; i < results.size(); ++i) {
        if (used[i] || !results[i].ok) continue;
        const TileResult& ref = results[i];
        blit(m.image, W, H, ref);
        used[i] = true;
        for (std::size_t j = i + 1; j < results.size(); ++j) {
            if (used[j] || !results[j].ok) continue;
            const TileResult& other = results[j];
            if (other.tile.x0 != ref.tile.x0 || other.tile.y0 != ref.tile.y0 ||
                other.tile.x1 != ref.tile.x1 || other.tile.y1 != ref.tile.y1)
                continue;  // different region
            used[j] = true;
            ++compared_pairs;
            int tw = ref.tile.width();
            std::vector<float> scratch_a, scratch_b;
            const std::vector<float>& pa = ref.floats(scratch_a);
            const std::vector<float>& pb = other.floats(scratch_b);
            std::size_t n = std::min(pa.size(), pb.size());
            std::size_t npix = n / 4;
            for (std::size_t p = 0; p < npix; ++p) {
                double pmax = 0;  // max channel diff at this pixel
                for (int ch = 0; ch < 4; ++ch) {
                    std::size_t k = p * 4 + ch;
                    double d = std::fabs((double)pa[k] - (double)pb[k]);
                    if (d > max_diff) {
                        max_diff = d;
                        m.worst_x = ref.tile.x0 + (int)(p % tw);
                        m.worst_y = ref.tile.y0 + (int)(p / tw);
                    }
                    sum_diff += d; ++diff_count;
                    if (d > pmax) pmax = d;
                }
                ++pixel_total;
                if (pmax <= tol_) ++hist_le_tol;
                else if (pmax <= 0.05) ++hist_small;
                else if (pmax <= 0.2) ++hist_med;
                else ++hist_large;
                // Heat map: red ∝ this pixel's max-channel diff.
                int lx = (int)(p % tw), ly = (int)(p / tw);
                int fx = ref.tile.x0 + lx, fy = ref.tile.y0 + ly;
                if (fx >= 0 && fx < W && fy >= 0 && fy < H) {
                    std::size_t di = ((std::size_t)fy * W + fx) * 4;
                    float dv = (float)pmax;
                    if (dv > m.diff_image[di]) {
                        m.diff_image[di + 0] = dv;
                        m.diff_image[di + 1] = dv * 0.3f;
                        m.diff_image[di + 2] = 0.0f;
                        m.diff_image[di + 3] = 1.0f;
                    }
                }
            }
        }
    }

    m.max_abs_diff = max_diff;
    m.mean_abs_diff = diff_count ? sum_diff / diff_count : 0.0;
    m.consistent = (max_diff <= tol_);

    std::ostringstream os;
    os << "compared " << compared_pairs << " path-pair(s); "
       << "max |Δ| = " << m.max_abs_diff;
    if (m.worst_x >= 0) os << " @ (" << m.worst_x << "," << m.worst_y << ")";
    os << ", mean |Δ| = " << m.mean_abs_diff
       << "  (tolerance " << tol_ << ")  → "
       << (m.consistent ? "CONSISTENT" : "DIVERGENT");
    if (pixel_total > 0) {
        auto pct = [&](std::size_t n) {
            return 100.0 * (double)n / (double)pixel_total;
        };
        os << "\n  pixel divergence: "
           << "≤tol " << pct(hist_le_tol) << "%, "
           << "(tol,0.05] " << pct(hist_small) << "%, "
           << "(0.05,0.2] " << pct(hist_med) << "%, "
           << ">0.2 " << pct(hist_large) << "%";
    }
    m.report = os.str();
    return m;
}

// ── Tile queue cursor ─────────────────────────────────────────────────────

TileQueueCursor::TileQueueCursor(std::size_t n_tiles, std::size_t n_execs, QueuePolicy p)
    : n_(n_tiles),
      policy_(p == QueuePolicy::Auto ? QueuePolicy::FastestTail : p),
      ema_(n_execs, 0.0) {}

std::size_t TileQueueCursor::take(std::size_t e, double last_ms) {
    std::lock_guard<std::mutex> lk(m_);
    if (e < ema_.size() && last_ms > 0)
        ema_[e] = ema_[e] > 0 ? 0.7 * ema_[e] + 0.3 * last_ms : last_ms;
    if (next_ >= n_) return npos;
    if (policy_ == QueuePolicy::FastestTail && e < ema_.size() && ema_[e] > 0) {
        double fastest = ema_[e];
        for (double v : ema_) if (v > 0 && v < fastest) fastest = v;
        const double k = ema_[e] / fastest;
        const std::size_t remaining = n_ - next_;
        // A tile here would outlast what the fastest executor finishes in
        // the same time; leave the tail to it. (The fast one keeps pulling
        // until the queue is empty, so nothing is left behind.)
        if (k > 1.5 && static_cast<double>(remaining) < k) return npos;
    }
    return next_++;
}

// ── Runner ────────────────────────────────────────────────────────────────

MultiPathResult MultiPathExecutor::run(const SceneGraph& scene, int W, int H,
                                       const std::vector<IExecutor*>& executors,
                                       RunMode mode) {
    MultiPathResult out;
    auto wall0 = clk::now();

    // Filter to available executors only.
    std::vector<IExecutor*> avail;
    for (auto* e : executors)
        if (e && e->available()) avail.push_back(e);

    auto tiles = decompose_.decompose(scene, W, H);

    if (mode == RunMode::DynamicQueue) {
        // Work-stealing: one worker thread per available executor, each
        // pulling the next tile from a shared atomic cursor until the queue
        // drains. A faster executor completes its tile sooner and grabs the
        // next, so it naturally renders proportionally more tiles — load
        // balances itself with no calibration. The first tile each worker
        // takes warms its compile cache (cold start); every later tile reuses
        // it. Each tile is rendered by exactly one executor (like a split),
        // so results tile the frame and StitchMerge reassembles them.
        out.tiles.resize(tiles.size());
        TileQueueCursor cursor(tiles.size(), avail.size(), queue_policy_);
        std::vector<std::thread> workers;
        workers.reserve(avail.size());
        for (std::size_t k = 0; k < avail.size(); ++k) {
            IExecutor* e = avail[k];
            workers.emplace_back([&, e, k] {
                double last_ms = 0;
                for (;;) {
                    std::size_t i = cursor.take(k, last_ms);
                    if (i == TileQueueCursor::npos) break;
                    auto t0 = clk::now();
                    out.tiles[i] = e->render(scene, W, H, tiles[i]);
                    last_ms = std::chrono::duration<double, std::milli>(clk::now() - t0).count();
                }
            });
        }
        for (auto& t : workers) t.join();

        for (const auto& r : out.tiles) out.timings.sum_render_ms += r.render_ms;
        out.timings.job_count  = (int)tiles.size();
        out.timings.path_count = (int)avail.size();
        out.merged = merge_.merge(out.tiles, W, H);
        out.timings.wall_ms = std::chrono::duration<double, std::milli>(clk::now() - wall0).count();
        return out;
    }

    auto jobs  = dispatch_.dispatch(tiles, avail);

    out.tiles.resize(jobs.size());

    if (mode == RunMode::Concurrent) {
        std::vector<std::future<void>> futs;
        futs.reserve(jobs.size());
        for (std::size_t i = 0; i < jobs.size(); ++i) {
            futs.push_back(std::async(std::launch::async, [&, i] {
                out.tiles[i] = jobs[i].executor->render(scene, W, H, jobs[i].tile);
            }));
        }
        for (auto& f : futs) f.get();
    } else {
        for (std::size_t i = 0; i < jobs.size(); ++i)
            out.tiles[i] = jobs[i].executor->render(scene, W, H, jobs[i].tile);
    }

    for (const auto& r : out.tiles) out.timings.sum_render_ms += r.render_ms;
    out.timings.job_count  = (int)jobs.size();
    out.timings.path_count = (int)avail.size();

    out.merged = merge_.merge(out.tiles, W, H);
    out.timings.wall_ms = std::chrono::duration<double, std::milli>(clk::now() - wall0).count();
    return out;
}

} // namespace frep::exec
