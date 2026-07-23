#pragma once
// core/dist/scheduler.hpp
//
// Tile scheduling for the distributed master. A scheduler decides which tile
// a worker renders next. Two models behind one interface:
//
//   PullScheduler  — work-stealing. A worker asks for work; the scheduler
//                    hands out the next tile from a shared queue. Self-
//                    balancing across heterogeneous workers and variable
//                    network latency (the distributed analogue of the local
//                    DynamicQueue run mode). This is the primary model.
//
//   PushScheduler  — pre-assignment. Tiles are partitioned among workers up
//                    front (round-robin here; a weighted variant could use a
//                    per-worker hint). Simpler, no rebalancing — useful when
//                    worker capabilities are known ahead of time.
//
// The scheduler is transport-agnostic: it works in tile indices, not sockets.
// The master drives it (see master.hpp): on a worker's TileRequest it calls
// next_for(worker_id); a -1 result means "no more work".

#include "core/exec/multipath.hpp"   // frep::Tile

#include <atomic>
#include <cstddef>
#include <mutex>
#include <vector>

namespace frep::dist {

class IScheduler {
public:
    virtual ~IScheduler() = default;

    // Return the index of the next tile for worker `worker_id`, or -1 if no
    // work remains for it. Must be thread-safe: the master may serve several
    // workers concurrently. `worker_id` is in [0, n_workers).
    virtual long next_for(int worker_id) = 0;

    // Total number of tiles to be rendered (for progress / sizing).
    virtual std::size_t tile_count() const = 0;
};

// Work-stealing: a single shared atomic cursor; every worker pulls the next
// untaken tile regardless of id. A fast worker simply calls back sooner and
// takes more tiles. worker_id is ignored (any worker can take any tile).
class PullScheduler final : public IScheduler {
public:
    explicit PullScheduler(std::size_t n_tiles) : n_(n_tiles) {}

    long next_for(int /*worker_id*/) override {
        std::size_t i = cursor_.fetch_add(1, std::memory_order_relaxed);
        return i < n_ ? static_cast<long>(i) : -1;
    }
    std::size_t tile_count() const override { return n_; }

private:
    std::size_t            n_;
    std::atomic<std::size_t> cursor_{0};
};

// Round-robin pre-assignment: tile i belongs to worker (i % n_workers). A
// worker only ever receives its own tiles, in order; once it has taken them
// all, next_for returns -1 for it. No rebalancing — if one worker is slow,
// its tiles wait for it rather than being stolen.
class PushScheduler final : public IScheduler {
public:
    PushScheduler(std::size_t n_tiles, int n_workers)
        : n_(n_tiles), n_workers_(n_workers < 1 ? 1 : n_workers),
          taken_(static_cast<std::size_t>(n_workers_), 0) {}

    long next_for(int worker_id) override {
        if (worker_id < 0 || worker_id >= n_workers_) return -1;
        // The k-th tile owned by this worker is index worker_id + k*n_workers.
        std::lock_guard<std::mutex> lk(mu_);
        std::size_t k = taken_[worker_id]++;
        std::size_t idx = static_cast<std::size_t>(worker_id) +
                          k * static_cast<std::size_t>(n_workers_);
        return idx < n_ ? static_cast<long>(idx) : -1;
    }
    std::size_t tile_count() const override { return n_; }

private:
    std::size_t          n_;
    int                  n_workers_;
    std::mutex           mu_;
    std::vector<std::size_t> taken_;
};

} // namespace frep::dist
