// core/tracer/pixel_buffers.cpp — see the header.

#include "core/tracer/pixel_buffers.hpp"

#include <mutex>
#include <utility>

namespace frep {

// A few large buffers of each type, under a mutex (the GUI renders from
// several threads). Only buffers of at least a 128² RGBA tile are kept; a
// reused one is shrunk to the requested size, which std::vector does without
// touching the elements, so nothing is zeroed. Nothing is retained beyond
// kKeep, so a burst of frames cannot grow the process.
namespace {
constexpr std::size_t kKeep    = 8;
constexpr std::size_t kMinKeep = 128u * 128u * 4u;
std::mutex                             g_m;
std::vector<std::vector<float>>        g_floats;
std::vector<std::vector<std::uint8_t>> g_bytes;

template <class V>
V take_from(std::vector<V>& pool, std::size_t n) {
    if (n >= kMinKeep) {
        std::lock_guard<std::mutex> lk(g_m);
        for (std::size_t i = 0; i < pool.size(); ++i) {
            if (pool[i].size() >= n) {
                V v = std::move(pool[i]);
                pool[i] = std::move(pool.back());
                pool.pop_back();
                v.resize(n);                  // shrink: no element is written
                return v;
            }
        }
    }
    return V(n);
}
template <class V>
void give_back(std::vector<V>& pool, V& v) {
    if (v.capacity() < kMinKeep) return;
    std::lock_guard<std::mutex> lk(g_m);
    if (pool.size() < kKeep) pool.push_back(std::move(v));
}
}  // namespace

std::vector<float>        take_floats(std::size_t n) { return take_from(g_floats, n); }
std::vector<std::uint8_t> take_bytes(std::size_t n)  { return take_from(g_bytes, n); }
void recycle(std::vector<float>& v)        { give_back(g_floats, v); }
void recycle(std::vector<std::uint8_t>& v) { give_back(g_bytes, v); }

}  // namespace frep
