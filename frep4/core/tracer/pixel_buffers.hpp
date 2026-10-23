// core/tracer/pixel_buffers.hpp
//
// Pixel buffers for a frame or a tile, recycled.
//
// Whoever consumes a result destroys it, and the executor allocates the
// next one. A std::vector value-initialises, and zeroing a 4 MB float frame
// whose cache lines the render's worker threads still owned cost 0.5–0.9 ms
// per frame on the CPU path at 512² -- and the render then paid again to
// take the lines back: with the buffer reused the sphere frame went from
// 1.85 to 0.60 ms. The GPU paths' 1 MB RGBA8 results carry the same tax on
// a 0.2 ms frame.
//
// take_*(n) hands back a buffer of exactly n elements, with stale contents
// when it is a reused one: the caller writes every element (every path
// writes its frame or tile in full -- the sky on a miss). recycle() keeps a
// few large buffers; TileResult's destructor calls it.

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

namespace frep {

std::vector<float>        take_floats(std::size_t n);
std::vector<std::uint8_t> take_bytes(std::size_t n);

void recycle(std::vector<float>& v);
void recycle(std::vector<std::uint8_t>& v);

}  // namespace frep
