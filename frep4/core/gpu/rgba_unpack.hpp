// core/gpu/rgba_unpack.hpp
//
// RGBA8 → float conversion for GPU readbacks. The GPU paths render into RGBA8
// images and the executors hand out float RGBA (the CPU path's native
// format), so every GPU frame crosses this loop once. Written as a multiply
// by a constant so the compiler vectorises it (zero-extend, convert,
// multiply); the division it replaces was also vectorised but a divide is an
// order of magnitude slower per lane, and at 512² the conversion alone took
// longer than the GPU trace. The two forms differ by at most one ulp, far
// below the 8-bit quantisation already in the data.

#pragma once

#include <cstddef>
#include <cstdint>

namespace frep::gpu {

inline void unpack_rgba8(const std::uint8_t* src, float* dst, std::size_t n) {
    constexpr float k = 1.0f / 255.0f;
    for (std::size_t i = 0; i < n; ++i) dst[i] = static_cast<float>(src[i]) * k;
}

} // namespace frep::gpu
