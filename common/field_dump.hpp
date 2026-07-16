#pragma once
// Cross-system visual parity: render each system's SDF through ONE identical
// orthographic projection, so the only variable is the evaluator.
//
// Every bench evaluates on the same canonical grid (-1.6 + 3.2*i/(N-1)). Here
// we march that grid along +Z and keep min_z f(x,y,z) per pixel -- an
// orthographic front view that depends only on scalar SDF evaluation, not on
// each system's native renderer (heightmap vs GPU-interval vs sphere-trace,
// which are NOT comparable). Same projection for all systems => the .f32
// outputs are directly pixel-diffable, and the .ppm is a recognizable preview.
//
// Writes two files:
//   <base>.f32 : R*R host-endian float32, row-major (iy outer, ix inner) = min_z f
//   <base>.ppm : colorized preview (blue outside, warm inside, black zero band)

#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <algorithm>

namespace fdump {

inline float half_extent() {
    if (const char* h = std::getenv("FDUMP_HALF")) { float v = std::atof(h); if (v > 0) return v; }
    return 1.6f;  // canonical benchmark box
}
inline float coord(int i, int N) { float H = half_extent(); return -H + 2 * H * i / (N - 1); }

// eval(x,y,z) -> signed distance (negative inside). Z = samples marched per pixel.
template <class F>
inline void dump_field(F eval, int R, int Z, const std::string& base) {
    std::vector<float> img((std::size_t)R * R);
    for (int iy = 0; iy < R; ++iy) {
        float y = coord(iy, R);
        for (int ix = 0; ix < R; ++ix) {
            float x = coord(ix, R);
            float m = INFINITY;
            for (int iz = 0; iz < Z; ++iz)
                m = std::min(m, eval(x, y, coord(iz, Z)));
            img[(std::size_t)iy * R + ix] = m;
        }
    }

    std::string f32 = base + ".f32";
    if (FILE* fp = std::fopen(f32.c_str(), "wb")) {
        std::fwrite(img.data(), sizeof(float), img.size(), fp);
        std::fclose(fp);
    }

    // Colorize: diverging ramp around 0, |f| normalized by the field's own
    // range so every scene fills the palette; a black band marks the surface.
    float vmax = 1e-6f;
    for (float v : img) if (std::isfinite(v)) vmax = std::max(vmax, std::fabs(v));
    std::string ppm = base + ".ppm";
    FILE* fp = std::fopen(ppm.c_str(), "wb");
    if (!fp) return;
    std::fprintf(fp, "P6\n%d %d\n255\n", R, R);
    for (float v : img) {
        unsigned char r, g, b;
        if (!std::isfinite(v)) { r = 255; g = 0; b = 255; }      // NaN/inf: magenta
        else if (std::fabs(v) < 0.02f * vmax) { r = g = b = 0; } // zero contour
        else if (v < 0) {                                        // inside: warm
            float t = std::min(1.0f, -v / vmax);
            r = 210; g = (unsigned char)(120 + 80 * (1 - t)); b = 70;
        } else {                                                 // outside: blue
            float t = std::min(1.0f, v / vmax);
            r = (unsigned char)(120 * (1 - t)); g = (unsigned char)(150 * (1 - t)); b = 210;
        }
        std::fputc(r, fp); std::fputc(g, fp); std::fputc(b, fp);
    }
    std::fclose(fp);
}

} // namespace fdump
