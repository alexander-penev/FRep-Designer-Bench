#pragma once
// core/io/bmp_loader.hpp
//
// Minimal Windows BMP loader. Supports the common 24-bit and 32-bit
// uncompressed variants — sufficient for hand-authored textures saved
// from any image editor (GIMP, Photoshop, Paint).
//
// Why BMP only: PNG decoding requires either a third-party library
// (libpng, stb_image) or a from-scratch DEFLATE implementation. BMP is
// trivially parsable in <100 LoC of plain C++ and covers the PoC use
// case (testing texture mapping on SDFs). PNG support can be added
// later by dropping in stb_image.h as a single-header dependency.
//
// On-disk layout (24-bit / 32-bit, no compression):
//   [BITMAPFILEHEADER 14 B]
//     "BM" (2)
//     file size (4)
//     reserved (4)
//     pixel-data offset (4)
//   [BITMAPINFOHEADER 40 B]
//     header size (4)
//     width (4, signed)
//     height (4, signed; negative = top-down)
//     planes (2)
//     bits per pixel (2)
//     compression (4) — 0 = BI_RGB, what we support
//     image size (4)
//     ... (rest unused)
//   [pixel data, BGR order, rows padded to 4 bytes]
//
// Output: an RGBA8 buffer in row-major top-down order (y = 0 at top).

#include <cstdint>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace frep::io {

struct Image {
    int                       width  = 0;
    int                       height = 0;
    std::vector<std::uint8_t> rgba;   // size = width * height * 4
    bool empty() const { return rgba.empty(); }
};

// Loads `path` as an Image. Returns an empty Image on any error
// (bad magic, unsupported format, truncated file). The caller can
// check img.empty() to detect failure.
inline Image load_bmp(const std::string& path) {
    Image img;
    std::ifstream f(path, std::ios::binary);
    if (!f) return img;

    // File header — 14 bytes, but we read it field by field to avoid
    // alignment / padding surprises across compilers.
    char magic[2];
    f.read(magic, 2);
    if (!f || magic[0] != 'B' || magic[1] != 'M') return img;
    f.seekg(10);  // skip to pixel data offset
    std::uint32_t pixel_offset = 0;
    f.read(reinterpret_cast<char*>(&pixel_offset), 4);

    // Info header.
    std::uint32_t info_size = 0;
    f.read(reinterpret_cast<char*>(&info_size), 4);
    std::int32_t  w = 0, h = 0;
    f.read(reinterpret_cast<char*>(&w), 4);
    f.read(reinterpret_cast<char*>(&h), 4);
    std::uint16_t planes = 0, bpp = 0;
    f.read(reinterpret_cast<char*>(&planes), 2);
    f.read(reinterpret_cast<char*>(&bpp), 2);
    std::uint32_t compression = 0;
    f.read(reinterpret_cast<char*>(&compression), 4);

    if (!f) return img;
    if (planes != 1) return img;
    if (compression != 0) return img;     // only BI_RGB
    if (bpp != 24 && bpp != 32) return img;
    if (w <= 0 || w > 8192) return img;
    if (h == 0 || std::abs(h) > 8192) return img;

    bool bottom_up = h > 0;
    int H = std::abs(h);
    int W = w;
    int bytes_per_px = bpp / 8;
    // Rows in 24-bit BMPs are padded to 4-byte boundary; 32-bit are
    // already aligned.
    int row_stride = ((W * bytes_per_px + 3) / 4) * 4;
    std::vector<std::uint8_t> row(row_stride);

    img.width  = W;
    img.height = H;
    img.rgba.assign(static_cast<std::size_t>(W) * H * 4, 0);

    f.seekg(pixel_offset);
    for (int y_disk = 0; y_disk < H; ++y_disk) {
        f.read(reinterpret_cast<char*>(row.data()), row_stride);
        if (!f) { img.rgba.clear(); img.width = img.height = 0; return img; }
        // Bottom-up BMPs store row 0 at the bottom of the image.
        int y_out = bottom_up ? (H - 1 - y_disk) : y_disk;
        for (int x = 0; x < W; ++x) {
            // BMP stores BGR (or BGRA); we want RGBA.
            std::uint8_t b = row[x * bytes_per_px + 0];
            std::uint8_t g = row[x * bytes_per_px + 1];
            std::uint8_t r = row[x * bytes_per_px + 2];
            std::uint8_t a = (bpp == 32) ? row[x * bytes_per_px + 3] : 255;
            std::size_t i = (static_cast<std::size_t>(y_out) * W + x) * 4;
            img.rgba[i + 0] = r;
            img.rgba[i + 1] = g;
            img.rgba[i + 2] = b;
            img.rgba[i + 3] = a;
        }
    }
    return img;
}

// Saves an RGBA8 image as a 24-bit bottom-up BMP. Returns false on I/O
// error. The alpha channel is discarded. Mostly useful for tests
// (round-trip load → save → load).
inline bool save_bmp(const Image& img, const std::string& path) {
    if (img.empty()) return false;
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;

    int W = img.width;
    int H = img.height;
    int row_stride = ((W * 3 + 3) / 4) * 4;
    std::uint32_t pixel_data_size = static_cast<std::uint32_t>(row_stride * H);
    std::uint32_t file_size = 14 + 40 + pixel_data_size;

    // File header.
    f.put('B'); f.put('M');
    f.write(reinterpret_cast<const char*>(&file_size), 4);
    std::uint32_t reserved = 0, off = 14 + 40;
    f.write(reinterpret_cast<const char*>(&reserved), 4);
    f.write(reinterpret_cast<const char*>(&off), 4);

    // Info header (40 bytes).
    std::uint32_t info_size = 40;
    std::int32_t  w = W, h = H;
    std::uint16_t planes = 1, bpp = 24;
    std::uint32_t compression = 0;
    std::uint32_t x_res = 2835, y_res = 2835;  // ~72 DPI
    std::uint32_t colors_used = 0, important = 0;
    f.write(reinterpret_cast<const char*>(&info_size), 4);
    f.write(reinterpret_cast<const char*>(&w), 4);
    f.write(reinterpret_cast<const char*>(&h), 4);
    f.write(reinterpret_cast<const char*>(&planes), 2);
    f.write(reinterpret_cast<const char*>(&bpp), 2);
    f.write(reinterpret_cast<const char*>(&compression), 4);
    f.write(reinterpret_cast<const char*>(&pixel_data_size), 4);
    f.write(reinterpret_cast<const char*>(&x_res), 4);
    f.write(reinterpret_cast<const char*>(&y_res), 4);
    f.write(reinterpret_cast<const char*>(&colors_used), 4);
    f.write(reinterpret_cast<const char*>(&important), 4);

    // Pixel data — bottom-up, BGR, padded rows.
    std::vector<std::uint8_t> row(row_stride, 0);
    for (int y_out = 0; y_out < H; ++y_out) {
        int y_in = H - 1 - y_out;
        for (int x = 0; x < W; ++x) {
            std::size_t i = (static_cast<std::size_t>(y_in) * W + x) * 4;
            row[x * 3 + 0] = img.rgba[i + 2]; // B
            row[x * 3 + 1] = img.rgba[i + 1]; // G
            row[x * 3 + 2] = img.rgba[i + 0]; // R
        }
        f.write(reinterpret_cast<const char*>(row.data()), row_stride);
    }
    return static_cast<bool>(f);
}

} // namespace frep::io
