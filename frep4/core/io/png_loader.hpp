#pragma once
// core/io/png_loader.hpp
//
// PNG file loading via libpng. Returns an Image with RGBA8 pixels in
// top-down row-major order — the same format as bmp_loader.hpp so
// downstream code (material.texture_rgba, GPU upload) doesn't care
// which loader produced the data.
//
// The header is conditionally compiled: when libpng isn't available
// at build time, FREP_HAS_LIBPNG is left undefined and load_png()
// returns an empty Image. Code that wants to support both formats
// should try load_png() first, then fall back to load_bmp().

#include "core/io/bmp_loader.hpp"  // for the Image struct

#include <cstdint>
#include <string>

#ifdef FREP_HAS_LIBPNG
extern "C" {
#include <png.h>
}
#include <cstdio>
#include <cstring>
#include <vector>
#endif

namespace frep::io {

inline Image load_png(const std::string& path) {
    Image img;
#ifndef FREP_HAS_LIBPNG
    (void)path;  // unused
    return img;  // empty — caller should fall back
#else
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return img;

    // Check magic.
    unsigned char header[8];
    if (std::fread(header, 1, 8, fp) != 8 || png_sig_cmp(header, 0, 8)) {
        std::fclose(fp);
        return img;
    }

    png_structp png = png_create_read_struct(PNG_LIBPNG_VER_STRING,
        nullptr, nullptr, nullptr);
    if (!png) { std::fclose(fp); return img; }
    png_infop info = png_create_info_struct(png);
    if (!info) {
        png_destroy_read_struct(&png, nullptr, nullptr);
        std::fclose(fp);
        return img;
    }
    if (setjmp(png_jmpbuf(png))) {
        // libpng error handler — jumps here on any decode error.
        png_destroy_read_struct(&png, &info, nullptr);
        std::fclose(fp);
        img.rgba.clear();
        img.width = img.height = 0;
        return img;
    }

    png_init_io(png, fp);
    png_set_sig_bytes(png, 8);
    png_read_info(png, info);

    int W       = png_get_image_width(png, info);
    int H       = png_get_image_height(png, info);
    auto ct     = png_get_color_type(png, info);
    auto bd     = png_get_bit_depth(png, info);

    // Normalise to RGBA8 regardless of source format:
    //   - 16-bit channels  → 8-bit
    //   - palette          → RGB
    //   - grayscale        → RGB
    //   - missing alpha    → add 0xFF
    if (bd == 16) png_set_strip_16(png);
    if (ct == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(png);
    if (ct == PNG_COLOR_TYPE_GRAY && bd < 8) png_set_expand_gray_1_2_4_to_8(png);
    if (png_get_valid(png, info, PNG_INFO_tRNS)) png_set_tRNS_to_alpha(png);
    if (ct == PNG_COLOR_TYPE_RGB || ct == PNG_COLOR_TYPE_GRAY
        || ct == PNG_COLOR_TYPE_PALETTE)
        png_set_filler(png, 0xFF, PNG_FILLER_AFTER);
    if (ct == PNG_COLOR_TYPE_GRAY || ct == PNG_COLOR_TYPE_GRAY_ALPHA)
        png_set_gray_to_rgb(png);
    png_read_update_info(png, info);

    img.width  = W;
    img.height = H;
    img.rgba.assign(static_cast<std::size_t>(W) * H * 4, 0);

    // libpng reads one row at a time; we need an array of row pointers.
    std::vector<png_bytep> row_ptrs(H);
    for (int y = 0; y < H; ++y)
        row_ptrs[y] = img.rgba.data() + y * W * 4;
    png_read_image(png, row_ptrs.data());
    png_read_end(png, nullptr);

    png_destroy_read_struct(&png, &info, nullptr);
    std::fclose(fp);
    return img;
#endif
}

// Convenience: try PNG first (if libpng is compiled in), then BMP.
// Picks the file format based on extension to avoid wasting one attempt
// on a wrong-format file.
inline Image load_image(const std::string& path) {
    // Case-insensitive extension check.
    auto ends_with = [&](const char* ext) {
        auto n = std::string(ext).size();
        if (path.size() < n) return false;
        for (std::size_t i = 0; i < n; ++i) {
            char a = path[path.size() - n + i];
            char b = ext[i];
            if (a >= 'A' && a <= 'Z') a = static_cast<char>(a - 'A' + 'a');
            if (b >= 'A' && b <= 'Z') b = static_cast<char>(b - 'A' + 'a');
            if (a != b) return false;
        }
        return true;
    };
    if (ends_with(".png")) {
        auto img = load_png(path);
        if (!img.empty()) return img;
    }
    // Default / fallback: BMP.
    return load_bmp(path);
}

} // namespace frep::io
