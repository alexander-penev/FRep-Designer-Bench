// tests/test_scene_io_relative_textures.cpp
//
// Regression tests for the texture-path portability behaviour added in
// the v4.0 polish sprint: when a scene file is saved into a directory
// that also contains the referenced textures, the saved JSON should
// hold a *relative* path so the directory tree can be moved without
// breaking the references. The matching loader must then resolve that
// relative path against the directory the .frep file came from.
//
// We deliberately keep the round-trip end-to-end (write a real PNG to
// disk, save the scene next to it, move both, reload) so the
// filesystem-level behaviour is what's actually exercised.

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/io/scene_io.hpp"
#include "core/io/png_loader.hpp"
#include "core/io/bmp_loader.hpp"

#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <random>

namespace fs = std::filesystem;

namespace {

// Write a small RGBA8 BMP at `path`. We use BMP rather than PNG so the
// test passes whether or not libpng was compiled in (`save_bmp` is
// always available; `save_png` is conditional). `load_image` will
// transparently dispatch to BMP via the file extension at load time.
bool write_test_image(const fs::path& path, int w = 8, int h = 8) {
    frep::io::Image img;
    img.width = w; img.height = h;
    img.rgba.resize(static_cast<std::size_t>(w) * h * 4);
    // Fill with a deterministic pattern so we can identify it later
    // if the round-trip ever decides to silently swap pixels.
    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            std::size_t i = (static_cast<std::size_t>(y) * w + x) * 4;
            img.rgba[i + 0] = static_cast<uint8_t>(x * 32);
            img.rgba[i + 1] = static_cast<uint8_t>(y * 32);
            img.rgba[i + 2] = 0;
            img.rgba[i + 3] = 255;
        }
    }
    return frep::io::save_bmp(img, path.string());
}

// Build a temp directory under the system temp area. Unique per test
// invocation so parallel ctest runs don't collide.
fs::path make_tmp_dir(const std::string& tag) {
    auto base = fs::temp_directory_path() / "frep4_scene_io_rel";
    std::random_device rd;
    auto dir = base / (tag + "_" + std::to_string(rd()));
    fs::create_directories(dir);
    return dir;
}

// Construct a one-object scene with the given texture path baked into
// its material. The texture pixels are loaded synchronously so the
// `texture_rgba` field is populated, matching what the GUI does after
// a Browse… click.
frep::SceneGraph make_scene_with_texture(const std::string& tex_path) {
    using namespace frep;
    SceneGraph s;
    auto sphere = std::make_shared<SphereNode>(1.0f, "s1");
    Material m;
    m.pattern        = Material::Pattern::Texture;
    m.pattern_scale  = 1.0f;
    m.texture_path   = tex_path;
    auto img = io::load_image(tex_path);
    if (!img.empty()) {
        m.texture_rgba   = std::move(img.rgba);
        m.texture_width  = img.width;
        m.texture_height = img.height;
    }
    s.add_object(sphere, m);
    return s;
}

} // anonymous namespace

TEST(SceneIORelativeTextures, RewritesAbsolutePathIntoRelativeOnSave) {
    auto dir = make_tmp_dir("rewrite");
    fs::path tex_path   = dir / "checker.bmp";
    fs::path scene_path = dir / "scene.frep";

    ASSERT_TRUE(write_test_image(tex_path))
        << "Couldn't write test PNG — does the build have libpng?";

    auto scene = make_scene_with_texture(tex_path.string());
    ASSERT_TRUE(frep::io::save_scene(scene, scene_path.string()));

    // The serialized text should contain just "checker.bmp", not the
    // full absolute path. The dir name varies per run so checking
    // its absence (rather than presence of a specific relative form)
    // is the most robust assertion.
    std::ifstream f(scene_path);
    std::string text((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    EXPECT_NE(text.find("\"checker.bmp\""), std::string::npos)
        << "Texture path should be saved relative.";
    EXPECT_EQ(text.find(tex_path.parent_path().string()), std::string::npos)
        << "Saved scene should not embed the parent directory of the texture.";

    fs::remove_all(dir);
}

TEST(SceneIORelativeTextures, ResolvesRelativePathOnLoadFromSameDir) {
    auto dir = make_tmp_dir("resolve");
    fs::path tex_path   = dir / "wood.bmp";
    fs::path scene_path = dir / "scene.frep";

    ASSERT_TRUE(write_test_image(tex_path, 16, 16));

    auto scene = make_scene_with_texture(tex_path.string());
    ASSERT_TRUE(frep::io::save_scene(scene, scene_path.string()));

    // Reload via the same file API: the relative path inside the .frep
    // should be resolved against the .frep file's directory, and the
    // RGBA buffer should be re-populated.
    auto loaded = frep::io::load_scene(scene_path.string());
    ASSERT_EQ(loaded.objects().size(), 1u);
    const auto& mat = loaded.objects().at("s1").material;
    EXPECT_EQ(mat.texture_width,  16);
    EXPECT_EQ(mat.texture_height, 16);
    EXPECT_EQ(mat.texture_rgba.size(), 16u * 16u * 4u);
    EXPECT_EQ(mat.pattern, frep::Material::Pattern::Texture);

    fs::remove_all(dir);
}

TEST(SceneIORelativeTextures, PortableAcrossDirectoryMove) {
    // The real point of relative paths: copy a whole scene-plus-textures
    // directory somewhere else, and it should still work. This test
    // saves to dir A, moves to dir B, then loads from dir B.
    auto dir_a = make_tmp_dir("src");
    auto dir_b = make_tmp_dir("dst");
    // dir_b was created by make_tmp_dir; remove it so rename succeeds.
    fs::remove_all(dir_b);

    fs::path tex_a   = dir_a / "marble.bmp";
    fs::path scene_a = dir_a / "scene.frep";
    ASSERT_TRUE(write_test_image(tex_a, 4, 4));
    auto scene = make_scene_with_texture(tex_a.string());
    ASSERT_TRUE(frep::io::save_scene(scene, scene_a.string()));

    // Physically move the whole directory. After this, the absolute
    // path that was baked into texture_path at construction time no
    // longer exists — so anything other than relative-path resolution
    // would fail to find the texture.
    fs::rename(dir_a, dir_b);
    fs::path scene_b = dir_b / "scene.frep";

    auto loaded = frep::io::load_scene(scene_b.string());
    ASSERT_EQ(loaded.objects().size(), 1u);
    const auto& mat = loaded.objects().at("s1").material;
    EXPECT_FALSE(mat.texture_rgba.empty())
        << "Texture failed to resolve after directory move — "
        << "relative-path support is not actually portable.";
    EXPECT_EQ(mat.texture_width,  4);
    EXPECT_EQ(mat.texture_height, 4);

    fs::remove_all(dir_b);
}

TEST(SceneIORelativeTextures, AbsolutePathOutsideBaseDirIsLeftAlone) {
    // If the texture happens to live outside the scene's directory tree,
    // rewriting it as ../../../.. would be surprising — we leave such
    // paths absolute. This test asserts that policy.
    auto dir_scene  = make_tmp_dir("scene");
    auto dir_assets = make_tmp_dir("assets");

    fs::path tex_path   = dir_assets / "external.bmp";
    fs::path scene_path = dir_scene  / "scene.frep";

    ASSERT_TRUE(write_test_image(tex_path));
    auto scene = make_scene_with_texture(tex_path.string());
    ASSERT_TRUE(frep::io::save_scene(scene, scene_path.string()));

    std::ifstream f(scene_path);
    std::string text((std::istreambuf_iterator<char>(f)),
                      std::istreambuf_iterator<char>());
    // The full absolute path should still appear — it's outside the
    // scene's directory so we don't try to make it relative.
    EXPECT_NE(text.find(tex_path.string()), std::string::npos)
        << "Out-of-tree texture path should be saved absolute (it isn't).";

    // And it should still load — absolute paths bypass base_dir resolution.
    auto loaded = frep::io::load_scene(scene_path.string());
    const auto& mat = loaded.objects().at("s1").material;
    EXPECT_FALSE(mat.texture_rgba.empty())
        << "Absolute texture path should still load (didn't).";

    fs::remove_all(dir_scene);
    fs::remove_all(dir_assets);
}

TEST(SceneIORelativeTextures, EmptyBaseDirIsBackwardCompatible) {
    // serialize_scene/deserialize_scene without a base_dir (the 1-arg
    // overload) should preserve exactly the behaviour the project had
    // before this change: no rewriting on serialize, raw path on load.
    auto dir = make_tmp_dir("compat");
    fs::path tex_path = dir / "compat.bmp";
    ASSERT_TRUE(write_test_image(tex_path));

    auto scene = make_scene_with_texture(tex_path.string());

    // 1-arg serialize should keep the absolute path verbatim.
    std::string json = frep::io::serialize_scene(scene);
    EXPECT_NE(json.find(tex_path.string()), std::string::npos)
        << "1-arg serialize_scene should not rewrite paths.";

    // 2-arg deserialize without base_dir (default empty) should load
    // the absolute path as-is.
    auto loaded = frep::io::deserialize_scene(json);
    EXPECT_FALSE(loaded.objects().at("s1").material.texture_rgba.empty());

    fs::remove_all(dir);
}
