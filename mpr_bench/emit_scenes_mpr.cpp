// Emit the canonical benchmark scenes as .frep archives using mpr's bundled
// libfive. mpr vendors libfive 8ca1b868, while the suite builds a much newer
// one; the archive opcode numbering differs between them, so archives written
// by the suite's libfive fail mpr's deserializer ("expected op > INVALID").
// The scene definitions themselves are version-agnostic, so we reuse them and
// only swap which libfive serializes.
//
// mpr's own mpr_emit.cpp covers the five canonical scenes but not the two
// complex ones, which is why the render table reported mpr as unable to
// render the gear and the colonnade; it renders both.
//
// Build (from the repository root):
//   g++ -std=c++17 -I. -I mpr/libfive/libfive/include \
//       -o /tmp/emit_scenes_mpr mpr_bench/emit_scenes_mpr.cpp \
//       -L mpr/build/libfive/libfive/src -lfive
//   LD_LIBRARY_PATH=mpr/build/libfive/libfive/src /tmp/emit_scenes_mpr scenes/mpr
#include "../libfive_bench/scenes_trees.hpp"
#include <libfive/tree/archive.hpp>
#include <cstdio>
#include <fstream>
#include <string>
int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : ".";
    auto scenes = canonical_scenes();
    for (auto& s : complex_scenes()) scenes.push_back(s);
    for (auto& [n, t] : scenes) {
        std::ofstream f(dir + "/" + n + ".frep", std::ios::binary);
        libfive::Archive a(t); a.serialize(f);
        std::printf("wrote %s/%s.frep\n", dir.c_str(), n.c_str());
    }
    return 0;
}
