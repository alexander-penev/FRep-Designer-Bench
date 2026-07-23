#pragma once
// tests/test_support.hpp
//
// Shared test infrastructure — JIT engine pool, type aliases,
// small C++26 utilities (pack indexing demonstration).

#include "core/compiler/jit_engine.hpp"

#include <array>
#include <memory>
#include <vector>

namespace frep::test {

using SdfFn = float(*)(float, float, float);

// Global pool — keeps JIT engine instances alive until the test executable exits.
inline std::vector<std::unique_ptr<JitEngine>>& jit_pool() {
    static std::vector<std::unique_ptr<JitEngine>> p;
    return p;
}

// C++26 pack indexing (P2662) — the Nth type from a parameter pack.
// Demonstrates that the project compiles with -std=c++26.
template <std::size_t N, typename... Ts>
using nth_type_t = Ts...[N];

static_assert(std::is_same_v<nth_type_t<0, int, char, double>, int>);
static_assert(std::is_same_v<nth_type_t<2, int, char, double>, double>);

} // namespace frep::test

