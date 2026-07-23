// tests/test_spirv.cpp
//
// Tests for the Khronos llvm-spirv integration.

#include "core/frep/scene.hpp"
#include "core/frep/primitives.hpp"
#include "core/compiler/codegen.hpp"
#include "core/compiler/retarget_spirv.hpp"
#include "core/compiler/spirv_external.hpp"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <gtest/gtest.h>
#include <cstdint>
#include <string>
#include <type_traits>
#include <memory>

using namespace frep;

namespace {

// Compile a small scene and clone its module ready for retargeting.
std::unique_ptr<llvm::Module> emit_test_module(llvm::LLVMContext& ctx) {
    SceneGraph s;
    s.add_object(std::make_shared<SphereNode>(1.0f, "ball"));
    SceneCodegen cg(ctx);
    cg.emit_render_tile(s);
    // Clone so the caller can inspect / retarget without disturbing the
    // codegen's own ownership.
    return llvm::CloneModule(*cg.module());
}

} // namespace

// ── Translator discovery ────────────────────────────────────────────────────

TEST(SpirvExternal, FindsTranslatorOrReportsNone) {
    auto path = spirv_ext::find_translator();
    // Either we found one (in CI it should be installed), or the lookup
    // returned empty cleanly — neither outcome should crash.
    SUCCEED() << "translator: '" << path << "'";
}

// ── SPIRVRetarget end-to-end ────────────────────────────────────────────────

TEST(SpirvRetarget, AlwaysEmitsTextIR) {
    llvm::LLVMContext ctx;
    auto mod = emit_test_module(ctx);
    SPIRVRetarget r;
    auto res = r.retarget(*mod);
    ASSERT_TRUE(res.has_value()) << res.error();
    // Assembly is always populated, even when the translator is missing.
    EXPECT_FALSE(res->assembly.empty());
    // It should look like LLVM IR.
    EXPECT_NE(res->assembly.find("target triple"), std::string::npos);
    EXPECT_NE(res->assembly.find("spirv64"), std::string::npos);
}

TEST(SpirvRetarget, EmitsBinaryWhenTranslatorAvailable) {
    if (spirv_ext::find_translator().empty()) {
        GTEST_SKIP() << "no llvm-spirv translator on PATH";
    }
    llvm::LLVMContext ctx;
    auto mod = emit_test_module(ctx);
    SPIRVRetarget r;
    auto res = r.retarget(*mod);
    ASSERT_TRUE(res.has_value()) << res.error();
    ASSERT_FALSE(res->bytes.empty()) << "translator should have produced bytes";

    // SPIR-V magic number is 0x07230203 in the host byte order.
    ASSERT_GE(res->bytes.size(), 4u);
    std::uint32_t magic =
        (static_cast<std::uint32_t>(res->bytes[0])      ) |
        (static_cast<std::uint32_t>(res->bytes[1]) <<  8) |
        (static_cast<std::uint32_t>(res->bytes[2]) << 16) |
        (static_cast<std::uint32_t>(res->bytes[3]) << 24);
    EXPECT_EQ(magic, 0x07230203u);
    EXPECT_FALSE(r.last_translator.empty());
}

TEST(SpirvRetarget, SetsSPIRKernelCallingConv) {
    llvm::LLVMContext ctx;
    auto mod = emit_test_module(ctx);
    SPIRVRetarget r;
    auto res = r.retarget(*mod);
    ASSERT_TRUE(res.has_value());
    // After retarget, render_tile should be SPIR_KERNEL.
    auto* fn = mod->getFunction("render_tile");
    ASSERT_NE(fn, nullptr);
    EXPECT_EQ(fn->getCallingConv(), llvm::CallingConv::SPIR_KERNEL);
    EXPECT_EQ(fn->getLinkage(), llvm::Function::ExternalLinkage);
}

TEST(SpirvRetarget, TripleIsSetOnModule) {
    llvm::LLVMContext ctx;
    auto mod = emit_test_module(ctx);
    SPIRVRetarget r;
    auto res = r.retarget(*mod);
    ASSERT_TRUE(res.has_value());
    // Portable across LLVM versions:
    //   - LLVM 20: getTargetTriple() returns std::string
    //   - LLVM 21+: getTargetTriple() returns llvm::Triple (with .str())
    // `if constexpr` discards statements only in templated contexts, so
    // we use a templated helper to keep each branch out of the other's
    // overload resolution.
    auto to_string = [](auto&& v) -> std::string {
        if constexpr (std::is_same_v<std::decay_t<decltype(v)>, std::string>) {
            return v;
        } else {
            return v.str();
        }
    };
    std::string triple = to_string(mod->getTargetTriple());
    EXPECT_NE(triple.find("spirv64"), std::string::npos);
}

// ── Direct call into the bitcode translator (no LLVM module required) ──────

TEST(SpirvExternal, EmptyBytesYieldFailure) {
    if (spirv_ext::find_translator().empty()) {
        GTEST_SKIP() << "no llvm-spirv translator on PATH";
    }
    std::vector<unsigned char> empty;
    auto res = spirv_ext::translate_bitcode_to_spirv(empty, false);
    EXPECT_FALSE(res.has_value())
        << "empty input should not produce a SPIR-V module";
}
