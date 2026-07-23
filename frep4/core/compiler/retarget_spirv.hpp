#pragma once
// core/compiler/retarget_spirv.hpp
//
// SPIR-V retarget plugin.
//
// Transforms an LLVM IR module into SPIR-V bytecode via LLVM's built-in
// SPIR-V backend (included since LLVM 18+).
//
// The resulting .spv module can run on:
//   - a Vulkan compute shader (any modern GPU)
//   - an OpenCL 2.1+ device
//   - Apple Metal (via SPIRV-Cross)
//
// For the PoC: we emit SPIR-V and validate it with `spirv-val`.
// A full Vulkan runtime (instance, device, pipeline, command buffer) is
// out of scope for this PoC — it requires 500+ lines of boilerplate.

#include "core/compiler/llvm_compat.hpp"
#include "core/compiler/spirv_external.hpp"
#include "core/plugin/plugin_api.hpp"

#include <llvm/Bitcode/BitcodeWriter.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/TargetParser/Triple.h>
#include <llvm/Support/raw_ostream.h>

#include <expected>
#include <optional>
#include <string>
#include <vector>

namespace frep {

// ─────────────────────────────────────────────────────────────────────────────
// SPIRVRetarget — turns LLVM IR into SPIR-V bytecode.
// Conforms to the RetargetPlugin concept.
//
// Strategy: the in-process LLVM 20 SPIR-V backend is still experimental, so
// we shell out to the Khronos llvm-spirv translator instead. The pipeline is:
//
//   1. Set the target triple to spirv64-unknown-unknown (NFC for the rest).
//   2. Mark `render_tile` as SPIR_KERNEL + strip AlwaysInline hints.
//   3. Serialize the module to LLVM bitcode (in-memory buffer).
//   4. Spawn llvm-spirv as a subprocess — it accepts bitcode and emits .spv.
//   5. If spirv-val is available, validate the result and record the verdict.
//
// All four steps are wrapped behind one std::expected. If the translator is
// not installed we fall back to assembly-only mode (RetargetResult with
// .assembly populated, .bytes empty) so the build does not hard-fail.
// ─────────────────────────────────────────────────────────────────────────────
class SPIRVRetarget {
public:
    SPIRVRetarget() = default;

    plugin::PluginInfo info() const {
        return {"spirv", "1.0", "LLVM IR → SPIR-V 1.5 (Vulkan/OpenCL compute)"};
    }

    std::string_view target_triple() const {
        // 64-bit pointer SPIR-V Logical
        return "spirv64-unknown-unknown";
    }

    // Extra fields the GUI / tools may want for the SPIR-V output panel.
    // Populated as a side-effect of retarget().
    bool        last_validated = false;
    std::string last_translator;
    std::string last_validator;
    std::string last_validator_message;

    std::expected<plugin::RetargetResult, std::string>
    retarget(llvm::Module& mod) {
        // Prepare the module for the SPIR-V target.
        // Uses a version-shim wrapper because LLVM 22 changed the parameter
        // type from StringRef to Triple.
        llvm_compat::set_target_triple(mod, target_triple());

        // Strip x86-specific intrinsics if there are any.
        // Force render_tile to be external + SPIR_KERNEL calling convention.
        for (auto& f : mod.functions()) {
            if (f.getName() == "render_tile") {
                f.setLinkage(llvm::Function::ExternalLinkage);
                f.setCallingConv(llvm::CallingConv::SPIR_KERNEL);
            }
            // Strip optimization hints that SPIR-V does not understand
            f.removeFnAttr(llvm::Attribute::AlwaysInline);
        }

        plugin::RetargetResult result;

        // Always emit the textual IR — it is useful for inspection and
        // provides a fallback path when the translator is missing.
        {
            std::string ir_text;
            llvm::raw_string_ostream os(ir_text);
            mod.print(os, nullptr);
            result.assembly = std::move(ir_text);
        }

        // ── Primary: the in-tree LLVM SPIR-V backend ────────────────────────
        // Since LLVM ~18 the SPIR-V target is built in and can lower a module
        // to SPIR-V via TargetMachine, with no external tool. This is the
        // preferred path (no fragile dependency on the Khronos llvm-spirv
        // binary being installed). Falls through to the external translator
        // only if the backend isn't available in this LLVM build.
        if (auto spv = emit_via_backend(mod)) {
            result.bytes           = std::move(*spv);
            last_translator        = "(in-tree SPIR-V backend)";
            last_validator         = {};
            last_validator_message = {};
            last_validated         = false;
            return result;
        }

        // ── Fallback: the Khronos translator ────────────────────────────────
        // Serialize bitcode into a buffer (no temp file on this side).
        llvm::SmallVector<char, 0> bc_buf;
        {
            llvm::raw_svector_ostream os(bc_buf);
            llvm::WriteBitcodeToFile(mod, os);
        }
        std::vector<unsigned char> bitcode(
            bc_buf.begin(), bc_buf.end());

        auto tr = spirv_ext::translate_bitcode_to_spirv(bitcode, /*validate=*/true);
        if (!tr) {
            // Neither backend nor translator available — keep the assembly
            // path. Surface the diagnostic but do not error out.
            last_translator        = "(not available)";
            last_validator         = {};
            last_validator_message = tr.error();
            last_validated         = false;
            return result;
        }

        result.bytes = std::move(tr->bytes);
        last_translator        = tr->translator_path;
        last_validator         = tr->validator_path;
        last_validator_message = tr->validator_message;
        last_validated         = tr->validated;
        return result;
    }

private:
    // Lower `mod` to SPIR-V using the in-tree LLVM SPIR-V backend. Returns
    // the .spv bytes, or nullopt if the backend isn't registered in this
    // build (then the caller falls back to the external translator). The
    // SPIR-V targets must be initialised by the process beforehand (the
    // executors do this).
    std::optional<std::vector<unsigned char>> emit_via_backend(llvm::Module& mod) {
        const std::string triple = "spirv64-unknown-unknown";
        llvm_compat::set_target_triple(mod, triple);

        std::string err;
        llvm::Triple T(triple);
#if LLVM_VERSION_MAJOR >= 22
        const llvm::Target* tgt = llvm::TargetRegistry::lookupTarget(T, err);
#else
        const llvm::Target* tgt = llvm::TargetRegistry::lookupTarget(triple, err);
#endif
        if (!tgt) return std::nullopt;   // SPIR-V backend not in this build

        std::unique_ptr<llvm::TargetMachine> tm(
#if LLVM_VERSION_MAJOR >= 22
            tgt->createTargetMachine(T, "", "", llvm::TargetOptions{},
                                     std::optional<llvm::Reloc::Model>())
#else
            tgt->createTargetMachine(triple, "", "", llvm::TargetOptions{},
                                     std::optional<llvm::Reloc::Model>())
#endif
        );
        if (!tm) return std::nullopt;

        llvm::SmallVector<char, 0> buf;
        llvm::raw_svector_ostream os(buf);
        llvm::legacy::PassManager pm;
        if (tm->addPassesToEmitFile(pm, os, nullptr,
                                    llvm::CodeGenFileType::ObjectFile))
            return std::nullopt;
        pm.run(mod);
        if (buf.empty()) return std::nullopt;
        return std::vector<unsigned char>(buf.begin(), buf.end());
    }

public:
};

static_assert(plugin::RetargetPlugin<SPIRVRetarget>,
              "SPIRVRetarget must satisfy the RetargetPlugin concept");

} // namespace frep
