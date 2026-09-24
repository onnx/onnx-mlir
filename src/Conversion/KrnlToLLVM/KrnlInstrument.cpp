
/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------ KrnlInstrument.cpp - Lower KrnlInstrumentOp -------------------===//
//
// Copyright 2019-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the KrnlInstrumentOp operator.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMAttrs.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"

#include "onnx-mlir/Compiler/OMCompilerRuntimeTypes.h"
#include "src/Conversion/KrnlToLLVM/KrnlToLLVMHelper.hpp"
#include "src/Dialect/Krnl/KrnlHelper.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

#define DEBUG_TYPE "krnl_to_llvm"

using namespace mlir;

namespace onnx_mlir {
namespace krnl {

// Module-attribute key under which we cache a single DICompileUnit for all
// instrument calls in a module. Caching at module scope keeps the DWARF
// compile-unit count at exactly one regardless of how many KrnlInstrumentOps
// are lowered; multiple CUs in the same object confuse dsymutil and some
// linkers.
static constexpr llvm::StringLiteral kInstrumentCUAttrName =
    "onnx-mlir.instrument.cu";

// Build (or look up) a DICompileUnit attribute hosted on the parent module.
// All fields are synthetic — the file is never opened; only the presence and
// shape of the attributes matter to the debug-info pipeline.
//
// --- DIFile shape requirement (macOS / dsymutil) ---
// The DIFile must carry BOTH a normal-looking filename AND a non-empty
// directory string. macOS ld (ld64 / ld-prime) silently skips writing an
// `N_OSO` debug-map entry for any object whose compile unit has either an
// angle-bracket synthetic filename (e.g. "<onnx-mlir-instrument>") or an
// empty `DW_AT_comp_dir`. Without an `N_OSO`, dsymutil cannot locate the
// object file and drops its DWARF entirely from the final `.dSYM` bundle,
// making the instrumentation invisible to Instruments.app and `llvm-dwarfdump`.
// Using a plain name + "/" satisfies both constraints without requiring any
// real file on disk.
//
// --- LLVM API break (llvm-project upgrade to ~43574226) ---
// Prior to this upgrade, DICompileUnitAttr::get had a 9-argument convenience
// overload whose first parameter was `MLIRContext *`:
//
//   DICompileUnitAttr::get(ctx, id, sourceLanguage/*unsigned*/, file,
//                          producer, isOptimized, emissionKind,
//                          nameTableKind, splitDebugFilename)
//
// That overload was removed. The new canonical form (matching what MLIR's own
// DebugImporter uses internally) requires:
//   1. An explicit `recId` (DistinctAttr — null for non-recursive types).
//   2. An `isRecSelf` flag (false for non-recursive types).
//   3. A `DISourceLanguageNameAttr` wrapper for the source language instead of
//      a bare `unsigned` DW_LANG_* constant.
//   4. An explicit `isDebugInfoForProfiling` bool (was defaulted before).
//   5. An explicit `importedEntities` array (was defaulted before).
//
// The new full signature is:
//   DICompileUnitAttr::get(ctx, recId, isRecSelf, id, sourceLanguage,
//                          file, producer, isOptimized, emissionKind,
//                          isDebugInfoForProfiling, nameTableKind,
//                          splitDebugFilename, importedEntities)
//
// --- DISourceLanguageNameAttr (DWARF 6 source-language model) ---
// DWARF 6 decouples the language identifier from its version and dialect into
// a (name, version, dialect) triple. DISourceLanguageNameAttr::get takes:
//   language  — the legacy DW_LANG_* integer (non-zero for DWARF ≤ 5 codes).
//   name      — the DWARF 6 SourceLanguageName enum value (0 when using legacy).
//   version   — optional DWARF 6 version; nullopt for legacy language codes.
//   dialect   — optional DWARF 6 dialect; 0 for none.
// For DW_LANG_C99 we are using the legacy code path: language=DW_LANG_C99,
// name=0, version=nullopt, dialect=0.  C99 is chosen as a safe neutral
// baseline — the CU is synthetic and the language tag does not affect
// correctness or tool behaviour for our instrumentation use-case.
static LLVM::DICompileUnitAttr getOrCreateInstrumentCU(ModuleOp module) {
  if (auto cached =
          module->getAttrOfType<LLVM::DICompileUnitAttr>(kInstrumentCUAttrName))
    return cached;

  MLIRContext *ctx = module.getContext();

  // Synthetic file: plain name + root directory satisfies macOS ld's N_OSO
  // heuristic (see block comment above). Never opened at runtime.
  auto fileAttr = LLVM::DIFileAttr::get(
      ctx, /*name=*/"onnx-mlir-instrument.mlir", /*directory=*/"/");
  auto producerAttr = StringAttr::get(ctx, "onnx-mlir");

  // Wrap the legacy DW_LANG_C99 integer in the new DISourceLanguageNameAttr
  // required by the updated DICompileUnitAttr::get API (see block comment).
  // name=0 and dialect=0 select the DWARF ≤ 5 (legacy) code path; version is
  // not applicable for legacy language codes.
  auto sourceLangAttr = LLVM::DISourceLanguageNameAttr::get(
      ctx, /*language=*/llvm::dwarf::DW_LANG_C99, /*name=*/0,
      /*version=*/std::nullopt, /*dialect=*/0);

  // recId=null / isRecSelf=false: this CU is not part of a recursive type
  // chain. id=fresh DistinctAttr uniquely identifies this compile unit within
  // the module. Full emission is needed so that DISubprograms we attach to
  // instrument call sites are retained in the final object.
  auto cu = LLVM::DICompileUnitAttr::get(ctx,
      /*recId=*/DistinctAttr{}, /*isRecSelf=*/false,
      /*id=*/DistinctAttr::create(UnitAttr::get(ctx)),
      sourceLangAttr, fileAttr, producerAttr,
      /*isOptimized=*/true, LLVM::DIEmissionKind::Full,
      /*isDebugInfoForProfiling=*/false, LLVM::DINameTableKind::Default,
      /*splitDebugFilename=*/StringAttr{},
      /*importedEntities=*/{});
  module->setAttr(kInstrumentCUAttrName, cu);
  return cu;
}

// Return a concrete FileLineColLoc used as the inner anchor of every
// FusedLocWith<DISubprogramAttr> we create (both for function-level and
// call-site-level DISubprograms).
//
// Why a FileLineColLoc instead of UnknownLoc or NameLoc?
// The MLIR→LLVM IR translator (mlir-translate / translateModuleToLLVMIR)
// converts a FusedLocWith<DISubprogramAttr> by first converting its inner
// location list to a DILocation.  It returns nullptr for any inner loc that
// cannot be translated — this includes UnknownLoc directly, and also NameLoc
// or FusedLoc chains that bottom out in UnknownLoc. When the inner
// translation returns nullptr the parent CallSiteLoc translation falls back
// to the caller's loc, silently dropping the inlined `__omip:` scope from
// the resulting `!dbg` metadata and therefore from the .dSYM / DWARF output.
//
// Production .onnx files almost never carry preserved source locations, so
// most KrnlInstrumentOps inherit UnknownLoc — which would trigger exactly
// this fallback.  Using a concrete FileLineColLoc avoids the nullptr path.
// The line/col values are 0 (unused); only the DISubprogram scope carried
// by the surrounding FusedLoc matters for the DWARF DW_TAG_inlined_subroutine
// DIE that tooling (addr2line, Instruments.app, profile-model.py) reads.
static Location syntheticAnchorLoc(MLIRContext *ctx) {
  return FileLineColLoc::get(
      StringAttr::get(ctx, "onnx-mlir-instrument.mlir"), 0, 0);
}

// Build a fresh DISubprogramAttr for one OMInstrumentPoint call site.
//
// The subprogram name has the form `__omip:<opName>:<nodeName>`. The double-
// colon prefix makes the symbol easy to grep / filter in dwarfdump output and
// in profile-model.py while being safe for all DWARF consumers (it is just a
// string). Each begin/end pair for the same op gets its own distinct
// DistinctAttr id, so LLVM's DwarfDebug emits separate DW_TAG_inlined_subroutine
// DIEs for them and addr2line can distinguish begin from end call sites.
//
// DISubprogramAttr::get still uses the pre-upgrade 13-argument convenience
// overload that begins with `(MLIRContext*, DistinctAttr id, ...)` — that
// overload was retained in the new LLVM. Only DICompileUnitAttr::get lost its
// MLIRContext*-first form (see getOrCreateInstrumentCU).
static LLVM::DISubprogramAttr buildInstrumentSubprogram(MLIRContext *ctx,
    LLVM::DICompileUnitAttr cuAttr, LLVM::DIFileAttr fileAttr, StringRef opName,
    StringRef nodeName) {
  std::string label = ("__omip:" + opName + ":" + nodeName).str();
  auto nameAttr = StringAttr::get(ctx, label);
  auto srTypeAttr = LLVM::DISubroutineTypeAttr::get(
      ctx, /*callingConvention=*/0, /*types=*/{});
  return LLVM::DISubprogramAttr::get(ctx,
      /*id=*/DistinctAttr::create(UnitAttr::get(ctx)),
      /*compileUnit=*/cuAttr, /*scope=*/fileAttr,
      /*name=*/nameAttr, /*linkageName=*/nameAttr, fileAttr,
      /*line=*/0, /*scopeLine=*/0,
      LLVM::DISubprogramFlags::Definition | LLVM::DISubprogramFlags::Optimized,
      srTypeAttr, /*retainedNodes=*/{}, /*annotations=*/{});
}

// Lazily attach a function-level DISubprogramAttr to `funcOp` and return its
// updated location.
//
// After attachment the function's location becomes a
// FusedLocWith<DISubprogramAttr>. mlir-translate uses that to emit a
// DW_TAG_subprogram DIE that covers the function's entire PC range, which
// is the LLVM DwarfDebug requirement for any DW_TAG_inlined_subroutine DIE
// that references this function as its DW_AT_abstract_origin parent.
// Without this anchor each `__omip:` inline scope is an orphan — it has no
// parent PC range to hang off — and LLVM's DwarfDebug pass silently drops
// every inlined subroutine DIE from the output object.
//
// Caching: we inspect funcOp.getLoc() and skip re-attachment if it is already
// a FusedLocWith<DISubprogramAttr>.  Multiple KrnlInstrumentOps inside the
// same function therefore share a single function-level DISubprogram, which
// is the correct DWARF shape.
static Location getOrAttachFuncDISubprogram(
    LLVM::LLVMFuncOp funcOp, LLVM::DICompileUnitAttr cuAttr) {
  MLIRContext *ctx = funcOp.getContext();
  if (isa<FusedLocWith<LLVM::DISubprogramAttr>>(funcOp.getLoc()))
    return funcOp.getLoc();
  auto fileAttr = cuAttr.getFile();
  auto srTypeAttr = LLVM::DISubroutineTypeAttr::get(
      ctx, /*callingConvention=*/0, /*types=*/{});
  auto funcSP = LLVM::DISubprogramAttr::get(ctx,
      /*id=*/DistinctAttr::create(UnitAttr::get(ctx)),
      /*compileUnit=*/cuAttr, /*scope=*/fileAttr,
      /*name=*/funcOp.getSymNameAttr(),
      /*linkageName=*/funcOp.getSymNameAttr(), fileAttr,
      /*line=*/0, /*scopeLine=*/0,
      LLVM::DISubprogramFlags::Definition | LLVM::DISubprogramFlags::Optimized,
      srTypeAttr, /*retainedNodes=*/{}, /*annotations=*/{});
  Location funcLoc = FusedLocWith<LLVM::DISubprogramAttr>::get(
      {syntheticAnchorLoc(ctx)}, funcSP, ctx);
  funcOp->setLoc(funcLoc);
  return funcLoc;
}

// Build the MLIR location to stamp on the OMInstrumentPoint LLVM call op.
//
// The shape we produce is:
//
//   CallSiteLoc(
//     callee = FusedLocWith<DISubprogramAttr>("__omip:<op>:<node>"),
//     caller = FusedLocWith<DISubprogramAttr>(<enclosing-function>)
//   )
//
// mlir-translate (translateModuleToLLVMIR) converts a CallSiteLoc into an
// LLVM DILocation whose `inlinedAt` field points at the caller's DILocation.
// LLVM's DwarfDebug pass then recognises that `inlinedAt` chain and emits a
// DW_TAG_inlined_subroutine DIE named `__omip:<op>:<node>` inside the
// enclosing function's DW_TAG_subprogram.
//
// The result is that any PC belonging to that call instruction (and, more
// usefully, any PC sampled between consecutive begin/end instrument pairs) is
// resolvable by external tooling:
//   addr2line --inlines    — maps a PC to the `__omip:` inline scope
//   llvm-dwarfdump --lookup — same, via DWARF lookup tables
//   profile-model.py       — uses the `__omip:` name to attribute cycles to
//                            the originating ONNX op without reading .rodata
//                            strings or recovering register dataflow
//
// The synthetic name is chosen at lowering time, so it always reflects the
// post-conversion op identity (e.g. `zhigh.MatMul` for ops that were
// `onnx.MatMul` before the ZHigh conversion pass).
static Location buildInstrumentMarkerLoc(MLIRContext *ctx,
    LLVM::LLVMFuncOp funcOp, ModuleOp module, Location originalLoc,
    StringRef opName, StringRef nodeName) {
  auto cuAttr = getOrCreateInstrumentCU(module);
  auto fileAttr = cuAttr.getFile();

  // Function-level anchor (created on demand, cached on the FuncOp).
  Location funcLoc = getOrAttachFuncDISubprogram(funcOp, cuAttr);

  // Inline scope for THIS call site.
  auto inlineSP =
      buildInstrumentSubprogram(ctx, cuAttr, fileAttr, opName, nodeName);
  (void)originalLoc; // intentionally not embedded; see syntheticAnchorLoc
  auto inlineLoc = FusedLocWith<LLVM::DISubprogramAttr>::get(
      {syntheticAnchorLoc(ctx)}, inlineSP, ctx);

  // CallSiteLoc(callee=inlineLoc, caller=funcLoc) → MLIR translates
  // this into a DILocation with `inlinedAt` chain pointing at funcLoc.
  return CallSiteLoc::get(/*callee=*/inlineLoc, /*caller=*/funcLoc);
}

class KrnlInstrumentOpLowering : public ConversionPattern {
public:
  explicit KrnlInstrumentOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(
            typeConverter, KrnlInstrumentOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto *context = op->getContext();
    KrnlInstrumentOpAdaptor operandAdaptor(operands);
    Location loc = op->getLoc();
    KrnlInstrumentOp instrumentOp = llvm::dyn_cast<KrnlInstrumentOp>(op);

    StringRef opNameStr = instrumentOp.getOpName();

    StringRef nodeName;
    if (instrumentOp.getNodeName().has_value()) {
      // If we can get it from the instrument op direct, do so
      nodeName = instrumentOp.getNodeName().value();
    } else {
      // Otherwise, backup by creating it from the op.
      std::string nodeNameStr = getNodeNameInPresenceOfOpt(op);
      nodeName = rewriter.getStringAttr(nodeNameStr).strref();
    }
    LLVM_DEBUG(
        llvm::dbgs() << "Instrumentation_nodeName: " << nodeName << "\n");

    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    const LLVMTypeConverter *typeConverter =
        static_cast<const LLVMTypeConverter *>(getTypeConverter());

    // Get (or lazily insert) the OMInstrumentPoint function declaration.
    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    auto instrumentRef = getOrInsertInstrument(rewriter, parentModule);

    LLVM::GlobalOp globalOpNameStr = krnl::getOrCreateGlobalString(
        opNameStr, loc, rewriter, parentModule, typeConverter);
    Value opNamePtr =
        krnl::getPtrToGlobalString(globalOpNameStr, loc, rewriter);
    // Encode the tag with the length of the op and node name strings
    uint64_t opNameLen = opNameStr.size();
    uint64_t nodeNameLen = nodeName.size();
    uint64_t tagWithLen = instrumentOp.getTag();
    SET_INSTRUMENT_OP_NAME_LEN(tagWithLen, opNameLen);
    SET_INSTRUMENT_NODE_NAME_LEN(tagWithLen, nodeNameLen);
    Value tag = create.llvm.constant(
        IntegerType::get(context, 64), static_cast<int64_t>(tagWithLen));
    LLVM::GlobalOp globalStr = krnl::getOrCreateGlobalString(
        nodeName, loc, rewriter, parentModule, typeConverter);
    Value nodeNamePtr = krnl::getPtrToGlobalString(globalStr, loc, rewriter);
    // Build the FusedLoc up front so we can stamp it onto the call op
    // at creation. Carries a synthetic DISubprogram named
    // `__omip:<opName>:<nodeName>`; after mlir-translate and LLVM
    // codegen this becomes a DW_TAG_subprogram DIE with explicit
    // PC-range coverage, so external tooling (addr2line --inlines,
    // llvm-dwarfdump --lookup, profile-model.py) can map any sampled
    // PC back to the originating ONNX op without reading .rodata
    // strings or doing per-arch register-dataflow recovery. The
    // synthetic name is chosen at lowering time, so it always reflects
    // the post-conversion op identity (e.g. `zhigh.MatMul` even for
    // ops that were `onnx.MatMul` upstream of the ZHigh conversion
    // pass).
    // The enclosing function is what anchors the synthetic
    // DISubprogram chain. KrnlInstrumentOp is always inside an
    // LLVM::LLVMFuncOp at this stage of lowering.
    auto parentFunc = op->getParentOfType<LLVM::LLVMFuncOp>();
    assert(parentFunc && "krnl.runtime_instrument outside an LLVM func");
    Location markerLoc = buildInstrumentMarkerLoc(
        context, parentFunc, parentModule, loc, opNameStr, nodeName);
    // Bypass `create.llvm.call` here because that helper returns a
    // `Value` (null for void calls) and we need the op handle to set
    // the location. Direct CallOp::create gives us both.
    LLVM::CallOp::create(rewriter, markerLoc, /*resultTypes=*/TypeRange{},
        instrumentRef, ValueRange{opNamePtr, tag, nodeNamePtr});

    rewriter.eraseOp(op);
    return success();
  }

private:
  // Create a function declaration for OMInstrumentPoint, the signature is:
  //   `void (ptr, i64, ptr)`. Not part of RuntimeAPI.hpp because we need this
  //   only when instrumentation is present.
  FlatSymbolRefAttr getOrInsertInstrument(
      PatternRewriter &rewriter, ModuleOp module) const {
    MLIRContext *context = module.getContext();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, module.getLoc());
    Type llvmVoidTy = LLVM::LLVMVoidType::get(context);
    Type llvmI64Ty = IntegerType::get(context, 64);
    Type opaquePtrTy = getI8PointerType(context);
    return create.llvm.getOrInsertSymbolRef(module,
        StringRef("OMInstrumentPoint"), llvmVoidTy,
        {opaquePtrTy, llvmI64Ty, opaquePtrTy});
  }
};

class KrnlInstrumentInitOpLowering : public ConversionPattern {
public:
  explicit KrnlInstrumentInitOpLowering(
      LLVMTypeConverter &typeConverter, MLIRContext *context)
      : ConversionPattern(typeConverter,
            KrnlInstrumentInitOp::getOperationName(), 1, context) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    auto *context = op->getContext();
    Location loc = op->getLoc();
    KrnlInstrumentInitOp initOp = llvm::dyn_cast<KrnlInstrumentInitOp>(op);

    ModuleOp parentModule = op->getParentOfType<ModuleOp>();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, loc);
    Type opaquePtrTy = getI8PointerType(context);
    Type llvmI64Ty = IntegerType::get(context, 64);

    // omCompilationInfo is always defined in the same module by
    // KrnlEntryPointOpLowering. Reference it by name without inserting a
    // separate declaration, which would cause a redefinition conflict when
    // the definition is emitted later in the same pass.
    auto compilationInfoRef =
        FlatSymbolRefAttr::get(context, "omCompilationInfo");
    auto infoCall = LLVM::CallOp::create(rewriter, loc, TypeRange{opaquePtrTy},
        compilationInfoRef, ValueRange{});
    Value compilationInfoPtr = infoCall.getResult();

    Value tag =
        create.llvm.constant(llvmI64Ty, static_cast<int64_t>(initOp.getTag()));

    auto initRef = getOrInsertInstrumentInit(rewriter, parentModule);
    LLVM::CallOp::create(rewriter, loc, TypeRange{}, initRef,
        ValueRange{tag, compilationInfoPtr});

    rewriter.eraseOp(op);
    return success();
  }

private:
  // Declare OMInstrumentPointInit: void (i64, ptr). Not part of RuntimeAPI.hpp
  // because we need this only when instrumentation is present.
  FlatSymbolRefAttr getOrInsertInstrumentInit(
      PatternRewriter &rewriter, ModuleOp module) const {
    MLIRContext *context = module.getContext();
    MultiDialectBuilder<LLVMBuilder> create(rewriter, module.getLoc());
    Type llvmVoidTy = LLVM::LLVMVoidType::get(context);
    Type llvmI64Ty = IntegerType::get(context, 64);
    Type opaquePtrTy = getI8PointerType(context);
    return create.llvm.getOrInsertSymbolRef(module,
        StringRef("OMInstrumentPointInit"), llvmVoidTy,
        {llvmI64Ty, opaquePtrTy});
  }
};

void populateLoweringKrnlInstrumentOpPattern(LLVMTypeConverter &typeConverter,
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.insert<KrnlInstrumentOpLowering>(typeConverter, ctx);
  patterns.insert<KrnlInstrumentInitOpLowering>(typeConverter, ctx);
}

} // namespace krnl
} // namespace onnx_mlir
