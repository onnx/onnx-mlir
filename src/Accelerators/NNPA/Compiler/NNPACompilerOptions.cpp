/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- NNPACompilerOptions.cpp --------------------===//
//
// Copyright 2022-2026 The IBM Research Authors.
//
// =============================================================================
//
// Compiler Options for NNPA
//
//===----------------------------------------------------------------------===//
#include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp"
#include "src/Accelerators/NNPA/Support/NNPALimit.hpp"

#include <atomic>
#include <cstdio>
#include <cstring>
#include <string>

#define DEBUG_TYPE "NNPACompilerOptions"

namespace onnx_mlir {

// Use external storage for the options so that they are globally accessible
std::vector<NNPAQuantOptions> nnpaQuantDynamic; // common for both
std::vector<std::string> nnpaQuantOpTypes;      // common for both

llvm::cl::opt<NNPAEmissionTargetType> nnpaEmissionTarget(
    llvm::cl::desc("[Optional] Choose NNPA-related target to emit "
                   "(once selected it will cancel the other targets):"),
    llvm::cl::values(
        clEnumVal(EmitZHighIR, "Lower model to ZHigh IR (ZHigh dialect)"),
        clEnumVal(EmitZLowIR, "Lower model to ZLow IR (ZLow dialect)"),
        clEnumVal(EmitZNONE, "Do not emit NNPA-related target (default)")),
    llvm::cl::init(EmitZNONE), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> nnpaDisableZHighToOnnx("disable-zhigh-to-onnx",
    llvm::cl::desc(
        "By default we convert a pattern `stick -> element-wise op -> "
        "unstick` back to an ONNX element-wise op. This conversion is called "
        "after applying all optimizations to remove stick/unstick at ZHigh "
        "level. Use this option to disable this optimization."),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> nnpaDisableZHighDecomposeStickUnstick(
    "disable-zhigh-decompose-stick-unstick",
    llvm::cl::desc(
        "Disable the converstion of zhigh.Stick to `zhigh.F32ToDLF16 -> "
        "onnx.LayoutTransform` and zhigh.Unstick to `onnx.LayoutTransform -> "
        "zhigh.DLF16ToF32`. Default is false."),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

// Enabled default now, could also enable it only if parallel is on as parallel
// stick/unstick is quite a bit faster than sequential.
llvm::cl::opt<bool> nnpaDisableCompilerStickUnstick(
    "disable-compiler-stick-unstick",
    llvm::cl::desc("Disable the compiler to generate some "
                   "stick/unstick code. Default is false."),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirCommonOptions));

llvm::cl::opt<bool> nnpaEnableScalarBcastBinary(
    "nnpa-enable-scalar-bcast-binary",
    llvm::cl::desc("Enable the lowering to NNPA of binary operations with "
                   "broadcasting of a scalar operand.\n"
                   "Currently only enable ONNXDiv. Default is false."),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirCommonOptions));

llvm::cl::opt<NNPAPlacementHeuristic> nnpaPlacementHeuristic{
    "nnpa-placement-heuristic",
    llvm::cl::desc(
        "[Optional] Choose NNPA-related heuristic to place operations "
        "on NNPA device:"),
    llvm::cl::values(
        clEnumVal(QualifyingOps, "Place all qualifying ops on NNPA (default)."),
        clEnumVal(FasterOps, "Place qualifying ops that are faster on NNPA."),
        clEnumVal(FasterOpsWSU, "FasterOps with stick/unstick cost."),
        clEnumVal(MuchFasterOpsWSU,
            "Much/Significantly FasterOps with stick/unstick cost.")),
    llvm::cl::init(QualifyingOps), llvm::cl::cat(OnnxMlirOptions)};

llvm::cl::opt<bool> nnpaDisableSaturation("nnpa-disable-saturation",
    llvm::cl::desc("Disable saturating f32 values before stickify them."
                   "Default is false."),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirCommonOptions));

namespace {

// Parsed value of --nnpa-saturation (default: mask).
enum class NNPASaturationKind { Mask, Max, Limit };
NNPASaturationKind nnpaSaturationKind = NNPASaturationKind::Mask;
float nnpaSaturationLimit = DLF16_MAX;

// Default limit of --nnpa-saturation=limit: DLF16_MAX / 4 (exactly
// representable in dlfloat16), which leaves room for the scores plus two
// masks added on the zAIU.
constexpr float defaultNNPASaturationLimit = DLF16_MAX / 4;

// Round a positive value down to a value representable in dlfloat16 (9
// fraction bits), so that the limit never exceeds what was asked for.
float roundDownToDLF16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  bits &= ~((1u << (23 - 9)) - 1); // Drop the f32 fraction bits dlf16 lacks.
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

// Parser of --nnpa-saturation that rejects invalid values when the option is
// parsed, and records the parsed kind and limit.
struct NNPASaturationParser : public llvm::cl::parser<std::string> {
  using llvm::cl::parser<std::string>::parser;

  bool parse(llvm::cl::Option &opt, llvm::StringRef argName,
      llvm::StringRef arg, std::string &value) {
    value = arg.str();
    if (arg == "mask") {
      nnpaSaturationKind = NNPASaturationKind::Mask;
      nnpaSaturationLimit = DLF16_MAX;
      return false;
    }
    if (arg == "max") {
      nnpaSaturationKind = NNPASaturationKind::Max;
      nnpaSaturationLimit = DLF16_MAX;
      return false;
    }
    if (arg == "limit") {
      nnpaSaturationKind = NNPASaturationKind::Limit;
      nnpaSaturationLimit = defaultNNPASaturationLimit;
      return false;
    }
    double limit;
    llvm::StringRef number = arg;
    if (number.consume_front("limit") && !number.getAsDouble(limit) &&
        limit >= 1.0 && limit <= DLF16_MAX) {
      nnpaSaturationKind = NNPASaturationKind::Limit;
      nnpaSaturationLimit = roundDownToDLF16(static_cast<float>(limit));
      return false;
    }
    return opt.error("invalid value '" + arg +
                     "': expected mask, max, limit, or limit<value> with "
                     "1 <= value <= DLF16_MAX (~8.57e9), e.g. limit1e9");
  }
};

} // namespace

static llvm::cl::opt<std::string, false, NNPASaturationParser> nnpaSaturation(
    "nnpa-saturation",
    llvm::cl::desc(
        "How f32 values are kept within the dlfloat16 range for NNPA. "
        "mask (default): saturate at +/-DLF16_MAX (~8.6e9), and raise large "
        "negative constants (e.g. -FLT_MAX) that only feed a Softmax, such "
        "as attention mask fills, to -1e9 (-1e4 for f16), so that fully "
        "masked rows do not overflow into NaN. max: saturate at "
        "+/-DLF16_MAX only (behavior before the mask handling). limit: "
        "saturate at +/-DLF16_MAX/4 (~2.1e9), no mask handling. "
        "limit<value>: saturate at +/-value (1 <= value <= DLF16_MAX, "
        "rounded down to a dlfloat16 value), e.g. limit1e9. "
        "--nnpa-disable-saturation turns off all of these, except for "
        "constants, which are always saturated."),
    llvm::cl::value_desc("mask|max|limit|limit<value>"), llvm::cl::init("mask"),
    llvm::cl::cat(OnnxMlirCommonOptions));

float getNNPASaturationLimit() { return nnpaSaturationLimit; }

bool isNNPASoftmaxMaskClampEnabled() {
  return nnpaSaturationKind == NNPASaturationKind::Mask &&
         !nnpaDisableSaturation;
}

// Statistics for -opt-report=NNPASaturation; atomic, since passes may run on
// several functions in parallel.
static std::atomic<uint64_t> statConstantValues{0}, statBeyondRange{0},
    statClippedByLimit{0}, statMaskValues{0};

void recordNNPAConstantSaturation(
    uint64_t numValues, uint64_t numBeyondRange, uint64_t numClippedByLimit) {
  statConstantValues += numValues;
  statBeyondRange += numBeyondRange;
  statClippedByLimit += numClippedByLimit;
}

void recordNNPAMaskClamp(uint64_t numValues) { statMaskValues += numValues; }

void printNNPASaturationReport() {
  // Same fields in every mode, all comma separated: the target type (dlf16
  // for now; other types, e.g. int8 or float8, would get their own line), the
  // mode, the mask values raised to the fill value (0 unless mask), the
  // values outside the dlfloat16 range, the limit and the values outside it
  // but inside the dlfloat16 range (0 unless limit < DLF16_MAX), and the
  // total number of constant values converted. With --nnpa-disable-saturation
  // the mode is "disabled": there is no mask handling (masked is 0), but
  // constants are still saturated, so their counts and the limit are reported
  // as they are.
  printf("==SATURATION-REPORT==, dlf16, mode, %s, masked, %llu, "
         "outside-range, %llu, outside-limit, %g, %llu, total, %llu\n",
      nnpaDisableSaturation ? "disabled" : nnpaSaturation.c_str(),
      static_cast<unsigned long long>(statMaskValues.exchange(0)),
      static_cast<unsigned long long>(statBeyondRange.exchange(0)),
      static_cast<double>(nnpaSaturationLimit),
      static_cast<unsigned long long>(statClippedByLimit.exchange(0)),
      static_cast<unsigned long long>(statConstantValues.exchange(0)));
  fflush(stdout);
}

llvm::cl::list<NNPAQuantOptions, std::vector<NNPAQuantOptions>>
    nnpaQuantDynamicOpt("nnpa-quant-dynamic",
        llvm::cl::desc(
            "Enable dynamic quantization of the input model. If enabled, it "
            "only quantizes from fp32 to i8. If an ONNX operation is already "
            "in i8, no quantization is applied to that operation. Optionally, "
            "a comma-separated list of quantization options can be specified "
            "as its value, e.g. -nnpa-quant-dynamic=symActivation,symWeight."),
        llvm::cl::values(clEnumVal(symWeight, "Symmetric quant for weights."),
            clEnumVal(asymWeight, "Asymmetric quant for weights."),
            clEnumVal(symActivation, "Symmetric quant for activations."),
            clEnumVal(asymActivation, "Asymmetric quant for activations."),
            // Use an empty string for the case where `--nnpa-quant-dynamic` is
            // specified on the command line WITHOUT value, which is different
            // from the case where `--nnpa-quant-dynamic` is NOT specified on
            // the command line.
            clEnumValN(autoQuantOpt, "",
                "Compiler automatically finds the best options. Once this "
                "option (an empty string) is in the list, the other options "
                "are ignored. This is the default option when "
                "`-nnpa-quant-dynamic` is specified without any value.")),
        llvm::cl::location(nnpaQuantDynamic), llvm::cl::ValueOptional,
        llvm::cl::CommaSeparated, llvm::cl::cat(OnnxMlirCommonOptions));

llvm::cl::list<std::string, std::vector<std::string>> nnpaQuantOpTypesOpt(
    "nnpa-quant-op-types",
    llvm::cl::desc(
        "A comma-separated list of types of operations that are quantized. "
        "E.g. 'MatMul,Conv'. Strings for types are the same as ONNX operator "
        "names in https://onnx.ai/onnx/operators/. Currently, only MatMul is "
        "supported. Without specifying this option, the compiler will "
        "determine the operation types by itself."),
    llvm::cl::location(nnpaQuantOpTypes), llvm::cl::ValueOptional,
    llvm::cl::CommaSeparated, llvm::cl::cat(OnnxMlirCommonOptions));

llvm::cl::opt<bool> nnpaUseDynamicQuantizeLinearOnCPU("nnpa-cpu-dql",
    llvm::cl::desc("Use dynamic quantized linear on CPU. Default is false"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirCommonOptions));

llvm::cl::opt<bool> nnpaUseDynamicQuantizeLinearOnCPUForScaleOffset(
    "nnpa-cpu-dql-scale",
    llvm::cl::desc("Use dynamic quantized linear computation of "
                   " scale and offset on CPU. Default is false"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirCommonOptions));

llvm::cl::opt<bool> nnpaDisableFusionOpStickUnstick(
    "nnpa-disable-fusion-op-stick-unstick",
    llvm::cl::desc("Disable fusion of eligible operations with "
                   " surrounding stick and unstick ops. Default is false"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirCommonOptions));

llvm::cl::opt<bool> nnpaDisableShapeRestriction(
    "nnpa-disable-shape-restriction",
    llvm::cl::desc("Disable shape restriction checks for operations with "
                   "dynamic shapes. When enabled, the compiler assumes that "
                   "dynamic shapes will satisfy NNPA constraints at runtime. "
                   "Helpful for Convolutions. Default is false"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirCommonOptions));

llvm::cl::opt<bool> nnpaDisableHugePageMalloc("nnpa-disable-hugepage-malloc",
    llvm::cl::desc("Disable replacement of malloc with OMHugePageMalloc. "
                   "Default is false"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> nnpaDisableExpandAttentionMask(
    "nnpa-disable-expand-attention-mask",
    llvm::cl::desc("Disable expansion of attention masks for broadcasting Add "
                   "operations. Default is false"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> nnpaDisableConstructSDPA3D("nnpa-disable-construct-sdpa-3d",
    llvm::cl::desc("Disable fusion of MatMul+Add+Softmax+MatMul (attention) "
                   "into a single fused op for NNPA. Default is false"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

} // namespace onnx_mlir
