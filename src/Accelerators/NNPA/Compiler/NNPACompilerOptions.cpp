/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- NNPACompilerOptions.cpp --------------------===//
//
// Copyright 2022-2025 The IBM Research Authors.
//
// =============================================================================
//
// Compiler Options for NNPA
//
//===----------------------------------------------------------------------===//
#include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp"

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

llvm::cl::opt<std::string> nnpaLoadConfigFile{"nnpa-load-config-file",
    llvm::cl::desc(
        "Load NNPA configuration such as device placement, quantization "
        "operations from a JSON file. To have a template for the JSON file, "
        "use "
        "--nnpa-save-config-file=cfg.json.\nNote that we can use regex for "
        "string values in the JSON file to match operations.\nThe compiler "
        "uses ECMAScript regular expressions for matching."),
    llvm::cl::init(""), llvm::cl::cat(OnnxMlirOptions)};

llvm::cl::opt<std::string> nnpaSaveConfigFile{"nnpa-save-config-file",
    llvm::cl::desc("Save NNPA configuration such as device placement and "
                   "quantization operations to a JSON file."),
    llvm::cl::init(""), llvm::cl::cat(OnnxMlirOptions)};

llvm::cl::opt<NNPAPlacementHeuristic> nnpaPlacementHeuristic{
    "nnpa-placement-heuristic",
    llvm::cl::desc(
        "[Optional] Choose NNPA-related heuristic to place operations "
        "on NNPA device:"),
    llvm::cl::values(
        clEnumVal(QualifyingOps, "Place all qualifying ops on NNPA."),
        clEnumVal(FasterOps, "Place qualifying ops that are faster on NNPA (default)."),
        clEnumVal(FasterOpsWSU, "FasterOps with stick/unstick cost."),
        clEnumVal(MuchFasterOpsWSU,
            "Much/Significantly FasterOps with stick/unstick cost.")),
    llvm::cl::init(FasterOps), llvm::cl::cat(OnnxMlirOptions)};

llvm::cl::opt<bool> nnpaDisableSaturation("nnpa-disable-saturation",
    llvm::cl::desc("Disable saturating f32 values before stickify them."
                   "Default is false."),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirCommonOptions));

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

} // namespace onnx_mlir
