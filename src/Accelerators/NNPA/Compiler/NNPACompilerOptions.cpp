/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- NNPACompilerOptions.cpp --------------------===//
//
// Copyright 2022 The IBM Research Authors.
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
std::vector<std::string> nnpaQuantOpTypes; // common for both

llvm::cl::opt<NNPAEmissionTargetType> nnpaEmissionTarget(
    llvm::cl::desc("[Optional] Choose NNPA-related target to emit "
                   "(once selected it will cancel the other targets):"),
    llvm::cl::values(
        clEnumVal(EmitZHighIR, "Lower model to ZHigh IR (ZHigh dialect)"),
        clEnumVal(EmitZLowIR, "Lower model to ZLow IR (ZLow dialect)"),
        clEnumVal(EmitZNONE, "Do not emit NNPA-related target (default)")),
    llvm::cl::init(EmitZNONE), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> nnpaClipToDLFloatRange("nnpa-clip-to-dlfloat-range",
    llvm::cl::desc("Clip CPU tensors to dlfloat range before stickification to "
                   "avoid out-of-range. Only clip Softmax inputs at this "
                   "moment. Default is true. This option will be removed and "
                   "replaced by --nnpa-saturation in the future."),
    llvm::cl::init(true), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> nnpaEnableZHighToOnnx("enable-zhigh-to-onnx",
    llvm::cl::desc(
        "Enabling this will convert a pattern `stick -> element-wise op -> "
        "unstick` back to an ONNX element-wise op. This conversion is called "
        "after applying all optimizations to remove stick/unstick at ZHigh "
        "level. Default is true."),
    llvm::cl::init(true), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> nnpaEnableZHighDecomposeStickUnstick(
    "enable-zhigh-decompose-stick-unstick",
    llvm::cl::desc(
        "[Experimental feature] Enabling this will convert zhigh.Stick to "
        "`zhigh.F32ToDLF16 -> onnx.LayoutTransform` and zhigh.Unstick to "
        "`onnx.LayoutTransform -> zhigh.DLF16ToF32`. "
        "Default is false."),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

// Enabled default now, could also enable it only if parallel is on as parallel
// stick/unstick is quite a bit faster than sequential.
llvm::cl::opt<bool> nnpaEnableCompilerStickUnstick(
    "enable-compiler-stick-unstick",
    llvm::cl::desc("[Experimental feature] Enable the compiler generate some "
                   "stick/unstick code. Default is true."),
    llvm::cl::init(true), llvm::cl::cat(OnnxMlirCommonOptions));

llvm::cl::opt<bool> nnpaEnableScalarBcastBinary(
    "nnpa-enable-scalar-bcast-binary",
    llvm::cl::desc("Enable the lowering to NNPA of binary operations with "
                   "broadcasting of a scalar operand.\n"
                   "Currently only enable ONNXDiv. Default is false."),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirCommonOptions));

llvm::cl::opt<std::string> nnpaLoadDevicePlacementFile{
    "nnpa-load-device-placement-file",
    llvm::cl::desc(
        "Load device placement configuration from a JSON file. To "
        "have a template for the JSON file, use "
        "--nnpa-save-device-placement-file=cfg.json.\nNote that we can use "
        "regex for "
        "string values in the JSON file to match operations.\nThe compiler "
        "uses "
        "C++ std::regex_match function for matching."),
    llvm::cl::init(""), llvm::cl::cat(OnnxMlirOptions)};

llvm::cl::opt<std::string> nnpaSaveDevicePlacementFile{
    "nnpa-save-device-placement-file",
    llvm::cl::desc("Save device placement configuration to a JSON file."),
    llvm::cl::init(""), llvm::cl::cat(OnnxMlirOptions)};

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

llvm::cl::opt<bool> nnpaEnableSaturation("nnpa-saturation",
    llvm::cl::desc("Enable saturating f32 values before stickify them."
                   "This option turns enable-compiler-stick-unstick on."
                   "Default is false."),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirCommonOptions));

llvm::cl::list<NNPAQuantOptions> nnpaQuantDynamic("nnpa-quant-dynamic",
    llvm::cl::desc(
        "Enable dynamic quantization of the input model. If enabled, it only "
        "quantizes from fp32 to i8. If an ONNX operation is already in i8, "
        "no quantization is applied to that operation. Optionally, a "
        "comma-separated list of quantization options can be specified as its "
        "value, e.g. -nnpa-quant-dynamic=SymmetricActivation,SymmetricWeight."),
    llvm::cl::values(clEnumVal(symWeight, "Symmetric quant for weights."),
        clEnumVal(asymWeight, "Asymmetric quant for weights."),
        clEnumVal(symActivation, "Symmetric quant for activations."),
        clEnumVal(asymActivation, "Asymmetric quant for activations."),
        // Use an empty string for the case where `--nnpa-quant-dynamic` is
        // specified on the command line WITHOUT value, which is different from
        // the case where `--nnpa-quant-dynamic` is NOT specified on the command
        // line.
        clEnumValN(autoQuantOpt, "",
            "Compiler automatically finds the best options.")),
    llvm::cl::ValueOptional, llvm::cl::CommaSeparated,
    llvm::cl::cat(OnnxMlirCommonOptions));

llvm::cl::list<std::string, std::vector<std::string>> nnpaQuantOpTypesOpt(
    "nnpa-quant-op-types",
    llvm::cl::desc(
        "A comma-separated list of types of operations that are quantized. "
        "E.g. 'MatMul,Conv'. Strings for types are the same as ONNX "
        "operator "
        "names in https://onnx.ai/onnx/operators/. By default or with "
        "specifying this option, the compiler will determine the operation "
        "types by itself."),
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

} // namespace onnx_mlir
