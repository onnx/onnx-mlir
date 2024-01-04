/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ CompilerOptions.cpp -------------------------===//
//
// Copyright 2022, 2023 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding options.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "ExternalUtil.hpp"
#include "onnx-mlir/Compiler/OMCompilerRuntimeTypes.h"
#include "onnx-mlir/Compiler/OMCompilerTypes.h"
#include "src/Compiler/CompilerOptions.hpp"

#define DEBUG_TYPE "compiler_options"

const std::string OnnxMlirEnvOptionName = "ONNX_MLIR_FLAGS";

namespace onnx_mlir {

// Use external storage for the options so that they are globally accessible
std::string inputFilename;                             // common for both
std::string outputBaseName;                            // common for both
std::vector<accel::Accelerator::Kind> maccel;          // common for both
OptLevel OptimizationLevel;                            // common for both
std::string mtriple;                                   // common for both
std::string mcpu;                                      // common for both
std::string march;                                     // common for both
InstrumentStages instrumentStage;                      // common for both
bool onnxConstPropRoundFPToInt;                        // common for both
int onnxConstPropExpansionBound;                       // common for both
std::vector<std::string> onnxConstPropDisablePatterns; // common for both
bool enableONNXHybridPass;                             // common for both
std::vector<std::string> functionsToDecompose;         // common for both
std::string opsForCall;                                // common for both
EmissionTargetType emissionTarget;                     // onnx-mlir only
bool invokeOnnxVersionConverter;                       // onnx-mlir only
bool preserveLocations;                                // onnx-mlir only
bool printIR;                                          // onnx-mlir only
bool preserveBitcode;                                  // onnx-mlir only
bool preserveLLVMIR;                                   // onnx-mlir only
bool preserveMLIR;                                     // onnx-mlir only
bool useOnnxModelTypes;                                // onnx-mlir only
int repeatOnnxTransform;                               // onnx-mlir only
std::string shapeInformation;                          // onnx-mlir only
ModelSize modelSize;                                   // onnx-mlir only
bool storeConstantsToFile;                             // onnx-mlir only
float constantsToFileTotalThreshold;                   // onnx-mlir only
float constantsToFileSingleThreshold;                  // onnx-mlir only
bool VerboseOutput;                                    // onnx-mlir only
std::vector<std::string> Xopt;                         // onnx-mlir only
std::vector<std::string> Xllc;                         // onnx-mlir only
std::string mllvm;                                     // onnx-mlir only
std::string instrumentOps;                             // onnx-mlir only
unsigned instrumentControlBits;                        // onnx-mlir only
bool instrumentONNXSignature;                          // onnx-mlir only
std::string ONNXOpStats;                               // onnx-mlir only
int onnxOpTransformThreshold;                          // onnx-mlir only
bool onnxOpTransformReport;                            // onnx-mlir only
bool enableParallel;                                   // onnx-mlir only
bool disableSimdOption;                                // onnx-mlir only
bool disableRecomposeOption;                           // onnx-mlir only
bool enableSimdDataLayout;                             // onnx-mlir only
bool verifyInputTensors;                               // onnx-mlir only
bool allowSorting;                                     // onnx-mlir only
std::string reportHeapBefore;                          // onnx-mlir only
std::string reportHeapAfter;                           // onnx-mlir only
std::string modelTag;                                  // onnx-mlir only
bool enableConvOptPass;                                // onnx-mlir only
bool disableConstantProp;                              // onnx-mlir only
std::vector<std::string> extraLibPaths;                // onnx-mlir only
std::vector<std::string> extraLibs;                    // onnx-mlir only
ProfileIRs profileIR;                                  // onnx-mlir only
OptReport optReport;                                   // onnx-mlir only
bool useOldBufferization;                              // onnx-mlir only
bool split_input_file;                                 // onnx-mlir-opt only
bool verify_diagnostics;                               // onnx-mlir-opt only
bool verify_passes;                                    // onnx-mlir-opt only
bool allowUnregisteredDialects;                        // onnx-mlir-opt only

// Category for common options shared between onnx-mlir and onnx-mlir-opt.
llvm::cl::OptionCategory OnnxMlirCommonOptions("common options",
    "These are options shared between onnx-mlir and onnx-mlir-opt");

// Category for options for onnx-mlir only.
llvm::cl::OptionCategory OnnxMlirOptions(
    "onnx-mlir options", "These are onnx-mlir frontend options");

// Category for options for onnx-mlir-opt only.
llvm::cl::OptionCategory OnnxMlirOptOptions(
    "onnx-mlir-opt options", "These are onnx-mlir-opt frontend options.");

// Common options shared between onnx-mlir and onnx-mlir-opt
static llvm::cl::opt<std::string, true> inputFilenameOpt(llvm::cl::Positional,
    llvm::cl::desc("<input file>"),
    llvm::cl::value_desc("Default read from stdin"),
    llvm::cl::location(inputFilename), llvm::cl::init("-"),
    llvm::cl::cat(OnnxMlirCommonOptions));

static llvm::cl::opt<std::string, true> outputBaseNameOpt("o",
    llvm::cl::desc("For onnx-mlir, specify the base path for output file, "
                   "extension will be added. Default is input filename "
                   "without the extension, or \"stdin\" if input is stdin.\n"
                   "For onnx-mlir-opt, specify the output filename. Default is "
                   "stdout."),
    llvm::cl::value_desc("path"), llvm::cl::location(outputBaseName),
    llvm::cl::init("-"), llvm::cl::cat(OnnxMlirCommonOptions),
    llvm::cl::ValueRequired);

static llvm::cl::list<accel::Accelerator::Kind,
    std::vector<accel::Accelerator::Kind>>
    maccelOpt("maccel",
        llvm::cl::desc("Specify an accelerator to generate code for"),
        llvm::cl::location(maccel),
        // clang-format off
        llvm::cl::values(
          APPLY_TO_ACCELERATORS(CREATE_ACCEL_CL_ENUM)
          clEnumValN(accel::Accelerator::Kind::NONE, "NONE", "No accelerator")
        ),
        // clang-format on
        llvm::cl::cat(OnnxMlirCommonOptions), llvm::cl::ValueRequired);

static llvm::cl::opt<OptLevel, true> OptimizationLevelOpt(
    llvm::cl::desc("Levels:"),
    llvm::cl::values(clEnumVal(O0, "Optimization level 0 (default):"),
        clEnumVal(O1, "Optimization level 1"),
        clEnumVal(O2, "Optimization level 2"),
        clEnumVal(O3, "Optimization level 3, SIMD is enabled")),
    llvm::cl::location(OptimizationLevel), llvm::cl::init(O0),
    llvm::cl::cat(OnnxMlirCommonOptions));

static llvm::cl::opt<std::string, true> mtripleOpt("mtriple",
    llvm::cl::desc("Override target triple for module"),
    llvm::cl::value_desc("LLVM target triple"), llvm::cl::location(mtriple),
    llvm::cl::init(kDefaultTriple), llvm::cl::cat(OnnxMlirCommonOptions),
    llvm::cl::ValueRequired);

static llvm::cl::opt<std::string, true> mcpuOpt("mcpu",
    llvm::cl::desc("Target cpu"),
    llvm::cl::value_desc("Target a specific CPU type"),
    llvm::cl::location(mcpu), llvm::cl::cat(OnnxMlirCommonOptions),
    llvm::cl::ValueRequired);

static llvm::cl::opt<std::string, true> marchOpt("march",
    llvm::cl::desc("Target architecture to generate code for"),
    llvm::cl::value_desc("Target a specific architecture type"),
    llvm::cl::location(march), llvm::cl::cat(OnnxMlirCommonOptions),
    llvm::cl::ValueRequired);

static llvm::cl::opt<InstrumentStages, true> instrumentStageOpt(
    "instrument-stage", llvm::cl::desc("Specify stage to be instrumented:"),
    llvm::cl::location(instrumentStage),
    llvm::cl::values(APPLY_TO_NO_ACCELERATORS(DEFAULT_INSTRUMENTSTAGE_CL_ENUM)
            APPLY_TO_ACCELERATORS(ACCEL_INSTRUMENTSTAGE_CL_ENUM)),
    llvm::cl::init(Onnx), llvm::cl::cat(OnnxMlirCommonOptions));

static llvm::cl::opt<bool, true> onnxConstPropRoundFPToIntOpt(
    "onnx-const-prop-round-fp-to-int",
    llvm::cl::desc("If true constant propagates onnx.Cast from a floating "
                   "point type to an integer type by rounding to nearest, "
                   "ties to even. If false truncates towards zero."),
    llvm::cl::location(onnxConstPropRoundFPToInt), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirCommonOptions));

static llvm::cl::opt<int, true> onnxConstPropExpansionBoundOpt(
    "onnx-const-prop-expansion-bound",
    llvm::cl::desc("ONNX dialect constant propagation maximum expansion factor."
                   " Constants are not propagated if their bytes size exceed"
                   " the aggregate operands' sizes by more than this factor."
                   " Set to -1 to always propagate, which is the default."),
    llvm::cl::location(onnxConstPropExpansionBound), llvm::cl::init(-1),
    llvm::cl::cat(OnnxMlirCommonOptions));

static llvm::cl::list<std::string, std::vector<std::string>>
    onnxConstPropDisablePatternsOpt("onnx-const-prop-disable-pattern",
        llvm::cl::desc("Named constant propagation pattern to disable.\n"
                       "Repeat the flag to disable multiple patterns."),
        llvm::cl::value_desc("named constant propagation pattern to disable"),
        llvm::cl::location(onnxConstPropDisablePatterns),
        llvm::cl::cat(OnnxMlirCommonOptions));

static llvm::cl::opt<bool, true> enableONNXHybridPassOpt("onnx-hybrid-pass",
    llvm::cl::desc("Enable ONNX hybrid pass (default=true)\n"
                   "Set to 'false' if you want to disable ONNX hybrid pass."),
    llvm::cl::location(enableONNXHybridPass), llvm::cl::init(true),
    llvm::cl::cat(OnnxMlirCommonOptions));

static llvm::cl::list<std::string, std::vector<std::string>>
    functionsToDecomposeOpt("functions-to-decompose",
        llvm::cl::desc("Specify ONNX functions to decompose"),
        llvm::cl::location(functionsToDecompose),
        llvm::cl::cat(OnnxMlirCommonOptions));

static llvm::cl::opt<bool, true> disableRecomposeOptionOpt("disable-recompose",
    llvm::cl::desc("Disable recomposition of ONNX operations."),
    llvm::cl::location(disableRecomposeOption), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

// Options for onnx-mlir only
static llvm::cl::opt<EmissionTargetType, true> emissionTargetOpt(
    llvm::cl::desc("Choose target to emit:"),
    llvm::cl::location(emissionTarget),
    llvm::cl::values(
        clEnumVal(EmitONNXBasic,
            "Ingest ONNX and emit the basic ONNX operations without "
            "inferred shapes."),
        clEnumVal(
            EmitONNXIR, "Ingest ONNX and emit corresponding ONNX dialect."),
        clEnumVal(EmitMLIR,
            "Lower the input to MLIR built-in transformation dialect."),
        clEnumVal(
            EmitLLVMIR, "Lower the input to LLVM IR (LLVM MLIR dialect)."),
        clEnumVal(EmitObj, "Compile the input into a object file."),
        clEnumVal(
            EmitLib, "Compile the input into a shared library (default)."),
        clEnumVal(EmitJNI, "Compile the input into a jar file.")),
    llvm::cl::init(EmitLib), llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> invokeOnnxVersionConverterOpt(
    "invokeOnnxVersionConverter",
    llvm::cl::desc(
        "call onnx version converter to convert ONNX model to current version"),
    llvm::cl::location(invokeOnnxVersionConverter), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> preserveLocationsOpt("preserveLocations",
    llvm::cl::desc("emit location data:"),
    llvm::cl::location(preserveLocations), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> printIROpt("printIR",
    llvm::cl::desc("print the IR to stdout:"), llvm::cl::location(printIR),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> preserveBitcodeOpt("preserveBitcode",
    llvm::cl::desc(
        "dont delete the bitcode files (optimized and unoptimized):"),
    llvm::cl::location(preserveBitcode), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> preserveLLVMIROpt("preserveLLVMIR",
    llvm::cl::desc("dont delete the LLVMIR files:"),
    llvm::cl::location(preserveLLVMIR), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> preserveMLIROpt("preserveMLIR",
    llvm::cl::desc("dont delete the MLIR files (input and llvm):"),
    llvm::cl::location(preserveMLIR), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> useOnnxModelTypesOpt("useOnnxModelTypes",
    llvm::cl::desc("use types and shapes from ONNX model"),
    llvm::cl::location(useOnnxModelTypes), llvm::cl::init(true),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<int, true> repeatOnnxTransformOpt("repeatOnnxTransform",
    llvm::cl::desc(
        "invoke extra onnx transform pass(shape inference, constant and etc.)"),
    llvm::cl::location(repeatOnnxTransform), llvm::cl::init(0),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<std::string, true> shapeInformationOpt("shapeInformation",
    llvm::cl::desc(
        "Custom shapes for the inputs of the ONNX model, e.g. setting static "
        "shapes for dynamic inputs.\n"
        "\"value\" is in the format of "
        "\"INPUT_ID1:D1xD2x...xDn,INPUT_ID2:D1xD2x...xDn, ...\",\n"
        "where \"INPUT_ID1, INPUT_ID2, ...\" are input indices (starting from "
        "0 or being -1 for all input indices), and\n"
        "\"D1, D2, ...\" are dimension sizes (positive integers or -1 for "
        "unknown dimensions)"),
    llvm::cl::value_desc("value"), llvm::cl::location(shapeInformation),
    llvm::cl::cat(OnnxMlirOptions));

// Default value is defined by the OnnxMlirEnvOptionName constant string
// variable, but the default setting mechanism here cannot be used here as we
// need to evaluate this value prior to the compiler options being set. Proper
// handling of the value of this compiler option is set by the calling the
// parseCustomEnvFlagsCommandLineOption(...) function.
static llvm::cl::opt<std::string, true> customEnvFlagsOpt("customEnvFlags",
    llvm::cl::desc("Override default option env var OnnxMlirEnvOptionName: "
                   "ONNX_MLIR_FLAGS"),
    llvm::cl::value_desc("option env var"), llvm::cl::location(customEnvFlags),
    llvm::cl::init(""), llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<ModelSize, true> modelSizeOpt("modelSize",
    llvm::cl::desc("Model to generate code"),
    llvm::cl::value_desc("Only support small or large"),
    llvm::cl::location(modelSize),
    llvm::cl::values(
        clEnumVal(small, "Generate code for the small model. "
                         "No special treatment at this moment. This is the "
                         "default code model"),
        clEnumVal(large,
            "Generate code for the large model. "
            "Global constants are put into large read-only data section.")),
    llvm::cl::init(small), llvm::cl::cat(OnnxMlirOptions),
    llvm::cl::ValueRequired);

static llvm::cl::opt<bool, true> storeConstantsToFileOpt(
    "store-constants-to-file",
    llvm::cl::desc(
        "Constants will be stored on a binary file instead of be embedded "
        "into the model.so. The binary file is in the same folder as the "
        "model.so and has the same name as the model with the extension of "
        ".constants.bin. For inference, model.constants.bin must be at the "
        "same folder as the inference program. If model.constants.bin is at "
        "another folder, use the environment variable OM_CONSTANT_PATH to set "
        "the constant folder. Windows will be supported soon."),
    llvm::cl::location(storeConstantsToFile), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<float, true> constantsToFileTotalThresholdOpt(
    "constants-to-file-total-threshold",
    llvm::cl::desc(
        "Put global constants to a file if the total size in "
        "bytes of constants is greater than this threshold. "
        "store-constants-to-file must be enabled for this to be effective. "
        "Only count constants whose size is greater than "
        "constants-to-file-single-threshold. Value is in GB. Default is 2GB."),
    llvm::cl::location(constantsToFileTotalThreshold), llvm::cl::init(2.0),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<float, true> constantsToFileSingleThresholdOpt(
    "constants-to-file-single-threshold",
    llvm::cl::desc(
        "Put global constants to a file if a single constant's size in "
        "bytes is greater than this threshold. "
        "store-constants-to-file must be enabled for this to be effective. "
        "Total sizes in bytes of satisfied constants must be greater than "
        "constants-to-file-total-threshold. Value is in KB. Default is 1KB."),
    llvm::cl::location(constantsToFileSingleThreshold), llvm::cl::init(1.0),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> VerboseOutputOpt("v",
    llvm::cl::desc("Use verbose output"), llvm::cl::location(VerboseOutput),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::list<std::string, std::vector<std::string>> XoptOpt("Xopt",
    llvm::cl::desc("Arguments to forward to LLVM's 'opt' option processing"),
    llvm::cl::value_desc("A valid LLVM's 'opt' option"),
    llvm::cl::location(Xopt), llvm::cl::cat(OnnxMlirOptions), llvm::cl::Hidden,
    llvm::cl::ValueRequired, llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated);

static llvm::cl::list<std::string, std::vector<std::string>> XllcOpt("Xllc",
    llvm::cl::desc("Arguments to forward to LLVM's 'llc' option processing"),
    llvm::cl::value_desc("A valid LLVM's 'llc' option"),
    llvm::cl::location(Xllc), llvm::cl::cat(OnnxMlirOptions), llvm::cl::Hidden,
    llvm::cl::ValueRequired, llvm::cl::ZeroOrMore, llvm::cl::CommaSeparated);

static llvm::cl::opt<std::string, true> mllvmOpt("mllvm",
    llvm::cl::desc(
        "Arguments to forward to LLVM's 'opt' and 'llc' option processing"),
    llvm::cl::value_desc("A valid LLVM's 'opt' and 'llc' option"),
    llvm::cl::location(mllvm), llvm::cl::cat(OnnxMlirOptions), llvm::cl::Hidden,
    llvm::cl::ValueRequired);

static llvm::cl::opt<std::string, true> instrumentOpsOpt("instrument-ops",
    llvm::cl::desc("Specify operations operations to be instrumented:\n"
                   "\"NONE\" or \"\" for no instrument,\n"
                   "\"ops1,ops2, ...\" for the multiple ops.\n"
                   "e.g. \"onnx.Conv,onnx.Add\" for Conv and Add ops.\n"
                   "Asterisk is also available.\n"
                   "e.g. \"onnx.*\" for all onnx operations.\n"),
    llvm::cl::location(instrumentOps), llvm::cl::init(""),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::bits<InstrumentActions, unsigned> instrumentControlBitsOpt(
    llvm::cl::desc("Specify what instrumentation actions at runtime:"),
    llvm::cl::location(instrumentControlBits),
    llvm::cl::values(
        clEnumVal(InstrumentBeforeOp, "insert instrument before op,"),
        clEnumVal(InstrumentAfterOp, "insert instrument after op,"),
        clEnumVal(
            InstrumentReportTime, "instrument runtime reports time usage,"),
        clEnumVal(InstrumentReportMemory,
            "instrument runtime reports memory usage.")),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> instrumentONNXSignatureOpt(
    "instrument-onnx-signature",
    llvm::cl::desc("Instrument ONNX ops to print the type of their inputs"),
    llvm::cl::location(instrumentONNXSignature), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<std::string, true> ONNXOpStatsOpt("onnx-op-stats",
    llvm::cl::desc(
        "Report the occurrence frequency of ONNX ops in JSON or TXT format:\n"
        "\"TXT\" for report as text,\n"
        "\"JSON\" for report as JSON.\n"
        "Requires targets like --EmitMLIR, --EmitLLVMIR, or binary-generating "
        "commands."),
    llvm::cl::location(ONNXOpStats), llvm::cl::init(""),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<int, true> onnxOpTransformThresholdOpt(
    "onnx-op-transform-threshold",
    llvm::cl::desc(
        "Max iteration for dynamic op transform passes (default=3).\n"
        "If set to 0, onnxOpTransformPass will be disabled, and\n"
        "static iteration will be used"),
    llvm::cl::location(onnxOpTransformThreshold), llvm::cl::init(3),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> onnxOpTransformReportOpt(
    "onnx-op-transform-report",
    llvm::cl::desc(
        "Report diagnostic info for ONNX op transform/optimization passes."),
    llvm::cl::location(onnxOpTransformReport), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> enableParallelOpt("parallel",
    llvm::cl::desc("Enable parallelization (default=false)\n"
                   "Set to 'true' if you want to enable parallelization."),
    llvm::cl::location(enableParallel), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> disableSimdOptionOpt("disable-simd",
    llvm::cl::desc("Disable SIMD optimizations (default=false). Set to `true` "
                   "to disable SIMD at O3."),
    llvm::cl::location(disableSimdOption), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> enableSimdDataLayoutOpt("simd-data-layout",
    llvm::cl::desc("Enable SIMD optimization for convolution (default=false)\n"
                   "Set to 'true' if you want to enable SIMD optimizations."),
    llvm::cl::location(enableSimdDataLayout), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<std::string, true> opsForCallOpt("ops-for-call",
    llvm::cl::desc("Specify which ops are lowered to knrl.call instead of"
                   "krnl loops. op name are used to check against this option."
                   "Names of opa are separated with space."
                   "Example: ops-for-call=Conv MatMul"
                   "The regexp match will be used to check against op name"),
    llvm::cl::location(opsForCall), llvm::cl::init(""),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> verifyInputTensorsOpt("verifyInputTensors",
    llvm::cl::desc(
        "Verify input tensors whenever the entry point function is called.\n"
        "Data type and shape are verified. Enable this may introduce overhead "
        "at runtime."),
    llvm::cl::location(verifyInputTensors), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> allowSortingOpt("allowSorting",
    llvm::cl::desc("Perform topological sort on onnx graph"),
    llvm::cl::location(allowSorting), llvm::cl::init(true),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<std::string, true> reportHeapBeforeOpt(
    "report-heap-before",
    llvm::cl::desc("Comma separated list of names of passes.\n"
                   "Before each heap statistics are dumped to "
                   "<output-files-base-path>.heap.log"),
    llvm::cl::location(reportHeapBefore), llvm::cl::init(""),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<std::string, true> reportHeapAfterOpt("report-heap-after",
    llvm::cl::desc("Comma separated list of names of passes.\n"
                   "After each heap statistics are dumped to "
                   "<output-files-base-path>.heap.log"),
    llvm::cl::location(reportHeapAfter), llvm::cl::init(""),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<std::string, true> modelTagOpt("tag",
    llvm::cl::desc(
        "Set a tag that will be used to postfix symbols in the generated "
        "LLVMIR to make the symbols unique across multiple generated models. "
        "By default, use the filename (without extension) of the input onnx "
        "model or the value passed to `-o`. The tag will be appended to "
        "global variable and function names. For backward compatibility, each "
        "function has two versions with the same signature and doing the same "
        "computation. For example, we will have two entry points: "
        "`run_main_graph` and `run_main_graph_tag`, where `run_main_graph` "
        "is just a wrapper of `run_main_graph_tag`. Users can call one of "
        "the entry points and expect the same result. Passing `NONE` to "
        "`--tag` will disable tag completely, meaning no tag is appended to "
        "the symbols."),
    llvm::cl::value_desc("a string that matches regex ([0-9a-z_.-]+)"),
    llvm::cl::location(modelTag), llvm::cl::init(""),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> enableConvOptPassOpt("enable-conv-opt-pass",
    llvm::cl::desc("Enable the ConvOptPass. Default is true."),
    llvm::cl::location(enableConvOptPass), llvm::cl::init(true),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool, true> disableConstantPropOpt("disable-constant-prop",
    llvm::cl::desc("Disable Constant Propagation (default is false)\n"
                   "Set to 'true' to disable Constant Propagation."),
    llvm::cl::location(disableConstantProp), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirCommonOptions));

static llvm::cl::list<std::string, std::vector<std::string>> extraLibPathsOpt(
    "L",
    llvm::cl::desc("Specify extra directories for libraries when compiling"
                   "an onnx model. Will be add used as -L in the linkage step."
                   "Each directory can be specified with one extra-lib-dirs"),
    llvm::cl::location(extraLibPaths), llvm::cl::Prefix,
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::list<std::string, std::vector<std::string>> extraLibsOpt("l",
    llvm::cl::desc("Specify extra libraries when compiling an onnx model."
                   "Will be add used as -l in the linkage step."
                   "Each lib can be specified with one extra-libs"),
    llvm::cl::location(extraLibs), llvm::cl::Prefix,
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<ProfileIRs, true> profileIROpt("profile-ir",
    llvm::cl::desc("Profile operations in an IR"),
    llvm::cl::location(profileIR),
    llvm::cl::values(clEnumVal(None, "No profiling. Default value."),
        clEnumVal(
            Onnx, "Profile operations in ONNXIR generated by --EmitONNXIR.")
            APPLY_TO_ACCELERATORS(ACCEL_PROFILEIR_CL_ENUM)),
    llvm::cl::init(ProfileIRs::None), llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<OptReport, true> optReportOpt("opt-report",
    llvm::cl::desc("Provide information on a specific compiler optimization."),
    llvm::cl::location(optReport),
    llvm::cl::values(clEnumVal(NoReport, "No report. Default value."),
        clEnumVal(Parallel,
            "Provide report on how OMP Parallel is applied to ONNX ops."),
        clEnumVal(Simd, "Provide report on how SIMD is applied to ONNX ops.")),
    llvm::cl::init(OptReport::NoReport), llvm::cl::cat(OnnxMlirOptions));

// Options for onnx-mlir-opt only
static llvm::cl::opt<bool, true> split_input_file_opt("split-input-file",
    llvm::cl::desc("Split the input file into pieces and process each "
                   "chunk independently"),
    llvm::cl::location(split_input_file), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptOptions));

static llvm::cl::opt<bool, true> verify_diagnostics_opt("verify-diagnostics",
    llvm::cl::desc("Check that emitted diagnostics match "
                   "expected-* lines on the corresponding line"),
    llvm::cl::location(verify_diagnostics), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptOptions));

static llvm::cl::opt<bool, true> verify_passes_opt("verify-each",
    llvm::cl::desc("Run the verifier after each transformation pass"),
    llvm::cl::location(verify_passes), llvm::cl::init(true),
    llvm::cl::cat(OnnxMlirOptOptions));

static llvm::cl::opt<bool, true> allowUnregisteredDialectsOpt(
    "allow-unregistered-dialect",
    llvm::cl::desc("Allow operation with no registered dialects"),
    llvm::cl::location(allowUnregisteredDialects), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptOptions));

// Removed once the new LLVM bufferization works without performance regression.
static llvm::cl::opt<bool, true> useOldBufferizationOpt("use-old-bufferization",
    llvm::cl::desc(
        "Enable the old LLVM bufferization mechanism (default=true)\n"
        "This option should be removed once the new LLVM bufferization works "
        "well in onnx-mlir"),
    llvm::cl::location(useOldBufferization), llvm::cl::init(true),
    llvm::cl::cat(OnnxMlirOptions));

// Configuration states associated with certain options.
// For example, when maccel is specified, NNPA can register
// dependent libdnn.
// This is just a simple string to vector map currently.
// If it gets more complicated in the future, it can be
// replaced by a class of its own.
std::map<std::string, std::vector<std::string>> CompilerConfigMap;
std::map<std::string, std::vector<size_t>> CompilerConfigStack;

// Must match ModelSize enum
const std::string modelSizeStr[] = {"small", "medium", "large", "huge"};

std::string customEnvFlags;

// =============================================================================
// Methods for setting and getting compiler variables.

// The customEnvFlags must be scanned before the normal options.
bool parseCustomEnvFlagsCommandLineOption(
    int argc, const char *const *argv, llvm::raw_ostream *errs) {
  // Use the default ONNX MLIR Environment variable, unless specified otherwise
  // by an argument, see below.
  std::string envVar = OnnxMlirEnvOptionName;
  // Customized version? -customEnvFlags=val and save its value.
  for (int i = argc - 1; i > 1; --i) {
    std::string arg(argv[i]);
    if (arg.find("--customEnvFlags") == 0) {
      envVar = arg.substr(sizeof("--customEnvFlags"));
      break;
    }
    if (arg.find("-customEnvFlags") == 0) {
      envVar = arg.substr(sizeof("-customEnvFlags"));
      break;
    }
  }
  // Check that the env var does not recursively hold another -customEnvFlags.
  const char *envValCstr;
  if ((envValCstr = std::getenv(envVar.c_str()))) {
    std::string envVal(envValCstr);
    if (envVal.find("-customEnvFlags") != std::string::npos) {
      if (errs)
        *errs << "Warning: recursive use of --customEnvFlags in "
                 "environment flag not permited\n";
      return false;
    }
  }
  // The envVar is verified, use it.
  setCustomEnvVar(envVar);
  return true;
}

// Support for customEnvFlags.
void setCustomEnvVar(const std::string &envVarName) {
  assert(envVarName != "" && "Expecting valid target envVarName description");
  LLVM_DEBUG(
      llvm::dbgs() << DEBUG_TYPE << "Set envVarName\"" << envVarName << "\"\n");
  customEnvFlags = envVarName;
}

void clearCustomEnvVar() { customEnvFlags.clear(); }

std::string getCustomEnvVarOption() {
  return (customEnvFlags != "") ? "--customEnvFlags=" + customEnvFlags : "";
}

// Support for Triple.
void setTargetTriple(const std::string &triple) {
  assert(triple != "" && "Expecting valid target triple description");
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set triple\"" << triple << "\"\n");
  mtriple = triple;
}

void clearTargetTriple() { mtriple.clear(); }

std::string getTargetTripleOption() {
  std::string targetOptions = "";
  // Command cannot tolerate extra spaces. Add only when needed.
  if (mtriple != "")
    targetOptions = "--mtriple=" + mtriple;
  else if (kDefaultTriple != "")
    targetOptions = "--mtriple=" + kDefaultTriple;
  return targetOptions;
}

// Support for Arch.
void setTargetArch(const std::string &arch) {
  assert(arch != "" && "Expecting valid target arch description");
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set arch\"" << arch << "\"\n");
  march = arch;
}

void clearTargetArch() { march.clear(); }

std::string getTargetArchOption() {
  return (march != "") ? "--march=" + march : "";
}

// Support for CPU.
void setTargetCPU(const std::string &cpu) {
  assert(cpu != "" && "Expecting valid target cpu description");
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set CPU\"" << cpu << "\"\n");
  mcpu = cpu;
}

void clearTargetCPU() { mcpu.clear(); }

std::string getTargetCPUOption() {
  return (mcpu != "") ? "--mcpu=" + mcpu : "";
}

// Support for Accel.
static bool getAccelKindFromString(
    accel::Accelerator::Kind &kind, const std::string &str) {
  // Test each existing accelerator, returning its Kind when found.
  APPLY_TO_ACCELERATORS(ACCEL_CL_ENUM_FROM_STRING, kind, str);
  // No specific accelerator found, check if we have Kind::NONE
  kind = accel::Accelerator::Kind::NONE;
  return str.compare(std::string("NONE")) == 0;
}

// Return 0 on success, nonzero on error.
int setTargetAccel(const std::string &str) {
  assert(str != "" && "Expecting valid accelerator description");
  accel::Accelerator::Kind accelKind;
  if (getAccelKindFromString(accelKind, str)) {
    setTargetAccel(accelKind);
    return 0;
  }
  return 1;
}

void setTargetAccel(const accel::Accelerator::Kind accel) {
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set accel\"" << accel << "\"\n";);
  // Add accel to maccel.
  maccel.push_back(accel);
}

void clearTargetAccel() {
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Clearing accel\n");
  maccel.clear();
}

std::string getTargetAccel() {
  std::stringstream ss;
  int accelCount = 0;
  for (accel::Accelerator::Kind accel : maccel) {
    if (accelCount++)
      ss << " ";
    ss << "--maccel=" << accel;
  }
  if (!accelCount)
    ss << "--maccel=NONE";
  return ss.str();
}

// Support for Optimization level.
void setOptLevel(const OptLevel level) {
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set opt level " << level << "\n");
  OptimizationLevel = level;
}

void clearOptLevel() { OptimizationLevel = OptLevel::O0; }

std::string getOptimizationLevelOption() {
  switch (OptimizationLevel) {
  case OptLevel::O0:
    return "-O0";
  case OptLevel::O1:
    return "-O1";
  case OptLevel::O2:
    return "-O2";
  case OptLevel::O3:
    return "-O3";
  }
  llvm_unreachable("Unexpected optimization level");
  return "";
}

// Support for Xopt.
void setXoptOption(const std::vector<std::string> &flags) {
  for (const std::string &flag : flags)
    Xopt.push_back(flag);
}

void clearXoptOption() { Xopt.clear(); }

std::vector<std::string> getXoptOption() {
  if (Xopt.empty())
    return std::vector<std::string>();

  std::vector<std::string> flags;
  for (std::string flag : Xopt)
    flags.push_back(flag);

  return flags;
}

// Support for Xllc.
void setXllcOption(const std::vector<std::string> &flags) {
  for (const std::string &flag : flags)
    Xllc.push_back(flag);
}

void clearXllcOption() { Xllc.clear(); }

std::vector<std::string> getXllcOption() {
  if (Xllc.empty())
    return std::vector<std::string>();

  std::vector<std::string> flags;
  for (std::string flag : Xllc)
    flags.push_back(flag);

  return flags;
}

// Support for LLVM.
void setLLVMOption(const std::string &flag) { mllvm = flag; }
void clearLLVMOption() { mllvm.clear(); }
std::string getLLVMOption() { return (mllvm != "") ? mllvm : std::string(); }

// Support for model tag
void setModelTag(const std::string &str) { modelTag = str; }
void clearModelTag() { modelTag = ""; }
std::string getModelTag() { return modelTag; }

// Support for Verbose Option
void setVerboseOption() { VerboseOutput = true; }
void clearVerboseOption() { VerboseOutput = false; }
std::string getVerboseOption() {
  return VerboseOutput ? std::string("-v") : std::string();
}

// =============================================================================
// Methods for OMCompilerOptions

int setCompilerOption(const OptionKind kind, const std::string &val) {
  switch (kind) {
  case OptionKind::TargetTriple:
    setTargetTriple(val);
    break;
  case OptionKind::TargetArch:
    setTargetArch(val);
    break;
  case OptionKind::TargetCPU:
    setTargetCPU(val);
    break;
  case OptionKind::TargetAccel:
    if (setTargetAccel(val) != 0)
      return InvalidCompilerOption;
    break;
  case OptionKind::CompilerOptLevel: {
    int level = atoi(val.c_str());
    if (level < 0 || level > 3)
      return InvalidCompilerOption;
    setOptLevel((OptLevel)level);
  } break;
  case OptionKind::OPTFlag:
    setXoptOption({val});
    break;
  case OptionKind::LLCFlag:
    setXllcOption({val});
    break;
  case OptionKind::LLVMFlag:
    setLLVMOption(val);
    break;
  case OptionKind::ModelTag:
    setModelTag(val);
    break;
  case OptionKind::Verbose:
    setVerboseOption();
    break;
    // Ignore options that were added but are unknown.
  }
  return CompilerSuccess;
}

void clearCompilerOption(const OptionKind kind) {
  switch (kind) {
  case OptionKind::TargetTriple:
    clearTargetTriple();
    break;
  case OptionKind::TargetArch:
    clearTargetArch();
    break;
  case OptionKind::TargetCPU:
    clearTargetCPU();
    break;
  case OptionKind::TargetAccel:
    clearTargetAccel();
    break;
  case OptionKind::CompilerOptLevel:
    clearOptLevel();
    break;
  case OptionKind::OPTFlag:
    clearXoptOption();
    break;
  case OptionKind::LLCFlag:
    clearXllcOption();
    break;
  case OptionKind::LLVMFlag:
    clearLLVMOption();
    break;
  case OptionKind::ModelTag:
    clearModelTag();
    break;
  case OptionKind::Verbose:
    clearVerboseOption();
    break;
    // Ignore options that were added but are unknown.
  }
}

std::string getCompilerOption(const OptionKind kind) {
  switch (kind) {
  case OptionKind::TargetTriple:
    return getTargetTripleOption();
  case OptionKind::TargetArch:
    return getTargetArchOption();
  case OptionKind::TargetCPU:
    return getTargetCPUOption();
  case OptionKind::TargetAccel:
    return getTargetAccel();
  case OptionKind::CompilerOptLevel:
    return getOptimizationLevelOption();
  case OptionKind::OPTFlag:
  case OptionKind::LLCFlag: {
    std::vector<std::string> flags =
        (kind == OptionKind::OPTFlag) ? getXoptOption() : getXllcOption();
    std::stringstream ss;
    for (int i = 0, n = flags.size(); i < n; ++i) {
      ss << flags.at(i);
      if (i != n - 1)
        ss << ' ';
    }
    return ss.str();
  }
  case OptionKind::LLVMFlag:
    return getLLVMOption();
  case OptionKind::ModelTag:
    return getModelTag();
  case OptionKind::Verbose:
    return getVerboseOption();
  }
  return std::string();
}

int setCompilerOptions(const CompilerOptionList &list) {
  for (const auto &pair : list) {
    int rc = setCompilerOption(pair.first, pair.second);
    if (rc != CompilerSuccess)
      return rc;
  }
  return CompilerSuccess;
}

// Get the string vector associated with the specified key
std::vector<std::string> getCompilerConfig(std::string k) {
  return CompilerConfigMap[k];
}

// Add strings in a vector to the string vector associated
// with the specified key
void addCompilerConfig(std::string k, std::vector<std::string> v) {
  std::vector<std::string> u = CompilerConfigMap[k];

  u.insert(u.end(), v.begin(), v.end());
  CompilerConfigMap[k] = u;
}

// Delete strings in a vector from the string vector associated
// with the specified key
void delCompilerConfig(std::string k, std::vector<std::string> v) {
  std::vector<std::string> u = CompilerConfigMap[k];

  u.erase(remove_if(begin(u), end(u),
              [&](auto x) { return find(begin(v), end(v), x) != end(v); }),
      end(u));
  CompilerConfigMap[k] = u;
}

std::optional<std::string> getEnvVar(std::string name) {
  if (const char *envVar = std::getenv(name.c_str()))
    return std::string(envVar);
  return std::nullopt;
}

// Find the path to the onnx-mlir executable
std::string getExecPath() {
  // argv0 is only used as a fallback for rare environments
  // where /proc isn't mounted and mainExecAddr is only needed for
  // unknown unix-like platforms
  auto execPath = llvm::sys::fs::getMainExecutable(nullptr, nullptr);
  if (execPath.empty()) {
    llvm::errs()
        << "Warning: Could not find path to current executable, falling "
           "back to default install path: "
        << kExecPath << "\n";
    return kExecPath;
  }
  return execPath;
}

// Directory contains all the libraries, jars, etc. that are necessary for
// running onnx-mlir. It's resolved in the following order:
//
//   - if ONNX_MLIR_LIBRARY_PATH is set, use it, otherwise
//   - get path from where onnx-mlir is run, if it's of the form
//     /foo/bar/bin/onnx-mlir,
//     the runtime directory is /foo/bar/lib (note that when onnx-mlir is
//     installed system wide, which is typically /usr/local/bin, this will
//     correctly resolve to /usr/local/lib), but some systems still have
//     lib64 so we check that first. If neither exists, then
//   - use CMAKE_INSTALL_PREFIX/lib, which is typically /usr/local/lib
//
// We now explicitly set CMAKE_INSTALL_LIBDIR to lib so we don't have
// to deal with lib64 anymore.
std::string getLibraryPath() {
  const auto &envDir = getEnvVar("ONNX_MLIR_LIBRARY_PATH");
  if (envDir && llvm::sys::fs::exists(envDir.value()))
    return envDir.value();

  std::string execDir = llvm::sys::path::parent_path(getExecPath()).str();
  if (llvm::sys::path::stem(execDir).str().compare("bin") == 0) {
    std::string p = execDir.substr(0, execDir.size() - 3);
    if (llvm::sys::fs::exists(p + "lib"))
      return p + "lib";
  }

  llvm::SmallString<8> instDir(kInstPath);
  llvm::sys::path::append(instDir, "lib");
  return llvm::StringRef(instDir).str();
}

// onnx-mlir currently requires llvm tools llc and opt and they are assumed
// to be under llvm-project/build/bin. This doesn't work with the case where
// llvm-project has been installed system wide (typically under /usr/local/...)
// and its source has been removed.
//
// To account for this scenario, we first search for the tools in the same
// directory where onnx-mlir is run. If they are found, it means both onnx-mlir
// and llvm-project have been installed system wide under the same directory,
// so we get them from that directory (typically /usr/local/bin). Otherwise,
// at least one of onnx-mlir and llvm-project has not been installed system
// wide. In this case, getToolPath returns the fallback directory where llvm
// is built which is typically llvm-project/build/bin.
//
// Note that this will not work if both onnx-mlir and llvm-project have been
// installed system wide but to different places and their sources have been
// removed. So we force CMAKE_INSTALL_PREFIX to be the same as that of
// llvm-project.
//
// If the flag is true, getToolPath will simply return the path detected by
// cmake at compile time. This is used for system wide tools such as cc, ld, ar,
// etc. Note that this means the path is valid only on the system where
// onnx-mlir is built. If onnx-mlir is subsequently run on a system that does
// not have these tools installed in the "standard" places, it will fail.
//
// Setting flag = true is also used to simply look up non-path config such
// as lrodataScript.
std::string getToolPath(
    const std::string &tool, bool flag /*false by default*/) {

  if (!flag) {
    std::string execDir = llvm::sys::path::parent_path(getExecPath()).str();
    llvm::SmallString<8> toolPath(execDir);
    llvm::sys::path::append(toolPath, tool);
    std::string p = llvm::StringRef(toolPath).str();
    if (llvm::sys::fs::can_execute(p))
      return p;
  }

  return toolPathMap.at(tool);
}

// This function is called before llvm::cl::ParseCommandLineOptions
// to remove unrelated options in addition to hiding them. Since
// hiding only means that unrelated options will not be printed by
// -h|--help but they can still be used and silently ignored. But
// the correct behavior is that using a unrelated option should
// result in a unknown option error.
void removeUnrelatedOptions(
    const std::vector<llvm::cl::OptionCategory *> Categories) {
  // Do not remove LLVM "internal" options such as --debug
  // that do not have a category (and therefore placed
  // under the general category). So we add the general
  // category to the list of not-really-hidden options.
  std::vector<llvm::cl::OptionCategory *> optCategories(Categories);
  optCategories.push_back(&llvm::cl::getGeneralCategory());
  llvm::cl::HideUnrelatedOptions(optCategories);

  llvm::StringMap<llvm::cl::Option *> &optMap =
      llvm::cl::getRegisteredOptions();
  for (auto n = optMap.begin(); n != optMap.end(); n++) {
    llvm::cl::Option *opt = n->getValue();
    if (opt->getOptionHiddenFlag() == llvm::cl::ReallyHidden)
      opt->removeArgument();
  }
}

// This function can be called after llvm::cl::ParseCommandLineOptions
// to create whatever options related compiler configuration states
// based on the parsed options. It can also check for option consistency.
//
// The reason we don't put llvm::cl::ParseCommandLineOptions and
// initCompilerConfig in a single function is that according to llvm doc
// llvm::cl::ParseCommandLineOptions should be called from main.
void initCompilerConfig() {
  // Test option requirements.
  if (!ONNXOpStats.empty() && emissionTarget <= EmitONNXIR)
    llvm::errs()
        << "Warning: --onnx-op-stats requires targets like --EmitMLIR, "
           "--EmitLLVMIR, or binary-generating emit commands.\n";

  // Library setup for EmitLib and EmitJNI targets
  if (emissionTarget == EmitLib || emissionTarget == EmitJNI) {
    // Add mandatory libs
    addCompilerConfig(CCM_SHARED_LIB_DEPS,
        emissionTarget == EmitLib
            ? std::vector<std::string>{"cruntime"}
            : std::vector<std::string>{"jniruntime", "cruntime"});
    addCompilerConfig(CCM_SHARED_LIB_PATH_DEPS, {getLibraryPath()});

    // Add OpenMP LLVM library if parallel is enabled.
    if (enableParallel)
      addCompilerConfig(CCM_SHARED_LIB_DEPS, {"ompruntime"});

    // Add user specified libs and their path
    // Multiple lib or directory can be specified with multiple options.
    // For example, -lextra1, -lextra2, -Lpath1, -Lpath2
    addCompilerConfig(CCM_SHARED_LIB_DEPS, extraLibs);
    addCompilerConfig(CCM_SHARED_LIB_PATH_DEPS, extraLibPaths);
  }
}

} // namespace onnx_mlir
