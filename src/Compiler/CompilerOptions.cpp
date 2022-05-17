/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ CompilerOptions.cpp -------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding options.
//
//===----------------------------------------------------------------------===//

#include "llvm/Support/Debug.h"

#include "ExternalUtil.hpp"
#include "onnx-mlir/Compiler/OMCompilerTypes.h"
#include "src/Compiler/CompilerOptions.hpp"

#define DEBUG_TYPE "compiler_options"

namespace onnx_mlir {
// Options for onnx-mlir only.
llvm::cl::OptionCategory OnnxMlirOptions(
    "ONNX-MLIR Options", "These are frontend options.");

// Common options shared between onnx-mlir and onnx-mlir-opt.
llvm::cl::OptionCategory OnnxMlirCommonOptions("ONNX-MLIR Common Options", "");

// the option is used in this file, so defined here
llvm::cl::opt<bool> invokeOnnxVersionConverter("invokeOnnxVersionConverter",
    llvm::cl::desc(
        "call onnx version converter to convert ONNX model to current version"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> preserveLocations("preserveLocations",
    llvm::cl::desc("emit location data:"), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> printIR("printIR",
    llvm::cl::desc("print the IR to stdout:"), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> preserveBitcode("preserveBitcode",
    llvm::cl::desc(
        "dont delete the bitcode files (optimized and unoptimized):"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> preserveLLVMIR("preserveLLVMIR",
    llvm::cl::desc("dont delete the LLVMIR files:"), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> preserveMLIR("preserveMLIR",
    llvm::cl::desc("dont delete the MLIR files (input and llvm):"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> useOnnxModelTypes("useOnnxModelTypes",
    llvm::cl::desc("use types and shapes from ONNX model"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<int> repeatOnnxTransform("repeatOnnxTransform",
    llvm::cl::desc(
        "invoke extra onnx transform pass(shape inference, constant and etc.)"),
    llvm::cl::init(0), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<std::string> shapeInformation("shapeInformation",
    llvm::cl::desc(
        "Custom shapes for the inputs of the ONNX model, e.g. setting static "
        "shapes for dynamic inputs.\n"
        "\"value\" is in the format of "
        "\"INPUT_ID1:D1xD2x...xDn,INPUT_ID2:D1xD2x...xDn, ...\",\n"
        "where \"INPUT_ID1, INPUT_ID2, ...\" are input indices starting from "
        "0, and\n"
        "\"D1, D2, ...\" are dimension sizes (positive integers of -1 for "
        "unknown dimensions)"),
    llvm::cl::value_desc("value"), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<std::string> mtriple("mtriple",
    llvm::cl::desc("Override target triple for module"),
    llvm::cl::value_desc("LLVM target triple"), llvm::cl::cat(OnnxMlirOptions),
    llvm::cl::ValueRequired);

llvm::cl::opt<std::string> mcpu("mcpu", llvm::cl::desc("Target cpu"),
    llvm::cl::value_desc("Target a specific CPU type"),
    llvm::cl::cat(OnnxMlirOptions), llvm::cl::ValueRequired);

llvm::cl::opt<std::string> march("march",
    llvm::cl::desc("Target architecture to generate code for"),
    llvm::cl::value_desc("Target a specific architecture type"),
    llvm::cl::cat(OnnxMlirOptions), llvm::cl::ValueRequired);

llvm::cl::list<accel::Accelerator::Kind> maccel("maccel",
    llvm::cl::desc("Specify an accelerator to generate code for"),
    // clang-format off
    llvm::cl::values(
      APPLY_TO_ACCELERATORS(CREATE_ACCEL_CL_ENUM)
      clEnumValN(accel::Accelerator::Kind::NONE, "NONE", "No accelerator")
    ),
    // clang-format on
    llvm::cl::cat(OnnxMlirCommonOptions), llvm::cl::ValueRequired);

llvm::cl::opt<bool> VerboseOutput("v", llvm::cl::desc("Use verbose output"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<std::string> Xopt("Xopt",
    llvm::cl::desc("Arguments to forward to LLVM's 'opt' option processing"),
    llvm::cl::value_desc("A valid LLVM's 'opt' option"),
    llvm::cl::cat(OnnxMlirOptions), llvm::cl::Hidden, llvm::cl::ValueRequired);

llvm::cl::opt<std::string> Xllc("Xllc",
    llvm::cl::desc("Arguments to forward to LLVM's 'llc' option processing"),
    llvm::cl::value_desc("A valid LLVM's 'llc' option"),
    llvm::cl::cat(OnnxMlirOptions), llvm::cl::Hidden, llvm::cl::ValueRequired);

llvm::cl::opt<std::string> mllvm("mllvm",
    llvm::cl::desc(
        "Arguments to forward to LLVM's 'opt' and 'llc' option processing"),
    llvm::cl::value_desc("A valid LLVM's 'opt' and 'llc' option"),
    llvm::cl::cat(OnnxMlirOptions), llvm::cl::Hidden, llvm::cl::ValueRequired);

llvm::cl::opt<OptLevel> OptimizationLevel(
    llvm::cl::desc("Optimization levels:"),
    llvm::cl::values(clEnumVal(O0, "Optimization level 0 (default)."),
        clEnumVal(O1, "Optimization level 1."),
        clEnumVal(O2, "Optimization level 2."),
        clEnumVal(O3, "Optimization level 3.")),
    llvm::cl::init(O0), llvm::cl::cat(OnnxMlirCommonOptions));

llvm::cl::opt<std::string> instrumentONNXOps("instrument-onnx-ops",
    llvm::cl::desc("Specify onnx ops to be instrumented\n"
                   "\"NONE\" or \"\" for no instrument\n"
                   "\"ALL\" for all ops. \n"
                   "\"op1 op2 ...\" for the specified ops."),
    llvm::cl::init(""), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::bits<InstrumentActions> instrumentControlBits(
    llvm::cl::desc("Specify what instrumentation actions at runtime:"),
    llvm::cl::values(
        clEnumVal(InstrumentBeforeOp, "insert instrument before op"),
        clEnumVal(InstrumentAfterOp, "insert instrument after op"),
        clEnumVal(
            InstrumentReportTime, "instrument runtime reports time usage"),
        clEnumVal(
            InstrumentReportMemory, "instrument runtime reports memory usage")),
    llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> enableMemoryBundling("enable-memory-bundling",
    llvm::cl::desc(
        "Enable memory bundling related optimizations (default=false)\n"
        "Set to 'false' if you experience significant compile time."),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<int> onnxOpTransformThreshold("onnx-op-transform-threshold",
    llvm::cl::desc(
        "Max iteration for dynamic op transform passes (default=3).\n"
        "If set to 0, onnxOpTransformPass will be disabled, and\n"
        "static iteration will be used"),
    llvm::cl::init(3), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> onnxOpTransformReport("onnx-op-transform-report",
    llvm::cl::desc("Report diagnostic info for op transform passes."),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

// Configuration states associated with certain options.
// For example, when maccel is specified, NNPA can register
// dependent libdnn.
// This is just a simple string to vector map currently.
// If it gets more complicated in the future, it can be
// replaced by a class of its own.
std::map<std::string, std::vector<std::string>> CompilerConfigMap;

// =============================================================================
// Methods for setting and getting compiler variables.

// Support for Triple.
void setTargetTriple(const std::string &triple) {
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set triple\"" << triple << "\"\n");
  mtriple = triple;
}

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
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set arch\"" << arch << "\"\n");
  march = arch;
}

std::string getTargetArchOption() {
  return (march != "") ? "--march=" + march : "";
}

// Support for CPU.
void setTargetCPU(const std::string &cpu) {
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set CPU\"" << cpu << "\"\n");
  mcpu = cpu;
}

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
  // Empty string means reset maccel.
  if (str.compare(std::string("RESET")) == 0) {
    LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set accel to empty\n");
    maccel.clear();
    return 0;
  }
  // Nonempty string means adding accelerator to current accelerator list.
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
void setXoptOption(const std::string &flag) { Xopt = flag; }

std::string getXoptOption() { return (Xopt != "") ? Xopt : std::string(); }

// Support for Xllc.
void setXllcOption(const std::string &flag) { Xllc = flag; }

std::string getXllcOption() { return (Xllc != "") ? Xllc : std::string(); }

// Support for LLVM.
void setLLVMOption(const std::string &flag) { mllvm = flag; }

std::string getLLVMOption() { return (mllvm != "") ? mllvm : std::string(); }

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
      return 1;
    break;
  case OptionKind::CompilerOptLevel: {
    int level = atoi(val.c_str());
    if (level < 0 || level > 3)
      return 1;
    setOptLevel((OptLevel)level);
  } break;
  case OptionKind::OPTFlag:
    setXoptOption(val);
    break;
  case OptionKind::LLCFlag:
    setXllcOption(val);
    break;
  case OptionKind::LLVMFlag:
    setLLVMOption(val);
    break;
    // Ignore options that were added but are unknown.
  }
  return 0;
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
    return getXoptOption();
  case OptionKind::LLCFlag:
    return getXllcOption();
  case OptionKind::LLVMFlag:
    return getLLVMOption();
  }
  return std::string();
}

int setCompilerOptions(const CompilerOptionList &list) {
  for (const auto &pair : list) {
    int rc = setCompilerOption(pair.first, pair.second);
    if (rc != 0)
      return rc;
  }
  return 0;
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

} // namespace onnx_mlir
