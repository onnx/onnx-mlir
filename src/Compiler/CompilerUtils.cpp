/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- CompilerUtils.cpp -------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/AffineToStandard/AffineToStandard.h"
#include "mlir/Conversion/ReconcileUnrealizedCasts/ReconcileUnrealizedCasts.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/DataLayout.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Target/TargetMachine.h"

#include "ExternalUtil.hpp"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Support/OMOptions.hpp"

#define DEBUG_TYPE "compiler_utils"

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

const string OnnxMlirEnvOptionName = "ONNX_MLIR_FLAGS";

llvm::cl::OptionCategory OnnxMlirOptions(
    "ONNX-MLIR Options", "These are frontend options.");

namespace {

static llvm::Optional<std::string> getEnvVar(std::string name) {
  if (const char *envVerbose = std::getenv(name.c_str()))
    return std::string(envVerbose);
  return llvm::None;
}

// This definition is here rather than in main.cpp because otherwise it's not
// found probably should be pulled out to a more common location
// TODO: Find a respectable home for the wain

// the option is used in this file, so defined here
static llvm::cl::opt<bool> invokeOnnxVersionConverter(
    "invokeOnnxVersionConverter",
    llvm::cl::desc(
        "call onnx version converter to convert ONNX model to current version"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool> preserveLocations("preserveLocations",
    llvm::cl::desc("emit location data:"), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool> printIR("printIR",
    llvm::cl::desc("print the IR to stdout:"), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool> preserveBitcode("preserveBitcode",
    llvm::cl::desc(
        "dont delete the bitcode files (optimized and unoptimized):"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool> preserveMLIR("preserveMLIR",
    llvm::cl::desc("dont delete the MLIR files (input and llvm):"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool> useOnnxModelTypes("useOnnxModelTypes",
    llvm::cl::desc("use types and shapes from ONNX model"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<int> repeatOnnxTransform("repeatOnnxTransform",
    llvm::cl::desc(
        "invoke extra onnx transform pass(shape inference, constant and etc.)"),
    llvm::cl::init(0), llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<string> shapeInformation("shapeInformation",
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

static llvm::cl::opt<std::string> mtriple("mtriple",
    llvm::cl::desc("Override target triple for module"),
    llvm::cl::value_desc("LLVM target triple>"), llvm::cl::cat(OnnxMlirOptions),
    llvm::cl::ValueRequired);

static llvm::cl::opt<std::string> mcpu("mcpu", llvm::cl::desc("Target cpu"),
    llvm::cl::value_desc("Target a specific CPU type"),
    llvm::cl::cat(OnnxMlirOptions), llvm::cl::ValueRequired);

static llvm::cl::opt<std::string> march("march",
    llvm::cl::desc("Target architecture to generate code for"),
    llvm::cl::value_desc("Target a specific architecture type"),
    llvm::cl::cat(OnnxMlirOptions), llvm::cl::ValueRequired);

static llvm::cl::opt<OptLevel> OptimizationLevel(
    llvm::cl::desc("Optimization levels:"),
    llvm::cl::values(clEnumVal(O0, "Optimization level 0 (default)."),
        clEnumVal(O1, "Optimization level 1."),
        clEnumVal(O2, "Optimization level 2."),
        clEnumVal(O3, "Optimization level 3.")),
    llvm::cl::init(O0), llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<bool> VerboseOutput("v",
    llvm::cl::desc("Use verbose output"), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

static llvm::cl::opt<std::string> Xopt("Xopt",
    llvm::cl::desc("Arguments to forward to LLVM's 'opt' option processing"),
    llvm::cl::value_desc("A valid LLVM's 'opt' option"),
    llvm::cl::cat(OnnxMlirOptions), llvm::cl::Hidden, llvm::cl::ValueRequired);

static llvm::cl::opt<std::string> Xllc("Xllc",
    llvm::cl::desc("Arguments to forward to LLVM's 'llc' option processing"),
    llvm::cl::value_desc("A valid LLVM's 'llc' option"),
    llvm::cl::cat(OnnxMlirOptions), llvm::cl::Hidden, llvm::cl::ValueRequired);

static llvm::cl::opt<std::string> mllvm("mllvm",
    llvm::cl::desc(
        "Arguments to forward to LLVM's 'opt' and 'llc' option processing"),
    llvm::cl::value_desc("A valid LLVM's 'opt' and 'llc' option"),
    llvm::cl::cat(OnnxMlirOptions), llvm::cl::Hidden, llvm::cl::ValueRequired);

// Make a function that forces preserving all files using the runtime arguments
// and/or the overridePreserveFiles enum.
enum class KeepFilesOfType { All, MLIR, Bitcode, Object, None };

// Value below override at compile time by effectively setting the requested
// flags.
static constexpr KeepFilesOfType overridePreserveFiles = KeepFilesOfType::None;

static bool keepFiles(KeepFilesOfType preserve) {
  // When wanting to preserve all files, do it regardles of isBitcode.
  if (overridePreserveFiles == KeepFilesOfType::All)
    return true;
  // When file is bitcode, check the runtime flag preserveBitcode.
  switch (preserve) {
  case KeepFilesOfType::Bitcode:
    return overridePreserveFiles == KeepFilesOfType::Bitcode || preserveBitcode;
  case KeepFilesOfType::MLIR:
    return overridePreserveFiles == KeepFilesOfType::MLIR || preserveMLIR;
  case KeepFilesOfType::Object:
    // Currently no option, enable using the overridePreserveFiles enum.
    return overridePreserveFiles == KeepFilesOfType::Object;
  default:
    // All, None should not be used in the parameter
    llvm_unreachable("illegal KeepFilesOfType enum value");
  }
  return false;
}

static std::string getExecPath() {
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

// Runtime directory contains all the libraries, jars, etc. that are
// necessary for running onnx-mlir. It's resolved in the following order:
//
//   - if ONNX_MLIR_RUNTIME_DIR is set, use it, otherwise
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
static std::string getRuntimeDir() {
  const auto &envDir = getEnvVar("ONNX_MLIR_RUNTIME_DIR");
  if (envDir && llvm::sys::fs::exists(envDir.getValue()))
    return envDir.getValue();

  string execDir = llvm::sys::path::parent_path(getExecPath()).str();
  if (llvm::sys::path::stem(execDir).str().compare("bin") == 0) {
    string p = execDir.substr(0, execDir.size() - 3);
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
// wide. In this case, getToolPath returns an empty string and we will fallback
// to llvm-project/build/bin.
//
// Note that this will not work if both onnx-mlir and llvm-project have been
// installed system wide but to different places and their sources have been
// removed. So we force CMAKE_INSTALL_PREFIX to be the same as that of
// llvm-project.
static std::string getToolPath(string tool) {
  string execDir = llvm::sys::path::parent_path(getExecPath()).str();
  llvm::SmallString<8> toolPath(execDir);
  llvm::sys::path::append(toolPath, tool);
  string p = llvm::StringRef(toolPath).str();
  if (llvm::sys::fs::can_execute(p))
    return p;
  else
    return string();
}

// Helper struct to make command construction and execution easy & readable.
struct Command {
  std::string _path;
  std::vector<std::string> _args;

  Command(std::string exePath)
      : _path(std::move(exePath)),
        _args({llvm::sys::path::filename(_path).str()}) {}

  // Append a single string argument.
  Command &appendStr(const std::string &arg) {
    if (arg.size() > 0)
      _args.emplace_back(arg);
    return *this;
  }

  // Append a single optional string argument.
  Command &appendStrOpt(const llvm::Optional<std::string> &arg) {
    if (arg.hasValue())
      _args.emplace_back(arg.getValue());
    return *this;
  }

  // Append a list of string arguments.
  Command &appendList(const std::vector<std::string> &args) {
    _args.insert(_args.end(), args.begin(), args.end());
    return *this;
  }

  // Reset arguments.
  Command &resetArgs() {
    auto exeFileName = _args.front();
    _args.clear();
    _args.emplace_back(exeFileName);
    return *this;
  }

  // Execute command in current work directory.
  //
  // If the optional wdir is specified, the command will be executed
  // in the specified work directory. Current work directory is
  // restored after the command is executed.
  void exec(std::string wdir = "") const {
    auto argsRef = std::vector<llvm::StringRef>(_args.begin(), _args.end());

    // If a work directory is specified, save the current work directory
    // and switch into it. Note that if wdir is empty, new_wdir will be
    // cur_wdir.
    SmallString<8> cur_wdir;
    SmallString<8> new_wdir(wdir);
    llvm::sys::fs::current_path(cur_wdir);
    llvm::sys::fs::make_absolute(cur_wdir, new_wdir);
    if (std::error_code ec = llvm::sys::fs::set_current_path(new_wdir)) {
      llvm::errs() << StringRef(new_wdir).str() << ": " << ec.message() << "\n";
      exit(ec.value());
    }

    if (VerboseOutput)
      llvm::errs() << "[" << StringRef(new_wdir).str() << "]" << _path << ": "
                   << llvm::join(argsRef, " ") << "\n";

    std::string errMsg;
    int rc = llvm::sys::ExecuteAndWait(_path, llvm::makeArrayRef(argsRef),
        /*Env=*/None, /*Redirects=*/None,
        /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

    if (rc != 0) {
      llvm::errs() << llvm::join(argsRef, " ") << "\n"
                   << "Error message: " << errMsg << "\n"
                   << "Program path: " << _path << "\n"
                   << "Command execution failed."
                   << "\n";
      exit(rc);
    }

    // Restore saved work directory.
    llvm::sys::fs::set_current_path(cur_wdir);
  }
};
} // namespace

// =============================================================================
// Methods for setting and getting compiler variables.

// Triple.
static void setTargetTriple(const std::string &triple) {
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set triple\"" << triple << "\"\n");
  mtriple = triple;
}

static std::string getTargetTripleOption() {
  string targetOptions = "";
  // Command cannot tolerate extra spaces. Add only when needed.
  if (mtriple != "")
    targetOptions = "--mtriple=" + mtriple;
  else if (kDefaultTriple != "")
    targetOptions = "--mtriple=" + kDefaultTriple;
  return targetOptions;
}

// Arch.
static void setTargetArch(const std::string &arch) {
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set arch\"" << arch << "\"\n");
  march = arch;
}

static std::string getTargetArchOption() {
  return (march != "") ? "--march=" + march : "";
}

// CPU.
static void setTargetCPU(const std::string &cpu) {
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set CPU\"" << cpu << "\"\n");
  mcpu = cpu;
}

static std::string getTargetCPUOption() {
  return (mcpu != "") ? "--mcpu=" + mcpu : "";
}

// Optimization level.
static void setOptLevel(const OptLevel level) {
  LLVM_DEBUG(llvm::dbgs() << DEBUG_TYPE << "Set opt level " << level << "\n");
  OptimizationLevel = level;
}

static std::string getOptimizationLevelOption() {
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

// Xopt.
static void setXoptOption(const std::string &flag) { Xopt = flag; }

static std::string getXoptOption() {
  return (Xopt != "") ? Xopt : std::string();
}

// Xllc.
static void setXllcOption(const std::string &flag) { Xllc = flag; }

static std::string getXllcOption() {
  return (Xllc != "") ? Xllc : std::string();
}

// LLVM.
static void setLLVMOption(const std::string &flag) { mllvm = flag; }

static std::string getLLVMOption() {
  return (mllvm != "") ? mllvm : std::string();
}

// =============================================================================
// Methods for OMCompilerOptions

int setCompilerOption(const OptionKind kind, const string &val) {
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

string getCompilerOption(const OptionKind kind) {
  switch (kind) {
  case OptionKind::TargetTriple:
    return getTargetTripleOption();
  case OptionKind::TargetArch:
    return getTargetArchOption();
  case OptionKind::TargetCPU:
    return getTargetCPUOption();
  case OptionKind::CompilerOptLevel:
    return getOptimizationLevelOption();
  case OptionKind::OPTFlag:
    return getXoptOption();
  case OptionKind::LLCFlag:
    return getXllcOption();
  case OptionKind::LLVMFlag:
    return getLLVMOption();
  }
  return string();
}

int setCompilerOptions(const CompilerOptionList &list) {
  for (const auto &pair : list) {
    int rc = setCompilerOption(pair.first, pair.second);
    if (rc != 0)
      return rc;
  }
  return 0;
}

// =============================================================================
// Methods for compiling and file processing.

void loadMLIR(string inputFilename, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module) {
  // Handle '.mlir' input to the ONNX-MLIR frontend.
  // The mlir format indicates that one or more of the supported
  // representations are used in the file.
  string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    exit(1);
  }
}

// Write LLVM optimized bitcode.
static void genLLVMBitcode(const mlir::OwningModuleRef &module,
    string optimizedBitcodePath, string outputBaseName) {
  error_code error;

  // Write bitcode to a file.
  string unoptimizedBitcodePath = outputBaseName + ".unoptimized.bc";
  llvm::FileRemover unoptimizedBitcodeRemover(
      unoptimizedBitcodePath, !keepFiles(KeepFilesOfType::Bitcode));

  // outputBaseName might contain a directory, which must exist.
  // Otherwise, a "No such file or directory" error will be returned.
  llvm::raw_fd_ostream moduleBitcodeStream(
      unoptimizedBitcodePath, error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << unoptimizedBitcodePath << ": " << error.message() << "\n";
    exit(error.value());
  }

  llvm::LLVMContext llvmContext;
  mlir::registerLLVMDialectTranslation(*(module.get().getContext()));
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate module to LLVMIR.\n";
    exit(1);
  }
  llvm::WriteBitcodeToFile(*llvmModule, moduleBitcodeStream);
  moduleBitcodeStream.flush();

  // Use the LLVM's 'opt' command to optimize the bitcode.
  string optPath = getToolPath("opt");
  Command optBitcode(/*exePath=*/!optPath.empty() ? optPath : kOptPath);
  optBitcode.appendStr(getOptimizationLevelOption())
      .appendStr(getTargetTripleOption())
      .appendStr(getTargetArchOption())
      .appendStr(getTargetCPUOption())
      .appendStr(getXoptOption())
      .appendStr(getLLVMOption())
      .appendList({"-o", optimizedBitcodePath})
      .appendStr(unoptimizedBitcodePath)
      .exec();
}

// Compile LLVM bitcode to object file.
static std::string genModelObject(string bitcodePath, string outputBaseName) {

#ifdef _WIN32
  string modelObjPath = outputBaseName + ".obj";
#else
  string modelObjPath = outputBaseName + ".o";
#endif

  string llcPath = getToolPath("llc");
  Command llvmToObj(/*exePath=*/!llcPath.empty() ? llcPath : kLlcPath);
  llvmToObj.appendStr(getOptimizationLevelOption())
      .appendStr(getTargetTripleOption())
      .appendStr(getTargetArchOption())
      .appendStr(getTargetCPUOption())
      .appendStr(getXllcOption())
      .appendStr(getLLVMOption())
      .appendStr("-filetype=obj")
      .appendStr("-relocation-model=pic")
      .appendList({"-o", modelObjPath})
      .appendStr(bitcodePath)
      .exec();
  return modelObjPath;
}

static void genJniObject(const mlir::OwningModuleRef &module,
    string jniSharedLibPath, string jniObjPath) {
  Command ar(/*exePath=*/kArPath);
  ar.appendStr("x")
      // old version of ar does not support --output so comment out
      // for now and use the optional wdir for exec() to get around
      // the problem.
      //.appendStr("--output")
      //.appendStr(llvm::sys::path::parent_path(jniObjPath).str())
      .appendStr(jniSharedLibPath)
      .appendStr(llvm::sys::path::filename(jniObjPath).str())
      .exec(llvm::sys::path::parent_path(jniObjPath).str());
}

// Link everything into a shared object.
static std::string genSharedLib(string outputBaseName, std::vector<string> opts,
    std::vector<string> objs, std::vector<string> libs,
    std::vector<string> libDirs) {

#ifdef _WIN32
  string sharedLibPath = outputBaseName + ".dll";
  std::vector<string> outputOpt = {"/Fe:" + sharedLibPath};
  // link has to be before def and libpath since they need to be passed through
  // to the linker
  std::vector<string> sharedLibOpts = {
      "/LD", "/link", "/NOLOGO", "/def:" + outputBaseName + ".def"};

  llvm::for_each(libs, [](string &lib) { lib = lib + ".lib"; });
  llvm::for_each(
      libDirs, [](string &libDir) { libDir = "/libpath:\"" + libDir + "\""; });
#else
  string sharedLibPath = outputBaseName + ".so";
  std::vector<string> outputOpt = {"-o", sharedLibPath};
  std::vector<string> sharedLibOpts = {"-shared", "-fPIC"};
  llvm::for_each(libs, [](string &lib) { lib = "-l" + lib; });
  llvm::for_each(libDirs, [](string &libDir) { libDir = "-L" + libDir; });
#endif

  Command link(kCxxPath);
  link.appendList(opts)
      .appendList(objs)
      .appendList(outputOpt)
      .appendList(sharedLibOpts)
      .appendList(libDirs)
      .appendList(libs)
      .exec();

  return sharedLibPath;
}

// Create jar containing java runtime and model shared library (which includes
// jni runtime).
static void genJniJar(const mlir::OwningModuleRef &module,
    string modelSharedLibPath, string modelJniJarPath) {
  llvm::SmallString<8> runtimeDir(getRuntimeDir());
  llvm::sys::path::append(runtimeDir, "javaruntime.jar");
  string javaRuntimeJarPath = llvm::StringRef(runtimeDir).str();

  // Copy javaruntime.jar to model jar.
  llvm::sys::fs::copy_file(javaRuntimeJarPath, modelJniJarPath);

  // Add shared library to model jar.
  Command jar(kJarPath);
  jar.appendStr("uf")
      .appendStr(modelJniJarPath)
      .appendStr("-C")
      .appendStr(llvm::sys::path::parent_path(modelSharedLibPath).str())
      .appendStr(llvm::sys::path::filename(modelSharedLibPath).str())
      .exec();
}

std::string compileModuleToObject(
    const mlir::OwningModuleRef &module, std::string outputBaseName) {
  string bitcodePath = outputBaseName + ".bc";
  genLLVMBitcode(module, bitcodePath, outputBaseName);
  llvm::FileRemover bitcodeRemover(
      bitcodePath, !keepFiles(KeepFilesOfType::Bitcode));

  return genModelObject(bitcodePath, outputBaseName);
}

std::string compileModuleToSharedLibrary(
    const mlir::OwningModuleRef &module, std::string outputBaseName) {
  string modelObjPath = compileModuleToObject(module, outputBaseName);
  llvm::FileRemover modelObjRemover(
      modelObjPath, !keepFiles(KeepFilesOfType::Object));

  return genSharedLib(
      outputBaseName, {}, {modelObjPath}, {"cruntime"}, {getRuntimeDir()});
}

void compileModuleToJniJar(
    const mlir::OwningModuleRef &module, std::string outputBaseName) {
  string modelObjPath = compileModuleToObject(module, outputBaseName);
  llvm::FileRemover modelObjRemover(
      modelObjPath, !keepFiles(KeepFilesOfType::Object));

  StringRef outputDir = llvm::sys::path::parent_path(outputBaseName);
  if (outputDir.empty())
    outputDir = StringRef(".");

  string jniSharedLibPath = getRuntimeDir() + "/libjniruntime.a";

  llvm::SmallString<8> jniObjDir(outputDir);
  llvm::sys::path::append(jniObjDir, "jnidummy.c.o");
  string jniObjPath = llvm::StringRef(jniObjDir).str();

  genJniObject(module, jniSharedLibPath, jniObjPath);
  llvm::FileRemover jniObjRemover(
      jniObjPath, !keepFiles(KeepFilesOfType::Object));

  llvm::SmallString<8> jniLibDir(outputDir);
  llvm::sys::path::append(jniLibDir, "libmodel");
  string jniLibBase = llvm::StringRef(jniLibDir).str();

  string modelSharedLibPath = genSharedLib(jniLibBase, {"-z", "noexecstack"},
      {modelObjPath, jniObjPath}, {"jniruntime", "cruntime"},
      {getRuntimeDir()});
  llvm::FileRemover modelSharedLibRemover(
      modelSharedLibPath, !keepFiles(KeepFilesOfType::Object));

  string modelJniJarPath = outputBaseName + ".jar";
  genJniJar(module, modelSharedLibPath, modelJniJarPath);
}

void registerDialects(mlir::MLIRContext &context) {
  // Load our Dialect in this MLIR Context.
  context.getOrLoadDialect<mlir::AffineDialect>();
  context.getOrLoadDialect<mlir::vector::VectorDialect>();
  context.getOrLoadDialect<mlir::LLVM::LLVMDialect>();
  context.getOrLoadDialect<mlir::scf::SCFDialect>();
  context.getOrLoadDialect<mlir::StandardOpsDialect>();
  context.getOrLoadDialect<mlir::shape::ShapeDialect>();
  context.getOrLoadDialect<mlir::math::MathDialect>();
  context.getOrLoadDialect<mlir::memref::MemRefDialect>();
  context.getOrLoadDialect<mlir::ONNXDialect>();
  context.getOrLoadDialect<mlir::KrnlOpsDialect>();
}

void addONNXToMLIRPasses(mlir::PassManager &pm) {
  // This is a transition from previous static passes to full dynamic passes
  // Static passes are kept and the dynamic pass is added as IF-THEN
  // with the static iteration.
  // The reasons are
  // 1. The debug flag, --print-ir-after/befor-all, can display IR for each
  //    static pass, but the dynamic pipeline will be viewed as one. MLIR
  //    may have solution that I am not aware of yet.
  // 2. Easy to compare two approaches.
  // In future, only the dynamic pass, ONNXOpTransformPass, will be used for
  // this function.

  pm.addNestedPass<FuncOp>(mlir::createDecomposeONNXToONNXPass());
  pm.addPass(mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createShapeInferencePass());
  // There are more opportunities for const propagation once all tensors have
  // inferred shapes.
  pm.addNestedPass<FuncOp>(mlir::createConstPropONNXToONNXPass());

  if (onnxOpTransformThreshold > 0) {
    // Dynamic iterate in ONNXOpTransformPass
    pm.addPass(mlir::createONNXOpTransformPass(onnxOpTransformThreshold));
  } else {
    // Statically add extra passes
    for (int i = 0; i < repeatOnnxTransform; i++) {
      pm.addPass(mlir::createCanonicalizerPass());
      pm.addPass(mlir::createShapeInferencePass());
      pm.addNestedPass<FuncOp>(mlir::createConstPropONNXToONNXPass());
    }
  }

  // Clean dead code.
  pm.addPass(mlir::createSymbolDCEPass());
}

void addONNXToKrnlPasses(mlir::PassManager &pm, int optLevel) {
  pm.addNestedPass<FuncOp>(mlir::createONNXPreKrnlVerifyPass());
  // Add instrumentation for Onnx Ops
  pm.addNestedPass<FuncOp>(mlir::createInstrumentONNXPass());
  pm.addPass(mlir::createLowerToKrnlPass(optLevel));
  // An additional pass of canonicalization is helpful because lowering
  // from ONNX dialect to Standard dialect exposes additional canonicalization
  // opportunities.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createDisconnectKrnlDimFromAllocPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void addKrnlToAffinePasses(mlir::PassManager &pm) {
  pm.addNestedPass<FuncOp>(mlir::createConvertKrnlToAffinePass());
  // Fuse loops in Affine dialect.
  //  pm.addPass(mlir::createLoopFusionPass());
}

void addKrnlToLLVMPasses(mlir::OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createLowerAffinePass());

  // Use MLIR buffer deallocation pass to emit buffer deallocs.
  // Currently this has to be done *after* lowering the affine dialect because
  // operations in that dialect do not conform to the requirements explained in
  // https://mlir.llvm.org/docs/BufferDeallocationInternals.
  pm.addNestedPass<FuncOp>(mlir::bufferization::createBufferDeallocationPass());
  if (enableMemoryBundling) {
    pm.addNestedPass<FuncOp>(mlir::createKrnlEnableMemoryPoolPass());
    pm.addNestedPass<FuncOp>(mlir::createKrnlBundleMemoryPoolsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(mlir::createKrnlOptimizeMemoryPoolsPass());
  }

  pm.addPass(mlir::createLowerToCFGPass());
  pm.addPass(mlir::createConvertKrnlToLLVMPass());
  pm.addPass(mlir::createReconcileUnrealizedCastsPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void processInputFile(string inputFilename, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module, std::string *errorMessage) {
  // Decide if the input file is an ONNX model or a model specified
  // in MLIR. The extension of the file is the decider.
  string extension = inputFilename.substr(inputFilename.find_last_of(".") + 1);
  bool inputIsONNX = (extension == "onnx");
  bool inputIsMLIR = (extension == "mlir");

  if (!inputIsONNX && !inputIsMLIR) {
    *errorMessage = "Invalid input file '" + inputFilename +
                    "': Either an ONNX model (.onnx), or an MLIR file (.mlir) "
                    "needs to be provided.";
    return;
  }

  if (inputIsONNX) {
    ImportOptions options;
    options.useOnnxModelTypes = useOnnxModelTypes;
    options.invokeOnnxVersionConverter = invokeOnnxVersionConverter;
    options.shapeInformation = shapeInformation;
    ImportFrontendModelFile(
        inputFilename, context, module, errorMessage, options);
  } else if (inputIsMLIR)
    loadMLIR(inputFilename, context, module);
}

void processInputArray(const void *onnxBuffer, int bufferSize,
    mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
  ImportOptions options;
  options.useOnnxModelTypes = useOnnxModelTypes;
  options.invokeOnnxVersionConverter = invokeOnnxVersionConverter;
  options.shapeInformation = shapeInformation;
  ImportFrontendModelArray(onnxBuffer, bufferSize, context, module, options);
}

InputIRLevelType determineInputIRLevel(mlir::OwningModuleRef &module) {
  Operation *moduleOp = module->getOperation();

  // Collect dialect namespaces.
  llvm::SmallDenseSet<StringRef> dialectNamespace;
  moduleOp->walk([&](mlir::Operation *op) {
    dialectNamespace.insert(op->getDialect()->getNamespace());
  });

  // If there are ONNX ops, the input level is ONNX.
  bool hasONNXOps = llvm::any_of(dialectNamespace,
      [&](StringRef ns) { return (ns == ONNXDialect::getDialectNamespace()); });
  if (hasONNXOps)
    return ONNXLevel;

  // If there are Krnl ops, the input level is MLIR.
  bool hasKrnlOps = llvm::any_of(dialectNamespace, [&](StringRef ns) {
    return (ns == KrnlOpsDialect::getDialectNamespace());
  });
  if (hasKrnlOps)
    return MLIRLevel;

  // Otherwise, set to the lowest level, LLVMLevel.
  return LLVMLevel;
}

void outputCode(
    mlir::OwningModuleRef &module, string filename, string extension) {
  mlir::OpPrintingFlags flags;
  if (preserveLocations)
    flags.enableDebugInfo();

  string errorMessage;
  auto output = openOutputFile(filename + extension, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  module->print(output->os(), flags);
  output->keep();
}

void emitOutputFiles(string outputBaseName, EmissionTargetType emissionTarget,
    mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
  // For EmitONNXIR and EmitMLIR the constant value are embedded in the code
  // thus making the code hard to read. These values can be elided by emitting
  // two versions of the same source code:
  // (1) a version with all the constant values included meant for being passed
  //     back to onnx-mlir for further processing and stored in:
  //
  //     <name>.onnx.mlir
  //
  // (2) a version without constants meant for being inspected by users and
  //     stored in:
  //
  //     <name>.tmp
  //
  // In the case of the LLVM Dialect IR the constant values are grouped
  // outside the function code at the beginning of the file in which case the
  // elision of these constants is not strictly required. Elision is also not
  // necessary when emitting the .bc file.
  switch (emissionTarget) {
  case EmitObj: {
    string modelObjPath = compileModuleToObject(module, outputBaseName);
    if (keepFiles(KeepFilesOfType::MLIR))
      outputCode(module, outputBaseName, ".llvm.mlir");

    if (VerboseOutput)
      printf("Object file %s.o has been compiled.\n", outputBaseName.c_str());
  } break;
  case EmitLib: {
    string sharedLib = compileModuleToSharedLibrary(module, outputBaseName);
    if (keepFiles(KeepFilesOfType::MLIR))
      outputCode(module, outputBaseName, ".llvm.mlir");
    if (VerboseOutput)
      printf("Shared library %s has been compiled.\n", sharedLib.c_str());
  } break;
  case EmitJNI: {
    compileModuleToJniJar(module, outputBaseName);
    if (keepFiles(KeepFilesOfType::MLIR))
      outputCode(module, outputBaseName, ".llvm.mlir");
    if (VerboseOutput)
      printf("JNI archive %s.jar has been compiled.\n", outputBaseName.c_str());
  } break;
  default: {
    // Emit the version with all constants included.
    outputCode(module, outputBaseName, ".onnx.mlir");
    if (VerboseOutput)
      printf("Full MLIR code written to: \n\t%s\n\n",
          (outputBaseName + ".onnx.mlir").c_str());

    // Apply specific passes to clean up the code where necessary.
    mlir::PassManager cleanSourcePM(
        &context, mlir::OpPassManager::Nesting::Implicit);
    if (emissionTarget == EmitONNXIR || emissionTarget == EmitONNXBasic)
      cleanSourcePM.addNestedPass<FuncOp>(mlir::createElideConstantValuePass());
    if (emissionTarget == EmitMLIR)
      cleanSourcePM.addNestedPass<FuncOp>(
          mlir::createElideConstGlobalValuePass());

    if (emissionTarget == EmitONNXBasic || emissionTarget == EmitONNXIR ||
        emissionTarget == EmitMLIR) {
      if (mlir::failed(cleanSourcePM.run(*module)))
        llvm::errs() << "Could not apply simplification passes.\n";
      outputCode(module, outputBaseName, ".tmp");
      if (VerboseOutput) {
        printf("Constant-free MLIR Code written to: \n\t%s\n\n",
            (outputBaseName + ".tmp").c_str());
        printf("Use:\n\t%s\nto continue lowering the code to other dialects.\n",
            (outputBaseName + ".onnx.mlir").c_str());
      }
    }
  }
  }
} // end anonymous namespace

// Get the LLVM Target object corresponding to the target triple (if valid).
static const llvm::Target *getLLVMTarget(
    const std::string &targetTriple, const Location &loc) {
  std::string error;
  const llvm::Target *LLVMTarget =
      llvm::TargetRegistry::lookupTarget(targetTriple, error);
  if (!LLVMTarget) {
    emitError(loc, Twine("Target architecture is unknown: ") + error);
    return nullptr;
  }

  return LLVMTarget;
}

static std::string getTargetTriple() {
  return (mtriple != "") ? mtriple.getValue() : kDefaultTriple;
}
static std::string getTargetCpu() {
  return (mcpu != "") ? mcpu.getValue() : "";
}

/// Return the module datalayout string. The datalayout string is determined
/// by creating a target machine using the target triple and target cpu.
static std::string getDataLayout(const Location &loc) {
  const std::string targetTriple = getTargetTriple();
  const std::string targetCpu = getTargetCpu();
  const llvm::Target &LLVMTarget = *getLLVMTarget(targetTriple, loc);
  llvm::TargetOptions ops;
  llvm::TargetMachine *targetMachine = LLVMTarget.createTargetMachine(
      targetTriple, targetCpu, "" /*features*/, ops, None);
  if (!targetMachine) {
    emitError(loc, "failed to create target machine");
    return nullptr;
  }

  const llvm::DataLayout &dl = targetMachine->createDataLayout();
  std::string dataLayoutString = dl.getStringRepresentation();
  assert(dataLayoutString != "" && "Expecting a valid target datalayout");

  return dataLayoutString;
}

void setupModule(mlir::OwningModuleRef &module, mlir::MLIRContext &context,
    std::string outputBaseName) {
  // Initialize the targets support for all targets LLVM was configured for.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  // Set the module target triple and datalayout.
  Operation &moduleOp = *(module->getOperation());
  Location loc = moduleOp.getLoc();
  moduleOp.setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
      StringAttr::get(&context, getTargetTriple()));
  moduleOp.setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
      StringAttr::get(&context, getDataLayout(loc)));

  if (keepFiles(KeepFilesOfType::MLIR)) {
    outputCode(module, outputBaseName, ".input.mlir");
    module.release();
    loadMLIR(outputBaseName + ".input.mlir", context, module);
  }
}

static void addPasses(mlir::OwningModuleRef &module, mlir::PassManager &pm,
    EmissionTargetType emissionTarget) {
  InputIRLevelType inputIRLevel = determineInputIRLevel(module);

  if (inputIRLevel <= ONNXLevel && emissionTarget >= EmitONNXIR)
    addONNXToMLIRPasses(pm);

  if (emissionTarget >= EmitMLIR) {
    if (inputIRLevel <= ONNXLevel)
      addONNXToKrnlPasses(pm, OptimizationLevel);
    if (inputIRLevel <= MLIRLevel)
      addKrnlToAffinePasses(pm);
  }

  if (inputIRLevel <= LLVMLevel && emissionTarget >= EmitLLVMIR)
    addKrnlToLLVMPasses(pm);
}

void emitOutput(mlir::OwningModuleRef &module, mlir::MLIRContext &context,
    std::string outputBaseName, mlir::PassManager &pm,
    EmissionTargetType emissionTarget) {
  if (printIR) {
    mlir::OpPrintingFlags flags;
    if (preserveLocations)
      flags.enableDebugInfo();
    module->print(llvm::outs(), flags);
  } else
    emitOutputFiles(outputBaseName, emissionTarget, context, module);
}

int compileModule(mlir::OwningModuleRef &module, mlir::MLIRContext &context,
    std::string outputBaseName, EmissionTargetType emissionTarget) {
  setupModule(module, context, outputBaseName);

  mlir::PassManager pm(&context, mlir::OpPassManager::Nesting::Implicit);
  addPasses(module, pm, emissionTarget);
  mlir::applyPassManagerCLOptions(pm);
  mlir::applyDefaultTimingPassManagerCLOptions(pm);

  if (mlir::failed(pm.run(*module)))
    return 4;
  emitOutput(module, context, outputBaseName, pm, emissionTarget);
  return 0;
}
