/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------------- MainUtils.cpp ---------------------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <regex>
#include <string>
#include <vector>

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"

#include "ExternalUtil.hpp"
#include "MainUtils.hpp"

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace std;
using namespace mlir;
using namespace onnx_mlir;

llvm::cl::OptionCategory OnnxMlirOptions(
    "ONNX MLIR Options", "These are frontend options.");

namespace {

llvm::Optional<std::string> getEnvVar(std::string name) {
  if (const char *envVerbose = std::getenv(name.c_str()))
    return std::string(envVerbose);
  return llvm::None;
}

// This definition is here rather than in main.cpp because otherwise it's not
// found probably should be pulled out to a more common location
// TODO: Find a respectable home for the wain

// the option is used in this file, so defined here
llvm::cl::opt<bool> invokeOnnxVersionConverter("invokeOnnxVersionConverter",
    llvm::cl::desc(
        "call onnx vesion converter to convert ONNX model to current version"),
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

llvm::cl::opt<bool> preserveMLIR("preserveMLIR",
    llvm::cl::desc("dont delete the MLIR files (input and llvm):"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> useOnnxModelTypes("useOnnxModelTypes",
    llvm::cl::desc("use types and shapes from ONNX model"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<string> mtriple("mtriple", llvm::cl::desc("Target architecture"),
    llvm::cl::value_desc("<llvm target triple>"),
    llvm::cl::cat(OnnxMlirOptions), llvm::cl::ValueRequired);

llvm::cl::opt<string> mcpu("mcpu", llvm::cl::desc("Target cpu"),
    llvm::cl::value_desc("<llvm cpu value>"), llvm::cl::cat(OnnxMlirOptions),
    llvm::cl::ValueRequired);

// Make a function that forces preserving all files using the runtime arguments
// and/or the overridePreserveFiles enum.
enum class KeepFilesOfType { All, MLIR, Bitcode, Object, None };

// Value below override at compile time by effectively setting the requested
// flags.
static const KeepFilesOfType overridePreserveFiles = KeepFilesOfType::None;

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

// Runtime directory contains all the libraries, jars, etc. that are
// necessary for running onnx-mlir. It's resolved in the following order:
//
//   - if ONNX_MLIR_RUNTIME_DIR is set, use it, otherwise
//   - get path from where onnx-mlir is run, if it's of the form
//   /foo/bar/bin/onnx-mlir,
//     the runtime directory is /foo/bar/lib (note that when onnx-mlir is
//     installed system wide, which is typically /usr/local/bin, this will
//     correctly resolve to /usr/local/lib), but some systems still have
//     lib64 so we check that first. If neither exists, then
//   - use CMAKE_INSTALL_PREFIX/lib, which is typically /usr/local/lib
string getRuntimeDir() {
  const auto &envDir = getEnvVar("ONNX_MLIR_RUNTIME_DIR");
  if (envDir && llvm::sys::fs::exists(envDir.getValue()))
    return envDir.getValue();

  string execDir = llvm::sys::path::parent_path(kExecPath).str();
  if (llvm::sys::path::stem(execDir).str().compare("bin") == 0) {
    string p = execDir.substr(0, execDir.size() - 3);
    if (llvm::sys::fs::exists(p + "lib64"))
      return p + "lib64";
    if (llvm::sys::fs::exists(p + "lib"))
      return p + "lib";
  }

  llvm::SmallString<8> instDir64(kInstPath);
  llvm::sys::path::append(instDir64, "lib64");
  string p = llvm::StringRef(instDir64).str();
  if (llvm::sys::fs::exists(p))
    return p;

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
// directory where onnx-mlir is run. If they are found, it  means both onnx-mlir
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
string getToolPath(string tool) {
  string execDir = llvm::sys::path::parent_path(kExecPath).str();
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

  // Execute command.
  void exec() {
    auto argsRef = std::vector<llvm::StringRef>(_args.begin(), _args.end());
    bool verbose = false;
    if (const auto &verboseStr = getEnvVar("ONNX_MLIR_VERBOSE"))
      istringstream(verboseStr.getValue()) >> verbose;

    // If in verbose mode, print out command before execution.
    if (verbose)
      cout << _path << ": " << llvm::join(argsRef, " ") << "\n";

    std::string errMsg;
    int rc = llvm::sys::ExecuteAndWait(_path, llvm::makeArrayRef(argsRef),
        /*Env=*/None, /*Redirects=*/None,
        /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

    if (rc != 0) {
      fprintf(stderr, "%s\n", llvm::join(argsRef, " ").c_str());
      fprintf(stderr, "Error message: %s\n", errMsg.c_str());
      fprintf(stderr, "Program path: %s\n", _path.c_str());
      llvm_unreachable("Command execution failed.");
    }
  }
};
} // namespace

void setExecPath(const char *argv0, void *fmain) {
  string p;
  if (!(p = llvm::sys::fs::getMainExecutable(argv0, fmain)).empty())
    kExecPath = p;
}

void LoadMLIR(string inputFilename, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module) {
  // Handle '.mlir' input to the ONNX MLIR frontend.
  // The mlir format indicates that one or more of the supported
  // representations are used in the file.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> fileOrErr =
      llvm::MemoryBuffer::getFileOrSTDIN(inputFilename);
  if (std::error_code EC = fileOrErr.getError()) {
    llvm::errs() << "Could not open input file: " << EC.message() << "\n";
    exit(1);
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    exit(1);
  }
}

string getTargetCpuOption() {
  string targetOptions = "";
  if (mcpu != "")
    targetOptions += "--mcpu=" + mcpu;
  return targetOptions;
}

string getTargetTripleOption() {
  string targetOptions = "";
  if (mtriple != "")
    targetOptions = "--mtriple=" + mtriple;
  // Comand cannot tolerate extra spaces. Add only when needed.
  return targetOptions;
}

// Write LLVM optimized bitcode.
void genLLVMBitcode(const mlir::OwningModuleRef &module,
    string optimizedBitcodePath, string outputBaseName) {
  error_code error;

  // Write bitcode to a file.
  string unoptimizedBitcodePath = outputBaseName + ".unoptimized.bc";
  llvm::FileRemover unoptimzedBitcodeRemover(
      unoptimizedBitcodePath, !keepFiles(KeepFilesOfType::Bitcode));

  llvm::raw_fd_ostream moduleBitcodeStream(
      unoptimizedBitcodePath, error, llvm::sys::fs::F_None);

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
  optBitcode.appendStr("-O3")
      .appendStr(getTargetTripleOption())
      .appendStr(getTargetCpuOption())
      .appendList({"-o", optimizedBitcodePath})
      .appendStr(unoptimizedBitcodePath)
      .exec();
}

// Compile LLVM bitcode to object file.
void genModelObject(const mlir::OwningModuleRef &module, string bitcodePath,
    string modelObjPath) {
  string llcPath = getToolPath("llc");
  Command llvmToObj(/*exePath=*/!llcPath.empty() ? llcPath : kLlcPath);
  llvmToObj.appendStr("-filetype=obj")
      .appendStr("-relocation-model=pic")
      .appendList({"-o", modelObjPath})
      .appendStr(bitcodePath)
      .exec();
}

void genJniObject(const mlir::OwningModuleRef &module, string jniSharedLibPath,
    string jniObjPath) {
  Command ar(/*exePath=*/kArPath);
  ar.appendStr("x").appendStr(jniSharedLibPath).appendStr(jniObjPath).exec();
}

// Link everything into a shared object.
void genSharedLib(const mlir::OwningModuleRef &module,
    string modelSharedLibPath, std::vector<string> opts,
    std::vector<string> objs, std::vector<string> libs) {

  string runtimeDirInclFlag = "-L" + getRuntimeDir();

  Command link(kCxxPath);
  link.appendList(opts)
      .appendList(objs)
      .appendList({"-o", modelSharedLibPath})
      .appendStrOpt(runtimeDirInclFlag)
      .appendList(libs)
      .exec();
}

// Create jar containing java runtime and model shared library (which includes
// jni runtime).
void genJniJar(const mlir::OwningModuleRef &module, string modelSharedLibPath,
    string modelJniJarPath) {
  llvm::SmallString<8> runtimeDir(getRuntimeDir());
  llvm::sys::path::append(runtimeDir, "javaruntime.jar");
  string javaRuntimeJarPath = llvm::StringRef(runtimeDir).str();

  // Copy javaruntime.jar to model jar.
  llvm::sys::fs::copy_file(javaRuntimeJarPath, modelJniJarPath);

  // Add shared library to model jar.
  Command jar(kJarPath);
  jar.appendList({"uf", modelJniJarPath}).appendStr(modelSharedLibPath).exec();
}

void compileModuleToSharedLibrary(
    const mlir::OwningModuleRef &module, std::string outputBaseName) {

  string bitcodePath = outputBaseName + ".bc";
  genLLVMBitcode(module, bitcodePath, outputBaseName);
  llvm::FileRemover bitcodeRemover(
      bitcodePath, !keepFiles(KeepFilesOfType::Bitcode));

  string modelObjPath = outputBaseName + ".o";
  genModelObject(module, bitcodePath, modelObjPath);
  llvm::FileRemover modelObjRemover(
      modelObjPath, !keepFiles(KeepFilesOfType::Object));

  string modelSharedLibPath = outputBaseName + ".so";
  genSharedLib(module, modelSharedLibPath, {"-shared", "-fPIC"}, {modelObjPath},
      {"-lcruntime"});
}

void compileModuleToJniJar(
    const mlir::OwningModuleRef &module, std::string outputBaseName) {

  string bitcodePath = outputBaseName + ".bc";
  genLLVMBitcode(module, bitcodePath, outputBaseName);
  llvm::FileRemover bitcodeRemover(
      bitcodePath, !keepFiles(KeepFilesOfType::Bitcode));

  string modelObjPath = outputBaseName + ".o";
  genModelObject(module, bitcodePath, modelObjPath);
  llvm::FileRemover modelObjRemover(
      modelObjPath, !keepFiles(KeepFilesOfType::Object));

  string jniSharedLibPath = getRuntimeDir() + "/libjniruntime.a";
  string jniObjPath = "jnidummy.c.o";
  genJniObject(module, jniSharedLibPath, jniObjPath);
  llvm::FileRemover jniObjRemover(
      jniObjPath, !keepFiles(KeepFilesOfType::Object));

  string modelSharedLibPath = "libmodel.so";
  genSharedLib(module, modelSharedLibPath,
      {"-shared", "-fPIC", "-z", "noexecstack"}, {modelObjPath, jniObjPath},
      {"-ljniruntime", "-lcruntime"});
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
  context.getOrLoadDialect<mlir::ONNXOpsDialect>();
  context.getOrLoadDialect<mlir::KrnlOpsDialect>();
}

void addONNXToMLIRPasses(mlir::PassManager &pm) {
  pm.addNestedPass<FuncOp>(mlir::createDecomposeONNXToONNXPass());
  pm.addPass(mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createShapeInferencePass());
  // There are more opportunities for const propagation once all tensors have
  // inferred shapes.
  pm.addNestedPass<FuncOp>(mlir::createConstPropONNXToONNXPass());
  // Clean dead code.
  pm.addPass(mlir::createSymbolDCEPass());
}

void addONNXToKrnlPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createLowerToKrnlPass());
  // An additional pass of canonicalization is helpful because lowering
  // from ONNX dialect to Standard dialect exposes additional canonicalization
  // oppertunities.
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createDisconnectKrnlDimFromAllocPass());

  // TODO: make this pass optional:
  pm.addNestedPass<FuncOp>(mlir::createKrnlEnableMemoryPoolPass());
  pm.addNestedPass<FuncOp>(mlir::createKrnlBundleMemoryPoolsPass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(mlir::createKrnlOptimizeMemoryPoolsPass());
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
  pm.addPass(mlir::createLowerToCFGPass());
  pm.addPass(mlir::createConvertKrnlToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void processInputFile(string inputFilename, mlir::MLIRContext &context,
    mlir::OwningModuleRef &module) {
  // Decide if the input file is an ONNX model or a model specified
  // in MLIR. The extension of the file is the decider.
  string extension = inputFilename.substr(inputFilename.find_last_of(".") + 1);
  bool inputIsONNX = (extension == "onnx");
  bool inputIsMLIR = (extension == "mlir");
  assert(inputIsONNX != inputIsMLIR &&
         "Either ONNX model or MLIR file needs to be provided.");

  if (inputIsONNX) {
    ImportOptions options;
    options.useOnnxModelTypes = useOnnxModelTypes;
    options.invokeOnnxVersionConverter = invokeOnnxVersionConverter;
    ImportFrontendModelFile(inputFilename, context, module, options);
  } else {
    LoadMLIR(inputFilename, context, module);
  }
}

InputIRLevelType determineInputIRLevel(mlir::OwningModuleRef &module) {
  Operation *moduleOp = module->getOperation();

  // Collect dialect namespaces.
  llvm::SmallDenseSet<StringRef> dialectNamespace;
  moduleOp->walk([&](mlir::Operation *op) {
    dialectNamespace.insert(op->getDialect()->getNamespace());
  });

  // If there are ONNX ops, the input level is ONNX.
  bool hasONNXOps = llvm::any_of(dialectNamespace, [&](StringRef ns) {
    return (ns == ONNXOpsDialect::getDialectNamespace());
  });
  if (hasONNXOps)
    return ONNXLevel;

  // If there are Krnl ops, the input level is MLIR.
  bool hasKrnlOps = llvm::any_of(dialectNamespace, [&](StringRef ns) {
    return (ns == KrnlOpsDialect::getDialectNamespace());
  });
  if (hasKrnlOps) {
    return MLIRLevel;
  }

  // Otherwise, set to the lowest level, LLVMLevel.
  return LLVMLevel;
}

void outputCode(
    mlir::OwningModuleRef &module, string filename, string extension) {
  string tempFilename = filename + extension;
  mlir::OpPrintingFlags flags;
  if (preserveLocations)
    flags.enableDebugInfo();

#ifdef _WIN32
  // copy original stderr file number
  int stderrOrigin = _dup(_fileno(stderr));
#else
  int stderrOrigin = dup(fileno(stderr));
#endif
  freopen(tempFilename.c_str(), "w", stderr);
  module->print(llvm::errs(), flags);
  fflush(stderr);
  // set modified stderr as original stderr
#ifdef _WIN32
  _dup2(stderrOrigin, _fileno(stderr));
#else
  dup2(stderrOrigin, fileno(stderr));
#endif
  if (printIR)
    module->print(llvm::outs(), flags);
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
  if (emissionTarget == EmitLib) {
    // Write LLVM bitcode to disk, compile & link.
    compileModuleToSharedLibrary(module, outputBaseName);
    if (keepFiles(KeepFilesOfType::MLIR))
      outputCode(module, outputBaseName, ".llvm.mlir");
    printf("Shared library %s.so has been compiled.\n", outputBaseName.c_str());
  } else if (emissionTarget == EmitJNI) {
    compileModuleToJniJar(module, outputBaseName);
    if (keepFiles(KeepFilesOfType::MLIR))
      outputCode(module, outputBaseName, ".llvm.mlir");
    printf("JNI archive %s.jar has been compiled.\n", outputBaseName.c_str());
  } else {
    // Emit the version with all constants included.
    outputCode(module, outputBaseName, ".onnx.mlir");
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
      printf("Constant-free MLIR Code written to: \n\t%s\n\n",
          (outputBaseName + ".tmp").c_str());

      printf("Use:\n\t%s\nto continue lowering the code to other dialects.\n",
          (outputBaseName + ".onnx.mlir").c_str());
    }
  }
}

int compileModule(mlir::OwningModuleRef &module, mlir::MLIRContext &context,
    std::string outputBaseName, EmissionTargetType emissionTarget) {
  mlir::PassManager pm(&context, mlir::OpPassManager::Nesting::Implicit);

  if (keepFiles(KeepFilesOfType::MLIR)) {
    outputCode(module, outputBaseName, ".input.mlir");
  }

  InputIRLevelType inputIRLevel = determineInputIRLevel(module);

  if (inputIRLevel <= ONNXLevel && emissionTarget >= EmitONNXIR) {
    addONNXToMLIRPasses(pm);
  }

  if (emissionTarget >= EmitMLIR) {
    if (inputIRLevel <= ONNXLevel)
      addONNXToKrnlPasses(pm);
    if (inputIRLevel <= MLIRLevel)
      addKrnlToAffinePasses(pm);
  }

  if (inputIRLevel <= LLVMLevel && emissionTarget >= EmitLLVMIR)
    addKrnlToLLVMPasses(pm);

  mlir::applyPassManagerCLOptions(pm);
  if (mlir::failed(pm.run(*module)))
    return 4;

  emitOutputFiles(outputBaseName, emissionTarget, context, module);
  return 0;
}
