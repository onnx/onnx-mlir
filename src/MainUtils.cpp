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

#include "mlir/Conversion/AffineToTVP/AffineToTVPPass.h"
#include "mlir/Conversion/StandardToTVP/ConvertStandardToTVP.h"
#include "mlir/Conversion/TVPToLLVM/ConvertTVPToLLVM.h"
#include "mlir/Conversion/VectorToTVP/ConvertVectorToTVP.h"
#include "mlir/Dialect/Affine/Passes.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h"
#include "mlir/Dialect/Nepal/Passes.h"
#include "mlir/Dialect/TVP/Passes.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Program.h"

#include "ExternalUtil.hpp"
#include "MainUtils.hpp"

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace std;
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
llvm::cl::opt<bool> preserveLocations("preserveLocations",
    llvm::cl::desc("emit location data:"), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> printIR("printIR",
    llvm::cl::desc("print the IR to stdout:"), llvm::cl::init(false),
    llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<bool> useOnnxModelTypes("useOnnxModelTypes",
    llvm::cl::desc("use types and shapes from ONNX model"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<string> mtriple("mtriple", llvm::cl::desc("Target architecture"),
    llvm::cl::value_desc("<llvm target triple>"),
    llvm::cl::cat(OnnxMlirOptions), llvm::cl::ValueRequired);

llvm::cl::opt<string> mcpu("mcpu", llvm::cl::desc("Target cpu"),
    llvm::cl::value_desc("<llvm cpu value>"), llvm::cl::cat(OnnxMlirOptions),
    llvm::cl::ValueRequired);

llvm::cl::opt<bool> npu("npu", llvm::cl::desc("Execute passes specific to NPU"),
    llvm::cl::init(false), llvm::cl::cat(OnnxMlirOptions));

llvm::cl::opt<int> virtualVectorSize("virtual-vector-size",
    llvm::cl::desc("Virtual vector size to affine-super-vectorize size"),
    llvm::cl::init(256), llvm::cl::cat(OnnxMlirOptions));

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
      cout << llvm::join(argsRef, " ") << "\n";

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
    return;
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(*fileOrErr), llvm::SMLoc());
  module = mlir::parseSourceFile(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    return;
  }
}

string getTargetOptions() {
  string targetOptions = "";
  if (mtriple != "")
    targetOptions = "--mtriple=" + mtriple;
  if (mcpu != "")
    targetOptions += " --mcpu=" + mcpu;
  return targetOptions;
}

// Write LLVM optimized bitcode.
void genLLVMBitcode(const mlir::OwningModuleRef &module,
    string optimizedBitcodePath, string outputBaseName) {
  error_code error;

  // Write bitcode to a file.
  string unoptimizedBitcodePath = outputBaseName + ".unoptimized.bc";
  llvm::FileRemover unoptimzedBitcodeRemover(unoptimizedBitcodePath);

  llvm::raw_fd_ostream moduleBitcodeStream(
      unoptimizedBitcodePath, error, llvm::sys::fs::F_None);

  llvm::LLVMContext llvmContext;
  mlir::registerLLVMDialectTranslation(*(module.get().getContext()));
  auto llvmModule = mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule)
    llvm::errs() << "Failed to translate module to LLVMIR.\n";
  llvm::WriteBitcodeToFile(*llvmModule, moduleBitcodeStream);
  moduleBitcodeStream.flush();

  // Use the LLVM's 'opt' command to optimize the bitcode.
  string optPath = getToolPath("opt");
  Command optBitcode(/*exePath=*/!optPath.empty() ? optPath : kOptPath);
  optBitcode.appendStr("-O2")
      .appendStr(getTargetOptions())
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
      .appendStr(getTargetOptions())
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
  llvm::FileRemover bitcodeRemover(bitcodePath);

  string modelObjPath = outputBaseName + ".o";
  genModelObject(module, bitcodePath, modelObjPath);
  llvm::FileRemover modelObjRemover(modelObjPath);

  string modelSharedLibPath = outputBaseName + ".so";
  genSharedLib(module, modelSharedLibPath, {"-shared", "-fPIC"}, {modelObjPath},
      {"-lcruntime"});
}

void compileModuleToJniJar(
    const mlir::OwningModuleRef &module, std::string outputBaseName) {

  string bitcodePath = outputBaseName + ".bc";
  genLLVMBitcode(module, bitcodePath, outputBaseName);
  llvm::FileRemover bitcodeRemover(bitcodePath);

  string modelObjPath = outputBaseName + ".o";
  genModelObject(module, bitcodePath, modelObjPath);
  llvm::FileRemover modelObjRemover(modelObjPath);

  string jniSharedLibPath = getRuntimeDir() + "/libjniruntime.a";
  string jniObjPath = "jnidummy.c.o";
  genJniObject(module, jniSharedLibPath, jniObjPath);
  llvm::FileRemover jniObjRemover(jniObjPath);

  string modelSharedLibPath = "libmodel.so";
  genSharedLib(module, modelSharedLibPath,
      {"-shared", "-fPIC", "-z", "noexecstack"}, {modelObjPath, jniObjPath},
      {"-ljniruntime", "-lcruntime"});
  llvm::FileRemover modelSharedLibRemover(modelSharedLibPath);

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
  context.getOrLoadDialect<mlir::linalg::LinalgDialect>();
  context.getOrLoadDialect<mlir::tvp::TVPDialect>();
  context.getOrLoadDialect<mlir::nepal::NepalDialect>();
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
  if (npu)
    pm.addNestedPass<FuncOp>(mlir::createConvertONNXToLinalgPass());
  pm.addPass(mlir::createLowerToKrnlPass());
  if (npu)
    pm.addPass(mlir::createConvertKrnlToStandardPass());
  // An additional pass of canonicalization is helpful because lowering
  // from ONNX dialect to Standard dialect exposes additional canonicalization
  // oppertunities.

  // MAKUDRYA-ISSUE_TODO: temporarily remove canonicalization pass as it
  // changes affine maps in affine loads/stores in such a way that
  // supervectorizer pass is not capable to generate correct mapping.
  // pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(createDisconnectKrnlDimFromAllocPass());

  // TODO: make this pass optional
  // NOTE-diprou-12/07/20: Conditionally disabling both mem-pool
  // passes for the Apollo pipeline. Introducing the pool leads
  // krnl.getref() statements. Such statements are later lowered
  // in krnl -> llvm pass. At that point Memrefs are converted
  // to a low-level representation which is incompatible with the
  // high-level view required for turning buffers to NEPAL ArrayRefs.
  // Disabling the passes leaves the original buffer allocations,
  // which can subsequently be optimized and turned into NEPAL
  // ArrayRefs in a straightforward manner.
  if (!npu) {
    pm.addNestedPass<FuncOp>(mlir::createKrnlEnableMemoryPoolPass());
    pm.addNestedPass<FuncOp>(mlir::createKrnlBundleMemoryPoolsPass());
    pm.addPass(mlir::createCanonicalizerPass());
    pm.addNestedPass<FuncOp>(mlir::createKrnlOptimizeMemoryPoolsPass());
    pm.addPass(mlir::createCanonicalizerPass());
  }
}

void addKrnlToAffinePasses(mlir::PassManager &pm) {
  pm.addNestedPass<FuncOp>(mlir::createConvertKrnlToAffinePass());

  // MAKUDRYA-ISSUE_TODO: see comment above. Add canonicalization
  // here instead, where it is safe.
  pm.addPass(mlir::createCanonicalizerPass());

  // Fuse loops in Affine dialect.
  //  pm.addPass(mlir::createLoopFusionPass());
  pm.addNestedPass<FuncOp>(mlir::createAffineLoopInvariantCodeMotionPass());
}

void addKrnlToLLVMPasses(mlir::OpPassManager &pm) {
  pm.addNestedPass<FuncOp>(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLowerToCFGPass());
  pm.addPass(mlir::createConvertKrnlToLLVMPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void processInputFile(string inputFilename, EmissionTargetType emissionTarget,
    mlir::MLIRContext &context, mlir::OwningModuleRef &module) {
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
    ImportFrontendModelFile(inputFilename, context, module, options);
  } else {
    LoadMLIR(inputFilename, context, module);
  }
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
    printf("Shared library %s.so has been compiled.\n", outputBaseName.c_str());
  } else if (emissionTarget == EmitJNI) {
    compileModuleToJniJar(module, outputBaseName);
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
  auto &tvpModulePM = pm.nest<tvp::TVPModuleOp>();

  if (emissionTarget >= EmitONNXIR) {
    addONNXToMLIRPasses(pm);
  }

  if (emissionTarget >= EmitMLIR) {
    addONNXToKrnlPasses(pm);
    addKrnlToAffinePasses(pm);
  }

  // For now, perform ALL passes on both nested tvp.module and top level module.
  // First, add them to the parent module, then add to the nested one.
  // We have to duplicate number of pass instances as they are passed
  // via unique_ptrs.

  if (emissionTarget >= EmitLLVMIR) {
    addKrnlToLLVMPasses(pm);
    addKrnlToLLVMPasses(tvpModulePM);
  }

  mlir::applyPassManagerCLOptions(pm);
  if (mlir::failed(pm.run(*module)))
    return 4;

  emitOutputFiles(outputBaseName, emissionTarget, context, module);
  return 0;
}

int compileModuleApollo(mlir::OwningModuleRef &module,
    mlir::MLIRContext &context, std::string outputBaseName,
    EmissionTargetType emissionTarget) {
  mlir::PassManager pm(&context, mlir::OpPassManager::Nesting::Explicit);
  pm.enableCrashReproducerGeneration(outputBaseName + ".crash.mlir", true);
#ifdef NDEBUG
  pm.enableVerifier();
#endif
  applyPassManagerCLOptions(pm);

  // EmitMLIR passes
  npu = true;
  addONNXToMLIRPasses(pm);
  addONNXToKrnlPasses(pm);
  addKrnlToAffinePasses(pm);

  // Outlining passes
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addNestedPass<FuncOp>(mlir::createAffineForToTVPPass());
  pm.addPass(mlir::createTVPKernelOutliningPass());
  pm.addNestedPass<tvp::TVPModuleOp>(mlir::createAssignKernelIdsPass());

  // TCP passes
  pm.addPass(mlir::createNepalDMAOpInsertionPass());
  pm.addPass(mlir::createNepalOptimizeDMAPass());
  pm.addPass(mlir::createAffineDMAToNepalConversionPass());
  pm.addPass(mlir::createNepalArgumentsStructPass());
  pm.addPass(mlir::createCSEPass());
  pm.addPass(mlir::createApplyArtemisCallingConventionPass());
  pm.addPass(mlir::createNepalGenerationPass(outputBaseName));

  // Remove TCP IR and only keep TVP kernels
  pm.addPass(mlir::createTVPKernelFilterPass());

  // TVP passes on TVP kernels
  pm.addNestedPass<FuncOp>(mlir::createTVPKernelMemSpaceRemovalPass());
  pm.addNestedPass<FuncOp>(mlir::createSuperVectorizePass({virtualVectorSize}));
  pm.addNestedPass<FuncOp>(mlir::createLowerAffinePass());
  pm.addNestedPass<FuncOp>(mlir::createConvertVectorToTVPPass());
  pm.addNestedPass<FuncOp>(mlir::createLowerToCFGPass());
  pm.addPass(mlir::createConvertStandardToTVPPass());
  pm.addNestedPass<FuncOp>(mlir::createConvertVectorToSCFPass());
  pm.addPass(mlir::createConvertTVPToLLVMPass());
  pm.addPass(mlir::LLVM::createGenerateKernelDispatcherPass());

  if (mlir::failed(pm.run(*module)))
    return 4;

  outputCode(module, outputBaseName, ".final.mlir");

  string mlirTranslatePath = getToolPath("mlir-translate");
  Command mlirTranslateCommand(
      /*exePath=*/!mlirTranslatePath.empty() ? mlirTranslatePath
                                             : kMlirTranslatePath);
  string mlirTranslateInput = outputBaseName + ".final.mlir";
  string mlirTranslateoOutput = outputBaseName + ".ll";
  mlirTranslateCommand.appendStr("--mlir-to-llvmir")
      .appendStr(mlirTranslateInput)
      .appendStr("-o=" + mlirTranslateoOutput)
      .exec();

  string llcPath = getToolPath("llc");
  Command llcCommand(
      /*exePath=*/!llcPath.empty() ? llcPath : kLlcPath);
  string llcInput = mlirTranslateoOutput;
  string llcOutput = outputBaseName + ".s";
  llcCommand.appendStr("-mtriple=apollo-none-none")
      .appendStr("-max-jump-table-size=0")
      .appendStr("-O2")
      .appendStr(llcInput)
      .appendStr("-o=" + llcOutput)
      .exec();

  return 0;
}
