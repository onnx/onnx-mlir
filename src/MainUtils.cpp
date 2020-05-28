//===--------------------------- main_utils.cpp ---------------------------===//
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
#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Program.h>
#include <string>

#include <mlir/Dialect/LLVMIR/LLVMDialect.h>
#include <mlir/IR/SymbolTable.h>

#include "src/ExternalUtil.hpp"
#include "src/MainUtils.hpp"

#include "MainUtils.hpp"

#ifdef _WIN32
#include <io.h>
#else
#include <unistd.h>
#endif

using namespace std;
using namespace onnx_mlir;

namespace {

llvm::Optional<std::string> getEnvVar(std::string name) {
  if (const char *envVerbose = std::getenv(name.c_str()))
    return std::string(envVerbose);
  return llvm::None;
}

int executeCommandAndWait(
    const std::string &exe, const std::vector<std::string> &cmds) {
  auto argsRef = std::vector<llvm::StringRef>(cmds.begin(), cmds.end());
  bool verbose = false;
  if (const auto &verboseStr = getEnvVar("VERBOSE"))
    std::istringstream(verboseStr.getValue()) >> verbose;

  // If in verbose mode, print out command before execution.
  if (verbose)
    std::cout << llvm::join(argsRef, " ") << "\n";
  int rc = llvm::sys::ExecuteAndWait(exe, llvm::makeArrayRef(argsRef));
  assert(rc == 0 && "Failed to execute:" && llvm::join(argsRef, " ").c_str());
  return rc;
}
} // namespace

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

void compileModuleToSharedLibrary(
    const mlir::OwningModuleRef &module, string outputBaseName) {
  // Write LLVM bitcode.
  string outputFilename = outputBaseName + ".bc";
  error_code error;
  llvm::raw_fd_ostream moduleBitcodeStream(
      outputFilename, error, llvm::sys::fs::F_None);

  auto linkWithConstPack = (*module).lookupSymbol<mlir::LLVM::GlobalOp>(
      mlir::KrnlPackedConstantOp::getConstPackFilePathSymbolName());
  auto constPackTempFileName =
      linkWithConstPack.valueAttr().dyn_cast_or_null<mlir::StringAttr>();
  auto constPackFileName = mlir::KrnlPackedConstantOp::getConstPackFileName();

  llvm::WriteBitcodeToFile(
      *mlir::translateModuleToLLVMIR(*module), moduleBitcodeStream);
  moduleBitcodeStream.flush();
  executeCommandAndWait(kLlcPath,
      {"llc", "-filetype=obj", "-relocation-model=pic", outputFilename});

  llvm::SmallVector<char, 10> wdPath;
  llvm::sys::fs::createUniqueDirectory("onnx-mlir", wdPath);

  // Move constant pack to working directory.
  llvm::SmallVector<char, 10> constPackPath = wdPath;
  llvm::sys::path::append(constPackPath, constPackFileName);
  llvm::sys::fs::rename(constPackTempFileName.getValue(), constPackPath);
  std::string constPackPathStr(constPackPath.begin(), constPackPath.end());

  llvm::SmallVector<char, 10> paramObjPath = wdPath;
  llvm::sys::path::append(paramObjPath, "param.o");
  std::string paramObjPathStr(paramObjPath.begin(), paramObjPath.end());
#if __APPLE__
  // Code to build object file with data and data loader.
  llvm::SmallVector<char, 10> cStubPath = wdPath;
  llvm::sys::path::append(cStubPath, "stub.cpp");

  // Create empty stub file.
  int descriptor;
  llvm::sys::fs::openFileForWrite(cStubPath, descriptor);
  llvm::FileRemover remover(cStubPath);
  close(descriptor);
  std::string stubPathStr(cStubPath.begin(), cStubPath.end());

  // Construct paths to the stub object file.
  llvm::SmallVector<char, 10> cStubObjPath = wdPath;
  llvm::sys::path::append(cStubObjPath, "stub.o");
  std::string cStubObjPathStr(cStubObjPath.begin(), cStubObjPath.end());

  // Compile empty stub src file to an empty sub object file.
  executeCommandAndWait(
      kCxxPath, {kCxxFileName, "-o", cStubObjPathStr, "-c", stubPathStr});

  // Create param.o holding packed parameter values.
  executeCommandAndWait(
      kLinkerPath, {kLinkerFileName, "-r", "-o", paramObjPathStr, "-sectcreate",
                       "binary", "param", constPackPathStr, cStubObjPathStr});
#elif __linux__
  // Create param.o holding packed parameter values.
  executeCommandAndWait(
      kLinkerPath, {kLinkerFileName, "-r", "-b", "binary", "-o",
                       paramObjPathStr, constPackPathStr});
#endif
  std::string runtimeDir = getEnvVar("RUNTIME_DIR").hasValue()
                               ? "-L" + getEnvVar("RUNTIME_DIR").getValue()
                               : "";
  // Link with runtime, dataloader.
  executeCommandAndWait(
      kCxxPath, {kCxxFileName, "-shared", "-fPIC", outputBaseName + ".o",
                    "param.o", "-o", outputBaseName + ".so", runtimeDir,
                    "-lEmbeddedDataLoader", "-lcruntime"});
}

void registerDialects() {
  mlir::registerDialect<mlir::AffineDialect>();
  mlir::registerDialect<mlir::LLVM::LLVMDialect>();
  mlir::registerDialect<mlir::scf::SCFDialect>();
  mlir::registerDialect<mlir::StandardOpsDialect>();
  mlir::registerDialect<mlir::ONNXOpsDialect>();
  mlir::registerDialect<mlir::KrnlOpsDialect>();
}

void addONNXToMLIRPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createDecomposeONNXToONNXPass());
  pm.addPass(mlir::createShapeInferencePass());
  pm.addPass(mlir::createCanonicalizerPass());
  pm.addPass(mlir::createAttributePromotionPass());
  pm.addPass(mlir::createShapeInferencePass());
  pm.addPass(mlir::createAttributePromotionPass());
}

void addONNXToKrnlPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createLowerToKrnlPass());
  pm.addPass(mlir::createPackKrnlGlobalConstantsPass());
  // An additional pass of canonicalization is helpful because lowering
  // from ONNX dialect to Standard dialect exposes additional canonicalization
  // oppertunities.
  pm.addPass(mlir::createCanonicalizerPass());
}

void addKrnlToAffinePasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createLowerKrnlPass());
}

void addKrnlToLLVMPasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createLowerAffinePass());
  pm.addPass(mlir::createLowerToCFGPass());
  pm.addPass(mlir::createKrnlLowerToLLVMPass());
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
    ImportFrontendModelFile(inputFilename, context, module);
  } else {
    LoadMLIR(inputFilename, context, module);
  }
}

void outputCode(
    mlir::OwningModuleRef &module, string filename, string extension) {
  // Start a separate process to redirect the model output. I/O redirection
  // changes will not be visible to the parent process.
  string tempFilename = filename + extension;
#ifdef _WIN32
  // copy original stderr file number
  int stderrOrigin = _dup(_fileno(stderr));
  freopen(tempFilename.c_str(), "w", stderr);
  module->dump();
  fflush(stderr);
  // set modified stderr as original stderr
  _dup2(stderrOrigin, _fileno(stderr));
#else
  if (fork() == 0) {
    freopen(tempFilename.c_str(), "w", stderr);
    module->dump();
    fclose(stderr);
    exit(0);
  }
#endif
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
    printf("Shared library %s.so has been compiled.", outputBaseName.c_str());
  } else {
    // Emit the version with all constants included.
    outputCode(module, outputBaseName, ".onnx.mlir");
    printf("Full MLIR code written to: \n\t%s\n\n",
        (outputBaseName + ".onnx.mlir").c_str());

    // Apply specific passes to clean up the code where necessary.
    mlir::PassManager cleanSourcePM(&context);
    if (emissionTarget == EmitONNXIR || emissionTarget == EmitONNXBasic)
      cleanSourcePM.addPass(mlir::createElideConstantValuePass());
    if (emissionTarget == EmitMLIR)
      cleanSourcePM.addPass(mlir::createElideConstGlobalValuePass());

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
