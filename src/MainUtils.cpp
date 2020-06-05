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
#include <regex>
#include <string>

#include <llvm/Support/FileSystem.h>
#include <llvm/Support/Program.h>
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

  if (rc != 0) {
    fprintf(stderr, "%s\n", llvm::join(argsRef, " ").c_str());
    llvm_unreachable("Command execution failed.");
  }

  return rc;
}

struct Command {
  std::string _path;
  std::vector<std::string> _args;

  Command(std::string exe, const std::string &exeFileName)
      : _path(std::move(exe)), _args({exeFileName}) {}

  void appendStr(const std::string &arg) { _args.emplace_back(arg); }

  void appendList(const std::vector<std::string> &args) {
    _args.insert(_args.end(), args.begin(), args.end());
  }

  void resetArgs() {
      auto exeFileName = _args.front();
      _args.clear();
      _args.emplace_back(exeFileName);
  }

  int exec() { return executeCommandAndWait(_path, _args); }
};
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
  // Extract constant pack file name, which is embedded as a symbol in the
  // module being compiled.
  auto constPackFilePathSym = (*module).lookupSymbol<mlir::LLVM::GlobalOp>(
      mlir::KrnlPackedConstantOp::getConstPackFilePathSymbolName());
  auto constPackFilePath = constPackFilePathSym.valueAttr()
                               .dyn_cast_or_null<mlir::StringAttr>()
                               .getValue()
                               .str();

  // Write LLVM bitcode.
  string outputFilename = outputBaseName + ".bc";
  error_code error;
  llvm::raw_fd_ostream moduleBitcodeStream(
      outputFilename, error, llvm::sys::fs::F_None);
  llvm::WriteBitcodeToFile(
      *mlir::translateModuleToLLVMIR(*module), moduleBitcodeStream);
  moduleBitcodeStream.flush();

  Command llvmToObj(/*exe=*/kLlcPath, /*exeFileName=*/"llc");
  llvmToObj.appendStr("-filetype=obj");
  llvmToObj.appendStr("-relocation-model=pic");
  llvmToObj.appendStr(outputFilename);
  llvmToObj.exec();

#if __APPLE__
  llvm::SmallVector<char, 20> stubSrcPath;
  llvm::sys::fs::createTemporaryFile("stub", "cpp", stubSrcPath);
  std::string stubSrcPathStr(stubSrcPath.begin(), stubSrcPath.end());
  llvm::FileRemover subSrcRemover(stubSrcPath);

  Command createStubObj(/*exe=*/kCxxPath, /*exeFileName=*/kCxxFileName);
  std::string stubObjPathStr = stubSrcPathStr + ".o";
  createStubObj.appendList({"-o", stubObjPathStr});
  createStubObj.appendList({"-c", stubSrcPathStr});
  createStubObj.exec();

  Command genParamObj(/*exe=*/kLinkerPath, /*exeFileName=*/kLinkerFileName);
  genParamObj.appendStr("-r");
  std::string constPackObjPath = constPackFilePath + ".o";
  genParamObj.appendList({"-o", constPackObjPath});
  genParamObj.appendList({"-sectcreate", "binary", "param", constPackFilePath});
  genParamObj.appendStr(stubObjPathStr);
  genParamObj.exec();

#elif __linux__
  std::regex e("[^0-9A-Za-z]");
  auto sanitizedName =
      "_binary_" + std::regex_replace(constPackFilePath, e, "_");

  // Create param.o holding packed parameter values.
  Command genParamObj(/*exe=*/kLinkerPath, /*exeFileName=*/kLinkerFileName);
  genParamObj.appendStr("-r");
  genParamObj.appendList({"-b", "binary"});
  std::string constPackObjPath = constPackFilePath + ".o";
  genParamObj.appendList({"-o", constPackObjPath});
  genParamObj.appendStr(constPackFilePath);
  genParamObj.exec();

  Command redefineSym(/*exe=*/kObjCopyPath, /*exeFileName=*/kObjCopyFileName);
  redefineSym.appendStr("--redefine-sym");
  redefineSym.appendStr(sanitizedName + "_start=_binary_param_bin_start");
  redefineSym.appendStr(constPackObjPath);
  redefineSym.exec();

  redefineSym.resetArgs();
  redefineSym.appendStr("--redefine-sym");
  redefineSym.appendStr(sanitizedName + "_end=_binary_param_bin_end");
  redefineSym.appendStr(constPackObjPath);
  redefineSym.exec();

#endif
  std::string runtimeDirInclFlag = "";
  if (getEnvVar("RUNTIME_DIR").hasValue())
    runtimeDirInclFlag = "-L" + getEnvVar("RUNTIME_DIR").getValue();

  Command link(kCxxPath, kCxxFileName);
  link.appendList({"-shared", "-fPIC"});
  link.appendStr(outputBaseName + ".o");
  link.appendStr(constPackObjPath);
  link.appendList({"-o", outputBaseName + ".so"});
  link.appendStr(runtimeDirInclFlag);
  link.appendList({"-lEmbeddedDataLoader", "-lcruntime"});
  link.exec();
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
