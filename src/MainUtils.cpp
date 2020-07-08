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

// Helper struct to make command construction and execution easy & readable.
struct Command {
  std::string _path;
  std::vector<std::string> _args;

  Command(std::string exePath)
      : _path(std::move(exePath)),
        _args({llvm::sys::path::filename(_path).str()}) {}

  // Append a single string argument.
  Command &appendStr(const std::string &arg) {
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
    if (const auto &verboseStr = getEnvVar("VERBOSE"))
      istringstream(verboseStr.getValue()) >> verbose;

    // If in verbose mode, print out command before execution.
    if (verbose)
      cout << llvm::join(argsRef, " ") << "\n";
    int rc = llvm::sys::ExecuteAndWait(_path, llvm::makeArrayRef(argsRef));

    if (rc != 0) {
      fprintf(stderr, "%s\n", llvm::join(argsRef, " ").c_str());
      llvm_unreachable("Command execution failed.");
    }
  }
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
  llvm::FileRemover constPackRemover(constPackFilePath);

  llvm::Optional<std::string> constPackObjPath;
#if __APPLE__
  // Create a empty stub file, compile it to an empty obj file.
  llvm::SmallVector<char, 20> stubSrcPath;
  llvm::sys::fs::createTemporaryFile("stub", "cpp", stubSrcPath);
  llvm::FileRemover subSrcRemover(stubSrcPath);
  std::string stubSrcPathStr(stubSrcPath.begin(), stubSrcPath.end());
  Command createStubObj(/*exePath=*/kCxxPath);
  std::string stubObjPathStr = stubSrcPathStr + ".o";
  createStubObj.appendList({"-o", stubObjPathStr})
      .appendList({"-c", stubSrcPathStr})
      .exec();
  llvm::FileRemover stubObjRemover(stubObjPathStr);

  // Embed data into the empty stub obj file.
  constPackObjPath = constPackFilePath + ".o";
  Command genParamObj(/*exePath=*/kLinkerPath);
  genParamObj.appendStr("-r")
      .appendList({"-o", constPackObjPath.getValue()})
      .appendList({"-sectcreate", "binary", "param", constPackFilePath})
      .appendStr(stubObjPathStr)
      .exec();
  llvm::FileRemover constPackObjRemover(constPackObjPath.getValue());

#elif __linux__
  // Create param.o holding packed parameter values.
  constPackObjPath = constPackFilePath + ".o";
  Command genParamObj(/*exePath=*/kLinkerPath);
  genParamObj.appendStr("-r")
      .appendList({"-b", "binary"})
      .appendList({"-o", constPackObjPath.getValue()})
      .appendStr(constPackFilePath)
      .exec();
  llvm::FileRemover constPackObjRemover(constPackObjPath.getValue());

  // Figure out what is the default symbol name describing the start/end
  // address of the embedded data.
  std::regex e("[^0-9A-Za-z]");
  auto sanitizedName =
      "_binary_" + std::regex_replace(constPackFilePath, e, "_");

  // Rename the symbols to saner ones expected by the runtime function.
  Command redefineSym(/*exePath=*/kObjCopyPath);
  redefineSym.appendStr("--redefine-sym")
      .appendStr(sanitizedName + "_start=_binary_param_bin_start")
      .appendStr(constPackObjPath.getValue())
      .exec();
  redefineSym.resetArgs()
      .appendStr("--redefine-sym")
      .appendStr(sanitizedName + "_end=_binary_param_bin_end")
      .appendStr(constPackObjPath.getValue())
      .exec();

#else
  llvm::SmallVector<char, 10> permConstPackFileName(
      constPackFilePath.begin(), constPackFilePath.end());
  llvm::sys::path::replace_extension(permConstPackFileName, "bin");
  std::string permConstPackFileNameStr(
      permConstPackFileName.begin(), permConstPackFileName.end());
  auto constPackFileName = llvm::sys::path::filename(outputBaseName) + "." +
                           llvm::sys::path::filename(permConstPackFileNameStr);
  llvm::sys::fs::rename(constPackFilePath, constPackFileName);

  mlir::Builder builder(*module);
  (*module)
      .lookupSymbol<mlir::LLVM::GlobalOp>(
          mlir::KrnlPackedConstantOp::getConstPackFileNameSymbolName())
      .valueAttr(builder.getStringAttr(constPackFileName.str()));
  (*module)
      .lookupSymbol<mlir::LLVM::GlobalOp>(
          mlir::KrnlPackedConstantOp::getConstPackFileNameStrLenSymbolName())
      .valueAttr(builder.getI64IntegerAttr(constPackFileName.str().size()));
#endif

  // Write LLVM bitcode.
  string outputFilename = outputBaseName + ".bc";
  error_code error;
  llvm::raw_fd_ostream moduleBitcodeStream(
      outputFilename, error, llvm::sys::fs::F_None);

  llvm::WriteBitcodeToFile(
      *mlir::translateModuleToLLVMIR(*module), moduleBitcodeStream);
  moduleBitcodeStream.flush();
  llvm::FileRemover bcRemover(outputFilename);

  // Compile LLVM bitcode to object file.
  Command llvmToObj(/*exePath=*/kLlcPath);
  llvmToObj.appendStr("-filetype=obj");
  llvmToObj.appendStr("-relocation-model=pic");
  llvmToObj.appendStr(outputFilename);
  llvmToObj.exec();
  std::string modelObjPath = outputBaseName + ".o";
  llvm::FileRemover modelObjRemover(modelObjPath);

  llvm::Optional<std::string> runtimeDirInclFlag;
  if (getEnvVar("RUNTIME_DIR").hasValue())
    runtimeDirInclFlag = "-L" + getEnvVar("RUNTIME_DIR").getValue();

  // Link everything into a shared object.
  Command link(kCxxPath);
  link.appendList({"-shared", "-fPIC"})
      .appendStr(modelObjPath)
      .appendStr(constPackObjPath.getValueOr(""))
      .appendList({"-o", outputBaseName + ".so"})
      .appendStrOpt(runtimeDirInclFlag)
      .appendList({"-lEmbeddedDataLoader", "-lcruntime"})
      .exec();
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
  pm.addPass(mlir::createConstPropONNXToONNXPass());
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

  // TODO: make this pass optional:
  pm.addPass(mlir::createKrnlEnableMemoryPoolPass());
  pm.addPass(mlir::createKrnlBundleMemoryPoolsPass());
  pm.addPass(mlir::createCanonicalizerPass());
}

void addKrnlToAffinePasses(mlir::PassManager &pm) {
  pm.addPass(mlir::createLowerKrnlPass());
  // Fuse loops in Affine dialect.
  //  pm.addPass(mlir::createLoopFusionPass());
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
    printf("Shared library %s.so has been compiled.\n", outputBaseName.c_str());
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

int compileModule(mlir::OwningModuleRef &module, mlir::MLIRContext &context,
    std::string outputBaseName, EmissionTargetType emissionTarget) {
  mlir::PassManager pm(&context);
  if (emissionTarget >= EmitONNXIR) {
    addONNXToMLIRPasses(pm);
  }

  if (emissionTarget >= EmitMLIR) {
    addONNXToKrnlPasses(pm);
    addKrnlToAffinePasses(pm);
  }

  if (emissionTarget >= EmitLLVMIR)
    addKrnlToLLVMPasses(pm);

  if (mlir::failed(pm.run(*module)))
    return 4;

  emitOutputFiles(outputBaseName, emissionTarget, context, module);
  return 0;
}
