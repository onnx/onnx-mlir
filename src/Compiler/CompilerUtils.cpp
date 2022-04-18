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

#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/IR/Constants.h"
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
#include "src/Accelerators/Accelerator.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Compiler/CompilerUtils.hpp"

#include "VCSVersion.inc"

#define DEBUG_TYPE "compiler_utils"

using namespace mlir;
using namespace onnx_mlir;

const std::string OnnxMlirEnvOptionName = "ONNX_MLIR_FLAGS";

namespace {

static llvm::Optional<std::string> getEnvVar(std::string name) {
  if (const char *envVerbose = std::getenv(name.c_str()))
    return std::string(envVerbose);
  return llvm::None;
}

// Make a function that forces preserving all files using the runtime arguments
// and/or the overridePreserveFiles enum.
enum class KeepFilesOfType { All, MLIR, LLVMIR, Bitcode, Object, None };

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
  case KeepFilesOfType::LLVMIR:
    return overridePreserveFiles == KeepFilesOfType::LLVMIR || preserveLLVMIR;
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
// wide. In this case, getToolPath returns an empty string and we will fallback
// to llvm-project/build/bin.
//
// Note that this will not work if both onnx-mlir and llvm-project have been
// installed system wide but to different places and their sources have been
// removed. So we force CMAKE_INSTALL_PREFIX to be the same as that of
// llvm-project.
static std::string getToolPath(std::string tool) {
  std::string execDir = llvm::sys::path::parent_path(getExecPath()).str();
  llvm::SmallString<8> toolPath(execDir);
  llvm::sys::path::append(toolPath, tool);
  std::string p = llvm::StringRef(toolPath).str();
  if (llvm::sys::fs::can_execute(p))
    return p;
  else
    return std::string();
}

static std::string getOnnxMlirFullVersion() {
  const std::string OnnxMlirVersion = "onnx-mlir version 1.0.0";
  return
#ifdef ONNX_MLIR_VENDOR
      ONNX_MLIR_VENDOR ", " + OnnxMlirVersion;
#elif defined(ONNX_MLIR_REPOSITORY) && defined(ONNX_MLIR_REVISION) &&          \
    defined(LLVM_REPOSITORY) && defined(LLVM_REVISION)
      OnnxMlirVersion + " (" ONNX_MLIR_REPOSITORY " " ONNX_MLIR_REVISION
                        ", " LLVM_REPOSITORY " " LLVM_REVISION ")";
#else
      OnnxMlirVersion;
#endif
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
// Methods for compiling and file processing.

void loadMLIR(std::string inputFilename, mlir::MLIRContext &context,
    mlir::OwningOpRef<ModuleOp> &module) {
  // Handle '.mlir' input to the ONNX-MLIR frontend.
  // The mlir format indicates that one or more of the supported
  // representations are used in the file.
  std::string errorMessage;
  auto input = openInputFile(inputFilename, &errorMessage);
  if (!input) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  // Parse the input mlir.
  llvm::SourceMgr sourceMgr;
  SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
  sourceMgr.AddNewSourceBuffer(std::move(input), llvm::SMLoc());
  module = mlir::parseSourceFile<ModuleOp>(sourceMgr, &context);
  if (!module) {
    llvm::errs() << "Error can't load file " << inputFilename << "\n";
    exit(1);
  }
}

// Tailor LLVMIR to add features that cannot be done with MLIR LLVMIR.
static void tailorLLVMIR(llvm::Module &llvmModule) {
  llvm::LLVMContext &ctx = llvmModule.getContext();
  // Emit metadata "zos_le_char_mode" for z/OS. Use EBCDIC codepage by default.
  if (llvm::Triple(getTargetTripleOption()).isOSzOS()) {
    StringRef charModeKey = "zos_le_char_mode";
    if (!llvmModule.getModuleFlag(charModeKey)) {
      auto val = llvm::MDString::get(ctx, "ebcdic");
      llvmModule.addModuleFlag(llvm::Module::Error, charModeKey, val);
    }
  }

  // Emit the onnx-mlir version as llvm.ident metadata.
  llvm::NamedMDNode *identMetadata =
      llvmModule.getOrInsertNamedMetadata("llvm.ident");
  llvm::Metadata *identNode[] = {
      llvm::MDString::get(ctx, getOnnxMlirFullVersion())};
  identMetadata->addOperand(llvm::MDNode::get(ctx, identNode));

  // Annotate functions to be accessible from DLL on Windows.
#ifdef _WIN32
  SmallVector<StringRef, 4> exportedFuncs;
  // Signature functions.
  exportedFuncs.emplace_back(StringRef("omInputSignature"));
  exportedFuncs.emplace_back(StringRef("omOutputSignature"));
  exportedFuncs.emplace_back(StringRef("omQueryEntryPoints"));
  // Entry point funtions.
  if (llvm::GlobalVariable *GV =
          llvmModule.getNamedGlobal(StringRef("_entry_point_arrays"))) {
    if (GV->isConstant() && GV->hasDefinitiveInitializer()) {
      llvm::Constant *initializer = GV->getInitializer();
      llvm::ArrayType *AT = dyn_cast<llvm::ArrayType>(initializer->getType());
      for (uint64_t i = 0; i < AT->getNumElements() - 1; ++i) {
        llvm::GlobalVariable *entryGV = llvmModule.getNamedGlobal(
            StringRef("_entry_point_" + std::to_string(i)));
        if (entryGV->isConstant()) {
          llvm::ConstantDataSequential *entry =
              dyn_cast<llvm::ConstantDataSequential>(entryGV->getInitializer());
          exportedFuncs.emplace_back(entry->getAsCString());
        }
      }
    }
  }
  for (StringRef funcName : exportedFuncs)
    if (llvm::GlobalValue *GV = llvmModule.getNamedValue(funcName)) {
      GV->setDSOLocal(true);
      GV->setDLLStorageClass(llvm::GlobalValue::DLLExportStorageClass);
    }
#endif
}

// Write LLVM optimized bitcode.
static void genLLVMBitcode(const mlir::OwningOpRef<ModuleOp> &module,
    std::string optimizedBitcodePath, std::string outputBaseName) {
  std::error_code error;

  // Write bitcode to a file.
  std::string unoptimizedBitcodePath = outputBaseName + ".unoptimized.bc";
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
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate module to LLVMIR.\n";
    exit(1);
  }

  // Tailor LLVMIR to add features that cannot be done with MLIR LLVMIR.
  tailorLLVMIR(*llvmModule);

  // Write LLVMIR to a file.
  std::string llvmirPath = outputBaseName + ".ll";
  llvm::FileRemover llvmirRemover(
      llvmirPath, !keepFiles(KeepFilesOfType::LLVMIR));
  llvm::raw_fd_ostream moduleLLVMIRStream(
      llvmirPath, error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << llvmirPath << ": " << error.message() << "\n";
    exit(error.value());
  }
  llvmModule->print(moduleLLVMIRStream, nullptr);
  moduleLLVMIRStream.flush();

  // Write unoptimized bitcode to a file.
  llvm::WriteBitcodeToFile(*llvmModule, moduleBitcodeStream);
  moduleBitcodeStream.flush();

  // Use the LLVM's 'opt' command to optimize the bitcode.
  std::string optPath = getToolPath("opt");
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
static std::string genModelObject(
    std::string bitcodePath, std::string outputBaseName) {

#ifdef _WIN32
  std::string modelObjPath = outputBaseName + ".obj";
#else
  std::string modelObjPath = outputBaseName + ".o";
#endif

  std::string llcPath = getToolPath("llc");
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

static void genJniObject(const mlir::OwningOpRef<ModuleOp> &module,
    std::string jniSharedLibPath, std::string jniObjPath) {
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
static std::string genSharedLib(std::string outputBaseName,
    std::vector<std::string> opts, std::vector<std::string> objs,
    std::vector<std::string> libs, std::vector<std::string> libDirs) {

#ifdef _WIN32
  std::string sharedLibPath = outputBaseName + ".dll";
  std::vector<std::string> outputOpt = {"/Fe:" + sharedLibPath};
  // link has to be before libpath since they need to be passed through to the
  // linker
  std::vector<std::string> sharedLibOpts = {"/LD", "/link", "/NOLOGO"};

  llvm::for_each(libs, [](std::string &lib) { lib = lib + ".lib"; });
  llvm::for_each(libDirs,
      [](std::string &libDir) { libDir = "/libpath:\"" + libDir + "\""; });
#else
  std::string sharedLibPath = outputBaseName + ".so";
  std::vector<std::string> outputOpt = {"-o", sharedLibPath};
  std::vector<std::string> sharedLibOpts = {"-shared", "-fPIC"};
  llvm::for_each(libs, [](std::string &lib) { lib = "-l" + lib; });
  llvm::for_each(libDirs, [](std::string &libDir) { libDir = "-L" + libDir; });
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
static void genJniJar(const mlir::OwningOpRef<ModuleOp> &module,
    std::string modelSharedLibPath, std::string modelJniJarPath) {
  llvm::SmallString<8> runtimeDir(getRuntimeDir());
  llvm::sys::path::append(runtimeDir, "javaruntime.jar");
  std::string javaRuntimeJarPath = llvm::StringRef(runtimeDir).str();

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
    const mlir::OwningOpRef<ModuleOp> &module, std::string outputBaseName) {
  std::string bitcodePath = outputBaseName + ".bc";
  genLLVMBitcode(module, bitcodePath, outputBaseName);
  llvm::FileRemover bitcodeRemover(
      bitcodePath, !keepFiles(KeepFilesOfType::Bitcode));

  return genModelObject(bitcodePath, outputBaseName);
}

std::string compileModuleToSharedLibrary(
    const mlir::OwningOpRef<ModuleOp> &module, std::string outputBaseName) {
  std::string modelObjPath = compileModuleToObject(module, outputBaseName);
  llvm::FileRemover modelObjRemover(
      modelObjPath, !keepFiles(KeepFilesOfType::Object));

  return genSharedLib(outputBaseName, {}, {modelObjPath},
      getCompilerConfig(CCM_SHARED_LIB_DEPS), {getRuntimeDir()});
}

void compileModuleToJniJar(
    const mlir::OwningOpRef<ModuleOp> &module, std::string outputBaseName) {
  std::string modelObjPath = compileModuleToObject(module, outputBaseName);
  llvm::FileRemover modelObjRemover(
      modelObjPath, !keepFiles(KeepFilesOfType::Object));

  StringRef outputDir = llvm::sys::path::parent_path(outputBaseName);
  if (outputDir.empty())
    outputDir = StringRef(".");

  std::string jniSharedLibPath = getRuntimeDir() + "/libjniruntime.a";

  llvm::SmallString<8> jniObjDir(outputDir);
  llvm::sys::path::append(jniObjDir, "jnidummy.c.o");
  std::string jniObjPath = llvm::StringRef(jniObjDir).str();

  genJniObject(module, jniSharedLibPath, jniObjPath);
  llvm::FileRemover jniObjRemover(
      jniObjPath, !keepFiles(KeepFilesOfType::Object));

  llvm::SmallString<8> jniLibDir(outputDir);
  llvm::sys::path::append(jniLibDir, "libmodel");
  std::string jniLibBase = llvm::StringRef(jniLibDir).str();

  std::string modelSharedLibPath = genSharedLib(jniLibBase,
      {"-z", "noexecstack"}, {modelObjPath, jniObjPath},
      getCompilerConfig(CCM_SHARED_LIB_DEPS), {getRuntimeDir()});
  llvm::FileRemover modelSharedLibRemover(
      modelSharedLibPath, !keepFiles(KeepFilesOfType::Object));

  std::string modelJniJarPath = outputBaseName + ".jar";
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

void processInputFile(std::string inputFilename, mlir::MLIRContext &context,
    mlir::OwningOpRef<ModuleOp> &module, std::string *errorMessage) {
  // Decide if the input file is an ONNX model or a model specified
  // in MLIR. The extension of the file is the decider.
  std::string extension =
      inputFilename.substr(inputFilename.find_last_of(".") + 1);
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
    mlir::MLIRContext &context, mlir::OwningOpRef<ModuleOp> &module) {
  ImportOptions options;
  options.useOnnxModelTypes = useOnnxModelTypes;
  options.invokeOnnxVersionConverter = invokeOnnxVersionConverter;
  options.shapeInformation = shapeInformation;
  ImportFrontendModelArray(onnxBuffer, bufferSize, context, module, options);
}

void outputCode(mlir::OwningOpRef<ModuleOp> &module, std::string filename,
    std::string extension) {
  mlir::OpPrintingFlags flags;
  if (preserveLocations)
    flags.enableDebugInfo();

  std::string errorMessage;
  auto output = openOutputFile(filename + extension, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    exit(1);
  }

  module->print(output->os(), flags);
  output->keep();
}

void emitOutputFiles(std::string outputBaseName,
    EmissionTargetType emissionTarget, mlir::MLIRContext &context,
    mlir::OwningOpRef<ModuleOp> &module) {
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
    std::string modelObjPath = compileModuleToObject(module, outputBaseName);
    if (keepFiles(KeepFilesOfType::MLIR))
      outputCode(module, outputBaseName, ".llvm.mlir");

    if (VerboseOutput)
      printf("Object file %s.o has been compiled.\n", outputBaseName.c_str());
  } break;
  case EmitLib: {
    addCompilerConfig(CCM_SHARED_LIB_DEPS, {"cruntime"});
    std::string sharedLib =
        compileModuleToSharedLibrary(module, outputBaseName);
    if (keepFiles(KeepFilesOfType::MLIR))
      outputCode(module, outputBaseName, ".llvm.mlir");
    if (VerboseOutput)
      printf("Shared library %s has been compiled.\n", sharedLib.c_str());
  } break;
  case EmitJNI: {
    addCompilerConfig(CCM_SHARED_LIB_DEPS, {"jniruntime", "cruntime"});
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
      cleanSourcePM.addNestedPass<FuncOp>(
          onnx_mlir::createElideConstantValuePass());
    if (emissionTarget == EmitMLIR)
      cleanSourcePM.addNestedPass<FuncOp>(
          onnx_mlir::createElideConstGlobalValuePass());

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

void setupModule(mlir::OwningOpRef<ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputBaseName) {
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

void emitOutput(mlir::OwningOpRef<ModuleOp> &module, mlir::MLIRContext &context,
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

int compileModule(mlir::OwningOpRef<ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputBaseName,
    EmissionTargetType emissionTarget) {
  setupModule(module, context, outputBaseName);

  mlir::PassManager pm(&context, mlir::OpPassManager::Nesting::Implicit);
  // Initialize accelerator(s) if required.
  if (!maccel.empty()) {
    onnx_mlir::accel::initAccelerators();
    for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators()) {
      if (!accel->isActive())
        continue;
      accel->getOrLoadDialects(context);
      accel->addPasses(module, pm, emissionTarget);
    }
  } else
    addPasses(module, pm, emissionTarget);
  mlir::applyPassManagerCLOptions(pm);
  mlir::applyDefaultTimingPassManagerCLOptions(pm);

  if (mlir::failed(pm.run(*module)))
    return 4;
  emitOutput(module, context, outputBaseName, pm, emissionTarget);
  return 0;
}
