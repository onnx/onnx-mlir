/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- CompilerUtils.cpp -------------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

#include "CompilerUtils.hpp"

#include <ctime>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/GlobalValue.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/ToolOutputFile.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetMachine.h"

#include "src/Accelerators/Accelerator.hpp"
#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Builder/ModelInputShaper.hpp"
#include "src/Compiler/CompilerDialects.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Compiler/HeapReporter.hpp"
#include "src/Version/Version.hpp"

using namespace mlir;
using namespace onnx_mlir;

mlir::DefaultTimingManager timingManager;
mlir::TimingScope rootTimingScope;
namespace onnx_mlir {

// Values to report the current phase of compilation.
// Increase TOTAL_COMPILE_PHASE when having more phases.
uint64_t CURRENT_COMPILE_PHASE = 1;
uint64_t TOTAL_COMPILE_PHASE = 5;

// Make a function that forces preserving all files using the runtime arguments
// and/or the overridePreserveFiles enum.
enum class KeepFilesOfType { All, MLIR, LLVMIR, Bitcode, Object, None };

// Value below override at compile time by effectively setting the requested
// flags.
static constexpr KeepFilesOfType overridePreserveFiles = KeepFilesOfType::None;

static bool keepFiles(KeepFilesOfType preserve) {
  // When wanting to preserve all files, do it regardless of isBitcode.
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

// Append a single string argument.
Command &Command::appendStr(const std::string &arg) {
  if (arg.size() > 0)
    _args.emplace_back(arg);
  return *this;
}

// Append a single optional string argument.
Command &Command::appendStrOpt(const std::optional<std::string> &arg) {
  if (arg.has_value())
    _args.emplace_back(arg.value());
  return *this;
}

// Append a list of string arguments.
Command &Command::appendList(const std::vector<std::string> &args) {
  _args.insert(_args.end(), args.begin(), args.end());
  return *this;
}

// Reset arguments.
Command &Command::resetArgs() {
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
//
// Return 0 on success, error value otherwise.
int Command::exec(std::string wdir) const {
  auto argsRef = std::vector<llvm::StringRef>(_args.begin(), _args.end());

  // If a work directory is specified, save the current work directory
  // and switch into it. Note that if wdir is empty, new_wdir will be
  // cur_wdir.
  llvm::SmallString<8> cur_wdir;
  llvm::SmallString<8> new_wdir(wdir);
  llvm::sys::fs::current_path(cur_wdir);
  llvm::sys::fs::make_absolute(cur_wdir, new_wdir);
  std::error_code ec = llvm::sys::fs::set_current_path(new_wdir);
  if (ec.value()) {
    llvm::errs() << llvm::StringRef(new_wdir).str() << ": " << ec.message()
                 << "\n";
    return ec.value();
  }

  if (VerboseOutput)
    llvm::outs() << "[" << llvm::StringRef(new_wdir).str() << "] " << _path
                 << ": " << llvm::join(argsRef, " ") << "\n";

  std::string errMsg;
  int rc = llvm::sys::ExecuteAndWait(_path, llvm::ArrayRef(argsRef),
      /*Env=*/std::nullopt, /*Redirects=*/std::nullopt,
      /*SecondsToWait=*/0, /*MemoryLimit=*/0, &errMsg);

  if (rc != 0) {
    llvm::errs() << llvm::join(argsRef, " ") << "\n"
                 << "Error message: " << errMsg << "\n"
                 << "Program path: " << _path << "\n"
                 << "Command execution failed."
                 << "\n";
    return rc;
  }

  // Restore saved work directory.
  llvm::sys::fs::set_current_path(cur_wdir);
  return 0;
}

void showCompilePhase(std::string msg) {
  time_t rawtime;
  struct tm *timeinfo;
  char buffer[80];

  // Get current date.
  time(&rawtime);
  timeinfo = localtime(&rawtime);
  strftime(buffer, 80, "%c", timeinfo);
  std::string currentTime(buffer);

  llvm::outs() << "[" << CURRENT_COMPILE_PHASE++ << "/" << TOTAL_COMPILE_PHASE
               << "] " << currentTime << " " << msg << "\n";
}

} // namespace onnx_mlir

namespace onnx_mlir {
// =============================================================================
// Methods for compiling and file processing.

static void loadMLIR(std::string inputFilename, mlir::MLIRContext &context,
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

  // Set shape information if required.
  // Only set shape if the module has a single function.
  uint64_t numOfFuncOp = 0;
  func::FuncOp funcOp;
  module->walk([&](func::FuncOp f) {
    funcOp = f;
    numOfFuncOp++;
  });
  if ((numOfFuncOp == 1) && (!shapeInformation.empty())) {
    ModelInputShaper modelInputShaper_;
    modelInputShaper_.setShapeInformation(shapeInformation);
    auto funcType = dyn_cast<FunctionType>(funcOp.getFunctionType());
    ArrayRef<Type> argTypes = funcType.getInputs();
    SmallVector<Type, 4> newArgTypes;
    for (uint64_t i = 0; i < argTypes.size(); ++i) {
      Type argTy = argTypes[i];
      // Get user's shape information.
      argTy = modelInputShaper_.reshape(i, argTy);
      // Update the arguments.
      funcOp.getBody().back().getArgument(i).setType(argTy);
      newArgTypes.emplace_back(argTy);
    }
    // Update the function type.
    FunctionType newType =
        FunctionType::get(&context, newArgTypes, funcType.getResults());
    funcOp.setType(newType);
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
      llvm::MDString::get(ctx, getOnnxMlirCommitVersion())};
  identMetadata->addOperand(llvm::MDNode::get(ctx, identNode));

#ifdef PRODUCT_VERSION_MAJOR
  int32_t ProductVersion = PRODUCT_VERSION_MAJOR;
  llvmModule.addModuleFlag(
      llvm::Module::Warning, "Product Major Version", ProductVersion);
#endif
#ifdef PRODUCT_VERSION_MINOR
  int32_t ProductRelease = PRODUCT_VERSION_MINOR;
  llvmModule.addModuleFlag(
      llvm::Module::Warning, "Product Minor Version", ProductRelease);
#endif
#ifdef PRODUCT_VERSION_PATCH
  int32_t ProductPatch = PRODUCT_VERSION_PATCH;
  llvmModule.addModuleFlag(
      llvm::Module::Warning, "Product Patchlevel", ProductPatch);
#endif
#ifdef PRODUCT_ID
  llvmModule.addModuleFlag(llvm::Module::Warning, "Product Id",
      llvm::MDString::get(ctx, PRODUCT_ID));
#endif

  // Annotate functions to be accessible from DLL on Windows.
#ifdef _WIN32
  SmallVector<StringRef, 4> exportedFuncs;
  std::string tag = "";
  assert(!modelTag.empty() && "Model tag was not set");
  if (!StringRef(modelTag).equals_insensitive("NONE"))
    tag = "_" + modelTag;
  // Signature functions.
  exportedFuncs.emplace_back(StringRef("omInputSignature"));
  exportedFuncs.emplace_back(StringRef("omOutputSignature"));
  exportedFuncs.emplace_back(StringRef("omQueryEntryPoints"));
  if (!tag.empty()) {
    exportedFuncs.emplace_back(StringRef("omInputSignature" + tag));
    exportedFuncs.emplace_back(StringRef("omOutputSignature" + tag));
    exportedFuncs.emplace_back(StringRef("omQueryEntryPoints" + tag));
  }
  // Entry point fuctions.
  if (llvm::GlobalVariable *GV =
          llvmModule.getNamedGlobal(StringRef("_entry_point_arrays" + tag))) {
    if (GV->isConstant() && GV->hasDefinitiveInitializer()) {
      llvm::Constant *initializer = GV->getInitializer();
      llvm::ArrayType *AT = dyn_cast<llvm::ArrayType>(initializer->getType());
      for (uint64_t i = 0; i < AT->getNumElements() - 1; ++i) {
        llvm::GlobalVariable *entryGV = llvmModule.getNamedGlobal(
            StringRef("_entry_point_" + std::to_string(i) + tag));
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

// Extend the input filename (with possibly a path but no extention) by the
// extention generated by the given emission target type. Names may be different
// depending on the underlying machine and/or operating system.
std::string getTargetFilename(
    const std::string filenameNoExt, EmissionTargetType target) {
  switch (target) {

#ifdef _WIN32
  case EmitObj:
    return filenameNoExt + ".obj";
  case EmitLib:
    return filenameNoExt + ".dll";
#else
  case EmitObj:
    return filenameNoExt + ".o";
  case EmitLib:
    return filenameNoExt + ".so";
#endif

  case EmitJNI:
    return filenameNoExt + ".jar";
  case EmitLLVMIR:
  case EmitONNXBasic:
  case EmitONNXIR:
  case EmitMLIR:
    return filenameNoExt + ".onnx.mlir";
  }
  llvm_unreachable("all cases should be handled in switch");
}

// Write LLVM optimized bitcode.
// Returns 0 on success, error code on failure.
static int genLLVMBitcode(const mlir::OwningOpRef<ModuleOp> &module,
    std::string outputNameNoExt, std::string optimizedBitcodeNameWithExt) {
  std::string msg =
      "Translating MLIR Module to LLVM and Generating LLVM Optimized Bitcode";
  showCompilePhase(msg);
  auto llvmTiming = rootTimingScope.nest("[onnx-mlir] " + msg);
  std::error_code error;

  // Write bitcode to a file.
  std::string unoptimizedBitcodeNameWithExt =
      outputNameNoExt + ".unoptimized.bc";
  llvm::FileRemover unoptimizedBitcodeRemover(
      unoptimizedBitcodeNameWithExt, !keepFiles(KeepFilesOfType::Bitcode));

  // outputNameNoExt might contain a directory, which must exist.
  // Otherwise, a "No such file or directory" error will be returned.
  llvm::raw_fd_ostream moduleBitcodeStream(
      unoptimizedBitcodeNameWithExt, error, llvm::sys::fs::OF_None);
  if (error) {
    llvm::errs() << unoptimizedBitcodeNameWithExt << ": " << error.message()
                 << "\n";
    return InvalidTemporaryFileAccess;
  }

  llvm::LLVMContext llvmContext;
  mlir::registerBuiltinDialectTranslation(*(module.get().getContext()));
  mlir::registerLLVMDialectTranslation(*(module.get().getContext()));
  std::unique_ptr<llvm::Module> llvmModule =
      mlir::translateModuleToLLVMIR(*module, llvmContext);
  if (!llvmModule) {
    llvm::errs() << "Failed to translate module to LLVMIR.\n";
    return CompilerFailureInMLIRToLLVM;
  }

  // Tailor LLVMIR to add features that cannot be done with MLIR LLVMIR.
  tailorLLVMIR(*llvmModule);

  // Write LLVMIR to a file.
  if (keepFiles(KeepFilesOfType::LLVMIR)) {
    std::string llvmirNameWithExt = outputNameNoExt + ".ll";
    llvm::FileRemover llvmirRemover(
        llvmirNameWithExt, !keepFiles(KeepFilesOfType::LLVMIR));
    llvm::raw_fd_ostream moduleLLVMIRStream(
        llvmirNameWithExt, error, llvm::sys::fs::OF_None);
    if (error) {
      llvm::errs() << llvmirNameWithExt << ": " << error.message() << "\n";
      return InvalidTemporaryFileAccess;
    }
    llvmModule->print(moduleLLVMIRStream, nullptr);
    moduleLLVMIRStream.flush();
  }

  // Write unoptimized bitcode to a file.
  llvm::WriteBitcodeToFile(*llvmModule, moduleBitcodeStream);
  moduleBitcodeStream.flush();

  // Use the LLVM's 'opt' command to optimize the bitcode.
  std::string optPath = getToolPath("opt");
  Command optBitcode(/*exePath=*/optPath);
  setXoptOption({"--code-model", modelSizeStr[modelSize]});
  int rc = optBitcode.appendStr(getOptimizationLevelOption())
               .appendStr(getTargetTripleOption())
               .appendStr(getTargetArchOption())
               .appendStr(getTargetCPUOption())
               .appendList(getXoptOption())
               .appendStr(getLLVMOption())
               .appendList({"-o", optimizedBitcodeNameWithExt})
               .appendStr(unoptimizedBitcodeNameWithExt)
               .exec();
  return rc != 0 ? CompilerFailureInLLVMOpt : CompilerSuccess;
}

// Compile LLVM bitcode to object file.
// Return 0 on success, error code on failure.
static int genModelObject(
    std::string bitcodeNameWithExt, std::string &modelObjNameWithExt) {
  std::string msg = "Generating Object from LLVM Bitcode";
  showCompilePhase(msg);
  auto objectTiming = rootTimingScope.nest("[onnx-mlir] " + msg);
  std::string llcPath = getToolPath("llc");
  Command llvmToObj(/*exePath=*/llcPath);
  setXllcOption({"--code-model", modelSizeStr[modelSize]});
  int rc = llvmToObj.appendStr(getOptimizationLevelOption())
               .appendStr(getTargetTripleOption())
               .appendStr(getTargetArchOption())
               .appendStr(getTargetCPUOption())
               .appendList(getXllcOption())
               .appendStr(getLLVMOption())
               .appendStr("-filetype=obj")
               .appendStr("-relocation-model=pic")
               .appendList({"-o", modelObjNameWithExt})
               .appendStr(bitcodeNameWithExt)
               .exec();
  return rc != 0 ? CompilerFailureInLLVMToObj : CompilerSuccess;
}

// Return 0 on success, error code on failure.
static int genJniObject(const mlir::OwningOpRef<ModuleOp> &module,
    std::string jniSharedLibPath, std::string jniObjPath) {
  std::string msg = "Generating JNI Object";
  showCompilePhase(msg);
  auto jniTiming = rootTimingScope.nest("[onnx-mlir] " + msg);
  Command ar(/*exePath=*/getToolPath("ar", true));
  int rc = ar.appendStr("x")
               // old version of ar does not support --output so comment out
               // for now and use the optional wdir for exec() to get around
               // the problem.
               //.appendStr("--output")
               //.appendStr(llvm::sys::path::parent_path(jniObjPath).str())
               .appendStr(jniSharedLibPath)
               .appendStr(llvm::sys::path::filename(jniObjPath).str())
               .exec(llvm::sys::path::parent_path(jniObjPath).str());
  return rc != 0 ? CompilerFailureInGenJniObj : CompilerSuccess;
}

// Link everything into a shared object.
// Return 0 on success, error code on failure.
static int genSharedLib(std::string sharedLibNameWithExt,
    std::vector<std::string> opts, std::vector<std::string> objs,
    std::vector<std::string> libs, std::vector<std::string> libDirs) {
  std::string msg = "Linking and Generating the Output Shared Library";
  showCompilePhase(msg);
  auto sharedLibTiming = rootTimingScope.nest("[onnx-mlir] " + msg);
#ifdef _WIN32
  std::vector<std::string> outputOpt = {"/Fe:" + sharedLibNameWithExt};
  // link has to be before libpath since they need to be passed through to the
  // linker
  std::vector<std::string> sharedLibOpts = {"/LD", "/link", "/NOLOGO"};

  llvm::for_each(libs, [](std::string &lib) { lib = lib + ".lib"; });
  llvm::for_each(libDirs,
      [](std::string &libDir) { libDir = "/libpath:\"" + libDir + "\""; });
#else
  std::vector<std::string> outputOpt = {"-o", sharedLibNameWithExt};
  std::vector<std::string> sharedLibOpts = {"-shared", "-fPIC"};
  llvm::for_each(libs, [](std::string &lib) { lib = "-l" + lib; });
  llvm::for_each(libDirs, [](std::string &libDir) { libDir = "-L" + libDir; });
#ifdef __s390x__
  llvm::SmallString<64> lds;
  if (modelSize == ModelSize::large) {
    if (auto ec =
            llvm::sys::fs::createTemporaryFile("s390x-lrodata", "ld", lds)) {
      llvm::errs() << ec.message() << "\n";
      return CompilerFailureInObjToLib;
    }

    std::string ldScript = std::string(lds);
    std::ofstream ofs(ldScript);
    ofs << getToolPath("lrodataScript", true);
    ofs.close();
    sharedLibOpts.push_back("-Wl,-T," + ldScript);
  }
  llvm::FileRemover ldsRemover(lds);
#endif
#endif

  Command link(getToolPath("cxx", true));
  int rc = link.appendList(opts)
               .appendList(objs)
               .appendList(outputOpt)
               .appendList(sharedLibOpts)
               .appendList(libDirs)
               .appendList(libs)
               .exec();
  return rc != 0 ? CompilerFailureInObjToLib : CompilerSuccess;
}

// Create jar containing java runtime and model shared library (which includes
// jni runtime).
// Return 0 on success, error code on failure.
static int genJniJar(const mlir::OwningOpRef<ModuleOp> &module,
    std::string modelSharedLibPath, std::string modelJniJarPath) {
  std::string msg = "Creating JNI Jar";
  showCompilePhase(msg);
  auto jniJarTiming = rootTimingScope.nest("[onnx-mlir] " + msg);
  llvm::SmallString<8> libraryPath(getLibraryPath());
  llvm::sys::path::append(libraryPath, "javaruntime.jar");
  std::string javaRuntimeJarPath = llvm::StringRef(libraryPath).str();

  // Copy javaruntime.jar to model jar.
  llvm::sys::fs::copy_file(javaRuntimeJarPath, modelJniJarPath);

  // Add shared library to model jar.
  Command jar(getToolPath("jar", true));
  int rc =
      jar.appendStr("uf")
          .appendStr(modelJniJarPath)
          .appendStr("-C")
          .appendStr(llvm::sys::path::parent_path(modelSharedLibPath).str())
          .appendStr(llvm::sys::path::filename(modelSharedLibPath).str())
          .exec();
  return rc != 0 ? CompilerFailureInGenJni : CompilerSuccess;
}

// Return 0 on success, error code on failure
static int compileModuleToObject(const mlir::OwningOpRef<ModuleOp> &module,
    std::string outputNameWithoutExt, std::string &objectNameWithExt) {
  std::string bitcodeNameWithExt = outputNameWithoutExt + ".bc";
  int rc = genLLVMBitcode(module, outputNameWithoutExt, bitcodeNameWithExt);
  if (rc != CompilerSuccess)
    return rc;
  llvm::FileRemover bitcodeRemover(
      bitcodeNameWithExt, !keepFiles(KeepFilesOfType::Bitcode));
  objectNameWithExt = getTargetFilename(outputNameWithoutExt, EmitObj);
  return genModelObject(bitcodeNameWithExt, objectNameWithExt);
}

// Return 0 on success, error code on failure
static int compileModuleToSharedLibrary(
    const mlir::OwningOpRef<ModuleOp> &module, std::string outputNameNoExt,
    std::string &libNameWithExt) {
  std::string modelObjNameWithExt;
  int rc = compileModuleToObject(module, outputNameNoExt, modelObjNameWithExt);
  if (rc != CompilerSuccess)
    return rc;
  llvm::FileRemover modelObjRemover(
      modelObjNameWithExt, !keepFiles(KeepFilesOfType::Object));
  libNameWithExt = getTargetFilename(outputNameNoExt, EmitLib);
  return genSharedLib(libNameWithExt, {}, {modelObjNameWithExt},
      getCompilerConfig(CCM_SHARED_LIB_DEPS),
      getCompilerConfig(CCM_SHARED_LIB_PATH_DEPS));
}

// Return 0 on success, error code on failure
static int compileModuleToJniJar(
    const mlir::OwningOpRef<ModuleOp> &module, std::string outputNameNoExt) {
  std::string modelObjNameWithExt;
  int rc = compileModuleToObject(module, outputNameNoExt, modelObjNameWithExt);
  if (rc != CompilerSuccess)
    return rc;
  llvm::FileRemover modelObjRemover(
      modelObjNameWithExt, !keepFiles(KeepFilesOfType::Object));

  StringRef outputDir = llvm::sys::path::parent_path(outputNameNoExt);
  if (outputDir.empty())
    outputDir = StringRef(".");

  std::string jniSharedLibPath = getLibraryPath() + "/libjniruntime.a";

  llvm::SmallString<8> jniObjDir(outputDir);
  llvm::sys::path::append(jniObjDir, "jnidummy.c.o");
  std::string jniObjPath = llvm::StringRef(jniObjDir).str();

  rc = genJniObject(module, jniSharedLibPath, jniObjPath);
  if (rc != CompilerSuccess)
    return rc;
  llvm::FileRemover jniObjRemover(
      jniObjPath, !keepFiles(KeepFilesOfType::Object));

  llvm::SmallString<8> jniLibDir(outputDir);
  llvm::sys::path::append(jniLibDir, "libmodel");
  std::string jniLibBase = llvm::StringRef(jniLibDir).str();

#if defined(__APPLE__) && defined(__clang__)
#define NOEXECSTACK                                                            \
  {}
#else
#define NOEXECSTACK                                                            \
  { "-z", "noexecstack" }
#endif
  std::string modelSharedLibPath = getTargetFilename(jniLibBase, EmitLib);
  rc = genSharedLib(modelSharedLibPath, NOEXECSTACK,
      {modelObjNameWithExt, jniObjPath}, getCompilerConfig(CCM_SHARED_LIB_DEPS),
      getCompilerConfig(CCM_SHARED_LIB_PATH_DEPS));
  if (rc != CompilerSuccess)
    return rc;
  llvm::FileRemover modelSharedLibRemover(
      modelSharedLibPath, !keepFiles(KeepFilesOfType::Object));

  std::string modelJniJarPath = getTargetFilename(outputNameNoExt, EmitJNI);
  return genJniJar(module, modelSharedLibPath, modelJniJarPath);
}

void loadDialects(mlir::MLIRContext &context) {
  context.appendDialectRegistry(registerDialects(maccel));
  context.loadAllAvailableDialects();
}

namespace {
std::string dirName(StringRef inputFilename) {
  llvm::SmallVector<char> path(inputFilename.begin(), inputFilename.end());
  llvm::sys::path::remove_filename(path);
  return std::string(path.data(), path.size());
}
} // namespace

// Return 0 on success, error number on failure.
int processInputFile(StringRef inputFilename, mlir::MLIRContext &context,
    mlir::OwningOpRef<ModuleOp> &module, std::string *errorMessage) {
  // Decide if the input file is an ONNX model (either ONNX protobuf, ONNX text,
  // or JSON) or a model specified in MLIR.
  // The extension of the file is the decider.
  bool inputIsSTDIN = (inputFilename == "-");
  bool inputIsONNX = inputFilename.ends_with(".onnx");
  bool inputIsONNXText = inputFilename.ends_with(".onnxtext");
  bool inputIsJSON = inputFilename.ends_with(".json");
  bool inputIsMLIR = inputFilename.ends_with(".mlir");

  if (!inputIsSTDIN && !inputIsONNX && !inputIsONNXText && !inputIsJSON &&
      !inputIsMLIR) {
    *errorMessage = "Invalid input file \"" + inputFilename.str() +
                    "\": Either an ONNX model (.onnx or .onnxtext or .json), "
                    "or an MLIR file (.mlir) needs to be provided.";
    return InvalidInputFile;
  }

  if (inputIsSTDIN || inputIsONNX || inputIsONNXText || inputIsJSON) {
    ImportOptions options;
    options.verboseOutput = VerboseOutput;
    options.useOnnxModelTypes = useOnnxModelTypes;
    options.invokeOnnxVersionConverter = invokeOnnxVersionConverter;
    options.shapeInformation = shapeInformation;
    options.dimParams = dimParams;
    options.allowSorting = allowSorting;
    options.externalDataDir = dirName(inputFilename);
    options.functionsToDecompose.insert(options.functionsToDecompose.end(),
        functionsToDecompose.begin(), functionsToDecompose.end());
    return ImportFrontendModelFile(
        inputFilename, context, module, errorMessage, options);
  } else if (inputIsMLIR)
    loadMLIR(inputFilename.str(), context, module);
  return CompilerSuccess;
}

// Return 0 on success, error code on error.
int processInputArray(const void *onnxBuffer, int bufferSize,
    mlir::MLIRContext &context, mlir::OwningOpRef<ModuleOp> &module,
    std::string *errorMessage) {
  ImportOptions options;
  options.useOnnxModelTypes = useOnnxModelTypes;
  options.invokeOnnxVersionConverter = invokeOnnxVersionConverter;
  options.allowSorting = allowSorting;
  options.shapeInformation = shapeInformation;
  return ImportFrontendModelArray(
      onnxBuffer, bufferSize, context, module, errorMessage, options);
}

static void outputModule(mlir::OwningOpRef<ModuleOp> &module, raw_ostream &os,
    int64_t largeElementLimit = -1) {
  mlir::OpPrintingFlags flags;
  if (preserveLocations)
    flags.enableDebugInfo();
  if (largeElementLimit >= 0) {
    flags.elideLargeElementsAttrs(largeElementLimit);
    flags.elideLargeResourceString(largeElementLimit);
  }
  module->print(os, flags);
}

// Return 0 on success, error code on error.
int outputCode(mlir::OwningOpRef<ModuleOp> &module, std::string filenameWithExt,
    int64_t largeElementLimit) {
  std::string errorMessage;
  auto output = openOutputFile(filenameWithExt, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return InvalidOutputFileAccess;
  }
  outputModule(module, output->os(), largeElementLimit);
  output->keep();
  return CompilerSuccess;
}

// Return 0 on success, error code on failure.
static int emitOutputFiles(std::string outputNameNoExt,
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
    std::string modelObjNameWithExt;
    int rc =
        compileModuleToObject(module, outputNameNoExt, modelObjNameWithExt);
    if (rc != CompilerSuccess)
      return rc;
    if (keepFiles(KeepFilesOfType::MLIR)) {
      rc = outputCode(module, outputNameNoExt + ".llvm.mlir");
      if (rc != CompilerSuccess)
        return rc;
    }
    if (VerboseOutput)
      printf(
          "Object file '%s' has been compiled.\n", modelObjNameWithExt.c_str());
  } break;
  case EmitLib: {
    std::string sharedLibNameWithExt;
    int rc = compileModuleToSharedLibrary(
        module, outputNameNoExt, sharedLibNameWithExt);
    if (rc != CompilerSuccess)
      return rc;
    if (keepFiles(KeepFilesOfType::MLIR)) {
      rc = outputCode(module, outputNameNoExt + ".llvm.mlir");
      if (rc != CompilerSuccess)
        return rc;
    }
    if (VerboseOutput)
      printf("Shared library '%s' has been compiled.\n",
          sharedLibNameWithExt.c_str());
  } break;
  case EmitJNI: {
    int rc = compileModuleToJniJar(module, outputNameNoExt);
    if (rc != CompilerSuccess)
      return rc;
    if (keepFiles(KeepFilesOfType::MLIR)) {
      rc = outputCode(module, outputNameNoExt + ".llvm.mlir");
      if (rc != CompilerSuccess)
        return rc;
    }
    if (VerboseOutput)
      printf(
          "JNI archive '%s.jar' has been compiled.\n", outputNameNoExt.c_str());
  } break;
  default: {
    // Emit the version with all constants included.
    std::string ouputNameWithExt =
        getTargetFilename(outputNameNoExt, emissionTarget);
    int rc = outputCode(module, ouputNameWithExt);
    if (VerboseOutput)
      printf("Full MLIR code written to: \n\t%s\n\n", ouputNameWithExt.c_str());
    if (rc != CompilerSuccess)
      return rc;

    // Elide element attributes if larger than 100.
    if (emissionTarget == EmitONNXBasic || emissionTarget == EmitONNXIR ||
        emissionTarget == EmitMLIR) {
      std::string tempNameWithExt = outputNameNoExt + ".tmp";
      int rc = outputCode(module, tempNameWithExt, /*largeElementLimit=*/100);
      if (VerboseOutput) {
        printf("Constant-free MLIR Code written to: \n\t%s\n\n",
            tempNameWithExt.c_str());
        printf("Use:\n\t%s\nto continue lowering the code to other dialects.\n",
            ouputNameWithExt.c_str());
      }
      if (rc != CompilerSuccess)
        return rc;
    }
  }
  }
  return CompilerSuccess;
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

/// Return the module datalayout string. The datalayout string is determined
/// by creating a target machine using the target triple and target cpu.
static std::string getDataLayout(const Location &loc) {
  const llvm::Target &LLVMTarget = *getLLVMTarget(mtriple, loc);
  llvm::TargetOptions ops;
  auto targetMachine =
      std::unique_ptr<llvm::TargetMachine>{LLVMTarget.createTargetMachine(
          mtriple, mcpu, "" /*features*/, ops, std::nullopt)};
  if (!targetMachine) {
    emitError(loc, "failed to create target machine");
    return nullptr;
  }

  const llvm::DataLayout &dl = targetMachine->createDataLayout();
  std::string dataLayoutString = dl.getStringRepresentation();
  assert(dataLayoutString != "" && "Expecting a valid target datalayout");

  return dataLayoutString;
}

// Return 0 on success, error code on failure.
static int setupModule(mlir::OwningOpRef<ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputNameNoExt) {
  // Initialize the targets support for all targets LLVM was configured for.
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmPrinters();
  llvm::InitializeAllAsmParsers();

  // Set the module target triple and datalayout.
  Operation &moduleOp = *(module->getOperation());
  Location loc = moduleOp.getLoc();
  moduleOp.setAttr(LLVM::LLVMDialect::getTargetTripleAttrName(),
      StringAttr::get(&context, mtriple));
  moduleOp.setAttr(LLVM::LLVMDialect::getDataLayoutAttrName(),
      StringAttr::get(&context, getDataLayout(loc)));

  // Set a tag that will be used to postfix symbols in the generated
  // LLVMIR. By default, use the filename (without extension) of the input onnx
  // model or the value passed to `-o`.
  // This tag makes the symbols unique across multiple generated models.
  // In particular, it will be appended to global variable and function names.
  // For example, we will have two entry points: `run_main_graph` and
  // `run_main_graph_tag`, doing the same computation.
  if (modelTag.empty())
    modelTag = llvm::sys::path::filename(outputNameNoExt).lower();
  // Verify modelTag value.
  if (!StringRef(modelTag).equals_insensitive("NONE") &&
      !std::regex_match(modelTag, std::regex("([0-9a-z_.-]+)"))) {
    llvm::outs() << "Tag is " << modelTag << "\n";
    emitError(loc,
        "Invalid value for --tag. If --tag is not given, it takes "
        "value from the model's filename or -o option. Make sure the tag value "
        "matches regex ([0-9a-z_.-]+)");
    return InvalidCompilerOption;
  }
  if (StringRef(modelTag).equals_insensitive("NONE"))
    moduleOp.setAttr("onnx-mlir.symbol-postfix", StringAttr::get(&context, ""));
  else
    moduleOp.setAttr(
        "onnx-mlir.symbol-postfix", StringAttr::get(&context, modelTag));

  // Set the module target accelerators.
  SmallVector<Attribute, 2> accelsAttr;
  for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators()) {
    std::ostringstream versionNumber;
    versionNumber << std::hex << accel->getVersionNumber();
    std::string accelStr = accel->getName() + "-0x" + versionNumber.str();
    accelsAttr.emplace_back(StringAttr::get(&context, accelStr));
  }
  if (!accelsAttr.empty())
    moduleOp.setAttr("onnx-mlir.accels", ArrayAttr::get(&context, accelsAttr));

  if (keepFiles(KeepFilesOfType::MLIR)) {
    std::string mlirNameWithExt = outputNameNoExt + ".input.mlir";
    int rc = outputCode(module, mlirNameWithExt);
    if (rc != CompilerSuccess)
      return rc;
    module.release();
    loadMLIR(mlirNameWithExt, context, module);
  }
  return CompilerSuccess;
}

static int emitOutput(mlir::OwningOpRef<ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputNameNoExt,
    mlir::PassManager &pm, EmissionTargetType emissionTarget) {
  if (printIR) {
    outputModule(module, llvm::outs());
    return CompilerSuccess;
  }
  return emitOutputFiles(outputNameNoExt, emissionTarget, context, module);
}

// Return 0 on success, error code on error.
int compileModule(mlir::OwningOpRef<ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputNameNoExt,
    EmissionTargetType emissionTarget) {
  std::string msg = "Compiling and Optimizing MLIR Module";
  showCompilePhase(msg);
  auto compileModuleTiming = rootTimingScope.nest("[onnx-mlir] " + msg);

  int rc = setupModule(module, context, outputNameNoExt);
  if (rc != CompilerSuccess)
    return rc;

  configurePasses();

  mlir::PassManager pm(
      module.get()->getName(), mlir::OpPassManager::Nesting::Implicit);
  // TODO(tung): Revise adding passes. The current mechanism does not work if
  // there are multiple accelerators enabled at the same time. It's because
  // each `accel->addPasses` is independent and controls the whole compilation
  // pipeline.
  bool hasAccel = false;
  for (auto *accel : onnx_mlir::accel::Accelerator::getAccelerators()) {
    hasAccel = true;
    accel->addPasses(module, pm, emissionTarget, outputNameNoExt);
  }
  if (!hasAccel)
    addPasses(module, pm, emissionTarget, outputNameNoExt);
  if (!reportHeapBefore.empty() || !reportHeapAfter.empty()) {
    std::string heapLogFileame = outputNameNoExt + ".heap.log";
    pm.addInstrumentation(std::make_unique<HeapReporter>(
        heapLogFileame, reportHeapBefore, reportHeapAfter));
  }
  (void)mlir::applyPassManagerCLOptions(pm);

  if (enableTiming) {
    pm.enableTiming(compileModuleTiming);
  }

  if (mlir::failed(pm.run(*module)))
    return CompilerFailure;
  compileModuleTiming.stop();
  return emitOutput(module, context, outputNameNoExt, pm, emissionTarget);
}

} // namespace onnx_mlir
