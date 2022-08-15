/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- CompilerUtils.hpp -------------------------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding passes and processing input files.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "llvm/Bitcode/BitcodeWriter.h"
#include "llvm/Support/FileUtilities.h"

#include "mlir/Conversion/SCFToControlFlow/SCFToControlFlow.h"
#include "mlir/Conversion/VectorToLLVM/ConvertVectorToLLVM.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "llvm/ADT/None.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"

#include "onnx-mlir/Compiler/OMCompilerTypes.h"

#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"
#include "src/Builder/FrontendDialectTransformer.hpp"
#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerPasses.hpp"
#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Pass/Passes.hpp"
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

#include "src/Accelerators/Accelerator.hpp"
#include "src/Version/Version.hpp"

namespace onnx_mlir {

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
  //
  // Return 0 on success, error value otherwise.
  int exec(std::string wdir = "") const {
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
      llvm::errs() << "[" << llvm::StringRef(new_wdir).str() << "]" << _path
                   << ": " << llvm::join(argsRef, " ") << "\n";

    std::string errMsg;
    int rc = llvm::sys::ExecuteAndWait(_path, llvm::makeArrayRef(argsRef),
        /*Env=*/llvm::None, /*Redirects=*/llvm::None,
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
};

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
std::string getToolPath(std::string tool);

void registerDialects(mlir::MLIRContext &context);

// ProcessInput* return 0 on success, OnnxMlirCompilerErrorCodes on error.
int processInputFile(std::string inputFilename, mlir::MLIRContext &context,
    mlir::OwningOpRef<mlir::ModuleOp> &module, std::string *errorMessage);
int processInputArray(const void *onnxBuffer, int bufferSize,
    mlir::MLIRContext &context, mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string *errorMessage);

onnx_mlir::InputIRLevelType determineInputIRLevel(
    mlir::OwningOpRef<mlir::ModuleOp> &module);

// Returns 0 on success, OnnxMlirCompilerErrorCodes on failure.
int outputCode(mlir::OwningOpRef<mlir::ModuleOp> &module,
    std::string filenameWithExt, int64_t largeElementLimit = -1);

// Process the input model given by its module and context into an output file
// according to the emission target type. Name of the output file can be
// constructed using the getTargetFilename function below.  When  generating
// libraries or jar files, the compiler will link in lightweight runtimes / jar
// files. If these libraries / jar files are not in the system wide directory
// (typically /usr/local/lib), the user can override the default location using
// the ONNX_MLIR_RUNTIME_DIR environment variable.
// Returns 0 on success,OnnxMlirCompilerErrorCodes on failure.
int compileModule(mlir::OwningOpRef<mlir::ModuleOp> &module,
    mlir::MLIRContext &context, std::string outputNameNoExt,
    onnx_mlir::EmissionTargetType emissionTarget);

// Extend the input filename (with possibly a path but no extention) by the
// extention generated by the given emission target type. Names may be different
// depending on the underlying machine and/or operating system.
std::string getTargetFilename(
    const std::string filenameNoExt, EmissionTargetType target);
} // namespace onnx_mlir