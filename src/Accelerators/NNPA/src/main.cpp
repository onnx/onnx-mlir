//===--------------------------- main.cpp ---------------------------------===//
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//
//

#include "src/Compiler/CompilerUtils.hpp"
#include "src/Compiler/DLCompilerUtils.hpp"
#include "src/Dialect/ZHigh/ZHighOps.hpp"
#include "src/Dialect/ZLow/ZLowOps.hpp"
#include "src/Pass/DLCPasses.hpp"
#include "src/Pass/Passes.hpp"

using namespace std;
using namespace onnx_mlir;

extern llvm::cl::OptionCategory OnnxMlirOptions;
extern llvm::cl::OptionCategory OMPassOptions;
extern llvm::cl::OptionCategory OMDLCPassOptions;

int main(int argc, char *argv[]) {
  mlir::MLIRContext context;
  registerDialects(context);

  llvm::cl::opt<string> inputFilename(llvm::cl::Positional,
      llvm::cl::desc("<input file>"), llvm::cl::init("-"),
      llvm::cl::cat(OnnxMlirOptions));

  llvm::cl::opt<string> outputBaseName("o",
      llvm::cl::desc("Base path for output files, extensions will be added."),
      llvm::cl::value_desc("path"), llvm::cl::cat(OnnxMlirOptions),
      llvm::cl::ValueRequired);
  llvm::cl::opt<EmissionTargetType> emissionTarget(
      llvm::cl::desc("Choose target to emit:"),
      llvm::cl::values(
          clEnumVal(EmitONNXBasic,
              "Ingest ONNX and emit the basic ONNX operations without "
              "inferred shapes."),
          clEnumVal(
              EmitONNXIR, "Ingest ONNX and emit corresponding ONNX dialect."),
          clEnumVal(
              EmitMLIR, "Lower model to MLIR built-in transformation dialect."),
          clEnumVal(EmitLLVMIR, "Lower model to LLVM IR (LLVM dialect)."),
          clEnumVal(EmitObj, "Compile the input into a object file."),
          clEnumVal(EmitLib, "Lower model to LLVM IR, emit (to file) "
                             "LLVM bitcode for model, compile and link it to a "
                             "shared library."),
          clEnumVal(EmitJNI, "Lower model to LLVM IR -> LLVM bitcode "
                             "-> JNI shared library -> jar")),
      llvm::cl::init(EmitLib), llvm::cl::cat(OnnxMlirOptions));

  llvm::cl::opt<DLCEmissionTargetType> dlcEmissionTarget(
      llvm::cl::desc("[Optional] Choose Z-related target to emit "
                     "(once selected it will cancel the other targets):"),
      llvm::cl::values(
          clEnumVal(EmitZHighIR, "Lower model to ZHigh IR (ZHigh dialect)"),
          clEnumVal(EmitZLowIR, "Lower model to ZLow IR (ZLow dialect)"),
          clEnumVal(EmitZNONE, "Do not emit Z-related target (default)")),
      llvm::cl::init(EmitZNONE), llvm::cl::cat(OnnxMlirOptions));

  llvm::cl::list<std::string> execNodesOnCpu{"execNodesOnCpu",
      llvm::cl::desc("Comma-separated list of node names in an onnx graph. The "
                     "specified nodes are forced to run on the CPU instead of "
                     "using the zDNN. The node name is an optional attribute "
                     "in onnx graph, which is `onnx_node_name` in ONNX IR"),
      llvm::cl::CommaSeparated, llvm::cl::ZeroOrMore,
      llvm::cl::cat(OnnxMlirOptions)};

  // llvm::cl::HideUnrelatedOptions(OnnxMlirOptions);
  mlir::registerPassManagerCLOptions();
  llvm::cl::ParseCommandLineOptions(
      argc, argv, "DLC++ modular optimizer driver\n");

  mlir::OwningModuleRef module;
  std::string errorMessage;
  processInputFile(inputFilename, context, module, &errorMessage);
  if (!errorMessage.empty()) {
    printf("%s\n", errorMessage.c_str());
    return 1;
  }

  // Input file base name, replace path if required.
  // outputBaseName must specify a file, so ignore invalid values
  // such as ".", "..", "./", "/.", etc.
  bool b = false;
  if (outputBaseName == "" ||
      (b = std::regex_match(outputBaseName, std::regex("(.*/)*\\.*$")))) {
    if (b)
      printf("Invalid -o option value %s ignored.\n", outputBaseName.c_str());
    outputBaseName = inputFilename.substr(0, inputFilename.find_last_of("."));
  }

  return compileModuleDLC(module, context, outputBaseName, emissionTarget,
      dlcEmissionTarget, execNodesOnCpu);
}
