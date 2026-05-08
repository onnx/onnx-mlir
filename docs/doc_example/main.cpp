#include <iostream>

#define USE_UNIFIED 1
#define USE_CONTAINER 1

#if USE_UNIFIED
#include "src/Compiler/OMUnifiedCompile.hpp"
#else
#include "src/Compiler/OMCompile.hpp"
#endif

#include "src/Runtime/ExecutionSession.hpp"

// Read the arguments from the command line and return a std::string
std::string readArgs(int argc, char *argv[]) {
  std::string commandLineStr;
  for (int i = 1; i < argc; i++) {
    commandLineStr.append(std::string(argv[i]) + " ");
  }
  return commandLineStr;
}

int main(int argc, char *argv[]) {
  // Read compiler options from command line.
  std::string flags = readArgs(argc, argv);
  flags += "-o add_cpp_interface -v";
  // And compile the doc example into a model library.
  #if USE_UNIFIED
  #if USE_CONTAINER
  onnx_mlir::OMUnifiedCompile compile(onnx_mlir::OMUnifiedCompile::ContainerEngine::Podman);
  #else
  onnx_mlir::OMUnifiedCompile compile({}, true);
  #endif
  #else
  onnx_mlir::OMCompile compile;
  #endif

  try {
    // For testing: log the compile output (stderr and stdout) in compile.log.
    compile.compile("add.onnx", flags);
  } catch (const onnx_mlir::OMCompileException &error) {
    std::cerr << error.what() << std::endl;
    return 1;
  }
  std::cout << "Compiled succeeded with results in file: "
            << compile.getOutputFilename() << std::endl;

  // Prepare the execution session.
  onnx_mlir::ExecutionSession session;
  try {
    session.loadModel(compile.getOutputFilename());
  } catch (const onnx_mlir::ExecutionSessionException &error) {
    std::cerr << "error while creating execution session: " << error.what()
              << std::endl;
    return 2;
  }

  // Get input signature and print it.
  std::string inputSignature;
  try {
    inputSignature = session.inputSignature();
  } catch (const onnx_mlir::ExecutionSessionException &error) {
    std::cerr << "error while loading input signature: " << error.what()
              << std::endl;
    return 3;
  }
  std::cout << "Compiled add.onnx model has input signature: \""
            << inputSignature << "\"." << std::endl;

  // Build the inputs, starts with shared shape & rank.
  int64_t shape[] = {3, 2};
  int64_t rank = 2;
  // Construct x1 omt filled with 1.
  float x1Data[] = {1., 1., 1., 1., 1., 1.};
  OMTensor *x1 = omTensorCreate(x1Data, shape, rank, ONNX_TYPE_FLOAT);
  // Construct x2 omt filled with 2.
  float x2Data[] = {2., 2., 2., 2., 2., 2.};
  OMTensor *x2 = omTensorCreate(x2Data, shape, rank, ONNX_TYPE_FLOAT);
  // Construct a list of omts as input.
  OMTensor *list[2] = {x1, x2};
  OMTensorList *input = omTensorListCreate(list, 2);

  // Call the compiled onnx model function.
  std::cout << "Start running model " << std::endl;
  OMTensorList *outputList;
  try {
    outputList = session.runDebug(input, /*debug: catch segfault in handler*/ true);
  } catch (const onnx_mlir::ExecutionSessionException &error) {
    std::cerr << "error while running model: " << error.what() << std::endl;
    return 5;
  }
  std::cout << "Finished running model " << std::endl;

  // Get the first omt as output.
  OMTensor *y = omTensorListGetOmtByIndex(outputList, 0);
  omTensorPrint("Result tensor: ", y);
  std::cout << std::endl;
  float *outputPtr = (float *)omTensorGetDataPtr(y);
  // Print its content, should be all 3.
  for (int i = 0; i < 6; i++) {
    if (outputPtr[i] != 3.0) {
      std::cerr << "Iteration " << i << ": expected 3.0, got " << outputPtr[i]
                << "." << std::endl;
      return 100;
    }
  }
  std::cout << "Model verified successfully" << std::endl;
  return 0;
}
