#include <errno.h>
#include <iostream>

#include <OnnxMlirCompiler.h>
#include <OnnxMlirRuntime.h>

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
  // Read compiler options from command line and compile the doc example into a
  // model library.
  char *errorMessage = nullptr;
  char *compiledFilename = nullptr;
  std::string flags = readArgs(argc, argv);
  flags += "-o add_cpp_interface";
  std::cout << "Compile with options \"" << flags << "\"\n";
  int rc = onnx_mlir::omCompileFromFile(
      "add.onnx", flags.c_str(), &compiledFilename, &errorMessage);
  if (rc != onnx_mlir::CompilerSuccess) {
    std::cerr << "Failed to compile add.onnx with error code " << rc;
    if (errorMessage)
      std::cerr << " and message \"" << errorMessage << "\"";
    std::cerr << "." << std::endl;
    free(compiledFilename);
    free(errorMessage);
    return rc;
  }
  std::string libFilename(compiledFilename);
  std::cout << "Compiled succeeded with results in file: " << libFilename
            << std::endl;
  free(compiledFilename);
  free(errorMessage);

  // Prepare the execution session.
  onnx_mlir::ExecutionSession *session;
  try {
    session = new onnx_mlir::ExecutionSession("./" + libFilename);
  } catch (const std::runtime_error &error) {
    std::cerr << "error while creating execution session: " << error.what()
              << " and errno " << errno << std::endl;
    return errno;
  }

  // Get input signature and print it.
  std::string inputSignature;
  try {
    inputSignature = session->inputSignature();
  } catch (const std::runtime_error &error) {
    std::cerr << "error while loading input signature: " << error.what()
              << " and errno " << errno << std::endl;
    return errno;
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
    outputList = session->run(input);
  } catch (const std::runtime_error &error) {
    std::cerr << "error while running model: " << error.what() << " and errno "
              << errno << std::endl;
    return errno;
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
  delete session;
  return 0;
}
