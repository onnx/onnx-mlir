#include <errno.h>
#include <iostream>

#include <OnnxMlirCompiler.h>
#include <OnnxMlirRuntime.h>

#include "src/Runtime/ExecutionSession.hpp"

int main() {
  // Read compiler options from environment first, and ensure -O3 is used.
  int rc;
  std::cout << "Start compiling add.onnx" << std::endl;
  rc = onnx_mlir::omSetCompilerOptionsFromEnv("TEST_DOC_EXAMPLE");
  if (rc != onnx_mlir::CompilerSuccess) {
    std::cerr << "Failed to process TEST_DOC_EXAMPLE compiler options with "
                 "error code "
              << rc << "." << std::endl;
    return rc;
  }
  rc = onnx_mlir::omSetCompilerOption(
      onnx_mlir::OptionKind::CompilerOptLevel, "3");
  if (rc != onnx_mlir::CompilerSuccess) {
    std::cerr << "Failed to process -O3 compiler option with error code " << rc
              << "." << std::endl;
    return rc;
  }
  // Compile the doc example into a model library.
  const char *errorMessage = NULL;
  const char *compiledFilename;
  rc = onnx_mlir::omCompileFromFile("add.onnx", "add-cppinterface",
      onnx_mlir::EmitLib, &compiledFilename, &errorMessage);
  if (rc != onnx_mlir::CompilerSuccess) {
    std::cerr << "Failed to compile add.onnx with error code " << rc;
    if (errorMessage)
      std::cerr << " and message \"" << errorMessage << "\"";
    std::cerr << "." << std::endl;
    return rc;
  }
  std::string libFilename(compiledFilename);
  std::cout << "Compiled succeeded with results in file: " << libFilename
            << std::endl;

  // Prepare the execution session and get input signature.
  onnx_mlir::ExecutionSession *session;
  try {
    session = new onnx_mlir::ExecutionSession("./" + libFilename);
  } catch (const std::runtime_error &error) {
    std::cerr << "error while creating execution session: " << error.what()
              << " and errno " << errno << std::endl;
    return errno;
  }
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
  OMTensorList *outputList;
  try {
    outputList = session->run(input);
  } catch (const std::runtime_error &error) {
    std::cerr << "error while running model: " << error.what() << " and errno "
              << errno << std::endl;
    return errno;
  }

  // Get the first omt as output.
  OMTensor *y = omTensorListGetOmtByIndex(outputList, 0);
  float *outputPtr = (float *)omTensorGetDataPtr(y);
  // Print its content, should be all 3.
  for (int i = 0; i < 6; i++) {
    std::cout << outputPtr[i];
    if (outputPtr[i] != 3.0) {
      std::cerr << "Iteration " << i << ": expected 3.0, got " << outputPtr[i]
                << "." << std::endl;
      return 100;
    }
  }
  delete session;
  std::cout << std::endl;
  return 0;
}
