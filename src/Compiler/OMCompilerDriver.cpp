// OMCompilerDriver.cpp
// Compile program using onnx-mlir, free of LLVM dependences.

#include "src/Compiler/OMCompilerDriver.hpp"

#include <cstdlib>
#include <filesystem>

#include "src/Compiler/Command.hpp"
#include "src/Compiler/DriverUtils.hpp"

namespace fs = std::filesystem;

extern int64_t omCompile(const std::string &inputFilename,
    const std::string &flags, std::string &outputFilename,
    std::string &errorMessage) {

  std::string onnxMlirPath = "onnx-mlir";
#if 0
  const char *envDir = std::getenv("ONNX_MLIR_HOME");
  if (envDir && fs::exists(envDir))
    onnxMlirPath = std::string(envDir) + "/bin/" + onnxMlirPath;
#endif

  Command compile(onnxMlirPath, /*verbose*/ true);
  std::vector<std::string> flagVect = parseFlags(flags);
  compile.appendList(flagVect);
  compile.appendStr(inputFilename);
  compile.print();
  int rc = compile.exec();
  if (rc != 0) {
    errorMessage = "Compiler failed with error code " + std::to_string(rc);
    outputFilename = "";
    return rc;
  }
  errorMessage = "";
  outputFilename = getOutputFilename(inputFilename, flagVect);
  return 0;
}

// clang++ ../src/Compiler/OMCompilerDriver.cpp ../src/Compiler/Command.cpp -I
// /Users/alexe/OM/onnx-mlir  -o alextest
int main() {
  std::string outputFilename, errorMessage;
  std::string model = "add1.mlir";
  int64_t results = omCompile(
      model, "-O3 -march=arm64  -o=bibi", outputFilename, errorMessage);
  fprintf(stderr, "  generated %s with error %d\n", outputFilename.c_str(), (int) results);
  results = omCompile(
      model, "-O3 -march=arm64  -o bobo/bibi", outputFilename, errorMessage);
        fprintf(stderr, "  generated %s with error %d\n", outputFilename.c_str(), (int) results);
  results = omCompile(
      model, "-O3 -march=arm64  ", outputFilename, errorMessage);
        fprintf(stderr, "  generated %s with error %d\n", outputFilename.c_str(), (int) results);

  return results;
}