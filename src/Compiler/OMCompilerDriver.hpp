
// OMCompilerDriver.hpp
// Compile program using onnx-mlir, free of LLVM dependences.

#include <string>
#include <vector>

extern int64_t omCompile(const std::string &inputFilename,
    const std::string &flags, std::string &outputFilename,
    std::string &errorMessage);

extern int64_t omCompileOuputFilename(const std::string &inputFilename,
    const std::string &flags, std::string &outputFilename);

extern int64_t omCompileModelTag(const std::string &flags, std::string tag);

