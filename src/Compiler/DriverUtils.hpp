#include <string>
#include <vector>

#include "include/onnx-mlir/Compiler/OMCompilerTypes.h"

std::vector<std::string> parseFlags(const std::string &flags);

std::string getOutputFilename(
    const std::string &inputFileName, const std::vector<std::string> &flagVect);
  
std::string getOutputFilename(
    const std::string &filenameNoExt, onnx_mlir::EmissionTargetType target);

