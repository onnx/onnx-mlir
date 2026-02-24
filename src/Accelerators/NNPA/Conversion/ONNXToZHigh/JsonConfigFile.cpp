#include <regex>

#include "mlir/IR/Operation.h"
#include "llvm/ADT/SetOperations.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"

#include "src/Accelerators/NNPA/Conversion/ONNXToZHigh/JsonConfigFile.hpp"

using namespace mlir;

namespace onnx_mlir {

// Global object to ease error reporting, it consumes errors and crash the
// application with a meaningful message.
static llvm::ExitOnError ExitOnErr;

NNPAJsonConfig::NNPAJsonConfig(std::string featureKey)
    : featureKey(featureKey) {}

void NNPAJsonConfig::saveConfigToFile(llvm::ArrayRef<Operation *> ops,
    std::string file,
    function_ref<void(llvm::json::Object *jsonObj, Operation *op)> updateFn) {
  // Parsing the module to JSON object.
  llvm::json::Array jsonArr;
  for (Operation *op : ops) {
    // Create a JSON object for this operation.
    llvm::json::Object jsonObj;
    // Insert user's attribute.
    updateFn(&jsonObj, op);
    if (jsonObj.empty())
      continue;
    // Insert node type.
    std::string nodeTypeStr = op->getName().getStringRef().str();
    jsonObj.insert({NODE_TYPE_KEY, nodeTypeStr});
    std::string nodeNameStr =
        op->getAttrOfType<mlir::StringAttr>(ONNX_NODE_NAME_KEY)
            ? op->getAttrOfType<mlir::StringAttr>(ONNX_NODE_NAME_KEY)
                  .getValue()
                  .str()
            : "";
    // Insert node name.
    jsonObj.insert({ONNX_NODE_NAME_KEY, nodeNameStr});

    jsonArr.emplace_back(llvm::json::Value(std::move(jsonObj)));
  }
  llvm::json::Value featureValue = llvm::json::Value(std::move(jsonArr));

  // Open the config file:
  // - If it already exists, open the file with the offset set to 0.
  // - If it does not already exist, create a new file.
  std::error_code EC;
  llvm::raw_fd_ostream fileOS(
      file, EC, llvm::sys::fs::CreationDisposition::CD_OpenAlways);
  if (EC)
    report_fatal_error(
        "Error when saving to a json file : " + StringRef(EC.message()));
  llvm::json::OStream jsonOS(fileOS, /*IndentSize=*/2);

  // Read the config file to check if it has content or not.
  auto Buf = ExitOnErr(
      errorOrToExpected(llvm::MemoryBuffer::getFile(file, /*bool IsText=*/true,
          /*RequiresNullTerminator=*/false)));

  // Exporting the JSON object to a file.
  if (Buf->getBuffer().empty()) {
    llvm::json::Object jsonContent{{featureKey, featureValue}};
    jsonOS.value(llvm::json::Value(std::move(jsonContent)));
  } else {
    auto jsonFile = ExitOnErr(llvm::json::parse(Buf->getBuffer()));
    llvm::json::Object *jsonExistingContent = jsonFile.getAsObject();
    jsonExistingContent->insert({featureKey, featureValue});
    jsonOS.value(llvm::json::Value(std::move(*jsonExistingContent)));
  }
  jsonOS.flush();
}
} // namespace onnx_mlir
