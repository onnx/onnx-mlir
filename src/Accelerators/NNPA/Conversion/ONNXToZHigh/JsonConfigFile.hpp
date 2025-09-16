#include <functional>
#include <string>

#include "llvm/Support/JSON.h"

using namespace mlir;

namespace onnx_mlir {
class NNPAJsonConfig {
  using OpSetType = DenseSet<Operation *>;

public:
  NNPAJsonConfig(std::string featureKey);

  /// Check if the json file is empty or not.
  // bool empty();

  /// Load the config file and set operations' attributes using the config
  /// information.
  ///
  /// JSON file example:
  /// ```json
  /// {
  ///   "feature_key": [
  ///     {
  ///       "feature_attribute": "cpu",
  ///       "node_type": "onnx.Relu",
  ///       "onnx_node_name": "Relu_[1,2]"
  ///     },
  ///     {
  ///       "feature_attribute": "nnpa",
  ///       "node_type": "onnx.Sigmoid",
  ///       "onnx_node_name": ".*"
  ///     }
  ///   ]
  /// }
  /// ```
  void loadConfigFromFile(llvm::ArrayRef<mlir::Operation *> ops,
      std::string file,
      function_ref<void(llvm::json::Object *jsonObj, mlir::Operation *op)>
          updateAttrFn);

  void saveConfigToFile(llvm::ArrayRef<mlir::Operation *> ops, std::string file,
      function_ref<void(llvm::json::Object *jsonObj, mlir::Operation *op)>
          updateFn);

private:
  std::string featureKey;
  std::string NODE_TYPE_KEY = "node_type";
  std::string ONNX_NODE_NAME_KEY = "onnx_node_name";
};
} // namespace onnx_mlir
