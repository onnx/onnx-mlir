#include <functional>
#include <string>

#include "llvm/Support/JSON.h"

using namespace mlir;

namespace onnx_mlir {
class NNPAJsonConfig {
  using OpSetType = DenseSet<Operation *>;

public:
  NNPAJsonConfig(std::string featureKey);

  void saveConfigToFile(llvm::ArrayRef<mlir::Operation *> ops, std::string file,
      function_ref<void(llvm::json::Object *jsonObj, mlir::Operation *op)>
          updateFn);

private:
  std::string featureKey;
  std::string NODE_TYPE_KEY = "node_type";
  std::string ONNX_NODE_NAME_KEY = "onnx_node_name";
};
} // namespace onnx_mlir
