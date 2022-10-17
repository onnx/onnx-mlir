/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- ModelInputShaper.cpp ---------------------------===//
//
// Helper class to override ONNX model input shapes.
//
//===----------------------------------------------------------------------===//

#include "src/Builder/ModelInputShaper.hpp"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"

#include <cstdlib>
#include <sstream>
#include <string>

using namespace mlir;

namespace onnx_mlir {

ModelInputShaper::ModelInputShaper() : force_dim_dynamic_enabled_(false) {
  if (const char *envInputString = std::getenv("IMPORTER_FORCE_DYNAMIC")) {
    force_dim_dynamic_enabled_ = true;
    std::stringstream envString;
    envString << envInputString;
    std::string dynamicInput;
    while (std::getline(envString, dynamicInput, '|')) {
      size_t pos = dynamicInput.find(':');
      std::string inputString = dynamicInput.substr(0, pos);
      std::string dimString = dynamicInput.substr(pos + 1);

      std::stringstream dimIndices(dimString);
      std::string dimIndex;
      std::vector<int> dims;
      while (std::getline(dimIndices, dimIndex, ',')) {
        dims.emplace_back(stoi(dimIndex));
      }
      // Default to the all dimensions if dims are not specified.
      if (dims.empty())
        dims.emplace_back(-1);
      forced_inputs_dims_.emplace(std::stoi(inputString), dims);
    }
    // Default to the all inputs and dimensions.
    if (forced_inputs_dims_.empty())
      forced_inputs_dims_.emplace(-1, std::vector<int>(1, -1));
  }
}

void ModelInputShaper::setShapeInformation(
    const std::string &shapeInformation) {
  if (!shapeInformation.empty()) {
    std::stringstream shapeInfoString(shapeInformation);
    std::string shapeString;
    while (std::getline(shapeInfoString, shapeString, ',')) {
      size_t pos = shapeString.find(':');
      std::string inputString = shapeString.substr(0, pos);
      std::string dimString = shapeString.substr(pos + 1);

      int64_t inputID = std::stoi(inputString);
      assert(inputID >= 0 && "input_id must be >= 0");

      std::stringstream dimSizes(dimString);
      std::string dimStr;
      std::vector<int64_t> dims;
      while (std::getline(dimSizes, dimStr, 'x')) {
        int64_t dimSize = std::stoi(dimStr);
        assert((dimSize == -1 || dimSize > 0) && "dim must be -1 or > 0");
        dims.emplace_back(dimSize);
      }
      inputs_shape_information_.insert(std::make_pair(inputID, dims));
    }
  }
}

namespace {
RankedTensorType forceShape(
    RankedTensorType tensorTy, const std::vector<int> &forcedDims) {
  auto shape = tensorTy.getShape();
  llvm::SmallVector<int64_t, 4> newDims;
  for (unsigned int i = 0; i < shape.size(); i++) {
    if (llvm::is_contained(forcedDims, -1) ||
        llvm::is_contained(forcedDims, i)) {
      newDims.push_back(-1);
    } else {
      newDims.push_back(shape[i]);
    }
  }
  return RankedTensorType::get(newDims, tensorTy.getElementType());
}
} // namespace

Type ModelInputShaper::reshape(int inputIndex, Type inputType) const {
  if (auto tensorTy = inputType.dyn_cast<TensorType>()) {
    // Make dims unknown (-1) if applicable.
    if (force_dim_dynamic_enabled_ && tensorTy.hasRank()) {
      auto rankedTensorTy = tensorTy.cast<RankedTensorType>();
      auto it = forced_inputs_dims_.find(-1);
      if (it != forced_inputs_dims_.end())
        return forceShape(rankedTensorTy, it->second);
      it = forced_inputs_dims_.find(inputIndex);
      if (it != forced_inputs_dims_.end())
        return forceShape(rankedTensorTy, it->second);
    }

    // Change to the custom shape if users provide.
    auto it = inputs_shape_information_.find(inputIndex);
    if (it != inputs_shape_information_.end())
      return RankedTensorType::get(it->second, tensorTy.getElementType());
  }

  // Default to not reshape.
  return inputType;
}

} // namespace onnx_mlir
