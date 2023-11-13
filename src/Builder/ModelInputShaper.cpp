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
  char *envInputString = NULL;
  char *envResultString = NULL;
  if (const char *envString = std::getenv("IMPORTER_FORCE_DYNAMIC")) {
    force_dim_dynamic_enabled_ = true;
    envInputString = strdup(envString);
    char *deliminaterPtr = strchr(envInputString, '%');
    envResultString = NULL;
    if (deliminaterPtr != NULL) { // if result part exists
      *deliminaterPtr = (char)0;  // null terminate envInputString
      envResultString = deliminaterPtr + 1;
    }
  }
  setForcedDims(envInputString, &forced_inputs_dims_);
  setForcedDims(envResultString, &forced_results_dims_);
  free(envInputString);
}

void ModelInputShaper::setShapeInformation(
    const std::string &shapeInformation) {
  if (!shapeInformation.empty()) {
    std::stringstream shapeInfoString(shapeInformation);
    std::string shapeString;
    bool hasAllInputSetting = false;
    while (std::getline(shapeInfoString, shapeString, ',')) {
      size_t pos = shapeString.find(':');
      std::string inputString = shapeString.substr(0, pos);
      std::string dimString = shapeString.substr(pos + 1);

      int64_t inputID = std::stoi(inputString);
      assert((inputID >= 0 || inputID == kUserAllInputs) &&
             "input_id must be -1 or >= 0");
      if (inputID == kUserAllInputs)
        hasAllInputSetting = true;

      std::stringstream dimSizes(dimString);
      std::string dimStr;
      std::vector<int64_t> dims;
      while (std::getline(dimSizes, dimStr, 'x')) {
        int64_t dimSize = std::stoi(dimStr);
        assert((dimSize == ModelInputShaper::kUserDynamic || dimSize > 0) &&
               "dim must be -1 or > 0");
        if (dimSize == ModelInputShaper::kUserDynamic)
          dimSize = ShapedType::kDynamic;
        dims.emplace_back(dimSize);
      }
      // The semantics of c++ map.insert() makes sure that only the first
      // setting of inputID is inserted.
      inputs_shape_information_.insert(std::make_pair(inputID, dims));
    }
    if (hasAllInputSetting && (inputs_shape_information_.size() > 1)) {
      llvm::outs()
          << "Warning: Found multiple settings that includes -1:d1xd2x...xdn "
             "for all inputs. Only the first -1:d1xd2x...xdn is effective and "
             "the other settings are ignored.\n";
    }
  }
}

namespace {
RankedTensorType forceShape(
    RankedTensorType tensorTy, const std::vector<int> &forcedDims) {
  auto shape = tensorTy.getShape();
  llvm::SmallVector<int64_t, 4> newDims;
  for (unsigned int i = 0; i < shape.size(); i++) {
    if (llvm::is_contained(forcedDims, ModelInputShaper::kUserDynamic) ||
        llvm::is_contained(forcedDims, i)) {
      newDims.push_back(ShapedType::kDynamic);
    } else {
      newDims.push_back(shape[i]);
    }
  }
  return RankedTensorType::get(newDims, tensorTy.getElementType());
}
} // namespace

void ModelInputShaper::setForcedDims(
    char *envString, std::map<int, std::vector<int>> *forced_dims) {
  if (envString != NULL) {
    std::stringstream envStringStream;
    envStringStream << envString;
    std::string dynamicString;
    while (std::getline(envStringStream, dynamicString, '|')) {
      size_t pos = dynamicString.find(':');
      std::string numString = dynamicString.substr(0, pos);
      std::string dimString = dynamicString.substr(pos + 1);
      std::stringstream dimIndices(dimString);
      std::string dimIndex;
      std::vector<int> dims;
      while (std::getline(dimIndices, dimIndex, ',')) {
        dims.emplace_back(stoi(dimIndex));
      }
      // Default to the all dimensions if dims are not specified.
      if (dims.empty())
        dims.emplace_back(ModelInputShaper::kUserDynamic);
      forced_dims->emplace(std::stoi(numString), dims);
    }
    // Default to the all inputs/results and dimensions.
    if (forced_dims->empty())
      forced_dims->emplace(ModelInputShaper::kUserDynamic,
          std::vector<int>(1, ModelInputShaper::kUserDynamic));
  }
}

Type ModelInputShaper::reshape(int inputIndex, Type inputType) const {
  return ModelInputShaper::reshapeInput(inputIndex, inputType);
}

Type ModelInputShaper::reshapeInput(int inputIndex, Type inputType) const {
  return ModelInputShaper::reshape(
      inputIndex, inputType, forced_inputs_dims_, inputs_shape_information_);
}

Type ModelInputShaper::reshapeResult(int resultIndex, Type resultType) const {
  return ModelInputShaper::reshape(resultIndex, resultType,
      forced_results_dims_, results_shape_information_);
}

Type ModelInputShaper::reshape(int index, Type type,
    std::map<int, std::vector<int>> forced_dims,
    std::map<int64_t, std::vector<int64_t>> shape_information) const {
  if (auto rankedTensorTy = type.dyn_cast<RankedTensorType>()) {
    ArrayRef<int64_t> origDims = rankedTensorTy.getShape();
    // Update the input dimensions based on internal information.
    if (force_dim_dynamic_enabled_) {
      auto it = forced_dims.find(kUserDynamic);
      if (it != forced_dims.end())
        return forceShape(rankedTensorTy, it->second);
      it = forced_dims.find(index);
      if (it != forced_dims.end())
        return forceShape(rankedTensorTy, it->second);
    }

    // Change to the custom shape if users provide.
    // Support partial custom shape counting from the outermost dimension. For
    // example, if custom shape is 0:3x5, it only changes the first and second
    // dimensions though the input has 3 dimensions. In that case, the third
    // dimension is unchanged.
    auto it = shape_information.find(kUserAllInputs);
    if (it == shape_information.end()) {
      // Users do not specify same dimensions for all inputs/results.
      // Find the specific input index.
      it = shape_information.find(index);
      if (it == shape_information.end()) {
        // Not found the specific input index, give up.
        return type;
      }
    }
    SmallVector<int64_t, 4> customDims;
    std::vector<int64_t> userDims = it->second;
    for (uint64_t i = 0; i < origDims.size(); ++i) {
      if (i < userDims.size())
        customDims.emplace_back(userDims[i]);
      else
        customDims.emplace_back(origDims[i]);
    }
    return RankedTensorType::get(customDims, rankedTensorTy.getElementType());
  }

  // Default to not reshape.
  return type;
}

} // namespace onnx_mlir
