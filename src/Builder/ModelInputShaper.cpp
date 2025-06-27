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

#include <algorithm>
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
        dims.emplace_back(ModelInputShaper::kUserDynamic);
      forced_inputs_dims_.emplace(std::stoi(inputString), dims);
    }
    // Default to the all inputs and dimensions.
    if (forced_inputs_dims_.empty())
      forced_inputs_dims_.emplace(ModelInputShaper::kUserDynamic,
          std::vector<int>(1, ModelInputShaper::kUserDynamic));
  }
}

void ModelInputShaper::setShapeInformation(
    const std::string &shapeInformation) {
  if (!shapeInformation.empty()) {
    std::stringstream shapeInfoString(shapeInformation);
    std::string shapeString;
    while (std::getline(shapeInfoString, shapeString, INPUT_SEP)) {
      size_t pos = shapeString.find(INPUT_DIM_SEP);
      std::string inputString = shapeString.substr(0, pos);
      std::string dimString = shapeString.substr(pos + 1);

      // Parse dimString.
      std::stringstream dimSizes(dimString);
      std::string dimStr;
      std::vector<int64_t> dims;
      while (std::getline(dimSizes, dimStr, DIM_SEP)) {
        int64_t dimSize = std::stoi(dimStr);
        assert((dimSize == ModelInputShaper::kUserDynamic || dimSize > 0) &&
               "dim must be -1 or > 0");
        if (dimSize == ModelInputShaper::kUserDynamic)
          dimSize = ShapedType::kDynamic;
        dims.emplace_back(dimSize);
      }

      // Parse inputString.
      assert(std::count(inputString.begin(), inputString.end(),
                 INPUT_RANGE_SEP) <= 1 &&
             "input_id is invalid");
      // Check if users input a range or not.
      size_t rangePos = inputString.find(INPUT_RANGE_SEP);
      std::string startString = inputString.substr(0, rangePos);
      std::string endString = inputString.substr(rangePos + 1);
      assert(endString != "" && "input_id has _ at the end");
      bool isRangeInput = (startString != "");
      // Insert (input_id, dim_value) to the shape info.
      SmallVector<int64_t> inputIDs;
      if (isRangeInput) {
        int64_t startID = std::stoi(startString);
        int64_t endID = std::stoi(endString);
        assert((startID >= 0) && "start_id must be >= 0");
        assert((endID >= 0) && "end_id must be >= 0");
        for (int64_t i = startID; i <= endID; ++i)
          inputIDs.emplace_back(i);
      } else {
        int64_t inputID = std::stoi(inputString);
        assert((inputID >= 0 || inputID == kUserAllInputs) &&
               "input_id must be -1 or >= 0");
        inputIDs.emplace_back(inputID);
      }
      for (int64_t inputID : inputIDs) {
        // The semantics of c++ map.insert() makes sure that only the first
        // setting of inputID is inserted.
        inputs_shape_information_.insert(std::make_pair(inputID, dims));
      }
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

Type ModelInputShaper::reshape(int inputIndex, Type inputType) const {
  if (auto rankedTensorTy = mlir::dyn_cast<RankedTensorType>(inputType)) {
    ArrayRef<int64_t> origDims = rankedTensorTy.getShape();
    // Update the input dimensions based on internal information.
    if (force_dim_dynamic_enabled_) {
      auto it = forced_inputs_dims_.find(kUserDynamic);
      if (it != forced_inputs_dims_.end())
        return forceShape(rankedTensorTy, it->second);
      it = forced_inputs_dims_.find(inputIndex);
      if (it != forced_inputs_dims_.end())
        return forceShape(rankedTensorTy, it->second);
    }

    // Change to the custom shape if users provide.
    // Support partial custom shape counting from the outermost dimension. For
    // example, if custom shape is 0:3x5, it only changes the first and second
    // dimensions though the input has 3 dimensions. In that case, the third
    // dimension is unchanged.

    // Find the specific input index first.
    auto it = inputs_shape_information_.find(inputIndex);
    if (it == inputs_shape_information_.end()) {
      // Not found the specific input index, find in the -1: for all inputs.
      it = inputs_shape_information_.find(kUserAllInputs);
      if (it == inputs_shape_information_.end()) {
        // Users do not specify same dimensions for all inputs. Give up.
        return inputType;
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
  return inputType;
}

} // namespace onnx_mlir
