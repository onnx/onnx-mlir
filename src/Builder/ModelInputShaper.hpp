/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------------- ModelInputShaper.hpp ---------------------------===//
//
// Helper class to override ONNX model input shapes.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_MODEL_INPUT_H
#define ONNX_MLIR_MODEL_INPUT_H

#include <map>
#include <string>
#include <vector>

#include "mlir/IR/Types.h"

namespace onnx_mlir {

// Sets shapes of ONNX model inputs.
//
// Reads environment variable IMPORTER_FORCE_DYNAMIC to change input
// shapes to unknown dimension.
// Temporarily added to use the test cases with static shape to test.
// The Backus–Naur Form (BNF) for IMPORTER_FORCE_DYNAMIC is as follows.
//
// <ImportForceDymanicExpr> :== `'` <expr> `'`
//                   <expr> ::= <inputString> | <inputString> `|` <expr>
//             <inputString ::= <inputIndex> `:` <dimString>
//              <dimString> ::= <dimIndex> | <dimIndex> `,` <dimString>
//             <inputIndex> ::= <index>
//               <dimIndex> ::= <index>
//                  <index> ::= -1 | <number>
//                 <number> ::= <digit> | <digit><number>
//                  <digit> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9
//
// Value `-1` semantically represents all inputs or all dimensions, and it
// has the highest priority. E.g. `'0: -1, 0'` means all dimensions of the
// first input will be changed. Input and dimension indices start from 0.
//
// Examples:
// 1. IMPORTER_FORCE_DYNAMIC='-1:-1'
//    - change all dimensions in all inputs to unknown dimensions.
// 2. IMPORTER_FORCE_DYNAMIC='-1:0'
//    - change the first dimension in all inputs to unknown dimensions.
// 3. IMPORTER_FORCE_DYNAMIC='1:-1'
//    - change all dimensions in the second input to unknown dimensions.
// 4. IMPORTER_FORCE_DYNAMIC='1:0,1'
//    - change the first and second dimensions in the second input to unknown
//    dimensions.
// 5. IMPORTER_FORCE_DYNAMIC='0:1|1:0,1'
//    - change the second dimension in the first input to unknown dimensions,
//    and
//    - change the first and second dimensions in the second input to unknown
//    dimensions
class ModelInputShaper {
public:
  // For users of onnx-mlir.
  // -1 is used for dynamic/unknown dimension.
  static constexpr int64_t kUserDynamic = -1;
  // -1 is used to indicate all input indices.
  static constexpr int64_t kUserAllInputs = -1;

  ModelInputShaper();

  // shapeInformation specifies custom shapes for the inputs of the ONNX model,
  // e.g. setting static shapes for dynamic inputs.
  // See the documentation of the shapeInformation flag in CompilerOptions.cpp.
  void setShapeInformation(const std::string &shapeInformation);

  // Takes the input type at the given input index and
  // returns the input type with any changes to the shape specified by
  // the environment variable IMPORTER_FORCE_DYNAMIC
  // or any shapeInformation set in setShapeInformation.
  mlir::Type reshape(int inputIndex, mlir::Type inputType) const;

private:
  // Whether environment variable IMPORTER_FORCE_DYNAMIC is set.
  bool force_dim_dynamic_enabled_;

  // A map from an input index to a list of dim indices those are changed to
  // dynamic. Default value corresponds to IMPORTER_FORCE_DYNAMIC='-1:-1'.
  std::map<int, std::vector<int>> forced_inputs_dims_;

  // Custom shape information for the graph inputs.
  std::map<int64_t, std::vector<int64_t>> inputs_shape_information_;
};

} // namespace onnx_mlir
#endif
