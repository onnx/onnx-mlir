// SPDX-License-Identifier: Apache-2.0

//===------------ ResultTypeInferenceOpInterface.td --------------===//
//===--------- Infer Data Type for Results Interface Definition --------===//
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the tablegen of the data type inference interface.
//
//===----------------------------------------------------------------------===//

#ifdef RESULT_TYPE_INFERENCE_OP_INTERFACE
#else
#define RESULT_TYPE_INFERENCE_OP_INTERFACE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def ResultTypeInferenceOpInterface : OpInterface<"ResultTypeInferenceOpInterface"> {
  let description = [{
    Interface to access a registered method to infer the data types for 
    the result of an operation
  }];

  let methods = [
      InterfaceMethod<"Infer output data type for this operation class.",
          "std::vector<mlir::Type>", "resultTypeInference"
      >
  ];
}

#endif // RESULT_TYPE_INFERENCE_OP_INTERFACE
