// SPDX-License-Identifier: Apache-2.0

//===- ShapeInferenceInterface.td - ShapeInference Interface -*- tablegen -===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Defines the operations of the Shape Inference Op Interface.
//
//===----------------------------------------------------------------------===//

#ifdef SHAPE_INFERENCE_OP_INTERFACE
#else
#define SHAPE_INFERENCE_OP_INTERFACE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def ShapeInferenceOpInterface : OpInterface<"ShapeInferenceOpInterface"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference. The shape inference
    interface method takes a shape inference function pointer as input, this
    allows shape inference to proceed recursively.

    Operations implicitly associated with other (sub) computation graphs can
    run shape inference on their dependent computation graphs; this is
    necessary because the shape inference procedure of such operations (e.g.,
    Loop operation) depends on the shape inference outcome of their associated
    (sub) computation graphs.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "mlir::LogicalResult", "inferShapes",
                    (ins "std::function<void(mlir::Region&)>":$shapeInferenceFunc)>
  ];
  
  let cppNamespace = "::mlir";
}

#endif // SHAPE_INFERENCE_OP_INTERFACE
