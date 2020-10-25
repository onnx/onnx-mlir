//===- ShapeInferenceInterface.td - ShapeInference Interface -*- tablegen -===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// Defines the operations of the Shape Inference Op Interface.
//
//===----------------------------------------------------------------------===//

#ifdef SHAPE_INFERENCE_INTERFACE
#else
#define SHAPE_INFERENCE_INTERFACE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
  let description = [{
    Interface to access a registered method to infer the return types for an
    operation that can be used during type inference.
  }];

  let methods = [
    InterfaceMethod<"Infer and set the output shape for the current operation.",
                    "LogicalResult", "inferShapes",
                    (ins "std::function<void(mlir::FuncOp)>":$shapeInferenceFunc)>
  ];
}

#endif // SHAPE_INFERENCE_INTERFACE
