// SPDX-License-Identifier: Apache-2.0

//===- ShapeHelperInterface.td - ShapeHelper Interface -*- tablegen -===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// Defines the operations of the Shape Helper Op Interface.
//
//===----------------------------------------------------------------------===//

#ifdef SHAPE_HELPER_OP_INTERFACE
#else
#define SHAPE_HELPER_OP_INTERFACE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def ShapeHelperOpInterface : OpInterface<"ShapeHelperOpInterface"> {
  let description = [{
    Interface to access a registered method to infer the return shapes for an
    operation that can be used during type inference. The shape inference
    interface method takes an operation and its operands as input, augmented
    by an index expression builder and an index expression scope as optional
    inputs. It returns a specialized shape helper object for that operation,
    casted as a generic ONNXOpShapeHelper.

    For operations that do not support shape helpers at this stage, a
    `ONNXUnimplementedOpShapeHelper` object is returned. This object does not
    compute shapes, and simply return success when `computeShape` is called
    on it. Users may verify if an operation has an actual implementation by
    calling `isImplemented()` on the shape helper object. An implementation
    is required when attempting to read the outputs of a shape helper object
    via the `getOutputDims` method.

    The new object is allocated on the heap and it is the responsability
    of the object user to free the memory after last use.
  }];

  let methods = [
    InterfaceMethod<"Provide a shape helper object that can be used to compute "
                    "the output shape for the current operation.",
                    "onnx_mlir::ONNXOpShapeHelper *", "getShapeHelper",
                    (ins "mlir::Operation *":$op,
                         "llvm::ArrayRef<mlir::Value>":$operands,
                         "onnx_mlir::IndexExprBuilder *":$ieBuilder,
                         "onnx_mlir::IndexExprScope *":$scope)>
  ];
  
  let cppNamespace = "::mlir";
}

#endif // SHAPE_HELPER_OP_INTERFACE
