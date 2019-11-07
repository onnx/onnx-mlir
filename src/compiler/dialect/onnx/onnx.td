//===- ONNXOps.td -- ONNX operation definitions ---------*- tablegen -*----===//
//
// Copyright 2019 The IBM Research Authors
//
// =============================================================================
//
// Defines MLIR ONNX operations.
//
//===----------------------------------------------------------------------===//

#ifdef ONNX_OPS
#else
#define ONNX_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

#ifdef SHAPE_INFERENCE_INTERFACE
#else
include "pass/shape_inference_interface.td"
#endif // SHAPE_INFERENCE_INTERFACE

def ONNX_Dialect : Dialect {
  let name = "onnx";
  let cppNamespace = "";
}

// Base class for ONNX dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class ONNX_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<ONNX_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ONNX Operations
//===----------------------------------------------------------------------===//

// We define an ONNX operation for adding two tensors elementwise.
def ONNXAddOp: ONNX_Op<"add",
    [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX add operation";
  let description = [{

    The "onnx.add" adds two tensors element-wise.

  }];

  // TODO: AnyTensor might be too wide for ONNX and may need to be constrained
  // to fewer valid types.
  // In the ONNX spec:
  //     T : tensor(uint32), tensor(uint64),
  //         tensor(int32), tensor(int64),
  //         tensor(float16), tensor(float), tensor(double)
  //
  let arguments = (ins AnyTensor:$lhs_in, AnyTensor:$rhs_in);
  let results = (outs AnyTensor);

  // Build an ONNX Add operation using two input operands.
  let builders = [
    OpBuilder<"Builder *b, OperationState &result, Value *lhs, Value *rhs", [{
      buildONNXAddOp(b, result, lhs, rhs);
    }]
  >];
}

#endif // ONNX_OPS
