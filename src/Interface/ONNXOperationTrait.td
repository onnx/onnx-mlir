// SPDX-License-Identifier: Apache-2.0

//===================- ONNXOperationTrait.td -*- tablegen -===================//
//
// ONNXOperationTrait defines static ONNX operation properties.
//
//===----------------------------------------------------------------------===//

#ifdef ONNX_OPERATION_TRAIT
#else
#define ONNX_OPERATION_TRAIT

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def ONNXOperationTrait : NativeOpTrait<"ONNXOperationTrait"> {
  let cppNamespace = "::mlir";
}

#endif // ONNX_OPERATION_TRAIT
