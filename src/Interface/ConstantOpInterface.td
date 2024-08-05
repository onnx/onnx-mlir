// SPDX-License-Identifier: Apache-2.0

//===------------------ ConstantOpInterface.hpp -----------------===//
//===------------ Constant Op Interface Definition -------------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the TableGen definition of the Constant Op
// Interface Definition.
//
//===----------------------------------------------------------------------===//

#ifdef CONSTANT_OP_INTERFACE
#else
#define CONSTANT_OP_INTERFACE

include "mlir/IR/OpBase.td"

def ConstantOpInterface : OpInterface<"ConstantOpInterface"> {
  let description = [{
  }];

  let methods = [
      InterfaceMethod<"Get the index of the region corresponding to the"
                      "onnx subgraph identified by its name.",
          "ArrayRef<char>", "getBuffer", (ins "mlir::Operation *":$op)
      >,
      InterfaceMethod<"Get the index of the region corresponding to the"
                      "onnx subgraph identified by its name.",
          "Attribute", "getValueAttr", (ins )
      >,
      InterfaceMethod<"Get the index of the region corresponding to the"
                      "onnx subgraph identified by its name.",
          "std::optional<uint64_t>", "getAlignment", (ins )
      >,
      InterfaceMethod<"Get the index of the region corresponding to the"
                      "onnx subgraph identified by its name.",
          "void", "setOffsetAttr", (ins "::mlir::IntegerAttr": $attr)
      >,
      InterfaceMethod<"Get the index of the region corresponding to the"
                      "onnx subgraph identified by its name.",
          "Attribute", "removeValueAttr", (ins )
      >
      
  ];
  
  let extraClassDeclaration = [{
    /// Return the single result of this op.
    ::mlir::Value getResult() {
      return getOperation()->getResult(0);
    }
  }];

  let cppNamespace = "::mlir";
}

#endif // SPECIALIZED_KERNEL_OP_INTERFACE
