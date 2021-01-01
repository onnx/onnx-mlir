//===----------------- HasOnnxSubgraphOpInterface.td -------------------===//
//===--------- Infer Data Type for Results Interface Definition --------===//
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the tablegen of the data type inference interface.
//
//===----------------------------------------------------------------------===//

#ifdef HAS_ONNX_SUBGRAPH_OP_INTERFACE
#else
#define HAS_ONNX_SUBGRAPH_OP_INTERFACE

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def HasOnnxSubgraphOpInterface : OpInterface<"HasOnnxSubgraphOpInterface"> {
  let description = [{
    Interface to access a registered method to infer the data types for 
    the result of an operation
  }];

  let methods = [
      InterfaceMethod<"Get the index of the region corresponding to the"
                      "onnx subgraph identified by its name.",
          "int64_t", "getSubgraphRegionIdx", (ins "const std::string &":$name)
      >
  ];
}

#endif // HAS_ONNX_SUBGRAPH_OP_INTERFACE
