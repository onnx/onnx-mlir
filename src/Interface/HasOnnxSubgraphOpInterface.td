// SPDX-License-Identifier: Apache-2.0

//===------------------- HasOnnxSubgraphOpInterface.hpp ------------------===//
//===------------- Has Onnx Subgraph Op Interface Definition -------------===//
//
// Copyright 2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the TableGen definition of the HasOnnxSubgraph Op
// Interface.
//
//===----------------------------------------------------------------------===//

#ifdef HAS_ONNX_SUBGRAPH_OP_INTERFACE
#else
#define HAS_ONNX_SUBGRAPH_OP_INTERFACE

include "mlir/IR/OpBase.td"

def HasOnnxSubgraphOpInterface : OpInterface<"HasOnnxSubgraphOpInterface"> {
  let description = [{
    Operation interface mapping the name of the subgraph (specified as the subgraph
    attribute name in ONNX) to region index of the imported MLIR operation.

    This operation interface essentially allows for the association of a name with
    an operation region, where the name is specified by the ONNX standard.
  }];

  let methods = [
      InterfaceMethod<"Get the index of the region corresponding to the"
                      "onnx subgraph identified by its name.",
          "int64_t", "getSubgraphRegionIdx", (ins "const std::string &":$name)
      >
  ];
}

#endif // HAS_ONNX_SUBGRAPH_OP_INTERFACE
