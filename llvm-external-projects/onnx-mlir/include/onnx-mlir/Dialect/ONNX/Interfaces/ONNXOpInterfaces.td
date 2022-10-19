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

def ShapeInferenceOpInterface : OpInterface<"ShapeInference"> {
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

#endif // HAS_ONNX_SUBGRAPH_OP_INTERFACE
