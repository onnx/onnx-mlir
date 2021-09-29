def ONNXCustomOp:ONNX_Op<"Custom",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX Custom operation";
  let description = [{
  "Allow call-out to a user defined operation. A single attribute"
  "is a string which names the operation, other inputs are"
  "passed to the user operation."
  "The number of inputs and outputs can vary."
  }];
  let arguments = (ins Variadic<AnyTypeOf<[AnyTensor, AnyMemRef]>>:$input, StrAttr:$function_name);
  let results = (outs Variadic<AnyTypeOf<[AnyTensor, AnyMemRef]>>:$outputs);
  let extraClassDeclaration = [{
    static int getNumberOfOperands() {
      return -1;
    }
    static int getNumberOfResults() {
      return -1;
    }
    static std::vector<int> getTypeMap() {
      return {20};
    }
  }];
}

def YieldOp : ONNX_Op<"yield", [NoSideEffect, ReturnLike, Terminator,
                               ParentOneOf<["ONNXSubgraphOp"]>]> {
  let summary = "subgraph yield and termination operation";
  let description = [{
    "onnx.yield" yields an SSA value from the ONNX dialect subgraph region and
    terminates the region.
  }];

  let arguments = (ins Variadic<AnyType>:$results);
  let builders = [OpBuilder<(ins), [{ /* nothing to do */ }]>];

  let assemblyFormat =
      [{  attr-dict ($results^ `:` type($results))? }];
}

def ONNXSubgraphOp:ONNX_Op<"Subgraph",
  [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>, OpInterface<"HasOnnxSubgraphOpInterface" >]> {
  let summary = "ONNX Subgraph operation";
  let description = [{
  "Used to outline a subgraph (currently a single node only)."
  "The number of inputs and outputs can vary."
  }];
  let arguments = (ins Variadic<AnyTypeOf<[AnyTensor, AnyMemRef]>>:$input);
  let results = (outs Variadic<AnyTypeOf<[AnyTensor, AnyMemRef]>>:$outputs);
  let regions = (region SizedRegion<1>:$body);
  
let extraClassDeclaration = [{
//    static int getNumberOfOperands() {
//      return -1;
//    }
//    static int getNumberOfResults() {
//      return -1;
//    }
  static std::vector<int> getTypeMap() {
      return {20};
    }
  int64_t getSubgraphRegionIdx(const std::string& name) {
    if (name == "body") return 0;
    llvm_unreachable("region with the specified name does not exist");
    }
  }];
}