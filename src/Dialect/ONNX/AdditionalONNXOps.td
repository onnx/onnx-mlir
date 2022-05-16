// SPDX-License-Identifier: Apache-2.0

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"

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

def ONNXCallOp : ONNX_Op<"ONNX_Call",
    [CallOpInterface, MemRefsNormalizable,
     DeclareOpInterfaceMethods<SymbolUserOpInterface>]> {
  let summary = "call operation";
  let description = [{
    The `call` operation represents a direct call to a function that is within
    the same symbol scope as the call. The operands and result types of the
    call must match the specified function type. The callee is encoded as a
    symbol reference attribute named "callee".

    Example:

    ```mlir
    %2 = call @my_add(%0, %1) : (f32, f32) -> f32
    ```
  }];

  let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyType>:$operands);
  //let arguments = (ins FlatSymbolRefAttr:$callee, Variadic<AnyTypeOf<[AnyTensor]>>:$operands);
  // TODO I would rather have <anyTypeOf<AnyTensor>> here, but the testcase for CustomOps supplies an empty parameter which fails that test --
  //will try to figure out why and revert, leaving this old style as a reminder
  //let results = (outs Variadic<AnyType>);
  let results = (outs Variadic<AnyTypeOf<[AnyTensor]>>);

  let builders = [
    OpBuilder<(ins "func::FuncOp":$callee, CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", SymbolRefAttr::get(callee));
      $_state.addTypes(callee.getFunctionType().getResults());
    }]>,
    OpBuilder<(ins "SymbolRefAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      $_state.addOperands(operands);
      $_state.addAttribute("callee", callee);
      $_state.addTypes(results);
    }]>,
    OpBuilder<(ins "StringAttr":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, SymbolRefAttr::get(callee), results, operands);
    }]>,
    OpBuilder<(ins "StringRef":$callee, "TypeRange":$results,
      CArg<"ValueRange", "{}">:$operands), [{
      build($_builder, $_state, StringAttr::get($_builder.getContext(), callee),
            results, operands);
    }]>];

  let assemblyFormat = [{
    $callee `(` $operands `)` attr-dict `:` functional-type($operands, results)
  }];

  let extraClassDeclaration = [{
    StringRef getCallee() { return callee(); }
    StringAttr getCalleeAttr() { return calleeAttr().getAttr(); }
    FunctionType getCalleeType();

    /// Get the argument operands to the called function.
    operand_range getArgOperands() {
      return {arg_operand_begin(), arg_operand_end()};
    }

    operand_iterator arg_operand_begin() { return operand_begin(); }
    operand_iterator arg_operand_end() { return operand_end(); }

    /// Return the callee of this operation.
    CallInterfaceCallable getCallableForCallee() {
      return (*this)->getAttrOfType<SymbolRefAttr>("callee");
    }
  }];
}
