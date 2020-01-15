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

//the tablegen code onnxop.in is generated with gen_doc.py
//clone and install onnx
//   git clone --recursive https://github.com/onnx/onnx.git
// set up env for anaconda3 and for ONNF (BOOSTROOT, cmake, gcc ...)
//   cd onnx
//install onnx
//  CC=gcc CXX=g++ pip install -e .
//run the script
//  python onnx/defs/gen_doc.py
//result is in docs/onnxop.inc
//current limitations:
// 1. Attributes are not processed
// 2. output type inference not implemented except Add
// 3. Type Attribute: 'optional' and 'Variadic hetergeneous' are ignored
// 4. type of string, complex64 and complex128 for input/output are ignored 
// 5. unsigned int are treated as signed one

include "dialect/onnx/onnxop.inc"

// Indicate entry point functions of ONNX graph.
def ONNXEntryPointOp: ONNX_Op<"EntryPoint"> {
  let summary = "Indicate ONNX entry point";
  let description = [{
    The "onnx.EntryPoint" function indicates the main entry point of ONNX model.
  }];

  let builders = [OpBuilder<[{Builder *builder, OperationState &state,
                             FuncOp function, int numInputs, int numOutputs}]>];

  let extraClassDeclaration = [{
    static ONNXEntryPointOp create(Location location, FuncOp& func,
                                   int numInputs, int numOutputs);

    static StringRef getEntryPointFuncAttrName() { return "func"; }
    static StringRef getNumInputsAttrName() { return "numInputs"; }
    static StringRef getNumOutputsAttrName() { return "numOutputs"; }
  }];
}


def ONNXFullGemmOp: ONNX_Op<"FullGemm",
    [NoSideEffect, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ONNX general matrix multiply operation";
  let description = [{

    The "onnx.gemm" generic matrix multiplication with bias.

  }];

  let arguments = (ins AnyTensor:$lhs_in, AnyTensor:$rhs_in, AnyTensor:$bias_in);
  let results = (outs AnyTensor);
}

def ONNXConv1Op:ONNX_Op<"Conv1", 
    [NoSideEffect]> {
  let summary = "ONNX Conv operation";
  let description = [{
    "The convolution operator consumes an input tensor and a filter, and"
    "computes the output."
  }];
  let arguments = (ins AnyTensor:$X);
  let results = (outs AnyTensor);
}

def ONNXConv3Op:ONNX_Op<"Conv3", 
    [NoSideEffect]> {
  let summary = "ONNX Conv operation";
  let description = [{
    "The convolution operator consumes an input tensor and a filter, and"
    "computes the output."
  }];
  let arguments = (ins AnyTensor:$X, AnyTensor:$W, AnyTensor:$B);
  let results = (outs AnyTensor);
}

def ONNXMaxPoolSingleOutOp: ONNX_Op<"MaxPoolSingleOut",
    [NoSideEffect]> {
  let summary = "ONNX MaxPool operation with a single output.";
  let description = [{
    "ONNX MaxPool operation with a single output."
    "See ONNXMaxPoolOp for a full description of the MaxPool semantics."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyTensor]>:$X);
  let results = (outs AnyTypeOf<[AnyMemRef, AnyTensor]>);
}

#endif // ONNX_OPS
