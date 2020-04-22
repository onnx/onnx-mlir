//===--- MLONNXOps.td -- ONNX ML Dialect Operation Definitions ----*- tablegen -===//
//
// Copyright 2019-2020 The IBM Research Authors
//
// =============================================================================
//
// Defines ONNX Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifdef MLONNX_OPS
#else
#define MLONNX_OPS

#ifdef OP_BASE
#else
include "mlir/IR/OpBase.td"
#endif // OP_BASE

#ifdef SHAPE_INFERENCE_INTERFACE
#else
include "src/Interface/ShapeInferenceInterface.td"
#endif // SHAPE_INFERENCE_INTERFACE

#ifdef PROMOTABLE_CONST_OPERANDS_OP_INTERFACE
#else
include "src/Interface/PromotableConstOperandsOpInterface.td"
#endif // PROMOTABLE_CONST_OPERANDS_OP_INTERFACE

def MLONNX_Dialect : Dialect {
  let name = "mlonnx";
  let cppNamespace = "";
}

// Base class for ONNX dialect operations. This operation inherits from the base
// `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class MLONNX_Op<string mnemonic, list<OpTrait> traits = []> :
    Op<MLONNX_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// MLONNX Operations
//===----------------------------------------------------------------------===//

//the tablegen code onnxop.in is generated with gen_doc.py
//clone and install onnx
//   git clone --recursive https://github.com/onnx/onnx.git
// set up env for anaconda3 and for ONNX MLIR (BOOSTROOT, cmake, gcc ...)
//   cd onnx
//install onnx
//  CC=gcc CXX=g++ pip install -e .
//run the script
//  python onnx/defs/gen_doc.py
//result is in docs/onnx_ops.td.inc
//current limitations:
// 1. Attributes are not processed
// 2. output type inference not implemented except Add
// 3. Type Attribute: 'optional' and 'Variadic hetergeneous' are ignored
// 4. type of string, complex64 and complex128 for input/output are ignored 
// 5. unsigned int are treated as signed one

include "mlir/Interfaces/SideEffects.td"
include "src/Dialect/MLONNX/MLONNXOps.td.inc"

#endif // MLONNX_OPS
