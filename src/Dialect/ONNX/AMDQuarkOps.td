// SPDX-License-Identifier: Apache-2.0

//===-- AMDQuarkOps.td -- AMD Quark Ops  -*- tablegen -===//
//
// Copyright 2025 Advanced Micro Devices, Inc. or its affiliates
//
// =============================================================================
//
// Defines Ops from AMD's Quark quantizer, version 0.10
//
//===----------------------------------------------------------------------===//

include "mlir/Interfaces/CallInterfaces.td"
include "mlir/IR/SymbolInterfaces.td"
include "src/IR/AttrBase.td"

//===----------------------------------------------------------------------===//
// BFPQuantizeDequantizeOp
def AMDQuarkBFPQuantizeDequantizeOp: ONNX_Op<"AMDQuarkBFPQuantizeDequantizeOp",
    [Pure,  OpVersionTrait<1>,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>, SameOperandsAndResultElementType]> {
  let summary = "BFPQuantizeDequantize";
  let description = [{
    Block Floating Point (BFP) groups numbers (e.g., tensors, arrays) into blocks, where each block shares a common exponent, and the values in the block are represented with individual mantissas (and the sign bit). This approach offers the performance and speed of 8-bit operations while bringing the precision closer to 16-bit operations.

    MicroeXponents (MX) extends the concept of BFP by introducing two levels of exponents: shared exponents for entire blocks and micro exponents for finer-grained sub-blocks. This two-level approach enables more precise scaling of individual elements within a block, reducing quantization error and improving the representational range. The paper https://arxiv.org/abs/2302.08007 introduces three specific formats: MX4, MX6 and MX9, which have different bits of mantissa.

    This operator converts floating-point values (typically 32-bit floating-point numbers) into BFP or MX values, then convert them back. It approximates the Quantize-Dequantize process and introduces quantization errors.
  }];

  let arguments = (ins TensorOf<[F32]>:$X,
                       DefaultValuedStrAttr<StrAttr, "to_bfp">:$bfp_method,
                       DefaultValuedAttr<SI64Attr, "1">:$axis,
                       DefaultValuedAttr<SI64Attr, "16">:$bit_width,
                       DefaultValuedAttr<SI64Attr, "8">:$block_size,
                       DefaultValuedAttr<SI64Attr, "0">:$rounding_mode,
                       DefaultValuedAttr<SI64Attr, "2">:$sub_block_size,
                       DefaultValuedAttr<SI64Attr, "1">:$sub_block_shift_bits
                      );
  let results = (outs TensorOf<[F32]>:$Y);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static int getNumberOfOperands() { return 1; }
    static int getNumberOfResults() { return 1; }
    static std::vector<int> getTypeMap() { return {30}; } // Same result element type as operand
    [[nodiscard]] bool isBFP16(bool ignoreAxis = false);
    [[nodiscard]] bool isMX4(bool ignoreAxis = false);
    [[nodiscard]] bool isMX6(bool ignoreAxis = false);
    [[nodiscard]] bool isMX9(bool ignoreAxis = false);
  }];
  
  let extraClassDefinition = [{  
    onnx_mlir::ONNXOpShapeHelper * AMDQuarkBFPQuantizeDequantizeOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new onnx_mlir::AMDQuarkBFPQuantizeDequantizeOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

