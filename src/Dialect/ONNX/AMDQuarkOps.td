// SPDX-License-Identifier: Apache-2.0

//===-- AMDQuarkOps.td -- AMD Quark Ops  -*- tablegen -===//
//
// Copyright 2025-2026 Advanced Micro Devices, Inc. or its affiliates
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

    Support for BF16 is an AMD extension in ONNX-MLIR to https://quark.docs.amd.com/latest/onnx/custom_operators/BFPQuantizeDequantize.html.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[BF16]>]>:$X,
                       DefaultValuedStrAttr<StrAttr, "to_bfp">:$bfp_method,
                       DefaultValuedAttr<SI64Attr, "1">:$axis,
                       DefaultValuedAttr<SI64Attr, "16">:$bit_width,
                       DefaultValuedAttr<SI64Attr, "8">:$block_size,
                       DefaultValuedAttr<SI64Attr, "0">:$rounding_mode,
                       DefaultValuedAttr<SI64Attr, "2">:$sub_block_size,
                       DefaultValuedAttr<SI64Attr, "1">:$sub_block_shift_bits
                      );
  let results = (outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[BF16]>]>:$Y);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static int getNumberOfOperands() { return 1; }
    static int getNumberOfResults() { return 1; }
    static std::vector<int> getTypeMap() { return {30}; } // Same result element type as operand
    [[nodiscard]] bool isBFP16(bool ignoreAxis = false);
    [[nodiscard]] bool isMX4(bool ignoreAxis = false);
    [[nodiscard]] bool isMX6(bool ignoreAxis = false);
    [[nodiscard]] bool isMX9(bool ignoreAxis = false);
    [[nodiscard]] std::optional<int64_t> getNormalizedAxis();
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

//===----------------------------------------------------------------------===//
// AMDQuarkExtendedQuantizeLinearOp
def AMDQuarkExtendedQuantizeLinearOp
    : ONNX_Op<"AMDQuarkExtendedQuantizeLinearOp",
          [Pure, OpVersionTrait<1>,
              DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
              DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ExtendedQuantizeLinear";
  let description = [{
    Extended quantization operator from AMD's Quark quantizer. Consumes a
    high-precision tensor, a scale, and a zero point to compute the
    low-precision/quantized tensor. The quantization formula is
    `y = saturate((x / scale) + zero_point)`.

    This is part of the `com.amd.quark` ONNX custom domain.

    Support for F16 and BF16 is an AMD extension in ONNX-MLIR to
    https://quark.docs.amd.com/latest/onnx/custom_operators/ExtendedQuantizeLinear.html.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>,
                       TensorOf<[BF16]>]>:$x,
      AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$y_scale,
      AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I16]>, TensorOf<[I8]>,
          TensorOf<[UI32]>, TensorOf<[UI16]>, TensorOf<[UI8]>, TensorOf<[F16]>,
          TensorOf<[BF16]>, NoneType]>:$y_zero_point,
      DefaultValuedAttr<SI64Attr, "1">:$axis);
  let results = (outs AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I16]>,
      TensorOf<[I8]>, TensorOf<[UI32]>, TensorOf<[UI16]>, TensorOf<[UI8]>,
      TensorOf<[F16]>, TensorOf<[BF16]>]>:$y);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static int getNumberOfOperands() { return 3; }
    static int getNumberOfResults() { return 1; }
    static std::vector<int> getTypeMap() { return {32}; }
    [[nodiscard]] std::optional<int64_t> getNormalizedAxis();
  }];

  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * AMDQuarkExtendedQuantizeLinearOp::getShapeHelper(
        mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh =
          new onnx_mlir::AMDQuarkExtendedQuantizeLinearOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

//===----------------------------------------------------------------------===//
// AMDQuarkExtendedDequantizeLinearOp
def AMDQuarkExtendedDequantizeLinearOp
    : ONNX_Op<"AMDQuarkExtendedDequantizeLinearOp",
          [Pure, OpVersionTrait<1>,
              DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
              DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ExtendedDequantizeLinear";
  let description = [{
    Extended dequantization operator from AMD's Quark quantizer. Consumes a
    quantized tensor, a scale, and a zero point to compute the full-precision
    tensor. The dequantization formula is `y = (x - zero_point) * scale`.

    This is part of the `com.amd.quark` ONNX custom domain.

    Support for F16 and BF16 is an AMD extension in ONNX-MLIR to
    https://quark.docs.amd.com/latest/onnx/custom_operators/ExtendedDequantizeLinear.html.
  }];

  let arguments = (ins AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I16]>,
                       TensorOf<[I8]>, TensorOf<[UI32]>, TensorOf<[UI16]>,
                       TensorOf<[UI8]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$x,
      AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$x_scale,
      AnyTypeOf<[TensorOf<[I32]>, TensorOf<[I16]>, TensorOf<[I8]>,
          TensorOf<[UI32]>, TensorOf<[UI16]>, TensorOf<[UI8]>, TensorOf<[F16]>,
          TensorOf<[BF16]>, NoneType]>:$x_zero_point,
      DefaultValuedAttr<SI64Attr, "1">:$axis);
  let results =
      (outs AnyTypeOf<[TensorOf<[F32]>, TensorOf<[F16]>, TensorOf<[BF16]>]>:$y);

  let hasVerifier = 1;

  let extraClassDeclaration = [{
    static int getNumberOfOperands() { return 3; }
    static int getNumberOfResults() { return 1; }
    static std::vector<int> getTypeMap() { return {31}; }
    [[nodiscard]] std::optional<int64_t> getNormalizedAxis();
  }];

  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * AMDQuarkExtendedDequantizeLinearOp::getShapeHelper(
        mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh =
          new onnx_mlir::AMDQuarkExtendedDequantizeLinearOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}
