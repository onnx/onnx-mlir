// SPDX-License-Identifier: Apache-2.0

//===-- ZLowOps.td -- ZLow Dialect Operation Definitions -*- tablegen ------==//
//
// Copyright 2019-2024 The IBM Research Authors
//
// =============================================================================
//
// Defines ZLow Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef ZLOW_OPS

#ifndef OP_BASE
include "mlir/IR/OpBase.td"
#endif // OP_BASE

def ZLow_Dialect : Dialect {
  let name = "zlow";
  let summary = "A low-level dialect for the ONNX NNPA accelerator ISA.";
  let cppNamespace = "::onnx_mlir::zlow";  
  let usePropertiesForAttributes = 0;
}

// Base class for ZLow dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class ZLow_Op<string mnemonic, list<Trait> traits = []> :
    Op<ZLow_Dialect, mnemonic, traits>;

// DLFloat16 is not natively supported in MLIR.
// F16 is used instead as a container.
def DLF16 : Type<CPred<"$_self.isF16()">, "dlfloat16 type">,
                 BuildableType<"$_builder.getF16Type()">;

// 0-rank MemRef for scalar.
def ODMemRefF32: MemRefRankOf<[F32], [0]>;

// MemRef-like type for zTensor.
def ZMemRef : MemRefOf<[DLF16]>;
// Quantized zTensor.
def ZQMemRef : MemRefOf<[DLF16, I8]>;

//===----------------------------------------------------------------------===//
// ZLow Operations
//===----------------------------------------------------------------------===//

include "mlir/Interfaces/SideEffectInterfaces.td"

def ZLowAddOp:ZLow_Op<"add", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow add operation";
  let description = [{
  ZLow operation to perform an add.
  }];
  let arguments = (ins ZMemRef:$X,
                       ZMemRef:$Y,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def ZLowSubOp:ZLow_Op<"sub", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow sub operation";
  let description = [{
  ZLow operation to perform a sub.
  }];
  let arguments = (ins ZMemRef:$X,
                       ZMemRef:$Y,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def ZLowMulOp:ZLow_Op<"mul", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow mul operation";
  let description = [{
  ZLow operation to perform a mul.
  }];
  let arguments = (ins ZMemRef:$X,
                       ZMemRef:$Y,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def ZLowDivOp:ZLow_Op<"div", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow div operation";
  let description = [{
  ZLow operation to perform a div.
  }];
  let arguments = (ins ZMemRef:$X,
                       ZMemRef:$Y,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def ZLowLogOp:ZLow_Op<"log", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow log operation";
  let description = [{
  ZLow operation to perform a log.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def ZLowExpOp:ZLow_Op<"exp", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow exp operation";
  let description = [{
  ZLow operation to perform a exp.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}


def ZLowInvSqrtOp:ZLow_Op<"invsqrt", [MemRefsNormalizable]> {
  let summary = "ZLow invsqrt operation";
  let description = [{
  ZLow operation to perform a invsqrt.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}


def ZLowMinOp:ZLow_Op<"min", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow min operation";
  let description = [{
  ZLow operation to perform a min.
  }];
  let arguments = (ins ZMemRef:$X,
                       ZMemRef:$Y,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def ZLowMaxOp:ZLow_Op<"max", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow max operation";
  let description = [{
  ZLow operation to perform a max.
  }];
  let arguments = (ins ZMemRef:$X,
                       ZMemRef:$Y,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def ZLowLeakyReluOp:ZLow_Op<"leakyrelu", [MemRefsNormalizable]> {
  let summary = "ZLow leakyrelu operation";
  let description = [{
  ZLow operation to perform a leakyrelu.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       DefaultValuedAttr<F32Attr, "0.01">:$alpha,
                       StrAttr:$layout);
}

def ZLowReluOp:ZLow_Op<"relu", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow relu operation";
  let description = [{
  ZLow operation to perform a relu.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def ZLowGeluOp:ZLow_Op<"gelu", [MemRefsNormalizable]> {
  let summary = "ZLow gelu operation";
  let description = [{
  ZLow operation to perform a gelu.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def ZLowTanhOp:ZLow_Op<"tanh", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow tanh operation";
  let description = [{
  ZLow operation to perform a tanh.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def ZLowSigmoidOp:ZLow_Op<"sigmoid", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow sigmoid operation";
  let description = [{
  ZLow operation to perform a sigmoid.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def ZLowSoftmaxOp:ZLow_Op<"softmax", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow softmax operation";
  let description = [{
  ZLow operation to perform a softmax.
  work_area: a 4K-aligned buffer.
  act_func: ACT_NONE or ACT_LOG.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[I8]>:$work_area,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$act_func);
}

def ZLowSqrtOp:ZLow_Op<"sqrt", [MemRefsNormalizable]> {
  let summary = "ZLow sqrt operation";
  let description = [{
  ZLow operation to perform a sqrt.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def ZLowReduceMaxOp:ZLow_Op<"reducemax", [MemRefsNormalizable]> {
  let summary = "ZLow reducemax operation";
  let description = [{
  ZLow operation to perform a reducemax.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[I8]>:$work_area,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def ZLowReduceMinOp:ZLow_Op<"reducemin", [MemRefsNormalizable]> {
  let summary = "ZLow reducemin operation";
  let description = [{
  ZLow operation to perform a reducemin.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[I8]>:$work_area,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

def ZLowMatMulOp:ZLow_Op<"matmul", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow matmul operation";
  let description = [{
  ZLow operation to perform a matmul.
  * In case of unstacked: X(m, n) * Y(n, p) + Bias(p)
  shape is a 1D MemRef (memref<3xi64>) whose items are:
    * 1st item: m
    * 2nd item: n
    * 3rd item: p
  * In case of stacked: X(s, m, n) * Y(s, n, p) + Bias(s, p)
       or broadcasting1: X(m, n) * Y(s, n, p) + Bias(s, p)
       or broadcasting23: X(s, m, n) * Y(n, p) + Bias(p)
  shape is a 1D MemRef (memref<4xi64>) whose items are:
    * 1st item: s
    * 2nd item: m
    * 3rd item: n
    * 4th item: p
  * is_bcast1:  -1 broadcasting1, 0: no broadcasting1.
  * is_bcast23: -1 broadcasting23, 0: no broadcasting23.
  * is_stacked: -1 stacked, 0: unstacked.
  * transposeA: !0 transpose A, 0: do not transpose A.
  * transposeB: !0 transpose B, 0: do not transpose B.
  }];
  let arguments = (ins ZMemRef:$X,
                       ZMemRef:$Y,
                       ZMemRef:$Bias,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$Out,
                       DefaultValuedAttr<SI64Attr, "0">:$is_bcast1,
                       DefaultValuedAttr<SI64Attr, "-1">:$is_bcast23,
                       DefaultValuedAttr<SI64Attr, "-1">:$is_stacked,
                       DefaultValuedAttr<SI64Attr, "0">:$transposeA,
                       DefaultValuedAttr<SI64Attr, "0">:$transposeB);
}

def ZLowQuantizedMatMulOp:ZLow_Op<"quantizedMatmul", [MemRefsNormalizable]> {
  let summary = "ZLow quantized matmul operation";
  let description = [{
  ZLow operation to perform a matmul.
  work_area: a 4K-aligned buffer having the same layout as bias but dlfloat16 type.
  * In case of unstacked: X(m, n) * Y(n, p) + Bias(p)
  shape is a 1D MemRef (memref<3xi64>) whose items are:
    * 1st item: m
    * 2nd item: n
    * 3rd item: p
  * In case of stacked: X(s, m, n) * Y(s, n, p) + Bias(s, p)
       or broadcasting: X(s, m, n) * Y(n, p) + Bias(p)
  shape is a 1D MemRef (memref<4xi64>) whose items are:
    * 1st item: s
    * 2nd item: m
    * 3rd item: n
    * 4th item: p
  * is_bcast: -1 broadcasting, 0: no broadcasting.
  * is_stacked: -1 stacked, 0: unstacked.
  * DequantizeOutput: -1 output is dequantized, 0: output is not dequantized.
  * PreComputedBias: -1 bias is re-computed, 0: bias is not pre-computed.

  Values for `q_type` are "DLFLOAT16", "INT8", "WEIGHTS", "UNDEFINED".

  }];
  let arguments = (ins ZQMemRef:$X, ODMemRefF32:$x_rec_scale, ODMemRefF32:$x_offset,
                       ZQMemRef:$Y, ODMemRefF32:$y_rec_scale, ODMemRefF32:$y_offset,
                       ZQMemRef:$Bias, ODMemRefF32:$bias_rec_scale, ODMemRefF32:$bias_offset,
                       AnyTypeOf<[ZQMemRef, NoneType]>:$work_area,
                       MemRefOf<[I64]>:$shape,
                       ZQMemRef:$Out, ODMemRefF32:$out_rec_scale, ODMemRefF32:$out_offset,
                       StrAttr:$x_q_type,
                       StrAttr:$y_q_type,
                       StrAttr:$bias_q_type,
                       StrAttr:$out_q_type,
                       DefaultValuedAttr<SI64Attr, "-1">:$is_bcast,
                       DefaultValuedAttr<SI64Attr, "-1">:$is_stacked,
                       DefaultValuedAttr<SI64Attr, "0">:$pre_computed_bias,
                       DefaultValuedAttr<SI64Attr, "0">:$disable_clipping,
                       DefaultValuedAttr<SI64Attr, "0">:$dequantize_output);
}

def ZLowLSTMOp:ZLow_Op<"lstm", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow lstm operation";
  let description = [{
  ZLow operation to perform a lstm.
  work_area: a 4K-aligned buffer.
  * shape is a 1D MemRef (memref<5xi64>) whose items are:
    * 1st item: direction
    * 2nd item: timestep
    * 3rd item: batchSize
    * 4th item: featureSize
    * 5th item: hiddenSize
  * direction accepts "forward", "reverse", or "bidirectional"
  * return_all_steps: -1 returns all timesteps, 0: returns only the last timestep
  * prev_layer for where input comes is "none", "uni", or "bidir"
  }];
  let arguments = (ins ZMemRef:$input,
                       ZMemRef:$h0,
                       ZMemRef:$c0,
                       ZMemRef:$input_weights,
                       ZMemRef:$input_bias,
                       ZMemRef:$hidden_weights,
                       ZMemRef:$hidden_bias,
                       MemRefOf<[I8]>:$work_area,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$hn_output,
                       ZMemRef:$cf_output,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       DefaultValuedAttr<SI64Attr, "-1">:$return_all_steps,
                       DefaultValuedStrAttr<StrAttr, "none">:$prev_layer);
}

def ZLowGRUOp:ZLow_Op<"gru", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow gru operation";
  let description = [{
  ZLow operation to perform a gru.
  * work_area: a 4K-aligned buffer.
  * shape is a 1D MemRef (memref<5xi64>) whose items are:;
    * 1st item: direction
    * 2nd item: timestep
    * 3rd item: batchSize
    * 4th item: featureSize
    * 5th item: hiddenSize
  * direction accepts "forward", "reverse", or "bidirectional"
  * return_all_steps: -1 returns all timesteps, 0: returns only the last timestep.
  * prev_layer for where input comes is "none", "uni", or "bidir"
  }];
  let arguments = (ins ZMemRef:$input,
                       ZMemRef:$h0,
                       ZMemRef:$input_weights,
                       ZMemRef:$input_bias,
                       ZMemRef:$hidden_weights,
                       ZMemRef:$hidden_bias,
                       MemRefOf<[I8]>:$work_area,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$hn_output,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       DefaultValuedAttr<SI64Attr, "-1">:$return_all_steps,
                       DefaultValuedStrAttr<StrAttr, "none">:$prev_layer);
}

def ZLowStickOp:ZLow_Op<"stick", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow stick operation";
  let description = [{
  "ZLow operation to perform a stick."
  }];
  let arguments = (ins MemRefOf<[F16, F32]>:$X,
                       ZMemRef:$Out,
                       OptionalAttr<StrAttr>:$layout,
                       OptionalAttr<SI64Attr>:$saturation);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$X, "::mlir::Value":$Out), [{
      build($_builder, $_state, X, Out, StringAttr(), IntegerAttr());
    }]>,
    OpBuilder<(ins "::mlir::Value":$X, "::mlir::Value":$Out, "::mlir::StringAttr":$layout), [{
      build($_builder, $_state, X, Out, layout, IntegerAttr());
    }]>
  ];
}

def ZLowStickForLSTMOp:ZLow_Op<"stickForLSTM", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow stick operation for LSTM";
  let description = [{
  ZLow operation to perform a stick for LSTM.
  Variadic: list of pointers for input data to be transformed: 
    - LSTM concatenated: 4 data pointers, one for each input gate in Forget, Input, Cell, Output (FICO) order.
  }];
  let arguments = (ins MemRefOf<[F16, F32]>:$f_gate,
                       MemRefOf<[F16, F32]>:$i_gate,
                       MemRefOf<[F16, F32]>:$c_gate,
                       MemRefOf<[F16, F32]>:$o_gate,
                       ZMemRef:$out,
                       DefaultValuedStrAttr<StrAttr, "none">:$prev_layer);
}

def ZLowStickForGRUOp:ZLow_Op<"stickForGRU", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow stick operation for GRU";
  let description = [{
  ZLow operation to perform a stick for GRU.
  Variadic: list of pointers for input data to be transformed: 
    - GRU concatenated: 3 data pointers, one for each input gate in (Z)update, Reset, Hidden, (ZRH) gate order.
  }];
  let arguments = (ins MemRefOf<[F16, F32]>:$z_gate,
                       MemRefOf<[F16, F32]>:$r_gate,
                       MemRefOf<[F16, F32]>:$h_gate,
                       ZMemRef:$out,
                       DefaultValuedStrAttr<StrAttr, "none">:$prev_layer);
}

def ZLowQuantizedStickOp:ZLow_Op<"quantizedStick", [MemRefsNormalizable]> {
  let summary = "ZLow stick operation for quantization";
  let description = [{
  "ZLow operation to perform a quantization stick."
  "Type is one of values: dlfloat16, int8, and weights."
  }];
  let arguments = (ins MemRefOf<[I8, F32]>:$X,
                       MemRefRankOf<[F32], [0]>:$rec_scale,
                       MemRefRankOf<[F32], [0]>:$offset,
                       ZQMemRef:$out,
                       StrAttr:$layout,
                       StrAttr:$q_type);
  let hasVerifier = 1;
}

def ZLowUnstickOp:ZLow_Op<"unstick", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow unstick operation";
  let description = [{
  ZLow operation to perform a unstick.
  }];
  let arguments = (ins ZMemRef:$X,
                       MemRefOf<[F16, F32]>:$Out,
                       OptionalAttr<StrAttr>:$layout);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$X, "::mlir::Value":$Out), [{
      build($_builder, $_state, X, Out, StringAttr());
    }]>
  ];
}

def ZLowConv2DOp:ZLow_Op<"conv2d", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow 2D convolution operation";
  let description = [{
  ZLow operation to perform 2D convolution.
  * shape is a 1D MemRef (memref<7xi64>) whose items are:
    * 1st item: batch size
    * 2nd item: channel in
    * 3rd item: height in
    * 4th item: width in
    * 5th item: channel out
    * 6th item: height out
    * 7th item: width out
  * kernel_shape: 1D array of kernel height and width
  * strides: 1D array of stride height and width
  * padding_type: SAME_PADDING or VALID_PADDING.
  * act_func: ACT_NONE or ACT_RELU.
  }];
  let arguments = (ins ZMemRef:$input,
                       ZMemRef:$input_kernel,
                       ZMemRef:$input_bias,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$output,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$strides,
                       DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type,
                       DefaultValuedStrAttr<StrAttr, "ACT_NONE">:$act_func);
}

def ZLowAvgPool2DOp:ZLow_Op<"avgpool2d", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow 2D average pooling operation";
  let description = [{
  ZLow operation to perform 2D average pooling.
  * shape is a 1D MemRef (memref<6xi64>) whose items are:
    * 1st item: batch size
    * 2nd item: channel
    * 3rd item: height in
    * 4th item: width in
    * 5th item: height out
    * 6th item: width out
  * kernel_shape: 1D array of kernel height and width
  * strides: 1D array of stride height and width
  * padding_type: SAME_PADDING or VALID_PADDING.
  }];
  let arguments = (ins ZMemRef:$input,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$output,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$strides,
                       DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type
                  );
}

def ZLowMaxPool2DOp:ZLow_Op<"maxpool2d", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow 2D max pooling operation";
  let description = [{
  ZLow operation to perform 2D max pooling.
  * shape is a 1D MemRef (memref<6xi64>) whose items are:
    * 1st item: batch size
    * 2nd item: channel
    * 3rd item: height in
    * 4th item: width in
    * 5th item: height out
    * 6th item: width out
  * kernel_shape: 1D array of kernel height and width
  * strides: 1D array of stride height and width
  * padding_type: SAME_PADDING or VALID_PADDING.
  }];
  let arguments = (ins ZMemRef:$input,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$output,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$strides,
                       DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type
                  );
}

def ZLowMeanReduce2DOp:ZLow_Op<"meanreduce2d", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow 2D mean reduce operation";
  let description = [{
  ZLow operation to perform 2D mean reduce.
  * shape is a 1D MemRef (memref<4xindex>) whose items are:;
    * 1st item: batch size": 1st dim of input
    * 2rd item: height": 2nd dim of input
    * 3th item: width": 3rd dim of input
    * 4nd item: channel": 4th dim of input
  }];
  let arguments = (ins ZMemRef:$input,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$output);
}

def ZLowBatchNormOp:ZLow_Op<"batchnorm", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "ZLow batchnorm operation";
  let description = [{
  ZLow operation to perform batchnorm.
  * shape is a 1D MemRef (memref<4xi64>) whose items are:
    * 1st item: batch size
    * 2nd item: height
    * 3rd item: width
    * 4th item: channel
  }];
  let arguments = (ins ZMemRef:$input,
                       ZMemRef:$A,
                       ZMemRef:$B,
                       MemRefOf<[I64]>:$shape,
                       ZMemRef:$output);
}

def ZLowDummyOp:ZLow_Op<"dummy", [MemRefsNormalizable]> {
  let summary = "ZLow dummy operation that behaves like identity";
  let description = [{
  ZLow operation to forward the input value to the output value.
  It will be removed if canonicalization is called.
  }];
  let arguments = (ins AnyType:$input);
  let results = (outs AnyType:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input), [{
      build($_builder, $_state, input.getType(), input);
    }]>
  ];
  let hasCanonicalizer = 1;
}

def ZLowConvertDLF16ToF32Op:ZLow_Op<"dlf16_to_f32", [Pure]> {
  let summary = "Convert a dlfloat16 value to a float32 value";
  let description = [{
  This operation converts a dlfloat16 value to a float32 value. 
  }];
  let arguments = (ins DLF16:$input);
  let results = (outs F32:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input), [{
      build($_builder, $_state, $_builder.getF32Type(), input);
    }]>
  ];
}

def ZLowConvertF32ToDLF16Op:ZLow_Op<"f32_to_dlf16", [Pure]> {
  let summary = "Convert a float32 value to a dlfloat16 value";
  let description = [{
  This operation converts a float32 value to a dlfloat16 value. 
  }];
  let arguments = (ins F32:$input);
  let results = (outs DLF16:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input), [{
      // F16 is used as a container for DLF16.
      build($_builder, $_state, $_builder.getF16Type(), input);
    }]>
  ];
  let hasCanonicalizer = 1;
}

def ZLowConvertDLF16ToF32VectorOp:ZLow_Op<"vec_dlf16_to_f32", [Pure]> {
  let summary = "Convert dlfloat16 values to float32 values";
  let description = [{
  This operation converts dlfloat16 values to float32 values. 
  }];
  let arguments = (ins AnyTypeOf<[VectorOfLengthAndType<[8], [F16]>]>:$input);
  let results = (outs AnyTypeOf<[VectorOfLengthAndType<[4], [F32]>]>:$output1,
                      AnyTypeOf<[VectorOfLengthAndType<[4], [F32]>]>:$output2);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input), [{
      mlir::Type resType = mlir::VectorType::get({4}, $_builder.getF32Type());
      build($_builder, $_state, {resType, resType}, input);
    }]>
  ];
}

def ZLowConvertF32ToDLF16VectorOp:ZLow_Op<"vec_f32_to_dlf16", [Pure]> {
  let summary = "Convert float32 values to dlfloat16 values";
  let description = [{
  This operation converts float32 values to dlfloat16 values. 
  }];
  let arguments = (ins AnyTypeOf<[VectorOfLengthAndType<[4], [F32]>]>:$input1,
                       AnyTypeOf<[VectorOfLengthAndType<[4], [F32]>]>:$input2);
  let results = (outs AnyTypeOf<[VectorOfLengthAndType<[8], [F16]>]>:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input1, "::mlir::Value":$input2), [{
      mlir::Type resType = mlir::VectorType::get({8}, $_builder.getF16Type());
      build($_builder, $_state, resType, input1, input2);
    }]>
  ];
}

def ZLowReshapeOp:ZLow_Op<"reshape", [MemRefsNormalizable]> {
  let summary = "ZLow Reshape operation";
  let description = [{
  ZLow operation to perform a reshape (no data movement).
  }];
  // Note that no shape is needed for this operation.
  let arguments = (ins ZMemRef:$X,
                       ZMemRef:$Out,
                       StrAttr:$layout);
}

#endif // ZLOW_OPS
