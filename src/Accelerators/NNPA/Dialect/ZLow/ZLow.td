// SPDX-License-Identifier: Apache-2.0

//===-- ZLowOps.td -- ZLow Dialect Operation Definitions -*- tablegen ------==//
//
// Copyright 2019-2020 The IBM Research Authors
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
  let cppNamespace = "::onnx_mlir::zlow";
}

// Base class for ZLow dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.
class ZLow_Op<string mnemonic, list<Trait> traits = []> :
    Op<ZLow_Dialect, mnemonic, traits>;

//===----------------------------------------------------------------------===//
// ZLow Operations
//===----------------------------------------------------------------------===//

include "mlir/Interfaces/SideEffectInterfaces.td"

def ZLowAddOp:ZLow_Op<"add", [MemRefsNormalizable]> {
  let summary = "ZLow add operation";
  let description = [{
  "ZLow operation to perform an add."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$X,
                       AnyTypeOf<[AnyMemRef]>:$Y,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$Out,
                       StrAttr:$layout);
}

def ZLowSubOp:ZLow_Op<"sub", [MemRefsNormalizable]> {
  let summary = "ZLow sub operation";
  let description = [{
  "ZLow operation to perform a sub."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$X,
                       AnyTypeOf<[AnyMemRef]>:$Y,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$Out,
                       StrAttr:$layout);
}

def ZLowMulOp:ZLow_Op<"mul", [MemRefsNormalizable]> {
  let summary = "ZLow mul operation";
  let description = [{
  "ZLow operation to perform a mul."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$X,
                       AnyTypeOf<[AnyMemRef]>:$Y,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$Out,
                       StrAttr:$layout);
}

def ZLowDivOp:ZLow_Op<"div", [MemRefsNormalizable]> {
  let summary = "ZLow div operation";
  let description = [{
  "ZLow operation to perform a div."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$X,
                       AnyTypeOf<[AnyMemRef]>:$Y,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$Out,
                       StrAttr:$layout);
}

def ZLowLogOp:ZLow_Op<"log", [MemRefsNormalizable]> {
  let summary = "ZLow log operation";
  let description = [{
  "ZLow operation to perform a log."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$X,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$Out,
                       StrAttr:$layout);
}

def ZLowExpOp:ZLow_Op<"exp", [MemRefsNormalizable]> {
  let summary = "ZLow exp operation";
  let description = [{
  "ZLow operation to perform a exp."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$X,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$Out,
                       StrAttr:$layout);
}

def ZLowMinOp:ZLow_Op<"min", [MemRefsNormalizable]> {
  let summary = "ZLow min operation";
  let description = [{
  "ZLow operation to perform a min."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$X,
                       AnyTypeOf<[AnyMemRef]>:$Y,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$Out,
                       StrAttr:$layout);
}

def ZLowMaxOp:ZLow_Op<"max", [MemRefsNormalizable]> {
  let summary = "ZLow max operation";
  let description = [{
  "ZLow operation to perform a max."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$X,
                       AnyTypeOf<[AnyMemRef]>:$Y,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$Out,
                       StrAttr:$layout);
}

def ZLowReluOp:ZLow_Op<"relu", [MemRefsNormalizable]> {
  let summary = "ZLow relu operation";
  let description = [{
  "ZLow operation to perform a relu."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$X,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$Out,
                       StrAttr:$layout);
}

def ZLowTanhOp:ZLow_Op<"tanh", [MemRefsNormalizable]> {
  let summary = "ZLow tanh operation";
  let description = [{
  "ZLow operation to perform a tanh."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$X,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$Out,
                       StrAttr:$layout);
}

def ZLowSigmoidOp:ZLow_Op<"sigmoid", [MemRefsNormalizable]> {
  let summary = "ZLow sigmoid operation";
  let description = [{
  "ZLow operation to perform a sigmoid."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$X,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$Out,
                       StrAttr:$layout);
}

def ZLowSoftmaxOp:ZLow_Op<"softmax", [MemRefsNormalizable]> {
  let summary = "ZLow softmax operation";
  let description = [{
  "ZLow operation to perform a softmax."
  "work_area: a 4K-aligned buffer."
  "act_func: ACT_NONE or ACT_LOG."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$X,
                       AnyTypeOf<[AnyMemRef]>:$work_area,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$Out,
                       StrAttr:$act_func);
}

def ZLowMatMulOp:ZLow_Op<"matmul", [MemRefsNormalizable]> {
  let summary = "ZLow matmul operation";
  let description = [{
  "ZLow operation to perform a matmul."
  "- In case of unstacked: X(m, n) * Y(n, p) + Bias(p)"
  "shape is a 1D MemRef (memref<3xi64>) whose items are:"
  "  - 1st item: m"
  "  - 2nd item: n"
  "  - 3rd item: p"
  "- In case of stacked: X(s, m, n) * Y(s, n, p) + Bias(s, p)"
  "     or broadcasting: X(s, m, n) * Y(n, p) + Bias(p)"
  "shape is a 1D MemRef (memref<4xi64>) whose items are:"
  "  - 1st item: s"
  "  - 2nd item: m"
  "  - 3rd item: n"
  "  - 4th item: p"
  "is_bcast: -1 broadcasting, 0: no broadcasting."
  "is_stacked: -1 stacked, 0: unstacked."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$X,
                       AnyTypeOf<[AnyMemRef]>:$Y,
                       AnyTypeOf<[AnyMemRef]>:$Bias,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$Out,
                       DefaultValuedAttr<SI64Attr, "-1">:$is_bcast,
                       DefaultValuedAttr<SI64Attr, "-1">:$is_stacked);
}

def ZLowLSTMOp:ZLow_Op<"lstm", [MemRefsNormalizable]> {
  let summary = "ZLow lstm operation";
  let description = [{
  "ZLow operation to perform a lstm."
  "work_area: a 4K-aligned buffer."
  "shape is a 1D MemRef (memref<5xi64>) whose items are:;
  "  - 1st item: direction"
  "  - 2nd item: timestep"
  "  - 3rd item: batchSize"
  "  - 4th item: featureSize"
  "  - 5th item: hiddenSize"
  "direction accepts "forward", "reverse", or "bidirectional"
  "return_all_steps: -1 returns all timesteps, 0: returns only the last timestep.
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$input,
                       AnyTypeOf<[AnyMemRef]>:$h0,
                       AnyTypeOf<[AnyMemRef]>:$c0,
                       AnyTypeOf<[AnyMemRef]>:$input_weights,
                       AnyTypeOf<[AnyMemRef]>:$input_bias,
                       AnyTypeOf<[AnyMemRef]>:$hidden_weights,
                       AnyTypeOf<[AnyMemRef]>:$hidden_bias,
                       AnyTypeOf<[AnyMemRef]>:$work_area,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$hn_output,
                       AnyTypeOf<[AnyMemRef]>:$cf_output,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       DefaultValuedAttr<SI64Attr, "-1">:$return_all_steps);
}

def ZLowGRUOp:ZLow_Op<"gru", [MemRefsNormalizable]> {
  let summary = "ZLow gru operation";
  let description = [{
  "ZLow operation to perform a gru."
  "work_area: a 4K-aligned buffer."
  "shape is a 1D MemRef (memref<5xi64>) whose items are:;
  "  - 1st item: direction"
  "  - 2nd item: timestep"
  "  - 3rd item: batchSize"
  "  - 4th item: featureSize"
  "  - 5th item: hiddenSize"
  "direction accepts "forward", "reverse", or "bidirectional"
  "return_all_steps: -1 returns all timesteps, 0: returns only the last timestep.
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$input,
                       AnyTypeOf<[AnyMemRef]>:$h0,
                       AnyTypeOf<[AnyMemRef]>:$input_weights,
                       AnyTypeOf<[AnyMemRef]>:$input_bias,
                       AnyTypeOf<[AnyMemRef]>:$hidden_weights,
                       AnyTypeOf<[AnyMemRef]>:$hidden_bias,
                       AnyTypeOf<[AnyMemRef]>:$work_area,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$hn_output,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       DefaultValuedAttr<SI64Attr, "-1">:$return_all_steps);
}

def ZLowStickOp:ZLow_Op<"stick", [MemRefsNormalizable]> {
  let summary = "ZLow stick operation";
  let description = [{
  "ZLow operation to perform a stick."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$X,
                       AnyTypeOf<[AnyMemRef]>:$Out,
                       OptionalAttr<StrAttr>:$layout);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$X, "::mlir::Value":$Out), [{
      build($_builder, $_state, X, Out, StringAttr());
    }]>
  ];
}

def ZLowStickForLSTMOp:ZLow_Op<"stickForLSTM", [MemRefsNormalizable]> {
  let summary = "ZLow stick operation for LSTM";
  let description = [{
  "ZLow operation to perform a stick for LSTM."
  "Variadic: list of pointers for input data to be transformed: "
  "  - LSTM concatenated: 4 data pointers, one for each input gate in Forget, Input, Cell, Output (FICO) order, "
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$f_gate,
                       AnyTypeOf<[AnyMemRef]>:$i_gate,
                       AnyTypeOf<[AnyMemRef]>:$c_gate,
                       AnyTypeOf<[AnyMemRef]>:$o_gate,
                       AnyTypeOf<[AnyMemRef]>:$out);
}

def ZLowStickForGRUOp:ZLow_Op<"stickForGRU", [MemRefsNormalizable]> {
  let summary = "ZLow stick operation for GRU";
  let description = [{
  "ZLow operation to perform a stick for GRU."
  "Variadic: list of pointers for input data to be transformed: "
  "  - GRU concatenated: 3 data pointers, one for each input gate in (Z)update, Reset, Hidden, (ZRH) gate order"
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$z_gate,
                       AnyTypeOf<[AnyMemRef]>:$r_gate,
                       AnyTypeOf<[AnyMemRef]>:$h_gate,
                       AnyTypeOf<[AnyMemRef]>:$out);
}

def ZLowUnstickOp:ZLow_Op<"unstick", [MemRefsNormalizable]> {
  let summary = "ZLow unstick operation";
  let description = [{
  "ZLow operation to perform a unstick."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$X,
                       AnyTypeOf<[AnyMemRef]>:$Out,
                       OptionalAttr<StrAttr>:$layout);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$X, "::mlir::Value":$Out), [{
      build($_builder, $_state, X, Out, StringAttr());
    }]>
  ];
}

def ZLowConv2DOp:ZLow_Op<"conv2d", [MemRefsNormalizable]> {
  let summary = "ZLow 2D convolution operation";
  let description = [{
  "ZLow operation to perform 2D convolution."
  "shape is a 1D MemRef (memref<7xi64>) whose items are:;
  "  - 1st item: batch size"
  "  - 2nd item: channel in"
  "  - 3rd item: height in"
  "  - 4th item: width in"
  "  - 5th item: channel out"
  "  - 6th item: height out"
  "  - 7th item: width out"
  "kernel_shape: 1D array of kernel height and width"
  "strides: 1D array of stride height and width"
  "padding_type: SAME_PADDING or VALID_PADDING."
  "act_func: ACT_NONE or ACT_RELU."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$input,
                       AnyTypeOf<[AnyMemRef]>:$input_kernel,
                       AnyTypeOf<[AnyMemRef]>:$input_bias,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$output,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$strides,
                       DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type,
                       DefaultValuedStrAttr<StrAttr, "ACT_NONE">:$act_func);
}

def ZLowAvgPool2DOp:ZLow_Op<"avgpool2d", [MemRefsNormalizable]> {
  let summary = "ZLow 2D average pooling operation";
  let description = [{
  "ZLow operation to perform 2D average pooling."
  "shape is a 1D MemRef (memref<6xi64>) whose items are:;
  "  - 1st item: batch size"
  "  - 2nd item: channel"
  "  - 3rd item: height in"
  "  - 4th item: width in"
  "  - 5th item: height out"
  "  - 6th item: width out"
  "kernel_shape: 1D array of kernel height and width"
  "strides: 1D array of stride height and width"
  "padding_type: SAME_PADDING or VALID_PADDING."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$input,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$output,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$strides,
                       DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type
                  );
}

def ZLowMaxPool2DOp:ZLow_Op<"maxpool2d", [MemRefsNormalizable]> {
  let summary = "ZLow 2D max pooling operation";
  let description = [{
  "ZLow operation to perform 2D max pooling."
  "shape is a 1D MemRef (memref<6xi64>) whose items are:;
  "  - 1st item: batch size"
  "  - 2nd item: channel"
  "  - 3rd item: height in"
  "  - 4th item: width in"
  "  - 5th item: height out"
  "  - 6th item: width out"
  "kernel_shape: 1D array of kernel height and width"
  "strides: 1D array of stride height and width"
  "padding_type: SAME_PADDING or VALID_PADDING."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$input,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$output,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$strides,
                       DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type
                  );
}

def ZLowMeanReduce2DOp:ZLow_Op<"meanreduce2d", [MemRefsNormalizable]> {
  let summary = "ZLow 2D mean reduce operation";
  let description = [{
  "ZLow operation to perform 2D mean reduce."
  "shape is a 1D MemRef (memref<4xindex>) whose items are:;
  "  - 1st item: batch size": 1st dim of input
  "  - 2rd item: height": 2nd dim of input
  "  - 3th item: width": 3rd dim of input
  "  - 4nd item: channel": 4th dim of input
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$input,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$output);
}

def ZLowBatchNormOp:ZLow_Op<"batchnorm", [MemRefsNormalizable]> {
  let summary = "ZLow batchnorm operation";
  let description = [{
  "ZLow operation to perform batchnorm."
  "shape is a 1D MemRef (memref<4xi64>) whose items are:;
  "  - 1st item: batch size"
  "  - 2nd item: height"
  "  - 3rd item: width"
  "  - 4th item: channel"
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$input,
                       AnyTypeOf<[AnyMemRef]>:$A,
                       AnyTypeOf<[AnyMemRef]>:$B,
                       AnyTypeOf<[MemRefOf<[I64]>]>:$shape,
                       AnyTypeOf<[AnyMemRef]>:$output);
}

def ZLowDummyOp:ZLow_Op<"dummy", [MemRefsNormalizable]> {
  let summary = "ZLow dummy operation that behaves like identity";
  let description = [{
  "ZLow operation to forward the input value to the output value."
  "It will be removed if canonicalization is called."
  }];
  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$input);
  let results = (outs AnyTypeOf<[AnyMemRef]>:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input), [{
      build($_builder, $_state, input.getType(), input);
    }]>
  ];
  let hasCanonicalizer = 1;
}

#endif // ZLOW_OPS
