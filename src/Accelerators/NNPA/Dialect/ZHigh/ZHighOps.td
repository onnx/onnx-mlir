// SPDX-License-Identifier: Apache-2.0

//===-- ZHighOps.td -- ZHigh Dialect Operation Definitions -*- tablegen ----==//
//
// Copyright 2019-2020 The IBM Research Authors
//
// =============================================================================
//
// Defines ZHigh Dialect operations.
//
//===----------------------------------------------------------------------===//

#ifndef ZHIGH_OPS
#define ZHIGH_OPS

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "src/Interface/ShapeInferenceOpInterface.td"

//===----------------------------------------------------------------------===//
// ZHigh Dialect
//===----------------------------------------------------------------------===//

def ZHigh_Dialect : Dialect {
  let name = "zhigh";
  let cppNamespace = "::onnx_mlir::zhigh";
}

//===----------------------------------------------------------------------===//
// ZHigh Attribute 
//===----------------------------------------------------------------------===//

// All of the Tensor attributes will extend this class.
class ZHigh_Attr<string name, list<Trait> traits = []>
	: AttrDef<ZHigh_Dialect, name, traits>;

// ztensor encoding attribute.
def ZTensorEncodingAttr : ZHigh_Attr<"ZTensorEncoding"> {
  let mnemonic = "encoding";

  let description = [{
    An attribute to encode information on properties of ztensors. A tensor with
    this encoding attribute indicates that it is a ztensor whose type is
    the original type of the pre-stickified data.

    `dataLayout` indicates the data layout of the pre-stickified data.

    TODO: should we add an affine_map to describe how the data is stickified?

    Example:

    ```mlir
    #ZAIU_NHWC= #zhigh.encoding<{
      dataLayout = "nhwc"
    }>


    ... tensor<8x8xf64, #ZAIU_NHWC> ...
    ```
  }];

  // Data in ztensor encoding.
  let parameters = (ins
    // A original data layout of the pre-stickified data.
    "ZTensorEncodingAttr::DataLayout":$dataLayout
  );

  let extraClassDeclaration = [{
    enum class DataLayout {
      UNDEFINED, _1D, _2D, _2DS, _3D, _3DS, _4D, _4DS,
      NCHW, NHWC, HWCK,
      FICO, ZRH, BFICO, BZRH
    };
  }];

  let cppNamespace = "::onnx_mlir::zhigh";
}

// Whether a ztensor type has the specified layout.
class DataLayoutOfPred<string layout> : And<[
  CPred<"($_self.cast<::mlir::RankedTensorType>()) &&"
        "($_self.cast<::mlir::RankedTensorType>()."
        "getEncoding().dyn_cast_or_null<ZTensorEncodingAttr>()) &&"
        "($_self.cast<::mlir::RankedTensorType>()."
        "getEncoding().cast<ZTensorEncodingAttr>().getDataLayout()"
        " == ZTensorEncodingAttr::DataLayout::" # layout # ")"> 
]>;

// So far ZTensor supports only F32 for pre-stickified data.
class ZTensorOf<string layout, list<int> ranks> :
    Type<And<[TensorOf<[F32]>.predicate, HasAnyRankOfPred<ranks>,
              DataLayoutOfPred<layout>]>,
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         TensorOf<[F32]>.summary # " with layout " # layout,
         "::mlir::TensorType">;

def UnrankedZTensor : UnrankedTensorOf<[F32]>;

def ZTensor_1D: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"_1D", [1]>]>;
def ZTensor_2D: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"_2D", [2]>]>;
def ZTensor_2DS: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"_2DS", [2]>]>;
def ZTensor_3D: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"_3D", [3]>]>;
def ZTensor_3DS: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"_3DS", [3]>]>;
def ZTensor_4D: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"_4D", [4]>]>;
def ZTensor_4DS: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"_4DS", [4]>]>;
def ZTensor_NHWC: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"NHWC", [4]>]>;
def ZTensor_NCHW: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"NCHW", [4]>]>;
def ZTensor_HWCK: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"HWCK", [4]>]>;
def ZTensor_FICO: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"FICO", [2, 3]>]>;
def ZTensor_ZRH: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"ZRH", [2, 3]>]>;
def ZTensor_BFICO: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"BFICO", [2, 3]>]>;
def ZTensor_BZRH: AnyTypeOf<[UnrankedZTensor, ZTensorOf<"BZRH", [2, 3]>]>;

def AnyZTensor: AnyTypeOf<[ZTensor_1D, ZTensor_2D, ZTensor_3D, ZTensor_4D,
      ZTensor_2DS, ZTensor_3DS, ZTensor_4DS,
      ZTensor_NCHW, ZTensor_NHWC, ZTensor_HWCK,
      ZTensor_FICO, ZTensor_ZRH, ZTensor_BFICO, ZTensor_BZRH]>;

//===----------------------------------------------------------------------===//
// ZHigh Operations
//===----------------------------------------------------------------------===//

// Op has the same operand and result layout.
def SameOperandsAndResultLayout: NativeOpTrait<"SameOperandsAndResultLayout">;

//===----------------------------------------------------------------------===//
// Base class for ZHigh dialect operations. This operation inherits from the
// base `Op` class in OpBase.td, and provides:
//   * The parent dialect of the operation.
//   * The mnemonic for the operation, or the name without the dialect prefix.
//   * A list of traits for the operation.

class ZHigh_Op<string mnemonic, list<Trait> traits = []> :
    Op<ZHigh_Dialect, mnemonic, traits>;

def ZHighStickOp:ZHigh_Op<"Stick", [NoSideEffect,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh Stick operation";
  let description = [{
    "ZHigh operation to perform a Stick."
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F32]>, NoneType]>:$In,
                       OptionalAttr<StrAttr>:$layout);
  let results = (outs AnyTypeOf<[ZTensor_1D, ZTensor_2D, ZTensor_3D, ZTensor_4D,
                                 ZTensor_2DS, ZTensor_3DS, ZTensor_4DS,
                                 ZTensor_NHWC,  ZTensor_NCHW, ZTensor_HWCK,
                                 NoneType]>:$Out);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$In, "::mlir::StringAttr":$layout)>
  ];
  let hasCanonicalizer = 1;
}

def ZHighUnstickOp:ZHigh_Op<"Unstick", [NoSideEffect,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh Unstick operation";
  let description = [{
    "ZHigh operation to perform a Unstick."
  }];
  let arguments = (ins AnyTypeOf<[ZTensor_1D, ZTensor_2D, ZTensor_3D, ZTensor_4D,
                                 ZTensor_2DS, ZTensor_3DS, ZTensor_4DS,
                                 ZTensor_NHWC, ZTensor_NCHW, ZTensor_HWCK]>:$In);
  let results = (outs TensorOf<[F32]>:$Out);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$In)>,
  ];
  let hasCanonicalizer = 1;
}

def ZHighAddOp:ZHigh_Op<"Add", [NoSideEffect, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh Add operation";
  let description = [{
    "ZHigh operation to perform an Add."
    "This operation does not support broadcasting."
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
}

def ZHighSubOp:ZHigh_Op<"Sub", [NoSideEffect, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh Sub operation";
  let description = [{
    "ZHigh operation to perform a Sub."
    "This operation does not support broadcasting."
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
}

def ZHighMulOp:ZHigh_Op<"Mul", [NoSideEffect, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh Mul operation";
  let description = [{
    "ZHigh operation to perform a Mul."
    "This operation does not support broadcasting."
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
}

def ZHighDivOp:ZHigh_Op<"Div", [NoSideEffect, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh Div operation";
  let description = [{
    "ZHigh operation to perform a Div."
    "This operation does not support broadcasting."
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
}

def ZHighMinOp:ZHigh_Op<"Min", [NoSideEffect, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh Min operation";
  let description = [{
    "ZHigh operation to perform a Min."
    "This operation does not support broadcasting."
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
}

def ZHighMaxOp:ZHigh_Op<"Max", [NoSideEffect, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh Max operation";
  let description = [{
    "ZHigh operation to perform a Max."
    "This operation does not support broadcasting."
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
}

def ZHighLogOp:ZHigh_Op<"Log", [NoSideEffect, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh Log operation";
  let description = [{
    "ZHigh operation to perform a Log."
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
}

def ZHighExpOp:ZHigh_Op<"Exp", [NoSideEffect, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh Exp operation";
  let description = [{
    "ZHigh operation to perform a Exp."
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
}

def ZHighReluOp:ZHigh_Op<"Relu", [NoSideEffect, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh Relu operation";
  let description = [{
    "ZHigh operation to perform a Relu."
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
}

def ZHighTanhOp:ZHigh_Op<"Tanh", [NoSideEffect, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh Tanh operation";
  let description = [{
    "ZHigh operation to perform a Tanh."
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
}

def ZHighSigmoidOp:ZHigh_Op<"Sigmoid", [NoSideEffect, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh Sigmoid operation";
  let description = [{
    "ZHigh operation to perform a Sigmoid."
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
}

def ZHighSoftmaxOp:ZHigh_Op<"Softmax", [NoSideEffect, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh Softmax operation";
  let description = [{
    "ZHigh operation to perform a Softmax."
    "act_func: ACT_NONE or ACT_LOG."
  }];
  let arguments = (ins ZTensor_3DS:$X,
                       DefaultValuedStrAttr<StrAttr, "ACT_NONE">:$act_func);
  let results = (outs ZTensor_3DS:$Out);
}

def ZHighMeanReduce2DOp:ZHigh_Op<"MeanReduce2d", [NoSideEffect,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh 2D mean reduce operation";
  let description = [{
    "ZHigh operation to perform 2D mean reduce. Given an input 4D tensor, "
    "returns a downsampled tensor reducing the middle 2nd and 3rd dimensions "
    "to a size of 1 based on the mean of the original values."
    " Input and Output tensors should be in the 3D layout."
  }];
  let arguments = (ins ZTensor_NHWC:$input);
  let results = (outs ZTensor_NHWC:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input), [{
      Type elementType = input.getType().cast<ShapedType>().getElementType();
      UnrankedTensorType resType = UnrankedTensorType::get(elementType);
      build($_builder, $_state, resType, input);
    }]>
  ];
}

def ZHighMaxPool2DOp:ZHigh_Op<"MaxPool2D", [NoSideEffect,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh 2D max pooling operation";
  let description = [{
    "ZHigh operation to perform 2D max pooling."
  }];
  let arguments = (ins ZTensor_NHWC:$input,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$strides,
                       DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type
  );
  let results = (outs ZTensor_NHWC:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "::mlir::ArrayAttr":$kernel_shape,
                   "::mlir::ArrayAttr":$strides, "::mlir::StringAttr":$padding_type), [{
      Type elementType = input.getType().cast<ShapedType>().getElementType();
      UnrankedTensorType resType = UnrankedTensorType::get(elementType);
      build($_builder, $_state, resType, input, kernel_shape, strides, padding_type);
    }]>
  ];
}

def ZHighAvgPool2DOp:ZHigh_Op<"AvgPool2D", [NoSideEffect,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh 2D average pooling operation";
  let description = [{
    "ZHigh operation to perform 2D average pooling."
  }];
  let arguments = (ins ZTensor_NHWC:$input,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$strides,
                       DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type
  );
  let results = (outs ZTensor_NHWC:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "::mlir::ArrayAttr":$kernel_shape,
                   "::mlir::ArrayAttr":$strides, "::mlir::StringAttr":$padding_type), [{
      Type elementType = input.getType().cast<ShapedType>().getElementType();
      UnrankedTensorType resType = UnrankedTensorType::get(elementType);
      build($_builder, $_state, resType, input, kernel_shape, strides, padding_type);
    }]>
  ];
}

def ZHighMatMulOp:ZHigh_Op<"MatMul", [NoSideEffect,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh MatMul operation";
  let description = [{
    "ZHigh operation to perform a MatMul."
  }];
  let arguments = (ins AnyTypeOf<[ZTensor_2D, ZTensor_3DS]>:$X,
                       AnyTypeOf<[ZTensor_2D, ZTensor_3DS]>:$Y,
                       AnyTypeOf<[ZTensor_1D, ZTensor_2DS, NoneType]>:$B);

  let results = (outs AnyTypeOf<[ZTensor_2D, ZTensor_3DS]>:$Out);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$X, "::mlir::Value":$Y, "::mlir::Value":$B), [{
      Type elementType = X.getType().cast<ShapedType>().getElementType();
      UnrankedTensorType resType = UnrankedTensorType::get(elementType);
      build($_builder, $_state, resType, X, Y, B);
    }]>
  ];
  let verifier = [{ return ::onnx_mlir::zhigh::verify(*this); }];
}

def ZHighLSTMOp:ZHigh_Op<"LSTM", [NoSideEffect,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh LSTM operation";
  let description = [{
    "zHigh operation to perform a LSTM."
    "Shape for input is [S, B, I]. Shape for h0 and c0 is [D, B, H]."
    "Shape for input_weights is  [D, I, 4*H]."
    "Shape for hidden_weights is  [D, H, 4*H]."
    "Shape for input_bias and hidden_bias is [D, 4*H]."
    "Shape for hn_output is [S, D, B, H] if return all timesteps "
    "and [1, D, B, H] if return the final step only."
    "Shape for cf_output is [1, D, B, H]."
    "S is timesteps, D is the number of directions (1 for unidirectional and "
    "2 for bidirectional), B is batch size, I is input size, and "
     H is hidden size."
    "direction accepts "forward", "reverse", or "bidirectional"
    "return_all_steps: -1 returns all timesteps, 0: returns only the last timestep.
  }];
  let arguments = (ins ZTensor_3DS:$input,
                       AnyTypeOf<[ZTensor_3DS, NoneType]>:$h0,
                       AnyTypeOf<[ZTensor_3DS, NoneType]>:$c0,
                       ZTensor_FICO:$input_weights,
                       AnyTypeOf<[ZTensor_FICO, NoneType]>:$input_bias,
                       ZTensor_FICO:$hidden_weights,
                       AnyTypeOf<[ZTensor_FICO, NoneType]>:$hidden_bias,
                       SI64Attr:$hidden_size,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       DefaultValuedAttr<SI64Attr, "-1">:$return_all_steps);
  let results = (outs ZTensor_4DS:$hn_output,
                      ZTensor_4DS:$cf_output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "::mlir::Value":$h0, "::mlir::Value":$c0,
                   "::mlir::Value":$input_weights, "::mlir::Value":$input_bias,
            		   "::mlir::Value":$hidden_weights, "::mlir::Value":$hidden_bias,
                   "::mlir::IntegerAttr":$hidden_size, "::mlir::StringAttr":$direction,
                   "::mlir::IntegerAttr":$return_all_steps), [{
      Type elementType = input.getType().cast<ShapedType>().getElementType();
      UnrankedTensorType resType = UnrankedTensorType::get(elementType);
      build($_builder, $_state, resType, resType,
            input, h0, c0, input_weights, input_bias, hidden_weights,
            hidden_bias, hidden_size, direction, return_all_steps);
    }]>
  ];
  let verifier = [{ return ::onnx_mlir::zhigh::verify(*this); }];
}

def ZHighGRUOp:ZHigh_Op<"GRU", [NoSideEffect,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh GRU operation";
  let description = [{
    "zHigh operation to perform a GRU."
    "Shape for input is [S, B, I]. Shape for h0 is [D, B, H]."
    "Shape for input_weights is  [D, I, 3*H]."
    "Shape for hidden_weights is  [D, H, 3*H]."
    "Shape for input_bias and hidden_bias is [D, 3*H]."
    "Shape for hn_output is [S, D, B, H] if return all timesteps "
    "and [1, D, B, H] if return the final step only."
    "S is timesteps, D is the number of directions (1 for unidirectional and "
    "2 for bidirectional), B is batch size, I is input size, and "
     H is hidden size."
    "direction accepts "forward", "reverse", or "bidirectional"
    "return_all_steps: -1 returns all timesteps, 0: returns only the last timestep.
  }];
  let arguments = (ins ZTensor_3DS:$input,
                       AnyTypeOf<[ZTensor_3DS, NoneType]>:$h0,
                       ZTensor_ZRH:$input_weights,
                       AnyTypeOf<[ZTensor_ZRH, NoneType]>:$input_bias,
                       ZTensor_ZRH:$hidden_weights,
                       AnyTypeOf<[ZTensor_ZRH, NoneType]>:$hidden_bias,
                       SI64Attr:$hidden_size,
                       DefaultValuedStrAttr<StrAttr, "forward">:$direction,
                       DefaultValuedAttr<SI64Attr, "-1">:$return_all_steps);
  let results = (outs ZTensor_4DS:$hn_output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "::mlir::Value":$h0, "::mlir::Value":$input_weights,
                   "::mlir::Value":$input_bias, "::mlir::Value":$hidden_weights,
                   "::mlir::Value":$hidden_bias, "::mlir::IntegerAttr":$hidden_size,
                   "::mlir::StringAttr":$direction, "::mlir::IntegerAttr":$return_all_steps), [{
      Type elementType = input.getType().cast<ShapedType>().getElementType();
      UnrankedTensorType resType = UnrankedTensorType::get(elementType);
      build($_builder, $_state, resType,
            input, h0, input_weights, input_bias, hidden_weights,
            hidden_bias, hidden_size, direction, return_all_steps);
    }]>
  ];
  let verifier = [{ return ::onnx_mlir::zhigh::verify(*this); }];
}

def ZHighStickForLSTMOp:ZHigh_Op<"StickForLSTM", [NoSideEffect,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh stick operation for LSTM";
  let description = [{
    "ZHigh operation to perform a stick for LSTM."
    "Variadic: list of pointers for input data to be transformed: "
    "  - LSTM concatenated: 4 data pointers, one for each input gate in "
    "Forget, Input, Cell, Output (FICO) order, "
  }];
  let arguments = (ins TensorOf<[F32]>:$f_gate,
                       TensorOf<[F32]>:$i_gate,
                       TensorOf<[F32]>:$c_gate,
                       TensorOf<[F32]>:$o_gate);
  let results = (outs ZTensor_FICO:$out);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$f_gate, "::mlir::Value":$i_gate,
                   "::mlir::Value":$c_gate, "::mlir::Value":$o_gate), [{
      Type elementType = f_gate.getType().cast<ShapedType>().getElementType();
      UnrankedTensorType resType = UnrankedTensorType::get(elementType);
      build($_builder, $_state, resType, f_gate, i_gate, c_gate, o_gate);
    }]>
  ];
}

def ZHighStickForGRUOp:ZHigh_Op<"StickForGRU", [NoSideEffect,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh stick operation for GRU";
  let description = [{
    "ZHigh operation to perform a stick for GRU."
    "Variadic: list of pointers for input data to be transformed: "
    "  - GRU concatenated: 3 data pointers, one for each input gate in " 
    "(Z)update, Reset, Hidden, (ZRH) gate order"
  }];
  let arguments = (ins TensorOf<[F32]>:$z_gate,
                       TensorOf<[F32]>:$r_gate,
                       TensorOf<[F32]>:$h_gate);
  let results = (outs ZTensor_ZRH:$out);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$z_gate, "::mlir::Value":$r_gate, "::mlir::Value":$h_gate), [{
      Type elementType = z_gate.getType().cast<ShapedType>().getElementType();
      UnrankedTensorType resType = UnrankedTensorType::get(elementType);
      build($_builder, $_state, resType, z_gate, r_gate, h_gate);
    }]>
  ];
}

def ZHighConv2DOp:ZHigh_Op<"Conv2D", [NoSideEffect,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh 2D convolution operation";
  let description = [{
    "ZHigh operation to perform 2D convolution. "
    "input: [num_batches, height_in, width_in, channels_in] "
    "input_kernel: [kernel_height, kernel_width, channels_in, channels_out] "
    "input_bias: [channels_out] "
    "kernel_shape: 1D array of kernel height and width "
    "strides: 1D array of stride height and width "
    "padding_type: SAME_PADDING or VALID_PADDING "
    "act_func: ACT_NONE or ACT_RELU "
    "output: [num_batches, height_out, width_out, channels_out]"
  }];
  let arguments = (ins ZTensor_NHWC:$input,
                       ZTensor_HWCK:$input_kernel,
                       AnyTypeOf<[ZTensor_1D, NoneType]>:$input_bias,
                       I64ArrayAttr:$kernel_shape,
                       I64ArrayAttr:$strides,
                       DefaultValuedStrAttr<StrAttr, "SAME_PADDING">:$padding_type,
                       DefaultValuedStrAttr<StrAttr, "ACT_NONE">:$act_func
  );
  let results = (outs ZTensor_NHWC:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "::mlir::Value":$input_kernel, "::mlir::Value":$input_bias,
                   "::mlir::ArrayAttr":$kernel_shape, "::mlir::ArrayAttr":$strides,
                   "::mlir::StringAttr":$padding_type, "::mlir::StringAttr":$act_func), [{
      Type elementType = input.getType().cast<ShapedType>().getElementType();
      UnrankedTensorType resType = UnrankedTensorType::get(elementType);
      build($_builder, $_state, resType, input, input_kernel, input_bias,
            kernel_shape, strides, padding_type, act_func);
    }]>
  ];
  let verifier = [{ return ::onnx_mlir::zhigh::verify(*this); }];
}

def ZHighBatchNormOp:ZHigh_Op<"BatchNorm", [NoSideEffect,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>]> {
  let summary = "ZHigh batchnorm operation";
  let description = [{
    "ZHigh operation to perform batchnorm."
  }];
  let arguments = (ins ZTensor_NHWC:$input,
                       ZTensor_1D:$a,
                       ZTensor_1D:$b
                  );
  let results = (outs ZTensor_NHWC:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$input, "::mlir::Value":$a, "::mlir::Value":$b), [{
      build($_builder, $_state, input.getType(), input, a, b);
    }]>
  ];
}

def ZHighStickifiedConstantOp:ZHigh_Op<"StickifiedConstant", [NoSideEffect]> {
  let summary = "ZHigh Stickified Constant operation";
  let description = [{
    "This operator produces a constant tensor to store stickified data." 
    "Stickified data is opaque and must be 4K-aligned. One who produces"
    "the stickified data must make sure its size in bytes consistent with"
    "the output tensor's size."
  }];
  let arguments = (ins OptionalAttr<AnyAttr>:$value,
                       DefaultValuedAttr<I64Attr, "4096">:$alignment);
  let results = (outs AnyZTensor:$output);
}

#endif // ZHIGH_OPS
