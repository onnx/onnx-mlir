// SPDX-License-Identifier: Apache-2.0

//===----- ZHigh.td -- ZHigh Dialect Operation Definitions -*- tablegen ----==//
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

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "src/Interface/ShapeHelperOpInterface.td"
include "src/Interface/ShapeInferenceOpInterface.td"
include "src/Interface/ResultTypeInferenceOpInterface.td"
include "src/Interface/HasOnnxSubgraphOpInterface.td"
include "src/IR/AttrBase.td"


//===----------------------------------------------------------------------===//
// ZHigh Dialect
//===----------------------------------------------------------------------===//

def ZHigh_Dialect : Dialect {
  let name = "zhigh";
  let summary = "A high-level dialect for the ONNX NNPA accelerator ISA.";
  let cppNamespace = "::onnx_mlir::zhigh";
  let useDefaultAttributePrinterParser = 1;  
  let usePropertiesForAttributes = 0;
}

//===----------------------------------------------------------------------===//
// ZHigh Attribute 
//===----------------------------------------------------------------------===//

// All of the Tensor attributes will extend this class.
class ZHigh_Attr<string name, list<Trait> traits = []>
	: BaseLayoutAttr<ZHigh_Dialect, name, traits>;

// ztensor encoding attribute.
def ZTensorEncodingAttr : ZHigh_Attr<"ZTensorEncoding"> {
  let hasCustomAssemblyFormat = 1;

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

// So far ZTensor supports only F16 for stickified data.
class ZTensorOf<string layout, list<int> ranks> :
    Type<And<[TensorOf<[F16]>.predicate, HasAnyRankOfPred<ranks>,
              DataLayoutOfPred<layout>]>,
         !interleave(!foreach(rank, ranks, rank # "D"), "/") # " " #
         TensorOf<[F16]>.summary # " with layout " # layout,
         "::mlir::TensorType">;

def UnrankedZTensor : UnrankedTensorOf<[F16]>;

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

def ZHighStickOp:ZHigh_Op<"Stick", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh Stick operation";
  let description = [{
    ZHigh operation to perform a Stick."
    If `layout`=`NHWC`, input must be in `NCHW` and output will be in `NHWC`.
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
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighStickOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighStickOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighUnstickOp:ZHigh_Op<"Unstick", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh Unstick operation";
  let description = [{
    ZHigh operation to perform a Unstick.
  }];
  let arguments = (ins AnyTypeOf<[ZTensor_1D, ZTensor_2D, ZTensor_3D, ZTensor_4D,
                                 ZTensor_2DS, ZTensor_3DS, ZTensor_4DS,
                                 ZTensor_NHWC, ZTensor_NCHW, ZTensor_HWCK]>:$In);
  let results = (outs TensorOf<[F32]>:$Out);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$In)>,
  ];
  let hasCanonicalizer = 1;
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighUnstickOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighUnstickOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighF32ToDLF16Op:ZHigh_Op<"F32ToDLF16", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh F32ToDLF16 operation";
  let description = [{
    ZHigh operation to convert a tensor of f32 to a tensor of dlfloat16.
  }];
  let arguments = (ins TensorOf<[F32]>: $In);
  let results = (outs TensorOf<[F16]>:$Out);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$In)>,
  ];
  let hasCanonicalizer = 1;
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighF32ToDLF16Op::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighUnaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighDLF16ToF32Op:ZHigh_Op<"DLF16ToF32", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh DLF16ToF32 operation";
  let description = [{
    ZHigh operation to convert a tensor of dlfloat16 to a tensor of f32.
  }];
  let arguments = (ins TensorOf<[F16]>: $In);
  let results = (outs TensorOf<[F32]>:$Out);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$In)>,
  ];
  let hasCanonicalizer = 1;
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighDLF16ToF32Op::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighUnaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighAddOp:ZHigh_Op<"Add", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh Add operation";
  let description = [{
    ZHigh operation to perform an Add.
    This operation does not support broadcasting.
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighAddOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighBinaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighSubOp:ZHigh_Op<"Sub", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh Sub operation";
  let description = [{
    ZHigh operation to perform a Sub.
    This operation does not support broadcasting.
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighSubOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighBinaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighMulOp:ZHigh_Op<"Mul", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh Mul operation";
  let description = [{
    ZHigh operation to perform a Mul.
    This operation does not support broadcasting.
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighMulOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighBinaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighDivOp:ZHigh_Op<"Div", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh Div operation";
  let description = [{
    ZHigh operation to perform a Div.
    This operation does not support broadcasting.
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighDivOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighBinaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighMinOp:ZHigh_Op<"Min", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh Min operation";
  let description = [{
    ZHigh operation to perform a Min.
    This operation does not support broadcasting.
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighMinOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighBinaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighMaxOp:ZHigh_Op<"Max", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh Max operation";
  let description = [{
    ZHigh operation to perform a Max.
    This operation does not support broadcasting.
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X,
                       AnyTypeOf<[AnyZTensor]>:$Y);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighMaxOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighBinaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighLogOp:ZHigh_Op<"Log", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh Log operation";
  let description = [{
    ZHigh operation to perform a Log.
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighLogOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighUnaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighExpOp:ZHigh_Op<"Exp", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh Exp operation";
  let description = [{
    ZHigh operation to perform a Exp.
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighExpOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighUnaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighReluOp:ZHigh_Op<"Relu", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh Relu operation";
  let description = [{
    "ZHigh operation to perform a Relu."
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighReluOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighUnaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighTanhOp:ZHigh_Op<"Tanh", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh Tanh operation";
  let description = [{
    ZHigh operation to perform a Tanh.
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighTanhOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighUnaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighSigmoidOp:ZHigh_Op<"Sigmoid", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh Sigmoid operation";
  let description = [{
    ZHigh operation to perform a Sigmoid.
  }];
  let arguments = (ins AnyTypeOf<[AnyZTensor]>:$X);
  let results = (outs AnyTypeOf<[AnyZTensor]>:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighSigmoidOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighUnaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighSoftmaxOp:ZHigh_Op<"Softmax", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh Softmax operation";
  let description = [{
    ZHigh operation to perform a Softmax.
    act_func: ACT_NONE or ACT_LOG.
  }];
  let arguments = (ins ZTensor_3DS:$X,
                       DefaultValuedStrAttr<StrAttr, "ACT_NONE">:$act_func);
  let results = (outs ZTensor_3DS:$Out);
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighSoftmaxOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighUnaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighMeanReduce2DOp:ZHigh_Op<"MeanReduce2d", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh 2D mean reduce operation";
  let description = [{
    ZHigh operation to perform 2D mean reduce. Given an input 4D tensor, 
    returns a downsampled tensor reducing the middle 2nd and 3rd dimensions 
    to a size of 1 based on the mean of the original values.
     Input and Output tensors should be in the 3D layout.
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
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighMeanReduce2DOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighMeanReduce2DOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighMaxPool2DOp:ZHigh_Op<"MaxPool2D", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh 2D max pooling operation";
  let description = [{
    ZHigh operation to perform 2D max pooling.
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
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighMaxPool2DOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighPoolingOpShapeHelper<ZHighMaxPool2DOp>(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighAvgPool2DOp:ZHigh_Op<"AvgPool2D", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh 2D average pooling operation";
  let description = [{
    ZHigh operation to perform 2D average pooling.
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
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighAvgPool2DOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighPoolingOpShapeHelper<ZHighAvgPool2DOp>(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighMatMulOp:ZHigh_Op<"MatMul", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh MatMul operation";
  let description = [{
    ZHigh operation to perform a MatMul.
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
  let hasVerifier = 1;
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighMatMulOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighMatMulOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighLSTMOp:ZHigh_Op<"LSTM", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh LSTM operation";
  let description = [{
    zHigh operation to perform a LSTM.
    * Shape for input is `[S, B, I]`. Shape for `h0` and `c0` is `[D, B, H]`.
    * Shape for input_weights is  `[D, I, 4*H]`.
    * Shape for hidden_weights is  `[D, H, 4*H]`.
    * Shape for input_bias and hidden_bias is `[D, 4*H]`.
    * Shape for hn_output is `[S, D, B, H]` if return all timesteps 
      and `[1, D, B, H]` if return the final step only.
    * Shape for cf_output is `[1, D, B, H]`.
    * S is timesteps, D is the number of directions (1 for unidirectional and 
    * 2 for bidirectional), B is batch size, I is input size, and 
    * H is hidden size.
    * direction accepts "forward", "reverse", or "bidirectional
    * return_all_steps: -1 returns all timesteps, 0: returns only the last timestep.
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
  let hasVerifier = 1;
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighLSTMOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighLSTMOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighGRUOp:ZHigh_Op<"GRU", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh GRU operation";
  let description = [{
    * zHigh operation to perform a GRU.
    * Shape for input is `[S, B, I]`. Shape for h0 is `[D, B, H]`.
    * Shape for input_weights is `[D, I, 3*H]`.
    * Shape for hidden_weights is `[D, H, 3*H]`.
    * Shape for input_bias and hidden_bias is `[D, 3*H]`.
    * Shape for hn_output is `[S, D, B, H]` if return all timesteps 
      and `[1, D, B, H]` if return the final step only.
    * S is timesteps, D is the number of directions (1 for unidirectional and 
    * 2 for bidirectional), B is batch size, I is input size, and 
    * H is hidden size.
    * direction accepts "forward", "reverse", or "bidirectional
    * return_all_steps: -1 returns all timesteps, 0: returns only the last timestep."
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
  let hasVerifier = 1;
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighGRUOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighGRUOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighStickForLSTMOp:ZHigh_Op<"StickForLSTM", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh stick operation for LSTM";
  let description = [{
    ZHigh operation to perform a stick for LSTM.
    Variadic: list of pointers for input data to be transformed: 
      - LSTM concatenated: 4 data pointers, one for each input gate in 
    Forget, Input, Cell, Output (FICO) order, 
  }];
  let arguments = (ins TensorOf<[F32]>:$f_gate,
                       TensorOf<[F32]>:$i_gate,
                       TensorOf<[F32]>:$c_gate,
                       TensorOf<[F32]>:$o_gate);
  let results = (outs ZTensor_FICO:$out);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$f_gate, "::mlir::Value":$i_gate,
                   "::mlir::Value":$c_gate, "::mlir::Value":$o_gate), [{
      Type elementType = $_builder.getF16Type();
      UnrankedTensorType resType = UnrankedTensorType::get(elementType);
      build($_builder, $_state, resType, f_gate, i_gate, c_gate, o_gate);
    }]>
  ];
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighStickForLSTMOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighStickForLSTMOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighStickForGRUOp:ZHigh_Op<"StickForGRU", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh stick operation for GRU";
  let description = [{
    ZHigh operation to perform a stick for GRU.
    Variadic: list of pointers for input data to be transformed: 
      - GRU concatenated: 3 data pointers, one for each input gate in
    (Z)update, Reset, Hidden, (ZRH) gate order
  }];
  let arguments = (ins TensorOf<[F32]>:$z_gate,
                       TensorOf<[F32]>:$r_gate,
                       TensorOf<[F32]>:$h_gate);
  let results = (outs ZTensor_ZRH:$out);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$z_gate, "::mlir::Value":$r_gate, "::mlir::Value":$h_gate), [{
      Type elementType = $_builder.getF16Type();
      UnrankedTensorType resType = UnrankedTensorType::get(elementType);
      build($_builder, $_state, resType, z_gate, r_gate, h_gate);
    }]>
  ];
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighStickForGRUOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighStickForGRUOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighConv2DOp:ZHigh_Op<"Conv2D", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh 2D convolution operation";
  let description = [{
    ZHigh operation to perform 2D convolution. 
    * input: `[num_batches, height_in, width_in, channels_in]`
    * input_kernel: `[kernel_height, kernel_width, channels_in, channels_out]` 
    * input_bias: `[channels_out] `
    * kernel_shape: 1D array of kernel height and width 
    * strides: 1D array of stride height and width 
    * padding_type: SAME_PADDING or VALID_PADDING 
    * act_func: ACT_NONE or ACT_RELU 
    * output: `[num_batches, height_out, width_out, channels_out]`
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
  let hasVerifier = 1;
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighConv2DOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighConv2DOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighBatchNormOp:ZHigh_Op<"BatchNorm", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh batchnorm operation";
  let description = [{
    ZHigh operation to perform batchnorm.
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
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighBatchNormOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighUnaryOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighStickifiedConstantOp:ZHigh_Op<"StickifiedConstant", [Pure]> {
  let summary = "ZHigh Stickified Constant operation";
  let description = [{
    This operator produces a constant tensor to store stickified data.
    Stickified data is opaque and must be 4K-aligned. One who produces
    the stickified data must make sure its size in bytes consistent with
    the output tensor's size.
  }];
  let arguments = (ins OptionalAttr<AnyAttr>:$value,
                       DefaultValuedAttr<I64Attr, "4096">:$alignment);
  let results = (outs AnyZTensor:$output);
}

def ZHighStickifiedConstantOfShapeOp:ZHigh_Op<"StickifiedConstantOfShape", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "ZHigh Stickified Constant operation for a dynamic shape";
  let description = [{
    This operator produces a constant tensor to store stickified data.
    The stickified data is defined by a f32 scalar value, a dynamic shape 
    and a layout. Stickified data is 4K-aligned.
  }];
  let arguments = (ins TensorOf<[I64]>:$shape,
                       F32Attr:$value,
                       OptionalAttr<StrAttr>:$layout);
  let results = (outs AnyZTensor:$output);
  let builders = [
    OpBuilder<(ins "::mlir::Value":$shape, "::mlir::FloatAttr":$value,
               "::mlir::StringAttr":$layout)>
  ];
  let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * ZHighStickifiedConstantOfShapeOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper, 
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighStickifiedConstantOfShapeOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighFixGRUYOp:ZHigh_Op<"FixGRUY", [Pure, SameOperandsAndResultLayout,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "Fix Y result of GRU for sequence_lens";
  let description = [{
    Fix Y result of GRU by padding value after sequence_lens.
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y,
    AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens,
    AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$initial_h);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$output);
    let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * $cppClass::getShapeHelper(mlir::Operation *op, llvm::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighFixGRUYOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighFixGRUYhOp:ZHigh_Op<"FixGRUYh", [Pure,
    DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>]> {
  let summary = "Fix Yh result of GRU for sequence_lens";
  let description = [{
    Fix Yh result of GRU by picking the value in Y according to sequence_lens.
  }];
  let arguments = (ins AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$Y,
    AnyTypeOf<[TensorOf<[I32]>, NoneType]>:$sequence_lens);
  let results = (outs AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, NoneType]>:$output);
    let extraClassDefinition = [{
    onnx_mlir::ONNXOpShapeHelper * $cppClass::getShapeHelper(mlir::Operation *op, llvm::ArrayRef<mlir::Value> oper,
        onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
      onnx_mlir::ONNXOpShapeHelper *sh = new ZHighFixGRUYhOpShapeHelper(op, oper, ieb, scope);
      assert(sh && "failed to allocate shape helper");
      return sh;
    }
  }];
}

def ZHighForkOp:ZHigh_Op<"Fork",
    [Pure, DeclareOpInterfaceMethods<ShapeInferenceOpInterface>,
    DeclareOpInterfaceMethods<ShapeHelperOpInterface>,
    DeclareOpInterfaceMethods<ResultTypeInferenceOpInterface>,
    OpInterface<"mlir::HasOnnxSubgraphOpInterface">]> {
  let summary = "ZHigh operation for forking threads.";
  let description = [{
    
  }];
  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>,
                  TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>>:$inputs);
  let results = (outs Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>,
                  TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>>:$results);
  let regions = (region SizedRegion<1>:$body);
  let skipDefaultBuilders = 1;
  let extraClassDeclaration = [{
      int64_t getSubgraphRegionIdx(const std::string& name) {
        if (name == "body") return 0;
        llvm_unreachable("region with the specified name does not exist");
      }
      using BodyBuilderFn =
          llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, mlir::ValueRange)>;
  }];
  let extraClassDefinition = [{
      onnx_mlir::ONNXOpShapeHelper * ZHighForkOp::getShapeHelper(mlir::Operation *op, mlir::ArrayRef<mlir::Value> oper,
          onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {
        onnx_mlir::ONNXOpShapeHelper *sh = new ZHighForkOpShapeHelper(op, oper, ieb, scope);
        assert(sh && "failed to allocate shape helper");
        return sh;
      }
  }];
  let builders = [
    OpBuilder<(ins "mlir::TypeRange":$resultTypes,
      "mlir::ValueRange":$operands,
      CArg<"llvm::function_ref<void(mlir::OpBuilder &, mlir::Location, mlir::ValueRange)>",
           "nullptr">:$bodyBuilder)>
  ];
}

def ZHighJoinOp : ZHigh_Op<"Join"> {
  let summary = "join threads";
  let description = [{
    The `zhigh.join` operation waits until the argument becomes ready.
  }];

  let arguments = (ins Variadic<AnyTypeOf<[TensorOf<[F16]>, TensorOf<[F32]>, TensorOf<[F64]>, TensorOf<[UI32]>,
                  TensorOf<[UI64]>, TensorOf<[I32]>, TensorOf<[I64]>, TensorOf<[BF16]>]>>:$operand);
}

#endif // ZHIGH_OPS
