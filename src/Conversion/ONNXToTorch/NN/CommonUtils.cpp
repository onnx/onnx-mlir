#include "CommonUtils.h"

std::vector<Value>
createPadsArrayAttribute(::mlir::ArrayAttr pads, Type ty, Location loc,
                         ConversionPatternRewriter &rewriter) {
  // Read ONNX side pads values and store inside a vector
  std::vector<Value> translatepadsList;
  if (!pads)  {
    for (unsigned i = 0; i < 2; i++) {
      Value zeroPaddingValue = rewriter.create<ConstantIntOp>(loc,
          IntegerAttr::get(ty, 0));
      translatepadsList.push_back(zeroPaddingValue);
    }
  } else {
    // Determine if padding is symmetrical
    // `onnx-mlir` padding has the following form
    // (pad_dim1_start, pad_dim2_start, pad_dim1_end, pad_dim2_end)
    bool is_symmetric = true;
    if (pads[0] != pads[2] || pads[1] != pads[3])
      is_symmetric = false;

    // Create appropriate padding vectors based on padding symmetry
    if (is_symmetric) {
      for (unsigned i = 0; i < pads.size(); i += 2) {
        auto pad = (pads[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
        auto padAttr = IntegerAttr::get(ty, pad);
        Value padValue = rewriter.create<ConstantIntOp>(loc, padAttr);
        translatepadsList.push_back(padValue);
      }
    } else {
      // `torch-mlir` only allows symmetric 2-dimensional padding for conv2d and
      // maxpool2d; therefore we pass the entire padding vector and insert
      // zeropad2d ops
      for (unsigned i = 0; i < pads.size(); i++) {
        auto pad = (pads[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
        auto padAttr = IntegerAttr::get(ty, pad);
        Value padValue = rewriter.create<ConstantIntOp>(loc, padAttr);
        translatepadsList.push_back(padValue);
      }
      // `torch-mlir` expects (pad_dim1_start, pad_dim1_end, ...)
      std::swap(translatepadsList[1], translatepadsList[2]);
    }
  }
  return translatepadsList;
}

std::vector<Value> createArrayAttribute(::mlir::ArrayAttr onnxArrayAttr,
                                        Type ty, Location loc,
                                        ConversionPatternRewriter &rewriter,
                                        int default_val) {
  std::vector<Value> operandArrayValues;
  if (onnxArrayAttr) {
    for (unsigned int i = 0; i < onnxArrayAttr.size(); i++) {
      auto f1 = IntegerAttr::get(
          ty,
          (onnxArrayAttr[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue());
      Value p1v = rewriter.create<ConstantIntOp>(loc, f1);
      operandArrayValues.push_back(p1v);
    }
  } else {
    auto f0 = IntegerAttr::get(ty, default_val);
    Value p0v = rewriter.create<ConstantIntOp>(loc, f0);
    operandArrayValues = {p0v, p0v};
  }
  return operandArrayValues;
}

/// Converts ONNX operand of type tensor to Torch tensor
///
/// Typical usage:
/// \code
///   auto operandType = toTorchType(context, unaryOp.getOperand().getType());
/// \endcode
///
/// \param ctx: context of the operand
/// \param t: tensor type of the operand
///
/// \returns Torch::ValueTensorType conversion from tensor
Torch::ValueTensorType toTorchType(mlir::MLIRContext *ctx, Type t) {
  auto type = t.template dyn_cast<TensorType>();
  return Torch::ValueTensorType::get(ctx, type.getShape(),
                                     type.getElementType());
}

Torch::ValueTensorType toSI64SignedType(mlir::MLIRContext *ctx, Type t) {
   auto type = t.template dyn_cast<TensorType>();
   auto elementType = IntegerType::get(type.getContext(), 64, IntegerType::Signed);
   return Torch::ValueTensorType::get(ctx, type.getShape(), elementType);
}

/// Get Torch tensor from mlir::Value tensor
///
/// \param operand: operand tensor
/// \param rewriter: rewriter object related to the operator
/// \param context: context related to operator
/// \param loc: location related to operator
///
/// \returns mlir::Value tensor of torch type
mlir::Value getTorchTensor(Value operand, ConversionPatternRewriter &rewriter,
                           mlir::MLIRContext *context, Location loc) {
  auto operandType = toTorchType(context, operand.getType());
  return rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
      loc, operandType, operand);
}

/// Get mlir::Value from int
///
/// \param val: input integer
/// \param rewriter: rewriter object related to the operator
/// \param context: context related to operator
/// \param loc: location related to operator
///
/// \returns mlir::Value of constant integer
Value getIntValue(int val, ConversionPatternRewriter &rewriter,
                  mlir::MLIRContext *context, Location loc) {
  auto iType = IntegerType::get(context, 64);
  auto iVal = IntegerAttr::get(iType, val);
  return rewriter.create<ConstantIntOp>(loc, iVal);
}

/// Get vector of ints from mlir::ArrayAttr<IntegerAttr>
///
/// \param operand: operand tensor
/// \param rewriter: rewriter object related to the operator
/// \param context: context related to operator
/// \param loc: location related to operator
///
/// \returns vector of integers
std::vector<int> toVector(mlir::ArrayAttr arr) {
  std::vector<int> elements;

  for (auto element : arr) {
    auto j = element.dyn_cast<IntegerAttr>();
    int64_t k = j.getValue().getSExtValue();
    elements.push_back(k);
  }

  return elements;
}
