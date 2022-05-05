#include "CommonUtils.h"

typedef struct dim_pads {
  int dim_start;
  int dim_end;
} dim_pads;

std::vector<Value> createPadsArrayAttribute(::mlir::ArrayAttr pads, Type ty,
    Location loc, ConversionPatternRewriter &rewriter) {
  // Reading the ONNX side pads values and store in the array.
  std::vector<Value> translatepadsList;
  if (!pads)
    return translatepadsList;

  bool is_symmetric = true;
  for (unsigned int i = 0; i < pads.size(); i += 2) {
    if (pads[i] != pads[i + 1]) {
      is_symmetric = false;
      break;
    }
  }
  assert(
      is_symmetric && "Frontend transformations only handle symmetric padding");

  dim_pads dimArray[pads.size()];
  if (is_symmetric) {
    for (unsigned int i = 0; i < pads.size(); i += 2) {
      auto pad_value =
          (pads[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
      auto f0 = IntegerAttr::get(ty, pad_value);
      Value p0v = rewriter.create<ConstantIntOp>(loc, f0);
      translatepadsList.push_back(p0v);
    }
  } else {
    int j = 0;
    for (unsigned int i = 0; i < pads.size(); i++) {
      dimArray[j].dim_start =
          (pads[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
      i++;
      dimArray[j].dim_end =
          (pads[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
      j++;
    }

    // read the onnx pad values from array(dim_start values)
    int k = 0;
    for (unsigned int i = 0; i < pads.size(); i = i + 2) {
      auto f0 = IntegerAttr::get(ty, (dimArray[k].dim_start));
      Value p0v = rewriter.create<ConstantIntOp>(loc, f0);
      translatepadsList.push_back(p0v);
      k++;
    }

    // read the onnx pad values from array(dim_end values)
    k = 0;
    for (unsigned int i = 0; i < pads.size(); i = i + 2) {
      auto f1 = IntegerAttr::get(ty, (dimArray[k].dim_end));
      Value p1v = rewriter.create<ConstantIntOp>(loc, f1);
      translatepadsList.push_back(p1v);
      k++;
    }
  }
  return translatepadsList;
}

std::vector<Value> createArrayAttribute(::mlir::ArrayAttr onnxArrayAttr,
    Type ty, Location loc, ConversionPatternRewriter &rewriter,
    int default_val) {
  std::vector<Value> operandArrayValues;
  if (onnxArrayAttr) {
    for (unsigned int i = 0; i < onnxArrayAttr.size(); i++) {
      auto f1 = IntegerAttr::get(ty,
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
   return Torch::ValueTensorType::get(ctx, type.getShape(), type.getElementType());
}
