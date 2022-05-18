#include "CommonUtils.h"
#include <set>
#include <vector>

typedef struct dim_pads {
  int dim_start;
  int dim_end;
} dim_pads;

std::vector<Value>
createPadsArrayAttribute(::mlir::ArrayAttr pads, Type ty, Location loc,
                         ConversionPatternRewriter &rewriter) {
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
  assert(is_symmetric &&
         "Frontend transformations only handle symmetric padding");

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

Value getIntValue(int val, ConversionPatternRewriter &rewriter,
                  mlir::MLIRContext *context, Location loc) {
  auto iType = IntegerType::get(context, 64);
  auto iVal = IntegerAttr::get(iType, val);
  return rewriter.create<ConstantIntOp>(loc, iVal);
}

std::vector<int> toUniqueAndNonNegative(std::vector<int> axes) {
  std::set<int> axesSet(axes.begin(), axes.end());
  std::vector<int> axesNonNeg;

  for (auto x : axesSet) {
    // positive integers are added as it
    // negative integers are normarlized to positive
    axesNonNeg.push_back((x > 0) ? x : (x + axesSet.size()));
  }
  return axesNonNeg;
}

std::vector<int> getSortedWithNegativeAxes(std::vector<int> axesRaw) {
  auto axesNonNegative = toUniqueAndNonNegative(axesRaw);
  auto axesSorted = axesNonNegative;

  std::sort(axesSorted.begin(), axesSorted.end());

  return axesSorted;
}

mlir::Value squeezeResult(std::vector<int> axes, mlir::Value dataTensor,
                          Torch::ValueTensorType resultType,
                          ConversionPatternRewriter &rewriter,
                          mlir::MLIRContext *context, Location loc) {
  Value result = dataTensor;

  if (axes.size() > 0) {
    for (auto i = 0; i < axes.size(); i++) {
      auto dataType = result.getType().dyn_cast<TensorType>();

      // With every successive deleting on dimension, the input axis
      // changes to `axis = axis - number_of_dimensions_deleted`
      // This works because, axes is sorted and normalized to possitive integers
      auto dim_raw = axes[i] - i;
      // assert((dataType.getShape()[dim_raw] == 1) && "Cannot squeeze for
      // dim");
      Value dim = getIntValue(dim_raw, rewriter, context, loc);
      result = rewriter.create<AtenSqueezeDimOp>(loc, resultType, result, dim);
    }
  } else {
    result = rewriter.create<AtenSqueezeOp>(loc, resultType, dataTensor);
  }

  return result;
}
