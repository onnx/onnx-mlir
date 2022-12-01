#include "CommonUtils.h"

dim_pads createPadsArrayAttribute(::mlir::ArrayAttr pads, Type ty, Location loc,
    ConversionPatternRewriter &rewriter) {
  // Read ONNX side pads values and store inside a vector
  if (!pads) {
    Value zeroPadding =
        rewriter.create<ConstantIntOp>(loc, IntegerAttr::get(ty, 0));
    // TODO: Derive number of pads based on input tensor
    return dim_pads{.padding = {zeroPadding, zeroPadding}, .isSymmetric = true};
  }

  // Determine if padding is symmetrical
  // `onnx-mlir` padding has the following form (pad_dim1_start,
  // pad_dim2_start, ..., pad_dim1_end, pad_dim2_end, ...)
  bool is_symmetric = true;
  std::vector<Value> padsList;
  auto padIndices = llvm::iota_range<unsigned>(0, pads.size() / 2, false);
  for (auto i : padIndices) {
    if (pads[i] != pads[i + (pads.size() / 2)]) {
      is_symmetric = false;
      break;
    }
  }

  // Create appropriate padding vectors based on padding symmetry
  if (is_symmetric) {
    for (auto i : llvm::reverse(padIndices)) {
      Value padValue =
          rewriter.create<ConstantIntOp>(loc, pads[i].cast<IntegerAttr>());
      padsList.push_back(padValue);
    }
  } else {
    // `torch-mlir` only allows symmetric 2-dimensional padding for conv2d
    // and maxpool2d; therefore we pass the entire padding vector and
    // insert zeropad2d ops. (pad_dimN_start, pad_dimN_end, ...,
    // pad_dim1_start, pad_dim1_end)
    for (auto i : llvm::reverse(padIndices)) {
      Value padValueStart =
          rewriter.create<ConstantIntOp>(loc, pads[i].cast<IntegerAttr>());
      padsList.push_back(padValueStart);
      Value padValueEnd = rewriter.create<ConstantIntOp>(
          loc, pads[i + (pads.size() / 2)].cast<IntegerAttr>());
      padsList.push_back(padValueEnd);
    }
  }
  return dim_pads{.padding = padsList, .isSymmetric = is_symmetric};
}

std::vector<Value> createArrayAttribute(::mlir::ArrayAttr onnxArrayAttr,
    Type ty, Location loc, ConversionPatternRewriter &rewriter,
    int default_val) {
  std::vector<Value> operandArrayValues;
  if (onnxArrayAttr) {
    for (unsigned int i = 0; i < onnxArrayAttr.size(); i++) {
      auto f1 = IntegerAttr::get(
          ty, (onnxArrayAttr[i].cast<IntegerAttr>()).getValue());
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

// Converts ONNX operand of type tensor to Torch tensor
//
// Typical usage:
// \code
//   auto operandType = toTorchType(context,
//   unaryOp.getOperand().getType());
// \endcode
//
// \param ctx: context of the operand
// \param t: tensor type of the operand
//
// \returns Torch::ValueTensorType conversion from tensor
Torch::ValueTensorType toTorchType(mlir::MLIRContext *ctx, Type t) {
  auto type = t.template dyn_cast<TensorType>();
  return Torch::ValueTensorType::get(
      ctx, type.getShape(), type.getElementType());
}

// Get Torch tensor from mlir::Value tensor
//
// \param operand: operand tensor
// \param rewriter: rewriter object related to the operator
// \param context: context related to operator
// \param loc: location related to operator
//
// \returns mlir::Value tensor of torch type
mlir::Value getTorchTensor(Value operand, ConversionPatternRewriter &rewriter,
    mlir::MLIRContext *context, Location loc) {
  auto operandType = toTorchType(context, operand.getType());
  return rewriter.create<torch::TorchConversion::FromBuiltinTensorOp>(
      loc, operandType, operand);
}

// Get mlir::Value from int
//
// \param val: input integer
// \param rewriter: rewriter object related to the operator
// \param context: context related to operator
// \param loc: location related to operator
//
// \returns mlir::Value of constant integer
Value getIntValue(int val, ConversionPatternRewriter &rewriter,
    mlir::MLIRContext *context, Location loc) {
  auto iType = IntegerType::get(context, 64);
  auto iVal = IntegerAttr::get(iType, val);
  return rewriter.create<ConstantIntOp>(loc, iVal);
}

// Get vector of ints from mlir::ArrayAttr<IntegerAttr>
//
// \param arr: mlir array attribute
//
// \returns vector of integers
std::vector<int> toVector(mlir::ArrayAttr arr) {
  std::vector<int> elements;
  for (auto element : arr) {
    auto j = element.dyn_cast<IntegerAttr>();
    int64_t k = j.getValue().getSExtValue();
    elements.push_back(k);
  }
  return elements;
}

// `torch-mlir` only supports 64-bit floats. Therefore, we need to
// consistently convert from 32-bit `onnx-mlir` floats.
mlir::FloatAttr convertToIEEEDouble(mlir::FloatAttr attr) {
  bool loosesInfo;
  llvm::APFloat value = attr.getValue();
  value.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
      &loosesInfo);
  assert(!loosesInfo && "conversion to 64-bit float failed");
  return FloatAttr::get(
      mlir::FloatType::getF64(attr.getContext()), std::move(value));
}


void setLayerNameAttr(Operation* source, Operation *target){
  if(source == nullptr || target == nullptr)
    return;
  if(source->hasAttr("onnx_node_name"))
    target->setAttr("layer_name", source->getAttr("onnx_node_name"));
}
