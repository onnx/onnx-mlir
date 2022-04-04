#include "CommonUtils.h"

typedef struct dim_pads{
  int dim_start;
  int dim_end;
} dim_pads;

std::vector<Value> createPadsArrayAttribute(::mlir::ArrayAttr pads, Type ty, 
		Location loc, ConversionPatternRewriter &rewriter) {
  // Reading the ONNX side pads values and store in the array.
  dim_pads dimArray[pads.size()];
  std::vector<Value> translatepadsList;
  if (pads) {
    int j = 0;
    for (unsigned int i = 0; i < pads.size(); i++) {
      dimArray[j].dim_start = (pads[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
      i++;
      dimArray[j].dim_end = (pads[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue();
      j++;
    }

    // read the onnx pad values from array(dim_start values) 
    int k = 0;
    for (unsigned int i = 0; i < pads.size(); i=i+2) {
      auto f0 = IntegerAttr::get(ty, (dimArray[k].dim_start));
      Value p0v = rewriter.create<ConstantIntOp>(loc, f0);
      translatepadsList.push_back(p0v);
      k++;
    }

    // read the onnx pad values from array(dim_end values)
    k = 0;
    for (unsigned int i = 0; i < pads.size(); i=i+2) {
      auto f1 = IntegerAttr::get(ty,(dimArray[k].dim_end));
      Value p1v = rewriter.create<ConstantIntOp>(loc, f1);
      translatepadsList.push_back(p1v);
      k++;
    }
  }
  return translatepadsList;  
}

std::vector<Value> createArrayAttribute(::mlir::ArrayAttr onnxArrayAttr, Type ty,
                Location loc, ConversionPatternRewriter &rewriter) {
  std::vector<Value> operandArrayValues;
  if (onnxArrayAttr) {
    for (unsigned int i = 0; i < onnxArrayAttr.size(); i++) {
      auto f1 = IntegerAttr::get(ty,
                  (onnxArrayAttr[i].dyn_cast<IntegerAttr>()).getValue().getZExtValue());
      Value p1v = rewriter.create<ConstantIntOp>(loc, f1);
      operandArrayValues.push_back(p1v);
    }
  }
  return operandArrayValues;
}
