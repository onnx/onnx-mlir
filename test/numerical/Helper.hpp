/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----------------------- Helper.hpp ----------------------------------===//
//
// This file defines some helper functions used for numerical tests.
//
//====---------------------------------------------------------------------===//

/// Sigmoid
float sigmoid(float x) { return 1 / (1 + exp(-x)); }

/// Build an ONNXConstantOp from an OMTensor.
ONNXConstantOp buildONNXConstantOp(MLIRContext *ctx, OpBuilder builder,
    unique_ptr<OMTensor, decltype(&omTensorDestroy)> &omt,
    RankedTensorType resultType) {
  int64_t numElems = omTensorGetNumElems(omt.get());
  auto bufferPtr = omTensorGetDataPtr(omt.get());
  float *arrayPtr = reinterpret_cast<float *>(bufferPtr);
  auto array = std::vector<float>(arrayPtr, arrayPtr + numElems);
  auto denseAttr =
      DenseElementsAttr::get(resultType, llvm::makeArrayRef(array));
  ONNXConstantOp constantTensor = builder.create<ONNXConstantOp>(
      UnknownLoc::get(ctx), resultType, Attribute(), denseAttr, FloatAttr(),
      ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(), ArrayAttr());
  return constantTensor;
}

std::string getSharedLibName(std::string sharedLibBaseName) {
#ifdef _WIN32
  return sharedLibBaseName + ".dll";
#else
  return sharedLibBaseName + ".so";
#endif
}
