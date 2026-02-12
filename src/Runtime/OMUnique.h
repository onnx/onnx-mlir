
#include "onnx-mlir/Runtime/OMTensor.h"
// hi alex #include "onnx-mlir/Runtime/OnnxDataType.h"

void omTensorUnique(OMTensor *totalTensor, const OMTensor *inputTensor,
    int64_t inputAxis, uint64_t sorted, OMTensor *Y, OMTensor *indices,
    OMTensor *inverse_indices, OMTensor *counts);
#edif
