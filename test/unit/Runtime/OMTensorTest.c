//===----------------- OMTensorTest.h - OMTensor Unit Test -----------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of OMTensor and data structures and
// helper functions.
//
//===----------------------------------------------------------------------===//
#include <assert.h>
#include <stdio.h>

#include "OnnxMlirRuntime.h"

void testOMTensorCtor() {
    float data[4] = {1.f, 1.f};
    int64_t shape[2] = {2, 2};
    OMTensor *tensor = omTensorCreate(data, shape, 2, ONNX_TYPE_FLOAT);
    assert(tensor);

    int64_t* shape_ptr = omTensorGetDataShape(tensor);
    assert(shape_ptr);
    assert(shape_ptr[0] == 2);
    assert(shape_ptr[1] == 2);

    int64_t* strides_ptr = omTensorGetStrides(tensor);
    assert(strides_ptr);
    assert(strides_ptr[0] == 1);
    assert(strides_ptr[1] == 2);
}

int main() {
    testOMTensorCtor();
    return 0;
}