/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ OMTensorListHelperTest.cpp - Input Builder Test ---------===//
//
// Copyright 2026 The IBM Research Authors.
//
// =============================================================================
//
// Unit test for omTensorListCreateFromInputSignature, the input builder behind
// RunONNXModel.py (fill_input_debug), run-onnx-lib, and profile-model.py. It
// parses hand-written signature strings, so no compiled model is needed.
//
//===----------------------------------------------------------------------===//

#include <stdio.h>

#include <vector>

#include "src/Runtime/OMTensorHelper.hpp"
#include "src/Runtime/OMTensorListHelper.hpp"

// Unlike assert, CHECK stays active in Release (NDEBUG) builds.
static int failures = 0;
#define CHECK(cond)                                                            \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond); \
      ++failures;                                                              \
    }                                                                          \
  } while (0)

// Check that list holds tensors of the given type and shapes, each with data.
static void checkList(OMTensorList *list, OM_DATA_TYPE type,
    const std::vector<std::vector<int64_t>> &shapes) {
  CHECK(list != nullptr);
  if (!list)
    return;
  CHECK(omTensorListGetSize(list) == (int64_t)shapes.size());
  for (int64_t i = 0; i < omTensorListGetSize(list); ++i) {
    OMTensor *t = omTensorListGetOmtByIndex(list, i);
    CHECK(t != nullptr);
    if (!t || i >= (int64_t)shapes.size())
      continue;
    CHECK(omTensorGetDataType(t) == type);
    CHECK(omTensorGetDataPtr(t) != nullptr);
    const std::vector<int64_t> &expected = shapes[i];
    CHECK(omTensorGetRank(t) == (int64_t)expected.size());
    if (omTensorGetRank(t) != (int64_t)expected.size())
      continue;
    const int64_t *shape = omTensorGetShape(t);
    for (size_t d = 0; d < expected.size(); ++d)
      CHECK(shape[d] == expected[d]);
  }
  omTensorListDestroy(list);
}

// Signature with symbolic dims, as emitted for an ONNX model using dim_param.
static const char *symbolicSig =
    R"([    { "type" : "i64" , "dims" : ["batch_size" , "sequence"] , "name" : "input_ids" }
 ,    { "type" : "i64" , "dims" : ["batch_size" , "sequence"] , "name" : "attention_mask" }
])";

// Symbolic, unnamed dynamic (-1), and static dims mixed in one signature.
static const char *mixedSig =
    R"([    { "type" : "f32" , "dims" : ["batch_size" , "sequence" , 4] , "name" : "X" }
 ,    { "type" : "f32" , "dims" : ["batch_size" , -1 , 4] , "name" : "Y" }
])";

static const char *staticSig =
    R"([ { "type" : "f32" , "dims" : [1 , 28 , 28] , "name" : "image" } ])";

// Symbolic dims are dynamic and take their values from shapeInfo.
void testSymbolicDims() {
  checkList(omTensorListCreateFromInputSignature(symbolicSig, "0:1x180,1:1x180"),
      ONNX_TYPE_INT64, {{1, 180}, {1, 180}});
}

// Symbolic, unnamed dynamic, and static dims all resolve together.
void testMixedDims() {
  checkList(omTensorListCreateFromInputSignature(mixedSig, "-1:2x5x4"),
      ONNX_TYPE_FLOAT, {{2, 5, 4}, {2, 5, 4}});
}

// Per-input, range, and all-inputs shapeInfo forms give the same shapes.
void testShapeInfoForms() {
  for (const char *shapeInfo : {"0:3x7x4,1:3x7x4", "0-1:3x7x4", "-1:3x7x4"})
    checkList(omTensorListCreateFromInputSignature(mixedSig, shapeInfo),
        ONNX_TYPE_FLOAT, {{3, 7, 4}, {3, 7, 4}});
}

// A fully static signature needs no shapeInfo.
void testStaticDims() {
  checkList(omTensorListCreateFromInputSignature(staticSig), ONNX_TYPE_FLOAT,
      {{1, 28, 28}});
}

// A dynamic dim left unresolved is an error, not a crash.
void testUnresolvedDynamicDim() {
  CHECK(omTensorListCreateFromInputSignature(symbolicSig) == nullptr);
  CHECK(omTensorListCreateFromInputSignature(symbolicSig, "0:1x180") ==
        nullptr);
}

// shapeInfo may not change a static dim.
void testStaticDimConflict() {
  CHECK(omTensorListCreateFromInputSignature(staticSig, "0:2x28x28") ==
        nullptr);
}

// Malformed signatures are rejected.
void testMalformedSignature() {
  CHECK(omTensorListCreateFromInputSignature(
            R"([ { "type" : "f32" , "dims" : ["batch_size , 4] } ])", "0:1x4") ==
        nullptr);
  CHECK(omTensorListCreateFromInputSignature(
            R"([ { "type" : "f32" , "dims" : [1 , 4 } ])") == nullptr);
  CHECK(omTensorListCreateFromInputSignature("") == nullptr);
  CHECK(omTensorListCreateFromInputSignature(nullptr) == nullptr);
}

int main() {
  omDefineSeed(42, /*hasSeedValue=*/1);
  testSymbolicDims();
  testMixedDims();
  testShapeInfoForms();
  testStaticDims();
  testUnresolvedDynamicDim();
  testStaticDimConflict();
  testMalformedSignature();
  if (failures) {
    fprintf(stderr, "OMTensorListHelperTest: %d check(s) failed\n", failures);
    return 1;
  }
  printf("OMTensorListHelperTest: all checks passed\n");
  return 0;
}
