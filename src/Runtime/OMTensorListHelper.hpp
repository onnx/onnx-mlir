/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- OMTensorListHelper.hpp - OMTensor List Helper Func header ------===//
//
// Copyright 2022-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of OMTensorList C++ helper functions.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_TENSOR_LIST_HELPER_H
#define ONNX_MLIR_TENSOR_LIST_HELPER_H

#include "OnnxMlirRuntime.h"

/*
 * Destroy the OMTensorList data structure (including the array of OMTensor when
 * ownership is asserted), but not the OMTensor themselves. Assumed here is that
 * their live range is managed explicitly or implicitly using a different
 * mechanism.
 */
void omTensorListDestroyShallow(OMTensorList *list);

/**
 * \brief Create an OMTensorList from a JSON input signature string.
 *
 * Parses the JSON signature returned by inputSignature() and creates one
 * OMTensor per entry. Shapes are taken from the signature; dynamic (negative)
 * dimensions are kept as-is unless overridden by shapeInfo. Data buffers are
 * filled with random values when valueInfo is provided, and left null otherwise.
 *
 * Bound resolution follows a three-level priority (highest to lowest):
 *   1. Per-tensor explicit bounds set via valueInfo (min/max/val).
 *   2. Per-type default bounds supplied via defaultLowerBound/defaultUpperBound.
 *   3. Built-in per-type defaults (floats: [-0.1, 0.1]; signed ints: [-10, 10];
 *      unsigned ints: [0, 10]; bool: {false, true}).
 *
 * @param inputSignatureStr JSON string from inputSignature(), e.g.
 *   [ { "type" : "f32" , "dims" : [1, 28, 28] , "name" : "image" } ]
 * @param shapeInfo Optional dimension overrides. Format:
 *   "INPUT_ID:D1xD2x...xDn, ..."
 *   INPUT_ID is an integer >= 0, a range (e.g. "5-17"), or -1 for all inputs.
 *   A dimension value of -1 keeps the value from the signature.
 *   Pass nullptr to use the signature dims unchanged.
 * @param valueInfo Optional per-tensor data fill specification. Format:
 *   "INPUT_ID:spec1spec2..., ..." where each spec is one of:
 *   - min<num>  lower bound for random fill (overrides priority 2 and 3)
 *   - max<num>  upper bound for random fill (overrides priority 2 and 3)
 *   - val<num>  constant fill, equivalent to min=max=num
 *   String tensors are not supported. Pass nullptr to leave data buffers null.
 * @param defaultLowerBound Optional per-type lower bound overrides (priority 2).
 *   Format: "typename:value, typename:value, ..."
 *   Supported type names: bool_ (or bool), int8, uint8, int16, uint16, int32,
 *   uint32, int64, uint64, float16, float32, float64.
 *   Applied when valueInfo does not supply an explicit min for a tensor.
 *   Pass nullptr to use the built-in defaults (priority 3).
 * @param defaultUpperBound Same format as defaultLowerBound, for upper bounds.
 * @param verbose When true, print the shape of each created tensor to stdout.
 * @return Pointer to the newly created OMTensorList, or null on error.
 */
OMTensorList *omTensorListCreateFromInputSignature(const char *inputSignatureStr,
    const char *shapeInfo = nullptr, const char *valueInfo = nullptr,
    const char *defaultLowerBound = nullptr,
    const char *defaultUpperBound = nullptr, bool verbose = false);

#endif
