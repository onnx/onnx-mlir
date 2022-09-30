/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----- OMTensorListHelper.hpp - OMTensor List Helper Func header ------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of OMTensorList C++ helper functions.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "OnnxMlirRuntime.h"

/*
 * Destroy the OMTensorList data structure (including the array of OMTensor when
 * ownership is asserted), but not the OMTensor themselves. Assumed here is that
 * their live range is managed explicitly or implicitly using a different
 * mechanism.
 */
void omTensorListDestroyShallow(OMTensorList *list);
