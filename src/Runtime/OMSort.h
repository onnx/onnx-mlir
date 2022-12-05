/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------- OMSort.h - OMTensor Helper Func header ----------===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains declaration of OMTensor C++ helper functions. At some
// point, this file needs to be merged into the OMTensor.h along with other C++
// APIs operating on OMTensor.
//
//===----------------------------------------------------------------------===//

#pragma once

#define OMSORT_DEFAULT OMSORT_BUBBLESORT
#define OMSORT_BUBBLESORT 0
#define OMSORT_COMBSORT 1
#define OMSORT_QUICKSORT 2
#define OMSORT_QUICKSORT_CUSTOM 3
