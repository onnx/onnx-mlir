/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-- ModelLib.hpp - Building Models for numerical and benchmark tests -===//
//
// Copyright 2022-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declarations for all the models that can be built.
// The result of each function is a .so built using the modelName.
//
//===----------------------------------------------------------------------===//

#include <string>

// Padding schemes
#define AUTO_PAD_NOTSET 0
#define AUTO_PAD_VALID 1
#define AUTO_PAD_LOWER 2
#define AUTO_PAD_UPPER 3
#define AUTO_PAD_UB 4
const std::string getAutoPadName(const int autoPad);

// Conv2D
bool generateCompiledConv2DModel(
    /* compile option */
    const std::string modelName, const CompilerOptionList &options,
    /* conv param in*/
    const int N, const int C, const int H, const int W, const int kH,
    const int kW, const int autoPad, const int pHBegin, const int pHEnd,
    const int pWBegin, const int pWEnd, const int stride, const int dilation,
    const int isDynamic,
    /* conv param out */
    int &NOut, int &COut, int &HOut, int &WOut);
