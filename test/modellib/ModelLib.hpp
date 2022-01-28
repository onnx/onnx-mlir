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
//
//===----------------------------------------------------------------------===//

bool generateCompiledConv2DModel(const string modelName, 
    /*in*/
    const int N,
    const int C, const int H, const int W, const int kH, const int kW,
    const int autoPad, const int stride, const int dilation,
    const int isDynamic,
    /* in/out */
    int &pHBegin, int &pHEnd, int &pWBegin, int &pWEnd,
    /* out */
    int &NOut, int &COut, int &HOut, int &WOut);
