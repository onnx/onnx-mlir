/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------- OMDebugSupport.inc - C/C++ Neutral OMDebugSupport Implementation--------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementations of OMDebugSupport data structures
// and helper functions.
//
//===----------------------------------------------------------------------===//

#ifdef __cplusplus
#include <cassert>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <typeinfo>
#include <vector>
#else
#include <assert.h>
#endif

#if defined(__APPLE__) || defined(__MVS__)
#include <stdlib.h>
#else
#include <malloc.h>
#endif

#include <stdint.h>
#include <stdio.h>
#include <string.h>


char * separator;

void printTensorElementFloat(double e) {
    printf("%s%d",separator,e);
    separator=", ";
}

void printTensorElementInt(uint64_t e) {
    printf("%s%i",separator,e);
    separator=", ";
}

void printTensorStart(uint64_t rank,void *bounds) {
    printf("%s","[");
    separator=" ";
}

void printTensorEnd() {
    printf("%s"," ]");
}

