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
int64_t tensorRank, currentDim, currentElement, *tensorBounds;

static void trackDim() {
    if(++currentElement==tensorBounds[currentDim]) {
      if(++currentDim<tensorRank) {
        printf("] [");
        separator = " ";
      } else {
        printf("]");
      }  
    }
}

void printTensorElementFloat(float e) {
    printf("%s%f",separator,e);
    separator=", ";
    trackDim();
}

void printTensorElementDouble(double e) {
    printf("%s%f",separator,e);
    separator=", ";
    trackDim();
}

void printTensorElementInt(int64_t e) {
    printf("%s%li",separator,e);
    separator=", ";
    trackDim();
}

void printTensorStart(int64_t rank,void *bounds) {
    tensorRank=rank;
    tensorBounds=(int64_t *)malloc(8*rank);
    currentDim=0;
    currentElement=0;
    for(int i=0;i<rank;++i) {
      tensorBounds[i]=((int64_t *)bounds)[i];
       printf("%s","[");
    }
    separator=" ";
}

void printTensorEnd() {
    for(int i=0;i<tensorRank;++i)
      printf("%s"," ]");
}

