/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- OMTopK.inc - OMTopK C/C++ Implementation --===//
//
// Copyright 2025 The IBM Research Authors.
//
// =============================================================================
//
// This file contains C/C++ implementation of OMTopK.
//
//===----------------------------------------------------------------------===//

// Include the header to get definitions for OMTensor,
// and the prototypes for getCompareFunction() and omTensorSort().
#include "OMSort.h" // Provides OMTensor, OnnxDataType, and function typedefs

#ifdef __cplusplus
#include <cassert>
#else
#include <assert.h>
#endif

#ifndef __USE_GNU
#define __USE_GNU 
#endif
#include <stdlib.h> // For qsort_r
#include <stdint.h> // For uint64_t, int64_t

#ifdef __cplusplus
extern "C" {
#endif

/* =========================================================================
 *
 * HEAP HELPER FUNCTIONS (for Stable TopK)
 *
 * ========================================================================= */

// Helper function to swap two 64-bit indices
static void swap_indices(uint64_t *a, uint64_t *b) {
  uint64_t temp = *a;
  *a = *b;
  *b = temp;
}

/**
 * Function to sift an element down the heap to maintain the heap property.
 */
static void heap_sift_down(uint64_t *heap, int64_t start, int64_t end,
#if defined(__APPLE__) || defined(_MSC_VER)
    int (*compare)(void *, const void *, const void *),
#else
    int (*compare)(const void *, const void *, void *),
#endif
    void *data) {
  
  int64_t root = start;
  while ((root * 2 + 1) <= end) { // While root has at least one child
    int64_t child = root * 2 + 1;
    int64_t swap_idx = root;

    // Check if root is "worse" than left child
    int comparison = 0;
#if defined(__APPLE__) || defined(_MSC_VER)
    comparison = compare(data, &heap[swap_idx], &heap[child]);
#else
    comparison = compare(&heap[swap_idx], &heap[child], data);
#endif
    if (comparison > 0) {
      swap_idx = child;
    }

    // Check if right child exists and is "better" than current swap_idx
    if (child + 1 <= end) {
      comparison = 0;
#if defined(__APPLE__) || defined(_MSC_VER)
      comparison = compare(data, &heap[swap_idx], &heap[child + 1]);
#else
      comparison = compare(&heap[swap_idx], &heap[child + 1], data);
#endif
      if (comparison > 0) {
        swap_idx = child + 1;
      }
    }

    if (swap_idx == root) {
      return;
    } else {
      swap_indices(&heap[root], &heap[swap_idx]);
      root = swap_idx;
    }
  }
}

/**
 * Function to turn an arbitrary array into a heap in O(K) time.
 */
static void make_heap(uint64_t *heap, int64_t heap_size,
#if defined(__APPLE__) || defined(_MSC_VER)
    int (*compare)(void *, const void *, const void *),
#else
    int (*compare)(const void *, const void *, void *),
#endif
    void *data) {
  // Start from the last non-leaf node and sift it down
  for (int64_t start = (heap_size / 2) - 1; start >= 0; start--) {
    heap_sift_down(heap, start, heap_size - 1, compare, data);
  }
}

/* =========================================================================
 *
 * Implementation of Stable TopK based on a partial Heapsort
 *
 * ========================================================================= */

/*
 * Performs a stable O(N log K) partial sort.
 * This function is emitted by the compiler for the ONNX TopK operation.
 */
void omTensorTopK(OMTensor *orderTensor, const OMTensor *inputTensor,
    uint64_t axis, uint64_t ascending, uint64_t k_u64, uint64_t sorted) {

  // Standard setup (same as omTensorSort)
  const OM_DATA_TYPE dataType = omTensorGetDataType(inputTensor);
  const uint64_t rank = omTensorGetRank(inputTensor);
  assert(rank <= 6 && "omTensorTopK assumes rank <= 6");
  assert(axis == (rank - 1) && "omTensorTopK assumes axis == (rank - 1)");
  const int64_t *inputShape = omTensorGetShape(inputTensor);
  const int64_t *inputStrides = omTensorGetStrides(inputTensor);
  assert(inputStrides[axis] == 1 && "omTensorTopK assumes strides[axis] == 1");

  void *orderPtr = omTensorGetDataPtr(orderTensor);
  uint64_t *order = (uint64_t *)orderPtr;
  void *dataPtr = omTensorGetDataPtr(inputTensor);
  int64_t sort_elems = inputShape[axis];
  int64_t k = (int64_t)k_u64; // Use signed int

  // If K is 0, or the axis is empty, do nothing
  if (k == 0 || sort_elems == 0)
    return;

  // If K is large, a full sort is faster
  // TODO: find better threshold
  if (k >= (int64_t)(sort_elems * 0.9)) {
    omTensorSort(orderTensor, inputTensor, axis, ascending);
    return;
  }

  // 1. Get the heap comparator.
  // To get Top-K "Largest" (ascending=0), we build a MIN-heap.
  // To get Top-K "Smallest" (ascending=1), we build a MAX-heap.
  // This uses the opposite ascending direction.
  compareFunctionType *heapCompare =
      getCompareFunction(1 - ascending, dataType);
  if (!heapCompare)
    return; // Unsupported data type

  // 2. Get the loop comparator.
  // This checks if a new element is "better" than the heap root.
  // This uses the correct ascending direction.
  compareFunctionType *loopCompare =
      getCompareFunction(ascending, dataType);

  // 3. Get the final sort comparator (same as loop).
  compareFunctionType *sortCompare = loopCompare;

  uint64_t datasize = OM_DATA_TYPE_SIZE[dataType];

  // Get the standard C sort function
  #if defined(__APPLE__)
    sortFunctionType *sortFunc = qsort_r;
  #elif defined(_MSC_VER)
    sortFunctionType *sortFunc = qsort_s;
  #elif defined(__linux) || defined(__linux__) || defined(linux)
    sortFunctionType *sortFunc = qsort_r;
  #else
    sortFunctionType *sortFunc = quick_sort_custom;
  #endif

  // Iterate over all 1D slices of the tensor
  int64_t shape[6] = {1, 1, 1, 1, 1, 1};
  int64_t strides[6] = {0, 0, 0, 0, 0, 0};
  for (uint64_t i = 0; i < rank; i++) {
    shape[i + (6 - rank)] = inputShape[i];
    strides[i + (6 - rank)] = inputStrides[i];
  }

  for (int dim0 = 0; dim0 < shape[0]; dim0++) {
    for (int dim1 = 0; dim1 < shape[1]; dim1++) {
      for (int dim2 = 0; dim2 < shape[2]; dim2++) {
        for (int dim3 = 0; dim3 < shape[3]; dim3++) {
          for (int dim4 = 0; dim4 < shape[4]; dim4++) {
            uint64_t off = dim0 * strides[0] + dim1 * strides[1] +
                           dim2 * strides[2] + dim3 * strides[3] +
                           dim4 * strides[4];
            void *data = ((char *)dataPtr) + datasize * off;
            uint64_t *idx = order + off;
            
            // --- Stable TopK Logic (Partial Heapsort) ---

            // STEP 1: Build the initial K-heap using the heap comparator.
            // This builds a Min-Heap (for TopK-Largest) or Max-Heap (for TopK-Smallest).
            // The root (idx[0]) is the "worst" item in the K-set.
            make_heap(idx, k, heapCompare, data);

            // STEP 2: Iterate through the remaining N-K elements.
            for (int64_t i = k; i < sort_elems; i++) {
              // Compare the new element (idx[i]) with the heap root (idx[0])
              // using the loop comparator.
              int comparison = 0;
#if defined(__APPLE__) || defined(_MSC_VER)
              comparison = loopCompare(data, &idx[i], &idx[0]);
#else
              comparison = loopCompare(&idx[i], &idx[0], data);
#endif
              // If the new element is "better" than the heap root (comp < 0)
              // replace the root and sift down.
              if (comparison < 0) {
                swap_indices(&idx[0], &idx[i]);
                // Sift down using the heap comparator to maintain the heap property.
                heap_sift_down(idx, 0, k - 1, heapCompare, data);
              }
            }

            // STEP 3: Sort only the final K elements, if "sorted" is not 0.
            if (sorted != 0) {
              #if defined(__APPLE__)
                sortFunc((void *)idx, k, sizeof(uint64_t), data,
                    sortCompare);
              #else
                sortFunc((void *)idx, k, sizeof(uint64_t), sortCompare,
                    data);
              #endif
            }
          }
        }
      }
    }
  }
}

#ifdef __cplusplus
} // extern "C"
#endif




