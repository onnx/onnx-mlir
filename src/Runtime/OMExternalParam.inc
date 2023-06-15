/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---- OMExternalParam.cpp - OMExternalParam C/C++ Implementation ------===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains C/C++ neutral implementation of OMExternalParam.
//
//===----------------------------------------------------------------------===//

#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <string.h>

#if defined(__APPLE__) || defined(__MVS__)
#include <stdlib.h>
#else
#include <malloc.h>
#endif

/// malloc_aligned and free_aligned are adopted from
/// https://github.com/IBM/zDNN/blob/main/zdnn/malloc4k.c

/// malloc() that does alignment
///
/// \param[in] size Size to be malloc'd
///
/// \return Pointer to the malloc'd area if successful, or NULL otherwise
///
void *malloc_aligned(size_t size, size_t align) {
  // request one more page + size of a pointer from the OS
  unsigned short extra_allocation = (align - 1) + sizeof(void *);

  void *ptr = malloc(size + extra_allocation);
  if (!ptr) {
    perror("Error during malloc");
    fprintf(stderr, "errno = %d\n", errno);
    return ptr;
  }

  // find the align boundary after ptr
  void *aligned_ptr =
      (void *)(((uintptr_t)ptr + extra_allocation) & ~(align - 1));
  // put the original malloc'd address right before aligned_ptr
  ((void **)aligned_ptr)[-1] = ptr;

  return aligned_ptr;
}

/// free() what was allocated via malloc_aligned()
///
/// \param[in] ptr Pointer returned by malloc_aligned()
///
/// \return None
///
void free_aligned(void *aligned_ptr) {
  if (aligned_ptr) {
    // get the original malloc'd address from where we put it and free it
    void *original_ptr = ((void **)aligned_ptr)[-1];
    free(original_ptr);
  }
}

void *omLoadExternalParam(
    char *fname, int64_t size, int64_t align, int64_t isLE) {
  void *buffer = malloc_aligned(size, align);

  FILE *fileptr;
  fileptr = fopen(fname, "rb"); // Open the file in binary mode
  if (fileptr == NULL) {
    perror("Error: ");
    return NULL;
  }

  long nr = fread(buffer, size, 1, fileptr);
  float *fb = (float *)buffer;
  fclose(fileptr);

  return buffer;
}
