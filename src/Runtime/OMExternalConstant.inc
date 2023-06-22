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
#include <stdlib.h>
#include <string.h>
#include <malloc.h>

const int i = 1;
#define IS_SYSTEM_LE() (!((*(const char *)&i) == 0))

#define XOR(a, b) (!(a) != !(b))

void checkEndianness(const char constPackIsLE) {
  if (XOR(IS_SYSTEM_LE(), constPackIsLE)) {
    fprintf(stderr, "Constant pack is stored in a byte order that is not "
                    "native to this current system.");
    exit(1);
  }
}

/// malloc() that does alignment
///
/// \param[in] size Size to be malloc'd
/// \param[in] align Alignment, -1 if no alignment
///
/// \return Pointer to the malloc'd area if successful, or NULL otherwise
///
/// TODO: find a new home for this function.
void *omMallocBuffer(int64_t size, int64_t align) {
  // No alignment.
  if (align == -1) {
    void *ptr = malloc(size);
    if (!ptr) {
      perror("Error during malloc");
      fprintf(stderr, "errno = %d\n", errno);
    }
    return ptr;
  }

  void *ptr = NULL;
#if defined(_MSC_VER)
  ptr = _aligned_malloc(size, align);
#else
  posix_memalign((void**) &ptr, align, size);
#endif

  if (!ptr) {
    perror("Error during malloc");
    fprintf(stderr, "errno = %d\n", errno);
    return ptr;
  }

  return ptr;
}

/// free() what was allocated via omMallocBuffer()
///
/// \param[in] ptr Pointer returned by omMallocBuffer()
///
/// \return None
///
/// TODO: find a new home for this function.
void omFreeBuffer(void *ptr) {
  if (ptr) {
    free(ptr);
    ptr = NULL;
  }
}

/// Open a file in binary mode
///
/// \param[in] fname File path
/// \param[in] isLE Data in the file is little-endian or not
///
/// \return A FILE pointer
/// TODO: find a new home for this function.
void *omOpenBinaryFile(char *fname, int64_t isLE) {
  checkEndianness(isLE);
  FILE *fileptr;
  fileptr = fopen(fname, "rb"); // Open the file in binary mode.
  if (fileptr == NULL) {
    fprintf(stderr, "Error while openning %s: %s\n", fname, strerror(errno));
    return NULL;
  }
  return (void *)fileptr;
}

/// Close a file
///
/// \param[in] fileptr A FILE pointer
///
/// \return None
/// TODO: find a new home for this function.
void omCloseFile(void *fileptr) {
  if (fileptr)
    fclose((FILE *)fileptr);
}

/// Load data from a binary file into an aligned buffer.
///
/// \param[in] ptr Pointer returned by omLoadExternalConstant()
/// \param[in] fname File name at the current folder
/// \param[in] offset Offset from the beginning of file
/// \param[in] size Size in bytes to copy data from the binary file
/// \param[in] align Alignment
/// \param[in] isLE Data in the binary file is little-endian or not
///
/// \return None
///
void omLoadExternalConstant(
    void **ptr, void *fileptr, int64_t offset, int64_t size, int64_t align) {
  if (ptr == NULL) {
    perror("Error: null pointer");
    return;
  }

  // Constant is already loaded. Nothing to do.
  if (ptr[0])
    return;

  // File is not opened.
  if (fileptr == NULL) {
    perror("Error: LoadExternalConstant");
    return;
  }

  void *buffer = omMallocBuffer(size, align);
  fseek((FILE *)fileptr, offset, SEEK_SET);
  fread(buffer, size, 1, (FILE *)fileptr);

  ptr[0] = buffer;
}
