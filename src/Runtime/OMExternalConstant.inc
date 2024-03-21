/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- OMExternalConstant.inc - OMExternalConstant C/C++ Implementation --===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// This file contains C/C++ neutral implementation of OMExternalConstant.
//
//===----------------------------------------------------------------------===//

#if defined(_WIN32)
/// Will support Windows soon.
typedef int make_iso_compilers_happy;
#else

#include <errno.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#if defined(_WIN32)
const char *DIR_SEPARATOR = "\\";
#else
const char *DIR_SEPARATOR = "/";
#endif

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

/// MMap data from a binary file into memory.
///
/// \param[in] constAddr lreturned address to a global variable in the IR.
/// \param[in] filename File name at the current folder
/// \param[in] size Size in bytes to copy data from the binary file
/// \param[in] isLE Data in the binary file is little-endian or not
///
/// \return None
///
/// This function is thread-safe.
///
void omMMapBinaryFile(
    void **constAddr, char *filename, int64_t size, int64_t isLE) {
  checkEndianness(isLE);
  char *fname = filename;
#ifdef __MVS__
  // Convert the file name to EBCDIC for the open call.
  char *tPath;
  size_t tLen = strlen(fname);
  tPath = (char *)malloc(tLen);
  if (!tPath) {
    fprintf(stderr, "Error while malloc");
    return;
  }
  memcpy(tPath, fname, tLen);
  __a2e_s(tPath);
  fname = tPath;
#endif

  if (constAddr == NULL) {
    perror("Error: null pointer");
    return;
  }

  if (constAddr[0] == NULL) {
    char *filePath;
    char *basePath = getenv("OM_CONSTANT_PATH");
    if (basePath) {
      size_t baseLen = strlen(basePath);
      size_t fnameLen = strlen(fname);
      size_t sepLen = strlen(DIR_SEPARATOR);
      size_t filePathLen = baseLen + sepLen + fnameLen;
      filePath = (char *)malloc(filePathLen);
      if (!filePath) {
        fprintf(stderr, "Error while malloc");
        return;
      }
      memcpy(filePath, basePath, baseLen);
      memcpy(filePath + baseLen, DIR_SEPARATOR, sepLen);
      memcpy(filePath + baseLen + sepLen, fname, fnameLen);
      filePath[filePathLen] = '\0';
    } else {
      filePath = (char *)fname;
    }
    int fd = open(filePath, O_RDONLY);
    if (fd < 0) {
      fprintf(stderr, "Error while opening %s\n", filePath);
      return;
    }

#ifdef __MVS__
    void *tempAddr = mmap(0, size, PROT_READ, __MAP_MEGA, fd, 0);
#else
    void *tempAddr = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
#endif

    if (tempAddr == MAP_FAILED) {
      fprintf(stderr, "Error while mmapping %s\n", fname);
      close(fd);
      return;
    }

    /* Prepare to compare-and-swap to setup the shared constAddr.
     * If we fail, another thread beat us so free our mmap.
     */
#ifdef __MVS__
    void *expected = NULL;
    if (cds((cds_t *)&expected, (cds_t *)&constAddr[0], *(cds_t *)tempAddr))
      munmap(tempAddr, size);
#else
    if (!__sync_bool_compare_and_swap(&constAddr[0], NULL, tempAddr))
      munmap(tempAddr, size);
#endif

    /* Either we succeeded in setting constAddr or someone else did it.
     * Either way, constAddr is now setup. We can close our fd without
     * invalidating the mmap.
     */
    close(fd);
    if (basePath)
      free(filePath);
  }

#ifdef __MVS__
  free(tPath);
#endif
}

/// Return the address of a constant at a given offset.
///
/// \param[in] outputAddr Returned address of a constant.
/// \param[in] baseAddr Base address to find the constant.
/// \param[in] offset Offset of the constant
///
/// \return None
///
void omGetExternalConstantAddr(
    void **outputAddr, void **baseAddr, int64_t offset) {
  if (outputAddr == NULL) {
    perror("Error: null pointer");
    return;
  }
  if (baseAddr == NULL) {
    perror("Error: null pointer");
    return;
  }
  // Constant is already loaded. Nothing to do.
  if (outputAddr[0])
    return;

  outputAddr[0] = (char *)baseAddr[0] + offset;
}

#endif
