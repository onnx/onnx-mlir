/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- OMExternalConstant.c   - OMExternalConstant C Implementation -----===//
//
// Copyright 2023-2026 The IBM Research Authors.
//
// =============================================================================
//
// This file contains C implementation of OMExternalConstant.
//
//===----------------------------------------------------------------------===//

#if defined(_WIN32)
/// Will support Windows soon.
typedef int make_iso_compilers_happy;
#else

#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHUNK_SIZE (1024LL * 1024LL * 1024LL)
#define PAGESIZE_IN_BYTES 4096

#if defined(_WIN32)
const char *DIR_SEPARATOR = "\\";
#else
const char *DIR_SEPARATOR = "/";
#endif

const int i = 1;
#define IS_SYSTEM_LE() (!((*(const char *)&i) == 0))

#define XOR(a, b) (!(a) != !(b))

__attribute__((unused)) static void checkEndianness(const char constPackIsLE) {
  if (XOR(IS_SYSTEM_LE(), constPackIsLE)) {
    fprintf(stderr, "Constant pack is stored in a byte order that is not "
                    "native to this current system.");
    exit(1);
  }
}

// Forward declarations for helper functions.
static void *mallocAndReadFile(const char *filePath, int64_t fileSize);
static void freeAlignedBuffer(void *alignedPtr);

/// Load data from a binary file into memory.
/// This function is called from the constructor of the .so file, so it runs
/// when the .so file is loaded.
///
/// \param[in] constAddr Returned address to a global variable in the IR.
/// \param[in] filename File name at the current folder
/// \param[in] size Size in bytes to copy data from the binary file
/// \param[in] isLE Data in the binary file is little-endian or not
///
/// \return true/false
///
bool omLoadConstantData(
    void **constAddr, char *fname, int64_t size, int64_t isLE) {
  if (constAddr == NULL) {
    fprintf(stderr, "Error: null pointer.");
    return false;
  }

  if (size <= 0) {
    fprintf(stderr, "File size is zero.");
    return false;
  }

  // Already loaded. Nothing to do.
  if (constAddr[0] != NULL)
    return true;

  char *filePath;
  char *basePath = getenv("OM_CONSTANT_PATH");
  if (basePath) {
    size_t baseLen = strlen(basePath);
    size_t fnameLen = strlen(fname);
    size_t sepLen = strlen(DIR_SEPARATOR);
    size_t filePathLen = baseLen + sepLen + fnameLen + 1;
    filePath = (char *)malloc(filePathLen);
    if (!filePath) {
      fprintf(stderr, "Error while malloc: %s", strerror(errno));
      return false;
    }
    snprintf(filePath, filePathLen, "%s%s%s", basePath, DIR_SEPARATOR, fname);
  } else {
    filePath = (char *)fname;
  }

  // Malloc a buffer and load the file into the buffer.
  constAddr[0] = mallocAndReadFile(filePath, size);

  if (basePath)
    free(filePath);

  if (constAddr[0] == NULL) {
    fprintf(stderr, "Error while loading the constant file %s\n", fname);
    return false;
  }

  // constAddr is now setup.
  return true;
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
    fprintf(stderr, "Error: null pointer.");
    return;
  }
  if (baseAddr == NULL) {
    fprintf(stderr, "Error: null pointer.");
    return;
  }
  // Constant is already loaded. Nothing to do.
  if (outputAddr[0])
    return;

  outputAddr[0] = (char *)baseAddr[0] + offset;
}

/// Free the preloaded constant buffer.
/// This function is called from the destructor of the .so file, so it runs
/// when the .so file is unloaded.
///
/// \param[in] constAddr Returned address to a global variable in the IR.
///
/// \return true/false
bool omUnloadConstantData(void **constAddr) {
  void *alignedPtr = constAddr[0];
  if (alignedPtr == NULL)
    return false;
  freeAlignedBuffer(alignedPtr);
  return true;
}

/// Load constants from file into memory using malloc/read.
///
/// \param[in] filename Name of the constants file
/// \param[in] fileSize Size of the file in bytes
///
/// \return true on success, false on failure
///
static void *mallocAndReadFile(const char *filePath, int64_t fileSize) {
  // Open the constants file
  int fd = open(filePath, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr, "Failed to open %s: %s\n", filePath, strerror(errno));
    return NULL;
  }

  // Large file - use malloc + chunked read with 4K alignment.
  // Allocate extra space to ensure 4K alignment.
  // Request one more page + size of a pointer from the OS, which is used for
  // tracking the original allocation.
  unsigned short extraAllocation = (PAGESIZE_IN_BYTES - 1) + sizeof(void *);
  void *ptr = malloc(fileSize + extraAllocation);

  if (!ptr) {
    fprintf(stderr, "Failed to allocate %lld bytes for constants\n",
        (long long)fileSize);
    close(fd);
    return NULL;
  }

  // Find the 4K boundary after ptr.
  void *alignedPtr =
      (void *)(((uintptr_t)ptr + extraAllocation) & ~(PAGESIZE_IN_BYTES - 1));
  // Put the original malloc'd address right before alignedPtr.
  // This is used when we free the buffer.
  ((void **)alignedPtr)[-1] = ptr;

  // Read file in 1GB chunks.
  int64_t remaining = fileSize;
  int64_t offset = 0;
  char *destPtr = (char *)alignedPtr;

  while (remaining > 0) {
    int64_t chunkSize = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
    int64_t totalRead = 0;

    // Handle short reads.
    while (totalRead < chunkSize) {
      ssize_t bytesRead =
          read(fd, destPtr + offset + totalRead, chunkSize - totalRead);

      if (bytesRead < 0) {
        fprintf(stderr, "Error reading constants at offset %lld: %s\n",
            (long long)(offset + totalRead), strerror(errno));
        freeAlignedBuffer(alignedPtr);
        close(fd);
        return NULL;
      }

      if (bytesRead == 0) {
        fprintf(stderr, "Unexpected EOF reading constants at offset %lld\n",
            (long long)(offset + totalRead));
        freeAlignedBuffer(alignedPtr);
        close(fd);
        return NULL;
      }

      totalRead += bytesRead;
    }

    offset += chunkSize;
    remaining -= chunkSize;
  }

  close(fd);

  return alignedPtr;
}

static void freeAlignedBuffer(void *alignedPtr) {
  if (alignedPtr) {
    // Was malloc'd - free the original pointer, not the aligned one
    // Get the original malloc'd address from where we put it and free it
    void *originalPtr = ((void **)alignedPtr)[-1];
    free(originalPtr);
  }
}

#endif
