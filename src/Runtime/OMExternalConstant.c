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

static void checkEndianness(const char constPackIsLE) {
  if (XOR(IS_SYSTEM_LE(), constPackIsLE)) {
    fprintf(stderr, "Constant pack is stored in a byte order that is not "
                    "native to this current system.");
    exit(1);
  }
}

// Forward declarations for helper functions.
static int mallocAndReadFile(void **constAddr, int fd, int64_t fileSize);
static int mmapAndReadFile(void **constAddr, int fd, int64_t fileSize);

/// Load data from a binary file into memory.
/// This function is called from the constructor of the .so file, so it runs
/// when the .so file is loaded.
///
/// \param[in] constAddr Returned address to a global variable in the IR.
/// \param[in] filename File name at the current folder.
/// \param[in] size Size in bytes to copy data from the binary file.
/// \param[in] isLE Data in the binary file is little-endian or not.
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

  // Ensure endianness is correct.
  checkEndianness(isLE);

  // Prepare an absolute path to the constants file.
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

  // Open the constants file.
  int fd = open(filePath, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr,
        "Error while opening %s: %s. Please set OM_CONSTANT_PATH to the folder "
        "that contains %s.\n",
        filePath, strerror(errno), fname);
    return false;
  }
  if (basePath)
    free(filePath);

  // Load the file into memory.
#ifdef __MVS__
  if (mallocAndReadFile(constAddr, fd, size)) {
#else
  if (mmapAndReadFile(constAddr, fd, size)) {
#endif
    fprintf(stderr, "Error while loading the constant file %s\n", fname);
    close(fd);
    return false;
  }

  // constAddr is now setup.
  close(fd);
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
bool omUnloadConstantData(void **constAddr, int64_t size) {
  void *ptr = constAddr[0];
  if (ptr == NULL)
    return false;
#ifdef __MVS__
  free(ptr);
#else
  munmap(ptr, size);
#endif
  return true;
}

/// Load constants from file into memory using malloc/read.
///
/// \param[in] constAddr Returned address to a global variable in the IR.
/// \param[in] fd File descriptor.
/// \param[in] fileSize Size of the file in bytes.
///
/// \return 0 on success, 1 on failure.
///
static int mallocAndReadFile(void **constAddr, int fd, int64_t fileSize) {
  // Large file - use malloc + chunked read with 4K alignment.
  // Allocate extra space to ensure 4K alignment.
  if (posix_memalign(
          constAddr, /*alignment*/ PAGESIZE_IN_BYTES, /*size*/ fileSize)) {
    fprintf(stderr, "Failed to allocate %lld bytes for constants\n",
        (long long)fileSize);
    return 1;
  }

  // Read file in 1GB chunks.
  int64_t remaining = fileSize;
  int64_t offset = 0;
  char *destAddr = (char *)constAddr[0];

  while (remaining > 0) {
    int64_t chunkSize = (remaining > CHUNK_SIZE) ? CHUNK_SIZE : remaining;
    int64_t totalRead = 0;

    // Handle short reads.
    while (totalRead < chunkSize) {
      ssize_t bytesRead =
          read(fd, destAddr + offset + totalRead, chunkSize - totalRead);

      if (bytesRead < 0) {
        fprintf(stderr, "Error reading constants at offset %lld: %s\n",
            (long long)(offset + totalRead), strerror(errno));
        free(constAddr[0]);
        return 1;
      }

      if (bytesRead == 0) {
        fprintf(stderr, "Unexpected EOF reading constants at offset %lld\n",
            (long long)(offset + totalRead));
        free(constAddr[0]);
        return 1;
      }

      totalRead += bytesRead;
    }

    offset += chunkSize;
    remaining -= chunkSize;
  }

  return 0;
}

/// Load constants from file into memory using mmap.
///
/// \param[in] constAddr Returned address to a global variable in the IR.
/// \param[in] fd File descriptor.
/// \param[in] size Size in bytes to copy data from the binary file.
///
/// \return 0 on success, 1 on failure.
///
/// This function is thread-safe.
///
static int mmapAndReadFile(void **constAddr, int fd, int64_t fileSize) {
#ifdef __MVS__
  void *tempAddr = mmap(0, fileSize, PROT_READ, __MAP_MEGA, fd, 0);
#else
  void *tempAddr = mmap(0, fileSize, PROT_READ, MAP_SHARED, fd, 0);
#endif

  if (tempAddr == MAP_FAILED) {
    fprintf(stderr, "Error while mmapping: %s\n", strerror(errno));
    return 1;
  }

  /* Prepare to compare-and-swap to setup the shared constAddr.
   * If we fail, another thread beat us so free our mmap.
   */
#ifdef __MVS__
  void *expected = NULL;
  if (cds((cds_t *)&expected, (cds_t *)&constAddr[0], *(cds_t *)&tempAddr))
    munmap(tempAddr, fileSize);
#else
  if (!__sync_bool_compare_and_swap(&constAddr[0], NULL, tempAddr))
    munmap(tempAddr, fileSize);
#endif

  /* Either we succeeded in setting constAddr or someone else did it.
   * Either way, constAddr is now setup. We can close our fd without
   * invalidating the mmap.
   */
  return 0;
}

#endif
