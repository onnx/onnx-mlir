/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- OMExternalConstant.c   - OMExternalConstant C Implementation -----===//
//
// Copyright 2023 The IBM Research Authors.
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
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef __MVS__
#define MAX_MMAP_SIZE (1024LL * 1024LL * 1024LL)
#endif

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

#ifdef __MVS__
// Global buffer to store preloaded constants (z/OS only)
static void *g_constant_buffer = NULL;
static int64_t g_constant_buffer_size = 0;

// Forward declarations for helper functions
static bool omMallocAndReadFile(const char *filename);
static void omFreeBuffer(void);
#endif

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
bool omMMapBinaryFile(
    void **constAddr, char *fname, int64_t size, int64_t isLE) {
  if (constAddr == NULL) {
    fprintf(stderr, "Error: null pointer.");
    return false;
  }

  // Already mmaped. Nothing to do.
  if (constAddr[0] != NULL)
    return true;

#ifdef __MVS__
  // On z/OS, return the preloaded constant buffer if available
  if (g_constant_buffer != NULL && g_constant_buffer_size > 0) {
    constAddr[0] = g_constant_buffer;
    return true;
  }
#endif

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
  int fd = open(filePath, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr,
        "Error while opening %s: %s. Please set OM_CONSTANT_PATH to the folder "
        "that contains %s.\n",
        filePath, strerror(errno), fname);
    if (basePath)
      free(filePath);
    return false;
  }

#ifdef __MVS__
  void *tempAddr = mmap(0, size, PROT_READ, __MAP_MEGA, fd, 0);
#else
  void *tempAddr = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
#endif

  if (tempAddr == MAP_FAILED) {
    fprintf(stderr, "Error while mmapping %s: %s\n", fname, strerror(errno));
    close(fd);
    if (basePath)
      free(filePath);
    return false;
  }

  /* Prepare to compare-and-swap to setup the shared constAddr.
   * If we fail, another thread beat us so free our mmap.
   */
#ifdef __MVS__
  void *expected = NULL;
  if (cds((cds_t *)&expected, (cds_t *)&constAddr[0], *(cds_t *)&tempAddr))
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

#ifdef __MVS__
/// Load constants from file into memory using malloc/read or mmap.
/// This function handles both large files (>1GB) using malloc + chunked read
/// and small files (≤1GB) using mmap.
///
/// \param[in] filename Name of the constants file
///
/// \return true on success, false on failure
///
static bool omMallocAndReadFile(const char *filename) {
  if (!filename) {
    return false;
  }

  // Get the base path from environment variable
  char *basePath = getenv("OM_CONSTANT_PATH");
  if (!basePath) {
    // Use current directory if no path specified
    basePath = ".";
  }

  // Construct full file path
  size_t baseLen = strlen(basePath);
  size_t fnameLen = strlen(filename);
  size_t sepLen = strlen(DIR_SEPARATOR);
  size_t filePathLen = baseLen + sepLen + fnameLen + 1;

  char *filePath = (char *)malloc(filePathLen);
  if (!filePath) {
    fprintf(stderr, "Error allocating memory for file path\n");
    return false;
  }

  snprintf(
      filePath, filePathLen, "%s%s%s", basePath, DIR_SEPARATOR, filename);

  // Open the constants file
  int fd = open(filePath, O_RDONLY);
  if (fd < 0) {
    // File doesn't exist - this is OK, might not have constants
    free(filePath);
    return false;
  }

  // Get file size
  struct stat st;
  if (fstat(fd, &st) != 0) {
    fprintf(stderr, "Error getting file size for %s: %s\n", filePath,
        strerror(errno));
    close(fd);
    free(filePath);
    return false;
  }

  int64_t fileSize = st.st_size;
  g_constant_buffer_size = fileSize;

  // Check if file is larger than 1GB (mmap limit on z/OS)
  if (fileSize > MAX_MMAP_SIZE) {
    // Large file - use malloc + chunked read
    g_constant_buffer = malloc(g_constant_buffer_size);

    if (!g_constant_buffer) {
      fprintf(stderr, "Failed to allocate %lld bytes for constants\n",
          (long long)g_constant_buffer_size);
      close(fd);
      free(filePath);
      return false;
    }

    // Read file in 1GB chunks
    int64_t remaining = g_constant_buffer_size;
    int64_t offset = 0;
    char *destPtr = (char *)g_constant_buffer;

    while (remaining > 0) {
      int64_t chunkSize =
          (remaining > MAX_MMAP_SIZE) ? MAX_MMAP_SIZE : remaining;
      int64_t totalRead = 0;

      // Handle short reads
      while (totalRead < chunkSize) {
        ssize_t bytesRead =
            read(fd, destPtr + offset + totalRead, chunkSize - totalRead);

        if (bytesRead < 0) {
          fprintf(stderr, "Error reading constants at offset %lld: %s\n",
              (long long)(offset + totalRead), strerror(errno));
          free(g_constant_buffer);
          g_constant_buffer = NULL;
          g_constant_buffer_size = 0;
          close(fd);
          free(filePath);
          return false;
        }

        if (bytesRead == 0) {
          fprintf(stderr, "Unexpected EOF reading constants at offset %lld\n",
              (long long)(offset + totalRead));
          free(g_constant_buffer);
          g_constant_buffer = NULL;
          g_constant_buffer_size = 0;
          close(fd);
          free(filePath);
          return false;
        }

        totalRead += bytesRead;
      }

      offset += chunkSize;
      remaining -= chunkSize;
    }

    close(fd);
    fprintf(stderr,
        "Successfully preloaded %lld bytes of constants via malloc (>1GB on "
        "z/OS)\n",
        (long long)g_constant_buffer_size);
  } else {
    // Small file on z/OS - use mmap
    g_constant_buffer = mmap(0, fileSize, PROT_READ, __MAP_MEGA, fd, 0);
    if (g_constant_buffer == MAP_FAILED) {
      fprintf(stderr, "Error while mmapping %s: %s\n", filePath,
          strerror(errno));
      g_constant_buffer = NULL;
      g_constant_buffer_size = 0;
      close(fd);
      free(filePath);
      return false;
    }
    close(fd);
    fprintf(stderr,
        "Successfully preloaded %lld bytes of constants via mmap (z/OS)\n",
        (long long)g_constant_buffer_size);
  }

  free(filePath);
  return true;
}

/// Free the preloaded constant buffer.
/// Handles both malloc'd and mmap'd buffers appropriately.
///
static void omFreeBuffer(void) {
  if (g_constant_buffer) {
    // Check if we used malloc (>1GB) or mmap (≤1GB)
    if (g_constant_buffer_size > MAX_MMAP_SIZE) {
      // Was malloc'd
      free(g_constant_buffer);
      fprintf(stderr, "Freed constant buffer (malloc)\n");
    } else {
      // Was mmap'd
      munmap(g_constant_buffer, g_constant_buffer_size);
      fprintf(stderr, "Freed constant buffer (munmap)\n");
    }
    g_constant_buffer = NULL;
    g_constant_buffer_size = 0;
  }
}

/// Constructor: Preload constants at library load time (z/OS only)
void __attribute__((constructor)) omCtor(void) {
  // Get the constant filename from environment variable
  char *const_filename = getenv("OM_CONSTANT_FILE");
  if (!const_filename) {
    // No constant file specified, nothing to preload
    return;
  }

  // Load the constants file
  omMallocAndReadFile(const_filename);
}

/// Destructor: Free preloaded constants at library unload time (z/OS only)
void __attribute__((destructor)) omDtor(void) {
  omFreeBuffer();
}
#endif

#endif
