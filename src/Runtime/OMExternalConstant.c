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
#include <unistd.h>

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
  // z/OS has different memory constraints for mmap usage,
  // to avoid certain system configurations for large constant sizes,
  // use malloc + read for large files
  #define MAX_MMAP_SIZE (1024LL * 1024LL * 1024LL)
  #define LOADING_SENTINEL ((void*)1)
  #define MAX_WAIT_MS 600000  // 600 seconds timeout
  #define SLEEP_MS 10         // 10ms between checks
  
  void *tempAddr;
  bool usedMalloc = false;
  
  if (size > MAX_MMAP_SIZE) {
    // For large files, use sentinel-based CAS to ensure only one thread loads
    void *expected = NULL;
    if (cds((cds_t *)&expected, (cds_t *)&constAddr[0], *(cds_t *)&LOADING_SENTINEL)) {
      // We won the race - we're responsible for loading
      tempAddr = malloc(size);
      if (!tempAddr) {
        fprintf(stderr, "Error allocating %lld bytes: %s\n",
                (long long)size, strerror(errno));
        // Reset sentinel to NULL so other threads can retry
        constAddr[0] = NULL;
        close(fd);
        if (basePath)
          free(filePath);
        return false;
      }
      usedMalloc = true;
      
      // Read file in 1GB chunks, handling short reads correctly
      int64_t remaining = size;
      int64_t offset = 0;
      char *destPtr = (char *)tempAddr;
      
      while (remaining > 0) {
        int64_t chunkSize = (remaining > MAX_MMAP_SIZE) ? MAX_MMAP_SIZE : remaining;
        int64_t totalRead = 0;
        
        // Inner loop to handle short reads (valid POSIX behavior for large reads)
        while (totalRead < chunkSize) {
          ssize_t bytesRead = read(fd, destPtr + offset + totalRead,
                                   chunkSize - totalRead);
          
          if (bytesRead < 0) {
            // Real error
            fprintf(stderr, "Error reading %s at offset %lld: %s\n",
                    fname, (long long)(offset + totalRead), strerror(errno));
            free(tempAddr);
            // Reset sentinel to NULL so other threads can retry
            constAddr[0] = NULL;
            close(fd);
            if (basePath)
              free(filePath);
            return false;
          }
          
          if (bytesRead == 0) {
            // EOF - should not happen unless file is truncated
            fprintf(stderr, "Unexpected EOF reading %s at offset %lld\n",
                    fname, (long long)(offset + totalRead));
            free(tempAddr);
            // Reset sentinel to NULL so other threads can retry
            constAddr[0] = NULL;
            close(fd);
            if (basePath)
              free(filePath);
            return false;
          }
          
          totalRead += bytesRead;
        }
        
        offset += chunkSize;
        remaining -= chunkSize;
      }
      
      // Successfully loaded - update constAddr with real pointer
      constAddr[0] = tempAddr;
      close(fd);
      if (basePath)
        free(filePath);
      
      // Clear errno before returning success
      errno = 0;
      return true;
    } else {
      // Another thread is loading or already loaded - wait for it
      close(fd);
      if (basePath)
        free(filePath);
      
      int waited_ms = 0;
      while (constAddr[0] == LOADING_SENTINEL) {
        if (waited_ms >= MAX_WAIT_MS) {
          fprintf(stderr, "Timeout waiting for constant loading after %d ms\n",
                  MAX_WAIT_MS);
          return false;
        }
        usleep(SLEEP_MS * 1000);  // Sleep 10ms to reduce CPU spinning
        waited_ms += SLEEP_MS;
      }
      
      // Check if the loading thread succeeded
      if (constAddr[0] == NULL) {
        fprintf(stderr, "Other thread failed to load constants\n");
        return false;
      }
      
      // Successfully loaded by another thread
      errno = 0;
      return true;
    }
  } else {
    // Use mmap for files <= 1GB (standard CAS approach)
    tempAddr = mmap(0, size, PROT_READ, __MAP_MEGA, fd, 0);
    if (tempAddr == MAP_FAILED) {
      fprintf(stderr, "Error while mmapping %s: %s\n", fname, strerror(errno));
      close(fd);
      if (basePath)
        free(filePath);
      return false;
    }
    
    // Standard CAS for mmap path
    void *expected = NULL;
    if (cds((cds_t *)&expected, (cds_t *)&constAddr[0], *(cds_t *)&tempAddr)) {
      // Another thread won, clean up our mmap
      munmap(tempAddr, size);
    }
    
    close(fd);
    if (basePath)
      free(filePath);
    
    errno = 0;
    return true;
  }
#else
  void *tempAddr = mmap(0, size, PROT_READ, MAP_SHARED, fd, 0);
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
  if (!__sync_bool_compare_and_swap(&constAddr[0], NULL, tempAddr))
    munmap(tempAddr, size);

  /* Either we succeeded in setting constAddr or someone else did it.
   * Either way, constAddr is now setup. We can close our fd without
   * invalidating the mmap.
   */
  close(fd);
  if (basePath)
    free(filePath);
  return true;
#endif
}
#else
  if (!__sync_bool_compare_and_swap(&constAddr[0], NULL, tempAddr))
    munmap(tempAddr, size);

  /* Either we succeeded in setting constAddr or someone else did it.
   * Either way, constAddr is now setup. We can close our fd without
   * invalidating the mmap.
   */
  close(fd);
  if (basePath)
    free(filePath);
  return true;
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

#endif
