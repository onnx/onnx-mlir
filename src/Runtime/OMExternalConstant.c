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
  fprintf(stderr, "[DEBUG] omMMapBinaryFile called: fname=%s, size=%lld, isLE=%lld, constAddr=%p\n",
          fname, (long long)size, (long long)isLE, (void*)constAddr);
  
  if (constAddr == NULL) {
    fprintf(stderr, "[ERROR] constAddr is NULL\n");
    return false;
  }

  // Already mmaped. Nothing to do.
  if (constAddr[0] != NULL) {
    fprintf(stderr, "[DEBUG] constAddr[0] already set to %p, returning early (idempotent)\n",
            constAddr[0]);
    return true;
  }

  char *filePath;
  char *basePath = getenv("OM_CONSTANT_PATH");
  if (basePath) {
    fprintf(stderr, "[DEBUG] OM_CONSTANT_PATH=%s\n", basePath);
    size_t baseLen = strlen(basePath);
    size_t fnameLen = strlen(fname);
    size_t sepLen = strlen(DIR_SEPARATOR);
    size_t filePathLen = baseLen + sepLen + fnameLen + 1;
    filePath = (char *)malloc(filePathLen);
    if (!filePath) {
      fprintf(stderr, "[ERROR] Failed to malloc %zu bytes for filePath: %s\n",
              filePathLen, strerror(errno));
      return false;
    }
    snprintf(filePath, filePathLen, "%s%s%s", basePath, DIR_SEPARATOR, fname);
    fprintf(stderr, "[DEBUG] Full file path: %s\n", filePath);
  } else {
    fprintf(stderr, "[DEBUG] OM_CONSTANT_PATH not set, using fname directly\n");
    filePath = (char *)fname;
  }
  
  fprintf(stderr, "[DEBUG] Opening file: %s\n", filePath);
  int fd = open(filePath, O_RDONLY);
  if (fd < 0) {
    fprintf(stderr,
        "[ERROR] Failed to open %s: %s. Please set OM_CONSTANT_PATH to the folder "
        "that contains %s.\n",
        filePath, strerror(errno), fname);
    if (basePath)
      free(filePath);
    return false;
  }
  fprintf(stderr, "[DEBUG] File opened successfully, fd=%d\n", fd);

#ifdef __MVS__
  // z/OS has different memory constraints for mmap usage,
  // to avoid certain system configurations for large constant sizes,
  // use malloc + read for large files
  #define MAX_MMAP_SIZE (1024LL * 1024LL * 1024LL)
  void *tempAddr;
  bool usedMalloc = false;
  
  if (size > MAX_MMAP_SIZE) {
    fprintf(stderr, "[DEBUG] z/OS: File size %lld bytes exceeds MAX_MMAP_SIZE, using malloc+read path\n",
            (long long)size);
    
    // Allocate memory and read file in chunks
    tempAddr = malloc(size);
    if (!tempAddr) {
      fprintf(stderr, "[ERROR] Failed to allocate %lld bytes: %s\n",
              (long long)size, strerror(errno));
      close(fd);
      if (basePath)
        free(filePath);
      return false;
    }
    fprintf(stderr, "[DEBUG] Successfully allocated %lld bytes at %p\n",
            (long long)size, tempAddr);
    usedMalloc = true;
    
    // Read file in 1GB chunks, handling short reads correctly
    int64_t remaining = size;
    int64_t offset = 0;
    char *destPtr = (char *)tempAddr;
    int chunkNum = 0;
    
    while (remaining > 0) {
      int64_t chunkSize = (remaining > MAX_MMAP_SIZE) ? MAX_MMAP_SIZE : remaining;
      int64_t totalRead = 0;
      chunkNum++;
      
      fprintf(stderr, "[DEBUG] Reading chunk %d: offset=%lld, chunkSize=%lld, remaining=%lld\n",
              chunkNum, (long long)offset, (long long)chunkSize, (long long)remaining);
      
      // Inner loop to handle short reads (valid POSIX behavior for large reads)
      int readAttempts = 0;
      while (totalRead < chunkSize) {
        ssize_t bytesRead = read(fd, destPtr + offset + totalRead,
                                 chunkSize - totalRead);
        readAttempts++;
        
        if (bytesRead < 0) {
          // Real error
          fprintf(stderr, "[ERROR] read() failed at offset %lld after %d attempts: %s\n",
                  (long long)(offset + totalRead), readAttempts, strerror(errno));
          free(tempAddr);
          close(fd);
          if (basePath)
            free(filePath);
          return false;
        }
        
        if (bytesRead == 0) {
          // EOF - should not happen unless file is truncated
          fprintf(stderr, "[ERROR] Unexpected EOF at offset %lld (expected %lld more bytes)\n",
                  (long long)(offset + totalRead), (long long)(chunkSize - totalRead));
          free(tempAddr);
          close(fd);
          if (basePath)
            free(filePath);
          return false;
        }
        
        totalRead += bytesRead;
        if (bytesRead < (chunkSize - totalRead + bytesRead)) {
          fprintf(stderr, "[DEBUG] Short read: got %lld bytes, %lld remaining in chunk (attempt %d)\n",
                  (long long)bytesRead, (long long)(chunkSize - totalRead), readAttempts);
        }
      }
      
      fprintf(stderr, "[DEBUG] Chunk %d complete: read %lld bytes in %d attempt(s)\n",
              chunkNum, (long long)totalRead, readAttempts);
      
      offset += chunkSize;
      remaining -= chunkSize;
    }
    
    fprintf(stderr, "[DEBUG] Successfully read all %lld bytes from %s\n",
            (long long)size, fname);
  } else {
    fprintf(stderr, "[DEBUG] z/OS: File size %lld bytes <= MAX_MMAP_SIZE, using mmap path\n",
            (long long)size);
    
    // Use mmap for files <= 1GB
    tempAddr = mmap(0, size, PROT_READ, __MAP_MEGA, fd, 0);
    if (tempAddr == MAP_FAILED) {
      fprintf(stderr, "[ERROR] mmap failed for %s: %s\n", fname, strerror(errno));
      close(fd);
      if (basePath)
        free(filePath);
      return false;
    }
    fprintf(stderr, "[DEBUG] Successfully mmapped %lld bytes at %p\n",
            (long long)size, tempAddr);
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
#endif

  /* Prepare to compare-and-swap to setup the shared constAddr.
   * If we fail, another thread beat us so free our mmap.
   */
#ifdef __MVS__
  fprintf(stderr, "[DEBUG] Attempting CAS to set constAddr[0] from NULL to %p\n", tempAddr);
  void *expected = NULL;
  if (cds((cds_t *)&expected, (cds_t *)&constAddr[0], *(cds_t *)&tempAddr)) {
    // CAS failed - another thread already set constAddr
    fprintf(stderr, "[DEBUG] CAS failed - another thread set constAddr[0] to %p, cleaning up our allocation\n",
            constAddr[0]);
    // Clean up based on allocation method
    if (usedMalloc)
      free(tempAddr);
    else
      munmap(tempAddr, size);
  } else {
    fprintf(stderr, "[DEBUG] CAS succeeded - constAddr[0] now set to %p\n", tempAddr);
  }
  
  /* Either we succeeded in setting constAddr or someone else did it.
   * Either way, constAddr is now setup. We can close our fd without
   * invalidating the mmap.
   */
  close(fd);
  if (basePath)
    free(filePath);
  
  // Clear errno before returning success to prevent stale errno from leaking
  errno = 0;
  fprintf(stderr, "[DEBUG] omMMapBinaryFile returning true for %s (constAddr[0]=%p)\n",
          fname, constAddr[0]);
  return true;
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
