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

#include <libgen.h>

#ifdef __MVS__
#define MAX_MMAP_SIZE (1024LL * 1024LL * 1024LL)
#include "metal_csvquery.h"
#else
#define _GNU_SOURCE
#include <dlfcn.h>
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

// Global buffer to store preloaded constants (all platforms)
static void *g_constant_buffer = NULL;
static int64_t g_constant_buffer_size = 0;
#ifndef __MVS__
static bool g_used_mmap = false; // Track if we used mmap (for munmap in destructor)
#endif

/// Constructor: Preload constants at library load time (all platforms)
void __attribute__ ((constructor)) omPreloadConstants(void) {
  char so_path[1024];
  
#ifdef __MVS__
  // z/OS: Use Metal C CSVQUERY to get the path to the current shared library
  int path_len = metal_get_module_path(so_path, sizeof(so_path));
  if (path_len == 0) {
    fprintf(stderr, "Failed to get library path via CSVQUERY\n");
    return;
  }
#else
  // Non-z/OS: Use dladdr to get the path to the current shared library
  Dl_info dl_info;
  if (!dladdr((void *)omPreloadConstants, &dl_info)) {
    fprintf(stderr, "Failed to get library path via dladdr\n");
    return;
  }
  strncpy(so_path, dl_info.dli_fname, sizeof(so_path) - 1);
  so_path[sizeof(so_path) - 1] = '\0';
#endif
  
  // Get directory and basename of the .so file
  char *so_path_copy1 = strdup(so_path);
  char *so_path_copy2 = strdup(so_path);
  if (!so_path_copy1 || !so_path_copy2) {
    free(so_path_copy1);
    free(so_path_copy2);
    return;
  }
  
  char *so_dir = dirname(so_path_copy1);
  char *so_basename = basename(so_path_copy2);
  
  // Check for OM_CONSTANT_PATH environment variable
  char *basePath = getenv("OM_CONSTANT_PATH");
  if (!basePath) {
    basePath = so_dir;
  }
  
  // Construct constants filename from .so name
  // e.g., "model.so" -> "model.constants.bin"
  size_t basename_len = strlen(so_basename);
  size_t const_filename_len = basename_len + 20; // extra space for ".constants.bin"
  char *const_filename = (char *)malloc(const_filename_len);
  if (!const_filename) {
    free(so_path_copy1);
    free(so_path_copy2);
    return;
  }
  
  // Copy basename, removing .so extension if present
  char *dot = strrchr(so_basename, '.');
  if (dot && strcmp(dot, ".so") == 0) {
    size_t name_len = dot - so_basename;
    strncpy(const_filename, so_basename, name_len);
    const_filename[name_len] = '\0';
  } else {
    strncpy(const_filename, so_basename, const_filename_len - 1);
    const_filename[const_filename_len - 1] = '\0';
  }
  
  // Safely append ".constants.bin"
  strncat(const_filename, ".constants.bin", const_filename_len - strlen(const_filename) - 1);
  
  size_t baseLen = strlen(basePath);
  size_t fnameLen = strlen(const_filename);
  size_t sepLen = strlen(DIR_SEPARATOR);
  size_t filePathLen = baseLen + sepLen + fnameLen + 1;
  
  char *filePath = (char *)malloc(filePathLen);
  if (!filePath) {
    free(const_filename);
    free(so_path_copy1);
    free(so_path_copy2);
    return;
  }
  
  snprintf(filePath, filePathLen, "%s%s%s", basePath, DIR_SEPARATOR, const_filename);
  
  // Open the constants file
  int fd = open(filePath, O_RDONLY);
  if (fd < 0) {
    // File doesn't exist - this is OK, might not have constants
    free(const_filename);
    free(filePath);
    free(so_path_copy1);
    free(so_path_copy2);
    return;
  }
  
  free(const_filename); // Done with this now
  
  // Get file size
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    free(filePath);
    free(so_path_copy1);
    free(so_path_copy2);
    return;
  }
  
  int64_t fileSize = st.st_size;
  g_constant_buffer_size = fileSize;
  
#ifdef __MVS__
  // z/OS: Check if file is larger than 1GB (mmap limit)
  if (fileSize > MAX_MMAP_SIZE) {
    // Large file - use malloc + chunked read
    g_constant_buffer = malloc(g_constant_buffer_size);
    
    if (!g_constant_buffer) {
      fprintf(stderr, "Failed to allocate %lld bytes for constants\n",
              (long long)g_constant_buffer_size);
      close(fd);
      free(filePath);
      free(so_path_copy1);
      free(so_path_copy2);
      return;
    }
    
    // Read file in 1GB chunks
    int64_t remaining = g_constant_buffer_size;
    int64_t offset = 0;
    char *destPtr = (char *)g_constant_buffer;
    
    while (remaining > 0) {
      int64_t chunkSize = (remaining > MAX_MMAP_SIZE) ? MAX_MMAP_SIZE : remaining;
      int64_t totalRead = 0;
      
      // Handle short reads
      while (totalRead < chunkSize) {
        ssize_t bytesRead = read(fd, destPtr + offset + totalRead,
                                 chunkSize - totalRead);
        
        if (bytesRead < 0) {
          fprintf(stderr, "Error reading constants at offset %lld: %s\n",
                  (long long)(offset + totalRead), strerror(errno));
          free(g_constant_buffer);
          g_constant_buffer = NULL;
          g_constant_buffer_size = 0;
          close(fd);
          free(filePath);
          free(so_path_copy1);
          free(so_path_copy2);
          return;
        }
        
        if (bytesRead == 0) {
          fprintf(stderr, "Unexpected EOF reading constants at offset %lld\n",
                  (long long)(offset + totalRead));
          free(g_constant_buffer);
          g_constant_buffer = NULL;
          g_constant_buffer_size = 0;
          close(fd);
          free(filePath);
          free(so_path_copy1);
          free(so_path_copy2);
          return;
        }
        
        totalRead += bytesRead;
      }
      
      offset += chunkSize;
      remaining -= chunkSize;
    }
    
    close(fd);
    fprintf(stderr, "Successfully preloaded %lld bytes of constants via malloc (>1GB on z/OS)\n",
            (long long)g_constant_buffer_size);
  } else {
    // Small file on z/OS - use mmap
    g_constant_buffer = mmap(0, fileSize, PROT_READ, __MAP_MEGA, fd, 0);
    if (g_constant_buffer == MAP_FAILED) {
      fprintf(stderr, "Error while mmapping %s: %s\n", filePath, strerror(errno));
      g_constant_buffer = NULL;
      g_constant_buffer_size = 0;
      close(fd);
      free(filePath);
      free(so_path_copy1);
      free(so_path_copy2);
      return;
    }
    close(fd);
    fprintf(stderr, "Successfully preloaded %lld bytes of constants via mmap (z/OS)\n",
            (long long)g_constant_buffer_size);
  }
#else
  // Non-z/OS: Always use mmap
  g_constant_buffer = mmap(0, fileSize, PROT_READ, MAP_SHARED, fd, 0);
  if (g_constant_buffer == MAP_FAILED) {
    fprintf(stderr, "Error while mmapping %s: %s\n", filePath, strerror(errno));
    g_constant_buffer = NULL;
    g_constant_buffer_size = 0;
    close(fd);
    free(filePath);
    free(so_path_copy1);
    free(so_path_copy2);
    return;
  }
  close(fd);
  g_used_mmap = true;
  fprintf(stderr, "Successfully preloaded %lld bytes of constants via mmap\n",
          (long long)g_constant_buffer_size);
#endif
  
  free(filePath);
  free(so_path_copy1);
  free(so_path_copy2);
}

/// Destructor: Free preloaded constants at library unload time (all platforms)
void __attribute__ ((destructor)) omFreeConstantBuffer(void) {
  if (g_constant_buffer) {
#ifdef __MVS__
    // z/OS: Check if we used malloc (>1GB) or mmap (≤1GB)
    if (g_constant_buffer_size > MAX_MMAP_SIZE) {
      // Was malloc'd
      free(g_constant_buffer);
      fprintf(stderr, "Freed constant buffer (malloc)\n");
    } else {
      // Was mmap'd
      munmap(g_constant_buffer, g_constant_buffer_size);
      fprintf(stderr, "Freed constant buffer (munmap)\n");
    }
#else
    // Non-z/OS: Always mmap'd
    if (g_used_mmap) {
      munmap(g_constant_buffer, g_constant_buffer_size);
      fprintf(stderr, "Freed constant buffer (munmap)\n");
    }
#endif
    g_constant_buffer = NULL;
    g_constant_buffer_size = 0;
  }
}

/// Return the address of preloaded constants.
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

  // Already loaded. Nothing to do.
  if (constAddr[0] != NULL)
    return true;

  // Return the preloaded constant buffer
  if (g_constant_buffer != NULL && g_constant_buffer_size > 0) {
    constAddr[0] = g_constant_buffer;
    errno = 0;
    return true;
  }

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

  /* Either we succeeded in setting constAddr or someone else did it.
   * Either way, constAddr is now setup. We can close our fd without
   * invalidating the mmap.
   */
  close(fd);
  if (basePath)
    free(filePath);
  return false;
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
