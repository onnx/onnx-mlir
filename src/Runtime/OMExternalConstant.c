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

#ifdef __MVS__
#include <dlfcn.h>
#include <libgen.h>
#include <sys/stat.h>
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
// Global buffer to store preloaded constants
static void *g_constant_buffer = NULL;
static int64_t g_constant_buffer_size = 0;

/// Constructor: Preload constants at library load time
void __attribute__ ((constructor)) omMallocAndReadFile(void) {
  Dl_info dl_info;
  
  // Get the path to the current shared library
  if (!dladdr((void *)omMallocAndReadFile, &dl_info)) {
    fprintf(stderr, "Failed to get library path\n");
    return;
  }
  
  // Get directory and basename of the .so file
  char *so_path = strdup(dl_info.dli_fname);
  if (!so_path) {
    return;
  }
  
  // Get directory
  char *so_path_copy = strdup(so_path);
  if (!so_path_copy) {
    free(so_path);
    return;
  }
  char *so_dir = dirname(so_path_copy);
  
  // Get basename (filename without directory)
  char *so_basename = basename(so_path);
  
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
    free(so_path);
    free(so_path_copy);
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
    free(so_path);
    return;
  }
  
  snprintf(filePath, filePathLen, "%s%s%s", basePath, DIR_SEPARATOR, const_filename);
  
  // Open the constants file
  int fd = open(filePath, O_RDONLY);
  if (fd < 0) {
    // File doesn't exist - this is OK, might not have constants
    free(const_filename);
    free(filePath);
    free(so_path);
    free(so_path_copy);
    return;
  }
  
  free(const_filename); // Done with this now
  
  // Get file size
  struct stat st;
  if (fstat(fd, &st) != 0) {
    close(fd);
    free(filePath);
    free(so_path);
    return;
  }
  
  int64_t fileSize = st.st_size;
  
  // Only preload if file is larger than 1GB
  #define MAX_MMAP_SIZE (1024LL * 1024LL * 1024LL)
  if (fileSize <= MAX_MMAP_SIZE) {
    // Small file - let mmap handle it at runtime
    close(fd);
    free(filePath);
    free(so_path);
    free(so_path_copy);
    return;
  }
  
  // Large file - preload into malloc'd buffer
  g_constant_buffer_size = fileSize;
  g_constant_buffer = malloc(g_constant_buffer_size);
  
  if (!g_constant_buffer) {
    fprintf(stderr, "Failed to allocate %lld bytes for constants\n",
            (long long)g_constant_buffer_size);
    close(fd);
    free(filePath);
    free(so_path);
    free(so_path_copy);
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
        free(so_path);
        free(so_path_copy);
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
        free(so_path);
        free(so_path_copy);
        return;
      }
      
      totalRead += bytesRead;
    }
    
    offset += chunkSize;
    remaining -= chunkSize;
  }
  
  close(fd);
  free(filePath);
  free(so_path);
  free(so_path_copy);
  
  fprintf(stderr, "Successfully preloaded %lld bytes of constants (>1GB)\n",
          (long long)g_constant_buffer_size);
  
  #undef MAX_MMAP_SIZE
}

/// Destructor: Free preloaded constants at library unload time
void __attribute__ ((destructor)) omFreeConstantBuffer(void) {
  if (g_constant_buffer) {
    free(g_constant_buffer);
    g_constant_buffer = NULL;
    g_constant_buffer_size = 0;
    fprintf(stderr, "Freed constant buffer\n");
  }
}

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
  // Check if we have a preloaded constant buffer from constructor
  if (g_constant_buffer != NULL && g_constant_buffer_size >= size) {
    constAddr[0] = g_constant_buffer;
    errno = 0;
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
  // z/OS: If model was not stored in malloc area, use mmap.
  void *tempAddr = mmap(0, size, PROT_READ, __MAP_MEGA, fd, 0);
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

#endif
