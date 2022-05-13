/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- OMRuntimeNNPA.c ---------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Onnx MLIR NNPA Accelerator Runtime
//
//===----------------------------------------------------------------------===//

// Include pthreads (need special treatment on Zos).
#ifdef __MVS__
#define _OPEN_THREADS
#endif
#include <pthread.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "zdnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Interface for device init and shutdown.
 *
 * For devices that requires initialization before execution, we suggest the
 * following interface. Assuming a device named X.
 *
 * 1. Define a variable OMIsInitAccelX initialized to zero. It should be safe
 *    to read this variable outside of a lock. Setting this value to one is done
 *    within OMInitAccelX and setting this value to zero is done within the
 *    OMShutdownAccelX.
 * 2. Define a function OMInitAccelX that initialize the device only once, and
 *    once it is initialized, set the OMIsInitAccelX value to 1. This function
 *    must be thread safe.
 * 3. Optionally define a function OMShutdownAccelX that shut down the device
 *    only once. This function is thread safe. Additional restrictions exist
 *    on this function, namely that it can only be called when provably no
 *    threads are using the accelerator. Failure to do so may result in
 *    incorrect result and/or execution failure.
 * 4. For models that use accelerator X, the compiler must insert a test of the
 *    type below before any use of accelerator's X functionality.
 *
 *    if (!OMIsInitAccelX) OMInitAccelX().
 *
 *    Calling OMInitAccelX() unconditionally is also appropriate.
 *
 * 5. Accelerators that requires a given level of support (e.g. the graph was
 *    compiled with code that requires level V), one may define a additional
 *    init function OMInitCompatibleAccelNNPA which passes the minimum level
 *    V as parameter. After initializing the function, the device is tested
 *    to see if it support level V. If not, an error is generated and the
 *    program abort.
 */

/* Init and shutdown for NNPA device.
 *
 * This test can be performed in the run_main_graph() without grabbing a lock,
 * as follows:
 *
 * if (!OMIsInitAccelNNPA) OMInitAccelNNPA();
 *
 * OMInitAccelNNPA() is thread safe, and is guaranteed to set
 * OMIsInitAccelNNPA=1 once any other threads are guaranteed to see the full
 * effects of the zdnn_init(). Because Z does not has a release consistency
 * memory subsystem, we don't need a hard memory fence between zdnn_init() and
 * OMIsInitAccelNNPA=1.
 *
 * For the OMShutdownAccelNNPA(), we simply set the OMIsInitAccelNNPA flag to
 * zero as there is currently no zdnn shutdown call. If one were added, then we
 * would follow the same code pattern as in the init function.
 */

// Define variable that tracks whether an accelerator is initialized or not.
// Initial value is uninitialized.
// Name must be OMIsInitAccelX where X=NNPA.
long OMIsInitAccelNNPA = 0;

// Mutex definitions for init and shutdown serialization.
pthread_mutex_t OMMutexForInitShutdownNNPA = PTHREAD_MUTEX_INITIALIZER;

/*!
 *  \brief Function that performs the initialization of the NNPA device.
 *
 *  Define function that performs the initialization as
 *  well as set the OMIsInitAccelNNPA to true. The function serialize
 *  potentially parallel accesses to this function so as to guarantee that the
 *  initializing code is executed exactly once.
 *  Function name must be OMInitAccelX where X=NNPA.
 */
void OMInitAccelNNPA() {
  if (!OMIsInitAccelNNPA) {
    /* Grab mutex. */
    pthread_mutex_lock(&OMMutexForInitShutdownNNPA);
    /* Test again in the mutex to see if accelerator is not initialized. */
    if (!OMIsInitAccelNNPA) {
      /* Still unitinitialized, actual init. */
      zdnn_init();
      /* No need for a fence due to strong consistency. */
      OMIsInitAccelNNPA = 1;
    } /* Release mutex. */
    pthread_mutex_unlock(&OMMutexForInitShutdownNNPA);
  }
}

/*!
 *  \brief Function that performs the initialization of the NNPA device and
 *   check that the NNPA version that the program was compiled for is compatible
 *   with the actual NNPA hardware.
 *
 *  Perform the same operations as OMInitAccelNNPA. In addition, it checks that
 *  the NNPA version that was embedded in the code when compiling the models is
 *  compatible with the NNPA version that is executing the code. If the versions
 *  are incompatible, an ERRNO=EPERM (Operation not permitted POSIX.1-2001) is
 *  set. The caller context is responsible to return NULL in the caller function
 *  so that users would know to check ERRNO to learn about why the call failed.
 *
 *  The assumption here is that all models are compiled for the same NNPA. If
 *  the compatibility succeeds once, it is assumed that all subsequent models
 *  were all compiled for the same NNPA version. However, if the compatibility
 *  fails during the first invocation of a model, the function will repetitively
 *  check if new models are compatible, in case the models were recompiled.
 *
 *  Function name must be OMInitAccelX where X=NNPA.
 *  @param versionNum Version of the library/Architecture level that models
 *  were compiled for.
 *  @return true for compatible versions or false on incompatible versions.
 */
uint64_t OMInitCompatibleAccelNNPA(uint64_t versionNum) {
  static int isCompatible = 0;
  if (!OMIsInitAccelNNPA) {
    /* Grab mutex. */
    pthread_mutex_lock(&OMMutexForInitShutdownNNPA);
    /* Test again in the mutex to see if accelerator is not initialized. */
    if (!OMIsInitAccelNNPA) {
      /* Still unitinitialized, actual init. */
      zdnn_init();
      /* Check if version is compatible */
      if (zdnn_is_version_runnable((uint32_t)versionNum))
        isCompatible = 1;
    }
    /* No need for a fence due to strong consistency. */
    OMIsInitAccelNNPA = 1;
    /* Release mutex. */
    pthread_mutex_unlock(&OMMutexForInitShutdownNNPA);
  }
  /* If not compatible, generate an error here. */
  if (!isCompatible) {
    /* Grab mutex. */
    pthread_mutex_lock(&OMMutexForInitShutdownNNPA);
    /* Check again if we have a compatible model. */
    if (zdnn_is_version_runnable((uint32_t)versionNum))
      isCompatible = 1;
    /* Release mutex. */
    pthread_mutex_unlock(&OMMutexForInitShutdownNNPA);
    if (!isCompatible) {
      /* Code below has to agree with zdnn.h convention. */
      unsigned long long ver_major = versionNum >> 16;
      unsigned long long ver_minor = (versionNum >> 8) & 0xff;
      unsigned long long ver_patch = versionNum & 0xff;
      fprintf(stderr,
          "Model is running on hardware that is not compatible with "
          "the zDNN library that this model was compiled for "
          "(version num %llu.%llu.%llu). Please check that the model is "
          "running on hardware with an integrated accelerator for AI "
          "(z16 +) that supports the required zDNN library version.\n ",
          ver_major, ver_minor, ver_patch);
      errno = EPERM;
      return false;
    }
  }
  return true;
}

/*!
 *  \brief Function that performs the shutdown of the NNPA device.
 *
 *  Function that performs the initialization as well as set the
 *  OMIsInitAccelNNPA to true. The function serialize potentially parallel
 *  accesses to this function so as to guarantee that the initializing code is
 *  executed exactly once.
 *  Function name must be OMInitAccelX where X=NNPA.
 */
void OMShutdownAccelNNPA() {
  if (OMIsInitAccelNNPA) {
    /* Grab mutex. */
    pthread_mutex_lock(&OMMutexForInitShutdownNNPA);
    /* Nothing to unitnitialize. */
    OMIsInitAccelNNPA = 0;
    /* Release mutex. */
    pthread_mutex_unlock(&OMMutexForInitShutdownNNPA);
  }
}

#ifdef __cplusplus
}
#endif
