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

#include <pthread.h>
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
 * effects of the zdnn_init(). Because Z has a release consistency memory
 * subsystem, we need a hard memory fence between zdnn_init() and
 * OMIsInitAccelNNPA=1. Because we are sticking to posix thread library here,
 * the easiest way to express a fence is to add an additional lock (the inner
 * mutex) as pthread_mutex_unlock() guarantees a fence in it (for release
 * consistency architectures).
 *
 * For the OMShutdownAccelNNPA(), we simply set the OMIsInitAccelNNPA flag to
 * zero as there is currently no zdnn shutdown call. If one were added, then we
 * would follow the same code pattern as in the init function.
 */

// Define variable that tracks whether an accelerator is initialized or not.
// Initial value is uninitialized.
// Name must be OMIsInitAccelX where X=NNPA.
long OMIsInitAccelNNPA = 0;

// Mutex definitions for init and shutdown serialization. A common set is used
// for all accelerators. Inner mutex is used to implement a fence around the
// init/shutdown code, to make sure that all side effects from such operations
// are completed prior to switching the globally readable OMIsInitAccelNNPA.
pthread_mutex_t OMOuterMutexForInitShutdownNNPA = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t OMInnerMutexForInitShutdownNNPA = PTHREAD_MUTEX_INITIALIZER;

// Define function that performs the serialization of the initialization as well
// as set the OMIsInitAccelNNPA to true.
// Name must be OMInitAccelX where X=NNPA.
void OMInitAccelNNPA() {
  if (!OMIsInitAccelNNPA) {
    /* Grab outer mutex. */
    pthread_mutex_lock(&OMOuterMutexForInitShutdownNNPA);
    /* Test again in the mutex to see if accelerator is not initialized. */
    if (!OMIsInitAccelNNPA) {
      /* Still unitinitialized, get inner mutex to fence init code. */
      pthread_mutex_lock(&OMInnerMutexForInitShutdownNNPA);
      /* Actual init. */
      zdnn_init();
      /* Release inner mutex, and then set accelerator to initialized. */
      pthread_mutex_unlock(&OMInnerMutexForInitShutdownNNPA);
      OMIsInitAccelNNPA = 1;
    } /* Release outer mutex. */
    pthread_mutex_unlock(&OMOuterMutexForInitShutdownNNPA);
  }
}

// Perform the same initialization and also check that the NNPA version that the
// program was compiled for is compatible with the actual NNPA hardware.
void OMInitCompatibleAccelNNPA(uint64_t versionNum) {
  if (!OMIsInitAccelNNPA) {
    int isCompatible = 1;
    /* Grab outer mutex. */
    pthread_mutex_lock(&OMOuterMutexForInitShutdownNNPA);
    /* Test again in the mutex to see if accelerator is not initialized. */
    if (!OMIsInitAccelNNPA) {
      /* Still unitinitialized, get inner mutex to fence init code. */
      pthread_mutex_lock(&OMInnerMutexForInitShutdownNNPA);
      /* Actual init. */
      zdnn_init();
      /* Release inner mutex, and then set accelerator to initialized. */
      pthread_mutex_unlock(&OMInnerMutexForInitShutdownNNPA);
      /* Check if version is compatible */
      isCompatible = zdnn_is_version_runnable((uint32_t)versionNum);
      OMIsInitAccelNNPA = 1;
    } /* Release outer mutex. */
    pthread_mutex_unlock(&OMOuterMutexForInitShutdownNNPA);
    /* If not compatible, generate an error here */
    if (!isCompatible) {
      fprintf(stderr,
          "Attempting to initialize zdnn with version num %llu, which is "
          "not compatible with current NNPA hardware\n",
          versionNum);
      exit(1);
    }
  }
}

// Define function that performs the serialization of the shutdown as well
// as set the OMIsInitAccelNNPA to false.
// Name must be OMShutdownAccelX where X=NNPA.
void OMShutdownAccelNNPA() {
  if (OMIsInitAccelNNPA) {
    /* Grab outer mutex. */
    pthread_mutex_lock(&OMOuterMutexForInitShutdownNNPA);
    /* Nothing to unitnitialize, so skip inner mutex. */
    OMIsInitAccelNNPA = 0;
    /* Release outer mutex. */
    pthread_mutex_unlock(&OMOuterMutexForInitShutdownNNPA);
  }
}

#ifdef __cplusplus
}
#endif
