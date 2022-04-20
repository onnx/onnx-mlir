/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- OMNNPA.c ---------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Onnx MLIR NNPA Accelerator Runtime
//
//===----------------------------------------------------------------------===//

#include <pthread.h>

#include "zdnn.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Interface for device init and shutdown.
 *
 * Testing if the code is initalized or not should be done prior to any zdnn
 * computations. This test can be performed in the run_main_graph() without
 * grabbing a lock, as follows:
 *
 * if (!OMIsInitAccelNNPA) OMInitAccelNNPA();
 *
 * OMInitAccelNNPA() is thread save, and is guaranteed to set
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
 * 
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
