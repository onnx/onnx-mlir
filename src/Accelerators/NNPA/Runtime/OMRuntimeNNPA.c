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

#include "zdnn.h"

#ifdef __cplusplus
extern "C" {
#endif

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
  if (OMIsInitAccelNNPA) { /* Grab outer mutex. */
    pthread_mutex_lock(&OMOuterMutexForInitShutdownNNPA);
    OMIsInitAccelNNPA = 0;
    /* Release outer mutex. */
    pthread_mutex_unlock(&OMOuterMutexForInitShutdownNNPA);
  }
}

#ifdef __cplusplus
}
#endif
