/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------- OMAccelerator.inc --------------------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// Implementation for Onnx Mlir Runtime variables/functions related to
// accelerators.
//
//===----------------------------------------------------------------------===//

#include <pthread.h>

#include "src/Accelerators/Accelerators.inc"

// Define variable that tracks whether an accelerator is initialized or not.
// Initial value is uninitialized.
#define DEFINE_OM_ACCEL_IS_INIT_VARIABLE(name) long OMIsInitAccel##name = 0;

// Declare internal function that performs the initialization and shutdown.
#define DECLARE_OM_ACCEL_INIT_INTERNAL_FUNCTION(name)                          \
  extern void InitAccel##name();
#define DECLARE_OM_ACCEL_SHUTDOWN_INTERNAL_FUNCTION(name)                      \
  extern void ShutdownAccel##name();

// Define function that performs the serialization of the initialization as well
// as set the OMIsInitAccel##name to true.
#define DEFINE_OM_ACCEL_INIT_FUNCTION(name)                                    \
  void OMInitAccel##name() {                                                   \
    if (!OMIsInitAccel##name) {                                                \
      /* Grab outer mutex. */                                                  \
      pthread_mutex_lock(&OMOuterMutexForInitShutdown);                        \
      /* Test again in the mutex to see if accelerator is not initialized. */  \
      if (!OMIsInitAccel##name) {                                              \
        /* Still unitinitialized, get inner mutex to fence init code. */       \
        pthread_mutex_lock(&OMInnerMutexForInitShutdown);                      \
        /* Actual init. */                                                     \
        InitAccel##name();                                                     \
        /* Release inner mutex, and then set accelerator to initialized. */    \
        pthread_mutex_unlock(&OMInnerMutexForInitShutdown);                    \
        OMIsInitAccel##name = 1;                                               \
      }                                                                        \
      /* Release outer mutex. */                                               \
      pthread_mutex_unlock(&OMOuterMutexForInitShutdown);                      \
    }                                                                          \
  }

// Define function that performs the serialization of the shutdown as well
// as set the OMIsInitAccel##name to false.
#define DEFINE_OM_ACCEL_SHUTDOWN_FUNCTION(name)                                \
  void OMShutdownAccel##name() {                                               \
    if (OMIsInitAccel##name) {                                                 \
      /* Grab outer mutex. */                                                  \
      pthread_mutex_lock(&OMOuterMutexForInitShutdown);                        \
      /* Test again in the mutex to see if accelerator is initialized. */      \
      if (OMIsInitAccel##name) {                                               \
        /* Still initinitialized, get inner mutex to fence init code. */       \
        pthread_mutex_lock(&OMInnerMutexForInitShutdown);                      \
        /* Actual shutdown. */                                                 \
        ShutdownAccel##name();                                                 \
        /* Release inner mutex, and then set accelerator to uninitialized. */  \
        pthread_mutex_unlock(&OMInnerMutexForInitShutdown);                    \
        OMIsInitAccel##name = 0;                                               \
      }                                                                        \
      /* Release outer mutex. */                                               \
      pthread_mutex_unlock(&OMOuterMutexForInitShutdown);                      \
    }                                                                          \
  }

#ifdef __cplusplus
extern "C" {
#endif

// Mutex definitions for init and shutdown serialization. A common set is used
// for all accelerators. Inner mutex is used to implement a fence around the
// init/shutdown code, to make sure that all side effects from such operations
// are completed prior to switching the globally readable OMIsInitAccel##name.
pthread_mutex_t OMOuterMutexForInitShutdown = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t OMInnerMutexForInitShutdown = PTHREAD_MUTEX_INITIALIZER;

// Define the OMIsInitAccel##name variable that records if an accelerator is
// initialized (nonzero) or is uninitialized (zero). Value is initially zero,
// can only be set in a mutex, but can be read without mutex.
APPLY_TO_ACCELERATORS(DEFINE_OM_ACCEL_IS_INIT_VARIABLE)

// Define internal init and shutdown functions. They will be linked in.
APPLY_TO_ACCELERATORS(DECLARE_OM_ACCEL_INIT_INTERNAL_FUNCTION)
APPLY_TO_ACCELERATORS(DECLARE_OM_ACCEL_SHUTDOWN_INTERNAL_FUNCTION)

// Define the external init and shutdown functions.
APPLY_TO_ACCELERATORS(DEFINE_OM_ACCEL_INIT_FUNCTION)
APPLY_TO_ACCELERATORS(DEFINE_OM_ACCEL_SHUTDOWN_FUNCTION)

#ifdef __cplusplus
}
#endif
