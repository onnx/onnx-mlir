/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===--------------- jnidummy.c - JNI wrapper dummy routine ---------------===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains a dummy routine to force the link editor to embed code
// in libjniruntime.a into libmodel.so.
//
//===----------------------------------------------------------------------===//

#include "com_ibm_onnxmlir_OMModel.h"

/* Dummy routine to force the link editor to embed code in libjniruntime.a
   into libmodel.so */
void __dummy_do_not_call__(JNIEnv *env, jclass cls, jobject obj) {
  Java_com_ibm_onnxmlir_OMModel_main_1graph_1jni(NULL, NULL, NULL);
  Java_com_ibm_onnxmlir_OMModel_query_1entry_1points(NULL, NULL);
  Java_com_ibm_onnxmlir_OMModel_input_1signature_1jni(NULL, NULL, NULL);
  Java_com_ibm_onnxmlir_OMModel_output_1signature_1jni(NULL, NULL, NULL);
}
