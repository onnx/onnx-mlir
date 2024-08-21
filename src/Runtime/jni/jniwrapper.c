/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- jniwrapper.c - JNI wrapper Implementation -------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file contains implementation of the JNI wrapper to allow Java users
// to call the model execution API.
//
//===----------------------------------------------------------------------===//

#if defined(__APPLE__)
#include <stdlib.h>
#elif defined(__MVS__)
#define _XOPEN_SOURCE_EXTENDED 1
#include <stdlib.h>
#include <unistd.h>
#else
#include <malloc.h>
#endif
#include <assert.h>
#include <limits.h>
#include <string.h>

#include "OnnxMlirRuntime.h"
#include "com_ibm_onnxmlir_OMModel.h"
#include "jnilog.h"

extern OMTensorList *run_main_graph(OMTensorList *);

/* Declare type var, make call and assign to var, check condition.
 * It's assumed that a Java exception has already been thrown so
 * this call simply returns NULL.
 */
#define CHECK_CALL(type, var, call, success, ...)                              \
  type(var) = (call);                                                          \
  do {                                                                         \
    if (!(success)) {                                                          \
      LOG_PRINTF(LOG_ERROR, __VA_ARGS__);                                      \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

/* Make a JNI call, log error and throw Java exception if the call
 * failed. stmt is out of do/while block because for a variable
 * definition it needs to be visible outside the do/while block.
 *
 * Note if the JNI call throws an exception, we simply return since
 * the exception will stay thrown until Java handles it. Some JNI
 * calls may fail with bad return value without throwing an exception.
 * In that case, we throw a new exception for Java to handle.
 */
#define JNI_CALL(env, stmt, success, ecpt, ...)                                \
  stmt;                                                                        \
  do {                                                                         \
    if ((*env)->ExceptionCheck(env)) {                                         \
      LOG_PRINTF(LOG_ERROR, __VA_ARGS__);                                      \
      return NULL;                                                             \
    } else if (!(success)) {                                                   \
      LOG_PRINTF(LOG_ERROR, __VA_ARGS__);                                      \
      if (ecpt)                                                                \
        (*env)->ThrowNew((env), (ecpt), jnistr[MSG_JNI_CALL_ERROR]);           \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

/* Make a JNI call and assign return value to var,
 * log error and throw Java exception if the call failed.
 */
#define JNI_VAR_CALL(env, var, call, success, ecpt, ...)                       \
  JNI_CALL((env), (var) = (call), (success), (ecpt), __VA_ARGS__)

/* Declare type var, make a JNI call and assign return value to var,
 * log error and throw Java exception if the call failed.
 */
#define JNI_TYPE_VAR_CALL(env, type, var, call, success, ecpt, ...)            \
  JNI_CALL((env), type(var) = (call), (success), (ecpt), __VA_ARGS__);

/* Make a native library call, check success condition,
 * log error and throw Java exception if native code failed.
 * stmt is out of do/while block because for a variable
 * definition it needs to be visible outside the do/while block.
 */
#define LIB_CALL(stmt, success, env, ecpt, ...)                                \
  stmt;                                                                        \
  do {                                                                         \
    if (!(success)) {                                                          \
      LOG_PRINTF(LOG_ERROR, __VA_ARGS__);                                      \
      if (ecpt)                                                                \
        (*env)->ThrowNew((env), (ecpt), jnistr[MSG_NATIVE_CODE_ERROR]);        \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

/* Make a native library call and assign return value to var,
 * log error and throw Java exception if the call failed.
 * Also check success condition.
 */
#define LIB_VAR_CALL(var, call, success, env, ecpt, ...)                       \
  LIB_CALL((var) = (call), (success), (env), (ecpt), __VA_ARGS__);

/* Declare type var, make a native library call and assign
 * return value to var, log error and throw Java exception
 * if the call failed. Also check success condition.
 */
#define LIB_TYPE_VAR_CALL(type, var, call, success, env, ecpt, ...)            \
  LIB_CALL(type(var) = (call), (success), (env), (ecpt), __VA_ARGS__);

/* Debug output of OMTensor fields */
#define OMT_DEBUG(                                                             \
    i, n, data, shape, strides, dataType, bufferSize, rank, owning)            \
  do {                                                                         \
    char tmp[1024];                                                            \
    LOG_BUF((dataType), (tmp), (data), (n));                                   \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:(data)=[%s]", (i), (tmp));                  \
    tmp[0] = '\0';                                                             \
    LOG_LONG_BUF(tmp, (shape), (rank));                                        \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:(shape)=[%s]", (i), tmp);                   \
    LOG_LONG_BUF(tmp, (strides), (rank));                                      \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:(strides)=[%s]", (i), tmp);                 \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:(dataType)=%d", (i), (dataType));           \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:(bufferSize)=%ld", (i), (bufferSize));      \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:(rank)=%ld", (i), (rank));                  \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:(owning)=%ld", (i), (owning));              \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:(numElems)=%ld", (i), (n));                 \
  } while (0)

/* Debug output of hex string */
#define HEX_DEBUG(label, string, n)                                            \
  do {                                                                         \
    char tmp[1024];                                                            \
    LOG_CHAR_XBUF(tmp, (string), (n));                                         \
    LOG_PRINTF(LOG_DEBUG, "%s(%d):[%s]", (label), (n), tmp);                   \
  } while (0)

/* Java classes and methods needed for making various JNI API calls */
typedef struct {
  jclass jecpt_cls;   /* java/lang/Exception class                */
  jclass jlong_cls;   /* java/lang/Long class                     */
  jclass jstring_cls; /* java/lang/String class                   */
  jclass jomt_cls;    /* com/ibm/onnxmlir/OMTensor class          */
  jclass jomtl_cls;   /* com/ibm/onnxmlir/OMTensorList class      */

  jmethodID jomt_constructor;   /* OMTensor constructor           */
  jmethodID jomt_getData;       /* OMTensor getData method        */
  jmethodID jomt_setData;       /* OMTensor setData method        */
  jmethodID jomt_getShape;      /* OMTensor getShape method       */
  jmethodID jomt_setShape;      /* OMTensor setShape method       */
  jmethodID jomt_getStrides;    /* OMTensor getStrides method     */
  jmethodID jomt_setStrides;    /* OMTensor setStrides method     */
  jmethodID jomt_getDataType;   /* OMTensor getType method        */
  jmethodID jomt_setDataType;   /* OMTensor setType method        */
  jmethodID jomt_getBufferSize; /* OMTensor getBufferSize method  */
  jmethodID jomt_getRank;       /* OMTensor getRank method        */
  jmethodID jomt_getNumElems;   /* OMTensor getNumOfElems method  */

  jmethodID jomtl_constructor; /* OMTensorList constructor        */
  jmethodID jomtl_getOmtArray; /* OMTensorList getOmtArray method */
} jniapi_t;

/* z/OS JNI expects ASCII but xlc encodes in EBCDIC so
 * convert string literals for JNI calls to ASCII on z/OS.
 */
#ifdef __MVS__
#pragma convert("ISO8859-1")
#endif
const char *jnistr[] = {
    "java/lang/Exception",             /* 0  CLS_JAVA_LANG_EXCEPTION          */
    "java/lang/Long",                  /* 1  CLS_JAVA_LANG_LONG               */
    "java/lang/String",                /* 2  CLS_JAVA_LANG_STRING             */
    "com/ibm/onnxmlir/OMTensor",       /* 3  CLS_COM_IBM_ONNXMLIR_OMTENSOR    */
    "com/ibm/onnxmlir/OMTensorList",   /* 4  CLS_COM_IBM_ONNXMLIR_OMTENSORLIST*/
    "<init>",                          /* 5  CTOR_INIT                        */
    "(Ljava/nio/ByteBuffer;[J[JI)V",   /* 6  CTOR_OMTENSOR                    */
    "([Lcom/ibm/onnxmlir/OMTensor;)V", /* 7  CTOR_OMTENSORLIST */
    "()Ljava/nio/ByteBuffer;",         /* 8  SIG_GET_DATA                     */
    "(Ljava/nio/ByteBuffer;)V",        /* 9  SIG_SET_DATA                     */
    "()[Lcom/ibm/onnxmlir/OMTensor;",  /* 10 SIG_GET_OMT_ARRAY                */
    "()[J",                            /* 11 SIG_GET_SHAPE, SIG_GET_STRIDES   */
    "([J)V",                           /* 12 SIG_SET_SHAPE, SIG_SET_STRIDES   */
    "()I",                             /* 13 SIG_GET_DATA_TYPE                */
    "(I)V",                            /* 14 SIG_SET_DATA_TYPE                */
    "()J",                             /* 15 SIG_GET_BUFFER_SIZE, SIG_GET_RANK,
                                             SIG_GET_NUM_ELEMS                */
    "getData",                         /* 16 FUNC_GET_DATA                    */
    "setData",                         /* 17 FUNC_SET_DATA                    */
    "getShape",                        /* 18 FUNC_GET_SHAPE                   */
    "setShape",                        /* 19 FUNC_SET_SHAPE                   */
    "getStrides",                      /* 20 FUNC_GET_STRIDES                 */
    "setStrides",                      /* 21 FUNC_SET_STRIDES                 */
    "getDataType",                     /* 22 FUNC_GET_DATA_TYPE               */
    "setDataType",                     /* 23 FUNC_SET_DATA_TYPE               */
    "getBufferSize",                   /* 24 FUNC_GET_BUFFER_SIZE             */
    "getRank",                         /* 25 FUNC_GET_RANK                    */
    "getNumElems",                     /* 26 FUNC_GET_NUM_ELEMS               */
    "getOmtArray",                     /* 27 FUNC_GET_OMT_ARRAY               */
    "JNI call error",                  /* 28 MSG_JNI_CALL_ERROR               */
    "native code error",               /* 29 MSG_NATIVE_CODE_ERROR            */
};
#ifdef __MVS__
#pragma convert(pop)
#endif

/* clang-format off */
enum {
  CLS_JAVA_LANG_EXCEPTION           = 0,  /* java/lang/Exception             */
  CLS_JAVA_LANG_LONG                = 1,  /* java/lang/Long                  */
  CLS_JAVA_LANG_STRING              = 2,  /* java/lang/String                */
  CLS_COM_IBM_ONNXMLIR_OMTENSOR     = 3,  /* com/ibm/onnxmlir/OMTensor       */
  CLS_COM_IBM_ONNXMLIR_OMTENSORLIST = 4,  /* com/ibm/onnxmlir/OMTensorList   */
  CTOR_INIT                         = 5,  /* <init>                          */
  CTOR_OMTENSOR                     = 6,  /* (Ljava/nio/ByteBuffer;[J[JI)V   */
  CTOR_OMTENSORLIST                 = 7,  /* ([Lcom/ibm/onnxmlir/OMTensor;)V */
  SIG_GET_DATA                      = 8,  /* ()Ljava/nio/ByteBuffer;         */
  SIG_SET_DATA                      = 9,  /* (Ljava/nio/ByteBuffer;)V        */
  SIG_GET_OMT_ARRAY                 = 10, /* ()[Lcom/ibm/onnxmlir/OMTensor;  */
  SIG_GET_SHAPE                     = 11, /* ()[J                            */
  SIG_GET_STRIDES                   = 11, /* ()[J                            */
  SIG_SET_SHAPE                     = 12, /* ([J)V                           */
  SIG_SET_STRIDES                   = 12, /* ([J)V                           */
  SIG_GET_DATA_TYPE                 = 13, /* ()I                             */
  SIG_SET_DATA_TYPE                 = 14, /* (I)V                            */
  SIG_GET_BUFFER_SIZE               = 15, /* ()J                             */
  SIG_GET_RANK                      = 15, /* ()J                             */
  SIG_GET_NUM_ELEMS                 = 15, /* ()J                             */
  FUNC_GET_DATA                     = 16, /* getData                         */
  FUNC_SET_DATA                     = 17, /* setData                         */
  FUNC_GET_SHAPE                    = 18, /* getShape                        */
  FUNC_SET_SHAPE                    = 19, /* setShape                        */
  FUNC_GET_STRIDES                  = 20, /* getStrides                      */
  FUNC_SET_STRIDES                  = 21, /* setStrides                      */
  FUNC_GET_DATA_TYPE                = 22, /* getDataType                     */
  FUNC_SET_DATA_TYPE                = 23, /* setDataType                     */
  FUNC_GET_BUFFER_SIZE              = 24, /* getBufferSize                   */
  FUNC_GET_RANK                     = 25, /* getRank                         */
  FUNC_GET_NUM_ELEMS                = 26, /* getNumElems                     */
  FUNC_GET_OMT_ARRAY                = 27, /* getOmtArray                     */
  MSG_JNI_CALL_ERROR                = 28, /* JNI call error                  */
  MSG_NATIVE_CODE_ERROR             = 29, /* native code error               */
};
/* clang-format on */

/* Find and initialize Java method IDs in struct jniapi */
jniapi_t *fill_jniapi(JNIEnv *env, jniapi_t *japi) {
  /* Get Java Exception, Long, String, OMTensor, and OMTensorList classes
   */
  assert(env);
  JNI_VAR_CALL(env, japi->jecpt_cls,
      (*env)->FindClass(env, jnistr[CLS_JAVA_LANG_EXCEPTION]),
      japi->jecpt_cls != NULL, NULL, "Class java/lang/Exception not found");
  JNI_VAR_CALL(env, japi->jlong_cls,
      (*env)->FindClass(env, jnistr[CLS_JAVA_LANG_LONG]),
      japi->jlong_cls != NULL, japi->jecpt_cls,
      "Class java/lang/Long not found");
  JNI_VAR_CALL(env, japi->jstring_cls,
      (*env)->FindClass(env, jnistr[CLS_JAVA_LANG_STRING]),
      japi->jstring_cls != NULL, japi->jecpt_cls,
      "Class java/lang/String not found");
  JNI_VAR_CALL(env, japi->jomt_cls,
      (*env)->FindClass(env, jnistr[CLS_COM_IBM_ONNXMLIR_OMTENSOR]),
      japi->jomt_cls != NULL, japi->jecpt_cls,
      "Class com/ibm/onnxmlir/OMTensor not found");
  JNI_VAR_CALL(env, japi->jomtl_cls,
      (*env)->FindClass(env, jnistr[CLS_COM_IBM_ONNXMLIR_OMTENSORLIST]),
      japi->jomtl_cls != NULL, japi->jecpt_cls,
      "Class com/ibm/onnxmlir/OMTensorList not found");

  /* Get method ID of constructor and various methods in OMTensor */
  JNI_VAR_CALL(env, japi->jomt_constructor,
      (*env)->GetMethodID(
          env, japi->jomt_cls, jnistr[CTOR_INIT], jnistr[CTOR_OMTENSOR]),
      japi->jomt_constructor != NULL, japi->jecpt_cls,
      "Method OMTensor.<init> not found");
  JNI_VAR_CALL(env, japi->jomt_getData,
      (*env)->GetMethodID(
          env, japi->jomt_cls, jnistr[FUNC_GET_DATA], jnistr[SIG_GET_DATA]),
      japi->jomt_getData != NULL, japi->jecpt_cls,
      "Method OMTensor.getData not found");
  JNI_VAR_CALL(env, japi->jomt_setData,
      (*env)->GetMethodID(
          env, japi->jomt_cls, jnistr[FUNC_SET_DATA], jnistr[SIG_SET_DATA]),
      japi->jomt_setData != NULL, japi->jecpt_cls,
      "Method OMTensor.setData not found");
  JNI_VAR_CALL(env, japi->jomt_getShape,
      (*env)->GetMethodID(
          env, japi->jomt_cls, jnistr[FUNC_GET_SHAPE], jnistr[SIG_GET_SHAPE]),
      japi->jomt_getShape != NULL, japi->jecpt_cls,
      "Method OMTensor.getShape not found");
  JNI_VAR_CALL(env, japi->jomt_setShape,
      (*env)->GetMethodID(
          env, japi->jomt_cls, jnistr[FUNC_SET_SHAPE], jnistr[SIG_SET_SHAPE]),
      japi->jomt_setShape != NULL, japi->jecpt_cls,
      "Method OMTensor.setShape not found");
  JNI_VAR_CALL(env, japi->jomt_getStrides,
      (*env)->GetMethodID(env, japi->jomt_cls, jnistr[FUNC_GET_STRIDES],
          jnistr[SIG_GET_STRIDES]),
      japi->jomt_getStrides != NULL, japi->jecpt_cls,
      "Method OMTensor.getStrides not found");
  JNI_VAR_CALL(env, japi->jomt_setStrides,
      (*env)->GetMethodID(env, japi->jomt_cls, jnistr[FUNC_SET_STRIDES],
          jnistr[SIG_SET_STRIDES]),
      japi->jomt_setStrides != NULL, japi->jecpt_cls,
      "Method OMTensor.setStrides not found");
  JNI_VAR_CALL(env, japi->jomt_getDataType,
      (*env)->GetMethodID(env, japi->jomt_cls, jnistr[FUNC_GET_DATA_TYPE],
          jnistr[SIG_GET_DATA_TYPE]),
      japi->jomt_getDataType != NULL, japi->jecpt_cls,
      "Method OMTensor.getDataType not found");
  JNI_VAR_CALL(env, japi->jomt_setDataType,
      (*env)->GetMethodID(env, japi->jomt_cls, jnistr[FUNC_SET_DATA_TYPE],
          jnistr[SIG_SET_DATA_TYPE]),
      japi->jomt_setDataType != NULL, japi->jecpt_cls,
      "Method OMTensor.setDataType not found");
  JNI_VAR_CALL(env, japi->jomt_getBufferSize,
      (*env)->GetMethodID(env, japi->jomt_cls, jnistr[FUNC_GET_BUFFER_SIZE],
          jnistr[SIG_GET_BUFFER_SIZE]),
      japi->jomt_getBufferSize != NULL, japi->jecpt_cls,
      "Method OMTensor.getBufferSize not found");
  JNI_VAR_CALL(env, japi->jomt_getRank,
      (*env)->GetMethodID(
          env, japi->jomt_cls, jnistr[FUNC_GET_RANK], jnistr[SIG_GET_RANK]),
      japi->jomt_getRank != NULL, japi->jecpt_cls,
      "Method OMTensor.getRank not found");
  JNI_VAR_CALL(env, japi->jomt_getNumElems,
      (*env)->GetMethodID(env, japi->jomt_cls, jnistr[FUNC_GET_NUM_ELEMS],
          jnistr[SIG_GET_NUM_ELEMS]),
      japi->jomt_getNumElems != NULL, japi->jecpt_cls,
      "Method OMTensor.getNumElems not found");

  /* Get method ID of constructor and various methods in OMTensorList */
  JNI_VAR_CALL(env, japi->jomtl_constructor,
      (*env)->GetMethodID(
          env, japi->jomtl_cls, jnistr[CTOR_INIT], jnistr[CTOR_OMTENSORLIST]),
      japi->jomtl_constructor != NULL, japi->jecpt_cls,
      "Method OMTensorList.<init> not found");
  JNI_VAR_CALL(env, japi->jomtl_getOmtArray,
      (*env)->GetMethodID(env, japi->jomtl_cls, jnistr[FUNC_GET_OMT_ARRAY],
          jnistr[SIG_GET_OMT_ARRAY]),
      japi->jomtl_getOmtArray != NULL, japi->jecpt_cls,
      "Method OMTensorList.getOmtArray not found");

  return japi;
}

/* Convert Java object to native data structure
 *
 *          +---------------------------+
 *          | TensorList                |
 *          |       +-----------------+ | (constructed by user)
 *          | _omts |   | o | ... |   | |
 *          |       +-----|-----------+ |
 *          +-------------|-------------+
 *                        v
 *                        +--------+
 *                  +---> | Tensor |
 *                  |     |        | (constructed by user)
 *                  |     | _data  |
 * Java world       |     +---|----+
 * -----------------|---------|------------------------------------------
 * Native world     |         v
 *                  |         +--------------------+ (constructed by
 *    +-------------+         | direct byte buffer |  user)
 *    |                       +--------------------+
 *   +|----------+            ^ ownership false, owned/freed by Java
 *   |o| | ... | |      +-----|---------+
 *   +-----------+      | _allocatedPtr | (constructed by jniwrapper)
 * jobj_omts, freed     |               |<---+
 * at the end of        | omTensor      |    |
 * omtl_java_to_native  +---------------+  freed by
 *                      ^                  omTensorListDestroy(jni_iomtl)
 *        +-------------|---------------+  at the end of
 *        |       +-----|-----------+   |  ..._main_1graph_1jni
 *        | _omts |   | o | ... |   |   |    |
 *        |       +-----------------+   |<---+
 *        | omTensorList                | (constructed by jniwrapper)
 *        +-----------------------------+
 */
OMTensorList *omtl_java_to_native(
    JNIEnv *env, jclass cls, jobject java_omtl, jniapi_t *japi) {

  /* Get OMTensor array Java object in OMTensorList */
  JNI_TYPE_VAR_CALL(env, jobjectArray, jomtl_omts,
      (*env)->CallObjectMethod(env, java_omtl, japi->jomtl_getOmtArray),
      jomtl_omts != NULL, japi->jecpt_cls, "jomtl_omts=%p", jomtl_omts);

  /* Get the number of OMTensors in the array */
  JNI_TYPE_VAR_CALL(env, jlong, jomtl_omtn,
      (*env)->GetArrayLength(env, jomtl_omts),
      jomtl_omtn >= 0 && jomtl_omtn <= INT_MAX, japi->jecpt_cls,
      "jomtl_omtn=%ld", jomtl_omtn);

  /* Allocate memory for holding each Java omt object and OMTensor pointers
   * for constructing native OMTensor array
   *
   * jobj_omts are the pointers to the Java OMTensor objects. They are used
   * to make JNI calls on the OMTensors to retrieve their internal fields
   * such as data, shape, strides, etc.
   *
   * jni_omts are the pointers to the native OMTensor structs we construct,
   * filled in with fields we retrieved from the corresponding Java OMTensor
   * objects, and then given to the native OMTensorList struct.
   */
  LIB_TYPE_VAR_CALL(jobject *, jobj_omts, malloc(jomtl_omtn * sizeof(jobject)),
      jobj_omts != NULL, env, japi->jecpt_cls, "jobj_omts=%p", jobj_omts);
  LIB_TYPE_VAR_CALL(OMTensor **, jni_omts,
      malloc(jomtl_omtn * sizeof(OMTensor *)), jni_omts != NULL, env,
      japi->jecpt_cls, "jni_omts=%p", jni_omts);

  /* Loop through all the jomtl_omts  */
  for (int i = 0; i < jomtl_omtn; i++) {
    JNI_VAR_CALL(env, jobj_omts[i],
        (*env)->GetObjectArrayElement(env, jomtl_omts, i), jobj_omts[i] != NULL,
        japi->jecpt_cls, "jobj_omts[%d]=%p", i, jobj_omts[i]);

    /* Get data, shape, strides, dataType, rank, and bufferSize by calling
     * corresponding methods
     */
    JNI_TYPE_VAR_CALL(env, jobject, jomt_data,
        (*env)->CallObjectMethod(env, jobj_omts[i], japi->jomt_getData),
        jomt_data != NULL, japi->jecpt_cls, "omt[%d]:data=%p", i, jomt_data);
    JNI_TYPE_VAR_CALL(env, jobject, jomt_shape,
        (*env)->CallObjectMethod(env, jobj_omts[i], japi->jomt_getShape),
        jomt_shape != NULL, japi->jecpt_cls, "omt[%d]:shape=%p", i, jomt_shape);
    JNI_TYPE_VAR_CALL(env, jobject, jomt_strides,
        (*env)->CallObjectMethod(env, jobj_omts[i], japi->jomt_getStrides),
        jomt_strides != NULL, japi->jecpt_cls, "omt[%d]:strides=%p", i,
        jomt_strides);
    JNI_TYPE_VAR_CALL(env, jint, jomt_dataType,
        (*env)->CallIntMethod(env, jobj_omts[i], japi->jomt_getDataType),
        jomt_dataType != ONNX_TYPE_UNDEFINED, japi->jecpt_cls,
        "omt[%d]:dataType=%d", i, jomt_dataType);
    JNI_TYPE_VAR_CALL(env, jlong, jomt_bufferSize,
        (*env)->CallLongMethod(env, jobj_omts[i], japi->jomt_getBufferSize),
        jomt_bufferSize >= 0, japi->jecpt_cls, "omt[%d]:bufferSize=%ld", i,
        jomt_bufferSize);
    JNI_TYPE_VAR_CALL(env, jlong, jomt_rank,
        (*env)->CallLongMethod(env, jobj_omts[i], japi->jomt_getRank),
        jomt_rank >= 0, japi->jecpt_cls, "omt[%d]:rank=%ld", i, jomt_rank);
    JNI_TYPE_VAR_CALL(env, jlong, jomt_numElems,
        (*env)->CallLongMethod(env, jobj_omts[i], japi->jomt_getNumElems),
        jomt_numElems >= 0, japi->jecpt_cls, "omt[%d]:numElems=%ld", i,
        jomt_numElems);

    /* Get direct buffer associated with data */
    JNI_TYPE_VAR_CALL(env, void *, jni_data,
        (*env)->GetDirectBufferAddress(env, jomt_data), jni_data != NULL,
        japi->jecpt_cls, "omt[%d]:jni_data=%p", i, jni_data);

    /* Get long array associated with data shape and strides */
    JNI_TYPE_VAR_CALL(env, jlong *, jni_shape,
        (*env)->GetLongArrayElements(env, jomt_shape, NULL), jni_shape != NULL,
        japi->jecpt_cls, "omt[%d]:jni_shape=%p", i, jni_shape);
    JNI_TYPE_VAR_CALL(env, jlong *, jni_strides,
        (*env)->GetLongArrayElements(env, jomt_strides, NULL),
        jni_strides != NULL, japi->jecpt_cls, "omt[%d]:jni_strides=%p", i,
        jni_strides);

    /* Primitive type int and long can be directly used */
    int jni_dataType = jomt_dataType;
    int64_t jni_bufferSize = jomt_bufferSize;
    int64_t jni_rank = jomt_rank;
    int64_t jni_numElems = jomt_numElems;

    /* Print debug info on what we got from the Java side */
    OMT_DEBUG(i, jni_numElems, jni_data, jni_shape, jni_strides, jni_dataType,
        jni_bufferSize, jni_rank, 0);

    /* Create native OMTensor struct. Note jni_data is owned by the
     * Java ByteBuffer object. So here the OMTensor is created with
     * owning=false. Therefore, later when we call omTensorListDestroy
     * the data buffer will not be freed. Java GC is responsible for
     * freeing the data buffer when it garbage collects the ByteBuffer
     * in the OMTensor.
     */
    LIB_VAR_CALL(jni_omts[i],
        omTensorCreate(jni_data, (int64_t *)jni_shape, jni_rank, jni_dataType),
        jni_omts[i] != NULL, env, japi->jecpt_cls, "omt[%d]:jni_omts=%p", i,
        jni_omts[i]);

    /* Release reference to the shape and strides Java objects */
    JNI_CALL(env,
        (*env)->ReleaseLongArrayElements(env, jomt_shape, jni_shape, 0), 1,
        NULL, "");
    JNI_CALL(env,
        (*env)->ReleaseLongArrayElements(env, jomt_strides, jni_strides, 0), 1,
        NULL, "");
  }

  /* We have constructed the native OMTensor structs so the pointers
   * to the corresponding Java OMTensor objects used for retrieving
   * their internal fields are no longer needed.
   */
  free(jobj_omts);

  /* Create OMTensorList to be constructed and passed to the model
   * shared library.
   */
  LIB_TYPE_VAR_CALL(OMTensorList *, jni_omtl,
      omTensorListCreate(jni_omts, (int64_t)jomtl_omtn), jni_omtl != NULL, env,
      japi->jecpt_cls, "jni_omtl=%p", jni_omtl);

  /* omTensorListCreate makes a copy of the tensor list, so we must free the
     list that we allocateed. */
  free(jni_omts);

  return jni_omtl;
}

/* Convert native data structure to Java object
 *
 *          +---------------------------+
 *          | TensorList                |
 *          |       +-----------------+ | (constructed by jniwrapper)
 *          | _omts |   | o | ... |   | |
 *          |       +-----|-----------+ |
 *          +-------------|-------------+
 *                        v
 *                        +--------+
 *                        | Tensor |
 *                        |        | (constructed by jniwrapper)
 *                        | _data  |
 * Java world             +---|----+
 * ---------------------------|------------------------------------------
 * Native world               v
 *                            +--------------------+ (constructed by
 *                            | native buffer      |  model runtime)
 *                            +--------------------+
 *                            ^ ownership true -> false, xferred to Java
 *                      +-----|---------+
 *                      | _allocatedPtr | (constructed by model runtime)
 *                      |               |<---+
 *                      | omTensor      |    |
 *                      +---------------+  freed by
 *                      ^                  omTensorListDestroy(jni_oomtl)
 *        +-------------|---------------+  at the end of
 *        |       +-----|-----------+   |  ..._main_1graph_1jni
 *        | _omts |   | o | ... |   |   |    |
 *        |       +-----------------+   |<---+
 *        | omTensorList                | (constructed by model runtime)
 *        +-----------------------------+
 */
jobject omtl_native_to_java(
    JNIEnv *env, jclass cls, OMTensorList *jni_omtl, jniapi_t *japi) {

  /* Get the OMTensor array in the OMTensorList */
  LIB_TYPE_VAR_CALL(OMTensor **, jni_omts, omTensorListGetOmtArray(jni_omtl),
      jni_omts != NULL, env, japi->jecpt_cls, "jni_omts=%p", jni_omts);

  /* Get the number of OMTensors in the OMTensorList */
  LIB_TYPE_VAR_CALL(int64_t, jni_omtn, omTensorListGetSize(jni_omtl),
      jni_omtn > 0 && jni_omtn <= INT_MAX, env, japi->jecpt_cls, "jni_omtn=%ld",
      jni_omtn);

  /* Create OMTensor java object array */
  JNI_TYPE_VAR_CALL(env, jobjectArray, jobj_omts,
      (*env)->NewObjectArray(env, jni_omtn, japi->jomt_cls, NULL),
      jobj_omts != NULL, japi->jecpt_cls, "jobj_omts=%p", jobj_omts);

  /* Loop through the native OMTensor structs */
  for (int i = 0; i < jni_omtn; i++) {

    LIB_TYPE_VAR_CALL(void *, jni_data, omTensorGetDataPtr(jni_omts[i]),
        jni_data != NULL, env, japi->jecpt_cls, "omt[%d]:data=%p", i, jni_data);
    LIB_TYPE_VAR_CALL(const int64_t *, jni_shape, omTensorGetShape(jni_omts[i]),
        jni_shape != NULL, env, japi->jecpt_cls, "omt[%d]:shape=%p", i,
        jni_shape);
    LIB_TYPE_VAR_CALL(const int64_t *, jni_strides,
        omTensorGetStrides(jni_omts[i]), jni_strides != NULL, env,
        japi->jecpt_cls, "omt[%d]:strides=%p", i, jni_strides);
    LIB_TYPE_VAR_CALL(OM_DATA_TYPE, jni_dataType,
        omTensorGetDataType(jni_omts[i]), jni_dataType != ONNX_TYPE_UNDEFINED,
        env, japi->jecpt_cls, "omt[%d]:dataType=%d", i, jni_dataType);
    LIB_TYPE_VAR_CALL(int64_t, jni_bufferSize,
        omTensorGetBufferSize(jni_omts[i]), jni_bufferSize > 0, env,
        japi->jecpt_cls, "omt[%ld]:bufferSize=%ld", i, jni_bufferSize);
    LIB_TYPE_VAR_CALL(int64_t, jni_rank, omTensorGetRank(jni_omts[i]),
        jni_rank >= 0, env, japi->jecpt_cls, "omt[%d]:rank=%ld", i, jni_rank);
    LIB_TYPE_VAR_CALL(int64_t, jni_owning, omTensorGetOwning(jni_omts[i]),
        jni_owning == 0 || jni_owning == 1, env, japi->jecpt_cls,
        "omt[%d]:owning=ld", i, jni_owning);
    LIB_TYPE_VAR_CALL(int64_t, jni_numElems, omTensorGetNumElems(jni_omts[i]),
        jni_numElems > 0, env, japi->jecpt_cls, "omt[%d]:numElems=%ld", i,
        jni_numElems);

    /* Print debug info on what we got from the native side */
    OMT_DEBUG(i, jni_numElems, jni_data, jni_shape, jni_strides, jni_dataType,
        jni_bufferSize, jni_rank, jni_owning);

    /* Primitive type int can be directly used */
    jint jomt_dataType = jni_dataType;
    jlong jomt_bufferSize = jni_bufferSize;
    jint jomt_rank = jni_rank;
    /*jlong jomt_numElems = jni_numElems;*/

    /* Create direct byte buffer Java object from native data buffer.
     *
     * When the Java OMTensor object is created, it will create a copy of the
     * tensor data to ensure Java ownership of the underlying tensor data.
     * Creation of the DirectByteBuffer here is merely a passthrough mechanism
     * to allow for the copy to occur.
     *
     * If jni_owning is false, it means the data buffer is not freeable
     * due to one of the two following cases:
     *
     *   - user has malloc-ed the data buffer so the user is
     *     responsible for freeing it
     *   - the data buffer is static
     */
    void *jbytebuffer_data = jni_data;
    JNI_TYPE_VAR_CALL(env, jobject, jomt_data,
        (*env)->NewDirectByteBuffer(env, jbytebuffer_data, jomt_bufferSize),
        jomt_data != NULL, japi->jecpt_cls, "omt[%d]:jomt_data=%p", i,
        jomt_data);

    /* Create data shape array Java object, fill in from native array */
    JNI_TYPE_VAR_CALL(env, jlongArray, jomt_shape,
        (*env)->NewLongArray(env, jomt_rank), jomt_shape != NULL,
        japi->jecpt_cls, "omt[%d]:jomt_shape=%p", i, jomt_shape);
    /* On z/OS, remove const specification for jni_shape */
#ifdef __MVS__
    JNI_CALL(env,
        (*env)->SetLongArrayRegion(
            env, jomt_shape, 0, jomt_rank, (jlong *)jni_shape),
        1, NULL, "");
#else
    JNI_CALL(env,
        (*env)->SetLongArrayRegion(
            env, jomt_shape, 0, jomt_rank, (const jlong *)jni_shape),
        1, NULL, "");
#endif

    /* Create data strides array Java object, fill in from native array */
    JNI_TYPE_VAR_CALL(env, jlongArray, jomt_strides,
        (*env)->NewLongArray(env, jomt_rank), jomt_strides != NULL,
        japi->jecpt_cls, "omt[%d]:jomt_strides=%p", i, jomt_strides);
    /* On z/OS, remove const specification for jni_strides */
#ifdef __MVS__
    JNI_CALL(env,
        (*env)->SetLongArrayRegion(
            env, jomt_strides, 0, jomt_rank, (jlong *)jni_strides),
        1, NULL, "");
#else
    JNI_CALL(env,
        (*env)->SetLongArrayRegion(
            env, jomt_strides, 0, jomt_rank, (const jlong *)jni_strides),
        1, NULL, "");
#endif

    /* Create the OMTensor Java object */
    JNI_TYPE_VAR_CALL(env, jobject, jobj_omt,
        (*env)->NewObject(env, japi->jomt_cls, japi->jomt_constructor,
            jomt_data, jomt_shape, jomt_strides, jomt_dataType),
        jobj_omt != NULL, japi->jecpt_cls, "omt[%d]:jobj_omt=%p", i, jobj_omt);

    /* Set the OMTensor object in the object array */
    JNI_CALL(env, (*env)->SetObjectArrayElement(env, jobj_omts, i, jobj_omt), 1,
        NULL, "");
  }

  /* Create the OMTensorList java object */
  JNI_TYPE_VAR_CALL(env, jobject, java_omtl,
      (*env)->NewObject(
          env, japi->jomtl_cls, japi->jomtl_constructor, jobj_omts),
      java_omtl != NULL, japi->jecpt_cls, "java_omtl=%p", java_omtl);

  return java_omtl;
}

JNIEXPORT jobject JNICALL Java_com_ibm_onnxmlir_OMModel_main_1graph_1jni(
    JNIEnv *env, jclass cls, jobject java_iomtl) {

  /* Apparently J9 cannot have the return pointer of FindClass shared
   * across threads. So move jniapi into stack so each thread has its
   * own copy.
   */
  jniapi_t jniapi;

  log_init();

  /* Find and initialize Java method IDs in struct jniapi */
  CHECK_CALL(jniapi_t *, japi, fill_jniapi(env, &jniapi), japi != NULL,
      "japi=%p", japi);

  /* Convert Java object to native data structure */
  CHECK_CALL(OMTensorList *, jni_iomtl,
      omtl_java_to_native(env, cls, java_iomtl, japi), jni_iomtl != NULL,
      "jni_iomtl=%p", jni_iomtl);

  /* Call model inference entry point */
  CHECK_CALL(OMTensorList *, jni_oomtl, run_main_graph(jni_iomtl),
      jni_oomtl != NULL, "jni_oomtl=%p", jni_oomtl);

  /* Convert native data structure to Java object */
  CHECK_CALL(jobject, java_oomtl,
      omtl_native_to_java(env, cls, jni_oomtl, japi), java_oomtl != NULL,
      "java_oomtl=%p", java_oomtl);

  /* Free intermediate data structures and return Java object */
  omTensorListDestroy(jni_iomtl);
  omTensorListDestroy(jni_oomtl);
  return java_oomtl;
}

#ifdef __MVS__
/* On z/OS, we convert entry point name in ASCII into EBCDIC for
 * the omInputSignature/omOutputSignaturee function using __a2e_s.
 * However, __a2e_s converts in place so we need to make a copy of
 * the entry point name.
 *
 * On z/OS, we convert signature in EBCDIC into ASCII for the JNI
 * NewStringUTF function using __e2a_s. However, __e2a_s converts
 * in place so we need to make a copy of the signature.
 */
#define __a2e(s) conv(s, __a2e_s)
#define __e2a(s) conv(s, __e2a_s)
static inline char *conv(const char *str, size_t (*fptr)(char *)) {
  char *dup = strdup(str);
  if (dup == NULL)
    return NULL;

  /* Convert ASCII to EBCDIC in place */
  if (fptr(dup) < 0) {
    free(dup);
    return NULL;
  }

  return dup;
}
#endif

JNIEXPORT jobjectArray JNICALL
Java_com_ibm_onnxmlir_OMModel_query_1entry_1points(JNIEnv *env, jclass cls) {

  log_init();

  /* Find and initialize Java Exception and String class */
  JNI_TYPE_VAR_CALL(env, jclass, jecpt_cls,
      (*env)->FindClass(env, jnistr[CLS_JAVA_LANG_EXCEPTION]),
      jecpt_cls != NULL, NULL, "Class java/lang/Exception not found");
  JNI_TYPE_VAR_CALL(env, jclass, jstring_cls,
      (*env)->FindClass(env, jnistr[CLS_JAVA_LANG_STRING]), jstring_cls != NULL,
      jecpt_cls, "Class java/lang/String not found");

  /* Call query entry points API */
  int64_t neps;
  const char *const *jni_eps = omQueryEntryPoints(&neps);

  /* Create entry points String array Java object */
  JNI_TYPE_VAR_CALL(env, jobjectArray, java_eps,
      (*env)->NewObjectArray(env, neps, jstring_cls, NULL), java_eps != NULL,
      jecpt_cls, "java_eps=%p", java_eps);

  /* Go through the entry points, convert them into Java String and
   * put them into the String array.
   */
  for (int64_t i = 0; i < neps; i++) {
    char ep[32];
    int num_chars_written = sprintf(ep, "ep[%lld]", (long long)i);
    assert(num_chars_written >= 0 && "sprintf write error to ep");
    HEX_DEBUG(ep, jni_eps[i], strlen(jni_eps[i]));
    LOG_PRINTF(LOG_DEBUG, "ep[%d](%ld):%s", i, strlen(jni_eps[i]), jni_eps[i]);

    /* On z/OS, convert entry point name in EBCDIC to ASCII */
#ifdef __MVS__
    CHECK_CALL(
        char *, epptr, __e2a(jni_eps[i]), epptr != NULL, "epptr=%p", epptr);
#else
    const char *epptr = jni_eps[i];
#endif

    /* Convert to Java String object */
    JNI_TYPE_VAR_CALL(env, jstring, jstr_ep, (*env)->NewStringUTF(env, epptr),
        jstr_ep != NULL, jecpt_cls, "jstr_ep=%p", jstr_ep);

    /* On z/OS, free the ASCII entry point name no longer needed */
#ifdef __MVS__
    free(epptr);
#endif

    /* Put Java String object into the String array */
    JNI_CALL(env, (*env)->SetObjectArrayElement(env, java_eps, i, jstr_ep), 1,
        NULL, "");
  }

  return java_eps;
}

JNIEXPORT jstring JNICALL Java_com_ibm_onnxmlir_OMModel_input_1signature_1jni(
    JNIEnv *env, jclass cls, jstring ep) {

  log_init();

  /* Find and initialize Java Exception class */
  JNI_TYPE_VAR_CALL(env, jclass, jecpt_cls,
      (*env)->FindClass(env, jnistr[CLS_JAVA_LANG_EXCEPTION]),
      jecpt_cls != NULL, NULL, "Class java/lang/Exception not found");

  /* Get reference to the signature Java String object */
  JNI_TYPE_VAR_CALL(env, const char *, jni_ep,
      (*env)->GetStringUTFChars(env, ep, NULL), jni_ep != NULL, jecpt_cls,
      "jni_ep=%p", jni_ep);

  /* On z/OS, convert entry point name in UTF-8 to EBCDIC */
#ifdef __MVS__
  CHECK_CALL(char *, epptr, __a2e(jni_ep), epptr != NULL, "epptr=%p", epptr);
#else
  const char *epptr = jni_ep;
#endif

  /* Entry point name in hex and string format for debugging */
  HEX_DEBUG("ep", epptr, strlen(epptr));
  LOG_PRINTF(LOG_DEBUG, "ep(%d):%s", strlen(epptr), epptr);

  /* Call model input signature API */
  CHECK_CALL(const char *, jni_isig, omInputSignature(epptr), jni_isig != NULL,
      "jni_isig=%p", jni_isig);

  /* On z/OS, free the EBCDIC entry point name no longer needed */
#ifdef __MVS__
  free(epptr);
#endif

  /* Release reference to the signature Java String object */
  JNI_CALL(env, (*env)->ReleaseStringUTFChars(env, ep, jni_ep), 1, NULL, "");

  /* Output signature in hex and string format for debugging */
  HEX_DEBUG("isig", jni_isig, strlen(jni_isig));
  LOG_PRINTF(LOG_DEBUG, "isig(%d):%s", strlen(jni_isig), jni_isig);

  /* On z/OS, convert input signature in EBCDIC to ASCII */
#ifdef __MVS__
  CHECK_CALL(
      char *, sigptr, __e2a(jni_isig), sigptr != NULL, "sigptr=%p", sigptr);
#else
  const char *sigptr = jni_isig;
#endif

  /* Convert to Java String object */
  JNI_TYPE_VAR_CALL(env, jstring, java_isig, (*env)->NewStringUTF(env, sigptr),
      java_isig != NULL, jecpt_cls, "java_isig=%p", java_isig);

  /* On z/OS, free the ASCII input signature no longer needed */
#ifdef __MVS__
  free(sigptr);
#endif

  return java_isig;
}

JNIEXPORT jstring JNICALL Java_com_ibm_onnxmlir_OMModel_output_1signature_1jni(
    JNIEnv *env, jclass cls, jstring ep) {

  log_init();

  /* Find and initialize Java Exception class */
  JNI_TYPE_VAR_CALL(env, jclass, jecpt_cls,
      (*env)->FindClass(env, jnistr[CLS_JAVA_LANG_EXCEPTION]),
      jecpt_cls != NULL, NULL, "Class java/lang/Exception not found");

  /* Get reference to the signature Java String object */
  JNI_TYPE_VAR_CALL(env, const char *, jni_ep,
      (*env)->GetStringUTFChars(env, ep, NULL), jni_ep != NULL, jecpt_cls,
      "jni_ep=%p", jni_ep);

  /* On z/OS, convert entry point name in UTF-8 to EBCDIC */
#ifdef __MVS__
  CHECK_CALL(char *, epptr, __a2e(jni_ep), epptr != NULL, "epptr=%p", epptr);
#else
  const char *epptr = jni_ep;
#endif

  /* Entry point name in hex and string format for debugging */
  HEX_DEBUG("ep", epptr, strlen(epptr));
  LOG_PRINTF(LOG_DEBUG, "ep(%d):%s", strlen(epptr), epptr);

  /* Call model output signature API */
  CHECK_CALL(const char *, jni_osig, omOutputSignature(epptr), jni_osig != NULL,
      "jni_osig=%p", jni_osig);

  /* On z/OS, free the EBCDIC entry point name no longer needed */
#ifdef __MVS__
  free(epptr);
#endif

  /* Release reference to the signature Java String object */
  JNI_CALL(env, (*env)->ReleaseStringUTFChars(env, ep, jni_ep), 1, NULL, "");

  /* Output signature in hex and string format for debugging */
  HEX_DEBUG("osig", jni_osig, strlen(jni_osig));
  LOG_PRINTF(LOG_DEBUG, "osig(%d):%s", strlen(jni_osig), jni_osig);

  /* On z/OS, convert output signature in EBCDIC to ASCII */
#ifdef __MVS__
  CHECK_CALL(
      char *, sigptr, __e2a(jni_osig), sigptr != NULL, "sigptr=%p", sigptr);
#else
  const char *sigptr = jni_osig;
#endif

  /* Convert to Java String object */
  JNI_TYPE_VAR_CALL(env, jstring, java_osig, (*env)->NewStringUTF(env, sigptr),
      java_osig != NULL, jecpt_cls, "java_osig=%p", java_osig);

  /* On z/OS, free the ASCII output signature no longer needed */
#ifdef __MVS__
  free(sigptr);
#endif

  return java_osig;
}
