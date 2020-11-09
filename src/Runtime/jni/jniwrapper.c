#include <assert.h>
#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif
#include <string.h>

#include "OnnxMlirRuntime.h"
#include "com_ibm_onnxmlir_DynEntryPoint.h"
#include "jnilog.h"

/* Declare type var, make call and assign to var, check against val.
 * It's assumed that a Java exception has already been thrown so
 * this call simply returns NULL.
 */
#define CHECK_CALL(type, var, call, val)                                       \
  type var = call;                                                             \
  if (var == val)                                                              \
  return NULL

/* Make a JNI call,  log error and throw Java exception if the call failed */
#define JNI_CALL(env, stmt)                                                    \
  stmt;                                                                        \
  do {                                                                         \
    jthrowable e = (*env)->ExceptionOccurred(env);                             \
    if (e) {                                                                   \
      LOG_PRINTF(LOG_ERROR, "JNI call exception occurred");                    \
      (*env)->Throw(env, e);                                                   \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

/* Make a JNI call and assign return value to var,
 * log error and throw Java exception if the call failed
 */
#define JNI_VAR_CALL(env, var, call) JNI_CALL(env, var = call)

/* Declare type var, make a JNI call and assign return value to var,
 * log error and throw Java exception if the call failed
 */
#define JNI_TYPE_VAR_CALL(env, type, var, call) JNI_CALL(env, type var = call);

/* Make a native library call, if cond is true (native code failed),
 * log error and throw Java exception
 */
#define LIB_CALL(stmt, check, env, cls, ...)                                   \
  stmt;                                                                        \
  do {                                                                         \
    if (check) {                                                               \
      LOG_PRINTF(LOG_ERROR, __VA_ARGS__);                                      \
      (*env)->ThrowNew(env, cls, "native code error");                         \
      return NULL;                                                             \
    }                                                                          \
  } while (0)

/* Make a native library call and assign return value to var,
 * log error and throw Java exception if the call failed
 */
#define LIB_VAR_CALL(var, call, val, env, cls, ...)                            \
  LIB_CALL(var = call, var == val, env, cls, __VA_ARGS__);

/* Declare type var, make a native library call and assign return value to var,
 * log error and throw Java exception if the call failed
 */
#define LIB_TYPE_VAR_CALL(type, var, call, val, env, cls, ...)                 \
  LIB_CALL(type var = call, var == val, env, cls, __VA_ARGS__);

/* Debug output of OMTensor fields */
#define OMT_DEBUG(i, n, data, shape, stride, dataType, dataSize, rank, owning) \
  do {                                                                         \
    char tmp[1024];                                                            \
    LOG_TYPE_BUF(dataType, tmp, data, n);                                      \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:data=[%s]", i, tmp);                        \
    LOG_LONG_BUF(tmp, shape, rank);                                            \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:shape=[%s]", i, tmp);                       \
    LOG_LONG_BUF(tmp, stride, rank);                                           \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:stride=[%s]", i, tmp);                      \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:dataType=%d", i, dataType);                 \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:dataSize=%ld", i, dataSize);                \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:rank=%d", i, rank);                         \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:owning=%d", i, owning);                     \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:numElems=%ld", i, n);                       \
  } while (0)

/* Java classes and methods needed for making various JNI API calls */
typedef struct {
  jclass jecpt_cls;   /* java/lang/Exception class                */
  jclass jlong_cls;   /* java/lang/Long class                     */
  jclass jstring_cls; /* java/lang/String class                   */
  jclass jomt_cls;    /* com/ibm/onnxmlir/OMTensor class          */
  jclass jomtl_cls;   /* com/ibm/onnxmlir/OMTensorList class      */

  jmethodID jomt_constructor; /* OMTensor constructor             */
  jmethodID jomt_getData;     /* OMTensor getData method          */
  jmethodID jomt_setData;     /* OMTensor setData method          */
  jmethodID jomt_getShape;    /* OMTensor getShape method         */
  jmethodID jomt_setShape;    /* OMTensor setShape method         */
  jmethodID jomt_getStride;   /* OMTensor getStride method        */
  jmethodID jomt_setStride;   /* OMTensor setStride method        */
  jmethodID jomt_getDataType; /* OMTensor getType method          */
  jmethodID jomt_setDataType; /* OMTensor setType method          */
  jmethodID jomt_getDataSize; /* OMTensor getDataSize method      */
  jmethodID jomt_getRank;     /* OMTensor getRank method          */
  jmethodID jomt_getNumElems; /* OMTensor getNumOfElems method    */

  jmethodID jomtl_constructor; /* OMTensorList constructor        */
  jmethodID jomtl_getOmtArray; /* OMTensorList getOmtArray method */
} jniapi_t;

jniapi_t jniapi;

/* Find and initialize Java method IDs in struct jniapi */
jniapi_t *fill_jniapi(JNIEnv *env, jniapi_t *japi) {
  /* Get Java Exception, Long, String, OMTensor, and OMTensorList classes
   */
  JNI_VAR_CALL(
      env, japi->jecpt_cls, (*env)->FindClass(env, "java/lang/Exception"));
  JNI_VAR_CALL(env, japi->jlong_cls, (*env)->FindClass(env, "java/lang/Long"));
  JNI_VAR_CALL(
      env, japi->jstring_cls, (*env)->FindClass(env, "java/lang/String"));
  JNI_VAR_CALL(
      env, japi->jomt_cls, (*env)->FindClass(env, "com/ibm/onnxmlir/OMTensor"));
  JNI_VAR_CALL(env, japi->jomtl_cls,
      (*env)->FindClass(env, "com/ibm/onnxmlir/OMTensorList"));

  /* Get method ID of constructor and various methods in OMTensor */
  JNI_VAR_CALL(env, japi->jomt_constructor,
      (*env)->GetMethodID(
          env, japi->jomt_cls, "<init>", "(Ljava/nio/ByteBuffer;[J[JI)V"));
  JNI_VAR_CALL(env, japi->jomt_getData,
      (*env)->GetMethodID(
          env, japi->jomt_cls, "getData", "()Ljava/nio/ByteBuffer;"));
  JNI_VAR_CALL(env, japi->jomt_setData,
      (*env)->GetMethodID(
          env, japi->jomt_cls, "setData", "(Ljava/nio/ByteBuffer;)V"));
  JNI_VAR_CALL(env, japi->jomt_getShape,
      (*env)->GetMethodID(env, japi->jomt_cls, "getShape", "()[J"));
  JNI_VAR_CALL(env, japi->jomt_setShape,
      (*env)->GetMethodID(env, japi->jomt_cls, "setShape", "([J)V"));
  JNI_VAR_CALL(env, japi->jomt_getStride,
      (*env)->GetMethodID(env, japi->jomt_cls, "getStride", "()[J"));
  JNI_VAR_CALL(env, japi->jomt_setStride,
      (*env)->GetMethodID(env, japi->jomt_cls, "setStride", "([J)V"));
  JNI_VAR_CALL(env, japi->jomt_getDataType,
      (*env)->GetMethodID(env, japi->jomt_cls, "getDataType", "()I"));
  JNI_VAR_CALL(env, japi->jomt_setDataType,
      (*env)->GetMethodID(env, japi->jomt_cls, "setDataType", "(I)V"));
  JNI_VAR_CALL(env, japi->jomt_getDataSize,
      (*env)->GetMethodID(env, japi->jomt_cls, "getDataSize", "()J"));
  JNI_VAR_CALL(env, japi->jomt_getRank,
      (*env)->GetMethodID(env, japi->jomt_cls, "getRank", "()I"));
  JNI_VAR_CALL(env, japi->jomt_getNumElems,
      (*env)->GetMethodID(env, japi->jomt_cls, "getNumElems", "()J"));

  /* Get method ID of constructor and various methods in OMTensorList */
  JNI_VAR_CALL(env, japi->jomtl_constructor,
      (*env)->GetMethodID(
          env, japi->jomtl_cls, "<init>", "([Lcom/ibm/onnxmlir/OMTensor;)V"));
  JNI_VAR_CALL(env, japi->jomtl_getOmtArray,
      (*env)->GetMethodID(env, japi->jomtl_cls, "getOmtArray",
          "()[Lcom/ibm/onnxmlir/OMTensor;"));

  return japi;
}

/* Convert Java object to native data structure */
OMTensorList *omtl_java_to_native(
    JNIEnv *env, jclass cls, jobject java_omtl, jniapi_t *japi) {

  /* Get OMTensor array Java object in OMTensorList */
  JNI_TYPE_VAR_CALL(env, jobjectArray, jomtl_omts,
      (*env)->CallObjectMethod(env, java_omtl, japi->jomtl_getOmtArray));

  /* Get the number of OMTensors in the array */
  JNI_TYPE_VAR_CALL(
      env, jsize, jomtl_omtn, (*env)->GetArrayLength(env, jomtl_omts));

  /* Allocate memory for holding each Java omt object and OMTensor pointers
   * for constructing native OMTensor array
   */
  LIB_TYPE_VAR_CALL(jobject *, jobj_omts, malloc(jomtl_omtn * sizeof(jobject)),
      NULL, env, japi->jecpt_cls, "jobj_omts=null");
  LIB_TYPE_VAR_CALL(OMTensor **, jni_omts,
      malloc(jomtl_omtn * sizeof(OMTensor *)), NULL, env, japi->jecpt_cls,
      "jni_omts=null");

  /* Loop through all the jomtl_omts  */
  for (int i = 0; i < jomtl_omtn; i++) {
    JNI_VAR_CALL(
        env, jobj_omts[i], (*env)->GetObjectArrayElement(env, jomtl_omts, i));

    /* Get data, shape, stride, dataType, rank, and dataSize by calling
     * corresponding methods
     */
    JNI_TYPE_VAR_CALL(env, jobject, jomt_data,
        (*env)->CallObjectMethod(env, jobj_omts[i], japi->jomt_getData));
    JNI_TYPE_VAR_CALL(env, jobject, jomt_shape,
        (*env)->CallObjectMethod(env, jobj_omts[i], japi->jomt_getShape));
    JNI_TYPE_VAR_CALL(env, jobject, jomt_stride,
        (*env)->CallObjectMethod(env, jobj_omts[i], japi->jomt_getStride));
    JNI_TYPE_VAR_CALL(env, jint, jomt_dataType,
        (*env)->CallIntMethod(env, jobj_omts[i], japi->jomt_getDataType));
    JNI_TYPE_VAR_CALL(env, jlong, jomt_dataSize,
        (*env)->CallLongMethod(env, jobj_omts[i], japi->jomt_getDataSize));
    JNI_TYPE_VAR_CALL(env, jint, jomt_rank,
        (*env)->CallIntMethod(env, jobj_omts[i], japi->jomt_getRank));
    JNI_TYPE_VAR_CALL(env, jlong, jomt_numElems,
        (*env)->CallLongMethod(env, jobj_omts[i], japi->jomt_getNumElems));

    /* Get direct buffer associated with data */
    JNI_TYPE_VAR_CALL(
        env, void *, jni_data, (*env)->GetDirectBufferAddress(env, jomt_data));

    /* Get long array associated with data shape and stride */
    JNI_TYPE_VAR_CALL(env, long *, jni_shape,
        (*env)->GetLongArrayElements(env, jomt_shape, NULL));
    JNI_TYPE_VAR_CALL(env, long *, jni_stride,
        (*env)->GetLongArrayElements(env, jomt_stride, NULL));

    /* Primitive type int and long can be directly used */
    int jni_dataType = jomt_dataType;
    long jni_dataSize = jomt_dataSize;
    int jni_rank = jomt_rank;
    long jni_numElems = jomt_numElems;

    /* Print debug info on what we got from the Java side */
    OMT_DEBUG(i, jni_numElems, jni_data, jni_shape, jni_stride, jni_dataType,
        jni_dataSize, jni_rank, 0);

    /* Create native OMTensor struct. Note jni_data is owned by the
     * Java ByteBuffer object. So here the OMTensor is created with
     * owning=false. Therefore, later when we call omTensorListDestroy
     * the data buffer will not be freed. Java GC is responsible for
     * freeing the data buffer when it garbage collects the ByteBuffer
     * in the OMTensor.
     */
    LIB_VAR_CALL(jni_omts[i],
        omTensorCreate(jni_data, jni_shape, jni_rank, jni_dataType), NULL, env,
        japi->jecpt_cls, "jni_omts[%d]=null", i);

    /* Release reference to the java objects */
    JNI_CALL(
        env, (*env)->ReleaseLongArrayElements(env, jomt_shape, jni_shape, 0));
    JNI_CALL(
        env, (*env)->ReleaseLongArrayElements(env, jomt_stride, jni_stride, 0));
  }

  /* Create OMTensorList to be constructed and passed to the
   * model shared library
   */
  LIB_TYPE_VAR_CALL(OMTensorList *, jni_omtl,
      omTensorListCreate(jni_omts, jomtl_omtn), NULL, env, japi->jecpt_cls,
      "jni_omtl=null");

  return jni_omtl;
}

/* Convert native data structure to Java object */
jobject omtl_native_to_java(
    JNIEnv *env, jclass cls, OMTensorList *jni_omtl, jniapi_t *japi) {

  /* Get the OMTensor array in the OMTensorList */
  LIB_TYPE_VAR_CALL(OMTensor **, jni_omts, omTensorListGetOmtArray(jni_omtl),
      NULL, env, japi->jecpt_cls, "jni_omts=null");
  /* Get the number of OMTensors in the OMTensorList */
  LIB_TYPE_VAR_CALL(int, jni_omtn, omTensorListGetSize(jni_omtl), 0, env,
      japi->jecpt_cls, "jni_omtn=0");

  /* Create OMTensor java object array */
  JNI_TYPE_VAR_CALL(env, jobjectArray, jobj_omts,
      (*env)->NewObjectArray(env, jni_omtn, japi->jomt_cls, NULL));

  /* Loop through the native OMTensor structs */
  for (int i = 0; i < jni_omtn; i++) {

    LIB_TYPE_VAR_CALL(void *, jni_data, omTensorGetDataPtr(jni_omts[i]), NULL,
        env, japi->jecpt_cls, "omt[%d]:data=null", i);
    LIB_TYPE_VAR_CALL(long *, jni_shape, omTensorGetShape(jni_omts[i]), NULL,
        env, japi->jecpt_cls, "omt[%d]:shape=null", i);
    LIB_TYPE_VAR_CALL(long *, jni_stride, omTensorGetStride(jni_omts[i]), NULL,
        env, japi->jecpt_cls, "omt[%d]:stride=null", i);
    LIB_TYPE_VAR_CALL(int, jni_dataType, omTensorGetDataType(jni_omts[i]), 0,
        env, japi->jecpt_cls, "omt[%d]:dataType=0", i);
    LIB_TYPE_VAR_CALL(long, jni_dataSize, omTensorGetDataSize(jni_omts[i]), 0,
        env, japi->jecpt_cls, "omt[%ld]:dataSize=0", i);
    LIB_TYPE_VAR_CALL(int, jni_rank, omTensorGetRank(jni_omts[i]), 0, env,
        japi->jecpt_cls, "omt[%d]:rank=0", i);
    LIB_TYPE_VAR_CALL(int, jni_owning, omTensorGetOwning(jni_omts[i]), -1, env,
        japi->jecpt_cls, "omt[%d]:owning=-1", i);
    LIB_TYPE_VAR_CALL(long, jni_numElems, omTensorGetNumElems(jni_omts[i]), 0,
        env, japi->jecpt_cls, "omt[%d]:numElems=0", i);

    /* Print debug info on what we got from the native side */
    OMT_DEBUG(i, jni_numElems, jni_data, jni_shape, jni_stride, jni_dataType,
        jni_dataSize, jni_rank, jni_owning);

    /* Create direct byte buffer Java object from native data buffer.
     * If data buffer is owned by the native code, we should make a
     * copy of it since the data buffer will be gone when we call
     * omTensorListDestroy. Java GC will be responsible for freeing
     * the data buffer copy when it garbage collects the ByteBuffer
     * in OMTensor.
     */
    void *unowned_data = jni_data;
    if (jni_owning) {
      LIB_VAR_CALL(unowned_data, malloc(jni_dataSize), NULL, env,
          japi->jecpt_cls, "unowned_data=null");
      memcpy(unowned_data, jni_data, jni_dataSize);
    }
    JNI_TYPE_VAR_CALL(env, jobject, jomt_data,
        (*env)->NewDirectByteBuffer(env, unowned_data, jni_dataSize));

    /* Create data shape array Java object, fill in from native array */
    JNI_TYPE_VAR_CALL(
        env, jlongArray, jomt_shape, (*env)->NewLongArray(env, jni_rank));
    JNI_CALL(env,
        (*env)->SetLongArrayRegion(env, jomt_shape, 0, jni_rank, jni_shape));

    /* Create data stride array Java object, fill in from native array */
    JNI_TYPE_VAR_CALL(
        env, jlongArray, jomt_stride, (*env)->NewLongArray(env, jni_rank));
    JNI_CALL(env,
        (*env)->SetLongArrayRegion(env, jomt_stride, 0, jni_rank, jni_stride));

    /* Primitive type int can be directly used. Call setDataType method */
    int jomt_dataType = jni_dataType;

    /* Create the OMTensor Java object */
    JNI_TYPE_VAR_CALL(env, jobject, jobj_omt,
        (*env)->NewObject(env, japi->jomt_cls, japi->jomt_constructor,
            jomt_data, jomt_shape, jomt_stride, jomt_dataType));

    /* Set the OMTensor object in the object array */
    JNI_CALL(env, (*env)->SetObjectArrayElement(env, jobj_omts, i, jobj_omt));
  }

  /* Create the OMTensorList java object */
  JNI_TYPE_VAR_CALL(env, jobject, java_omtl,
      (*env)->NewObject(
          env, japi->jomtl_cls, japi->jomtl_constructor, jobj_omts));

  return java_omtl;
}

JNIEXPORT jobject JNICALL Java_com_ibm_onnxmlir_DynEntryPoint_main_1graph_1jni(
    JNIEnv *env, jclass cls, jobject java_iomtl) {

  /* Find and initialize Java method IDs in struct jniapi */
  CHECK_CALL(jniapi_t *, japi, fill_jniapi(env, &jniapi), NULL);

  /* Convert Java object to native data structure */
  CHECK_CALL(OMTensorList *, jni_iomtl,
      omtl_java_to_native(env, cls, java_iomtl, japi), NULL);

  /* Call model inference entry point */
  CHECK_CALL(OMTensorList *, jni_oomtl, run_main_graph(jni_iomtl), NULL);

  /* Convert native data structure to Java object */
  CHECK_CALL(jobject, java_oomtl,
      omtl_native_to_java(env, cls, jni_oomtl, japi), NULL);

  /* Free intermediate data structures and return Java object */
  omTensorListDestroy(jni_iomtl);
  omTensorListDestroy(jni_oomtl);
  return java_oomtl;
}
