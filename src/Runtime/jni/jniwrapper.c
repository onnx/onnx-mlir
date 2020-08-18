#include <assert.h>
#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif
#include <string.h>

#include "RtMemRef.h"
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

/* Debug output of RtMemRef fields */
#define RMR_DEBUG(                                                             \
    i, n, data, dataSizes, dataStrides, dataType, dataBufferSize, rank, name)  \
  do {                                                                         \
    char tmp[1024];                                                            \
    LOG_TYPE_BUF(dataType, tmp, data, n);                                      \
    LOG_PRINTF(LOG_DEBUG, "rmr[%d]:data=[%s]", i, tmp);                        \
    LOG_LONG_BUF(tmp, dataSizes, rank);                                        \
    LOG_PRINTF(LOG_DEBUG, "rmr[%d]:dataSizes=[%s]", i, tmp);                   \
    LOG_LONG_BUF(tmp, dataStrides, rank);                                      \
    LOG_PRINTF(LOG_DEBUG, "rmr[%d]:dataStrides=[%s]", i, tmp);                 \
    LOG_PRINTF(LOG_DEBUG, "rmr[%d]:dataType=%d", i, dataType);                 \
    LOG_PRINTF(LOG_DEBUG, "rmr[%d]:dataBufferSize=%ld", i, dataBufferSize);    \
    LOG_PRINTF(LOG_DEBUG, "rmr[%d]:rank=%d", i, rank);                         \
    LOG_PRINTF(LOG_DEBUG, "rmr[%d]:name=%s", i, name);                         \
    LOG_PRINTF(LOG_DEBUG, "rmr[%d]:numOfElems=%ld", i, n);                     \
  } while (0)

/* Model shared library entry point */
extern RtMemRefList *_dyn_entry_point_main_graph(RtMemRefList *);

/* Java classes and methods needed for making various JNI API calls */
typedef struct {
  jclass ecpt_cls;     /* java/lang/Exception class                  */
  jclass long_cls;     /* java/lang/Long class                       */
  jclass string_cls;   /* java/lang/String class                     */
  jclass rmr_cls;      /* com/ibm/onnxmlir/RtMemRef class            */
  jclass rmr_list_cls; /* com/ibm/onnxmlir/RtMemRefList class */

  jmethodID rmr_constructor;       /* RtMemRef constructor              */
  jmethodID rmr_getData;           /* RtMemRef getData method           */
  jmethodID rmr_setData;           /* RtMemRef setData method           */
  jmethodID rmr_getDataSizes;      /* RtMemRef getDataSizes method      */
  jmethodID rmr_setDataSizes;      /* RtMemRef setDataSizes method      */
  jmethodID rmr_getDataStrides;    /* RtMemRef getDataStrides method    */
  jmethodID rmr_setDataStrides;    /* RtMemRef setDataStrides method    */
  jmethodID rmr_getDataType;       /* RtMemRef getDataType method       */
  jmethodID rmr_setDataType;       /* RtMemRef setDataType method       */
  jmethodID rmr_getDataBufferSize; /* RtMemRef getDataBufferSize method */
  jmethodID rmr_getRank;           /* RtMemRef getRank method           */
  jmethodID rmr_getName;           /* RtMemRef getName method           */
  jmethodID rmr_setName;           /* RtMemRef setName method           */
  jmethodID rmr_getNumOfElems;     /* RtMemRef getNumOfElems method     */

  jmethodID rmr_list_constructor; /* RtMemRefList constructor    */
  jmethodID rmr_list_getRmrs;     /* RtMemRefList getRmrs method */
} jniapi_t;

jniapi_t jniapi;

/* Fill in struct jniapi */
jniapi_t *fill_jniapi(JNIEnv *env, jniapi_t *japi) {
  /* Get Java Exception, Long, String, RtMemRef, and RtMemRefList classes
   */
  JNI_VAR_CALL(
      env, japi->ecpt_cls, (*env)->FindClass(env, "java/lang/Exception"));
  JNI_VAR_CALL(env, japi->long_cls, (*env)->FindClass(env, "java/lang/Long"));
  JNI_VAR_CALL(
      env, japi->string_cls, (*env)->FindClass(env, "java/lang/String"));
  JNI_VAR_CALL(
      env, japi->rmr_cls, (*env)->FindClass(env, "com/ibm/onnxmlir/RtMemRef"));
  JNI_VAR_CALL(env, japi->rmr_list_cls,
      (*env)->FindClass(env, "com/ibm/onnxmlir/RtMemRefList"));

  /* Get method ID of constructor and various methods in RtMemRef */
  JNI_VAR_CALL(env, japi->rmr_constructor,
      (*env)->GetMethodID(env, japi->rmr_cls, "<init>", "(I)V"));
  JNI_VAR_CALL(env, japi->rmr_getData,
      (*env)->GetMethodID(
          env, japi->rmr_cls, "getData", "()Ljava/nio/ByteBuffer;"));
  JNI_VAR_CALL(env, japi->rmr_setData,
      (*env)->GetMethodID(
          env, japi->rmr_cls, "setData", "(Ljava/nio/ByteBuffer;)V"));
  JNI_VAR_CALL(env, japi->rmr_getDataSizes,
      (*env)->GetMethodID(env, japi->rmr_cls, "getDataSizes", "()[J"));
  JNI_VAR_CALL(env, japi->rmr_setDataSizes,
      (*env)->GetMethodID(env, japi->rmr_cls, "setDataSizes", "([J)V"));
  JNI_VAR_CALL(env, japi->rmr_getDataStrides,
      (*env)->GetMethodID(env, japi->rmr_cls, "getDataStrides", "()[J"));
  JNI_VAR_CALL(env, japi->rmr_setDataStrides,
      (*env)->GetMethodID(env, japi->rmr_cls, "setDataStrides", "([J)V"));
  JNI_VAR_CALL(env, japi->rmr_getDataType,
      (*env)->GetMethodID(env, japi->rmr_cls, "getDataType", "()I"));
  JNI_VAR_CALL(env, japi->rmr_setDataType,
      (*env)->GetMethodID(env, japi->rmr_cls, "setDataType", "(I)V"));
  JNI_VAR_CALL(env, japi->rmr_getDataBufferSize,
      (*env)->GetMethodID(env, japi->rmr_cls, "getDataBufferSize", "()J"));
  JNI_VAR_CALL(env, japi->rmr_getRank,
      (*env)->GetMethodID(env, japi->rmr_cls, "getRank", "()I"));
  JNI_VAR_CALL(env, japi->rmr_getName,
      (*env)->GetMethodID(
          env, japi->rmr_cls, "getName", "()Ljava/lang/String;"));
  JNI_VAR_CALL(env, japi->rmr_setName,
      (*env)->GetMethodID(
          env, japi->rmr_cls, "setName", "(Ljava/lang/String;)V"));
  JNI_VAR_CALL(env, japi->rmr_getNumOfElems,
      (*env)->GetMethodID(env, japi->rmr_cls, "getNumOfElems", "()J"));

  /* Get method ID of constructor and various methods in RtMemRefList */
  JNI_VAR_CALL(env, japi->rmr_list_constructor,
      (*env)->GetMethodID(env, japi->rmr_list_cls, "<init>",
          "([Lcom/ibm/onnxmlir/RtMemRef;)V"));
  JNI_VAR_CALL(env, japi->rmr_list_getRmrs,
      (*env)->GetMethodID(env, japi->rmr_list_cls, "getRmrs",
          "()[Lcom/ibm/onnxmlir/RtMemRef;"));

  return japi;
}

/* Convert Java object to native data structure */
RtMemRefList *rmr_list_java_to_native(
    JNIEnv *env, jclass cls, jobject obj, jniapi_t *japi) {

  /* Get RtMemRef array Java object in RtMemRefList */
  JNI_TYPE_VAR_CALL(env, jobjectArray, rmr_list_rmrs,
      (*env)->CallObjectMethod(env, obj, japi->rmr_list_getRmrs));

  /* Get the number of RtMemRefs in the array */
  JNI_TYPE_VAR_CALL(
      env, jsize, rmr_list_nrmr, (*env)->GetArrayLength(env, rmr_list_rmrs));

  /* Allocate memory for holding each Java rmr object and RtMemRef pointers
   * for constructing native RtMemRef array
   */
  LIB_TYPE_VAR_CALL(jobject *, obj_rmrs,
      malloc(rmr_list_nrmr * sizeof(jobject)), NULL, env, japi->ecpt_cls,
      "obj_rmrs=null");
  LIB_TYPE_VAR_CALL(RtMemRef **, jni_rmrs,
      malloc(rmr_list_nrmr * sizeof(RtMemRef *)), NULL, env, japi->ecpt_cls,
      "jni_rmrs=null");

  /* Loop through all the rmr_list_rmrs  */
  for (int i = 0; i < rmr_list_nrmr; i++) {
    JNI_VAR_CALL(
        env, obj_rmrs[i], (*env)->GetObjectArrayElement(env, rmr_list_rmrs, i));

    /* Get data, dataSizes, dataStrides, dataType, rank, name and
     * dataBufferSize by calling corresponding methods
     */
    JNI_TYPE_VAR_CALL(env, jobject, rmr_data,
        (*env)->CallObjectMethod(env, obj_rmrs[i], japi->rmr_getData));
    JNI_TYPE_VAR_CALL(env, jobject, rmr_dataSizes,
        (*env)->CallObjectMethod(env, obj_rmrs[i], japi->rmr_getDataSizes));
    JNI_TYPE_VAR_CALL(env, jobject, rmr_dataStrides,
        (*env)->CallObjectMethod(env, obj_rmrs[i], japi->rmr_getDataStrides));
    JNI_TYPE_VAR_CALL(env, jint, rmr_dataType,
        (*env)->CallIntMethod(env, obj_rmrs[i], japi->rmr_getDataType));
    JNI_TYPE_VAR_CALL(env, jlong, rmr_dataBufferSize,
        (*env)->CallLongMethod(env, obj_rmrs[i], japi->rmr_getDataBufferSize));
    JNI_TYPE_VAR_CALL(env, jint, rmr_rank,
        (*env)->CallIntMethod(env, obj_rmrs[i], japi->rmr_getRank));
    JNI_TYPE_VAR_CALL(env, jstring, rmr_name,
        (*env)->CallObjectMethod(env, obj_rmrs[i], japi->rmr_getName));
    JNI_TYPE_VAR_CALL(env, jlong, rmr_numOfElems,
        (*env)->CallLongMethod(env, obj_rmrs[i], japi->rmr_getNumOfElems));

    /* Get direct buffer associated with data */
    JNI_TYPE_VAR_CALL(
        env, void *, jni_data, (*env)->GetDirectBufferAddress(env, rmr_data));

    /* Get long array associated with data sizes and strides */
    JNI_TYPE_VAR_CALL(env, long *, jni_dataSizes,
        (*env)->GetLongArrayElements(env, rmr_dataSizes, NULL));
    JNI_TYPE_VAR_CALL(env, long *, jni_dataStrides,
        (*env)->GetLongArrayElements(env, rmr_dataStrides, NULL));

    /* Primitive type int and long can be directly used */
    int jni_dataType = rmr_dataType;
    long jni_dataBufferSize = rmr_dataBufferSize;
    int jni_rank = rmr_rank;
    long jni_numOfElems = rmr_numOfElems;

    /* Get name string */
    JNI_TYPE_VAR_CALL(env, char *, jni_name,
        (char *)(*env)->GetStringUTFChars(env, rmr_name, NULL));

    /* Print debug info on what we got from the Java side */
    RMR_DEBUG(i, jni_numOfElems, jni_data, jni_dataSizes, jni_dataStrides,
        jni_dataType, jni_dataBufferSize, jni_rank, jni_name);

    /* Create native RtMemRef struct and fill in its fields */
    LIB_VAR_CALL(jni_rmrs[i], rmr_create(jni_rank), NULL, env, japi->ecpt_cls,
        "jni_rmrs[%d]=null", i);
    rmr_setData(jni_rmrs[i], jni_data);
    rmr_setDataSizes(jni_rmrs[i], jni_dataSizes);
    rmr_setDataStrides(jni_rmrs[i], jni_dataStrides);
    rmr_setDataType(jni_rmrs[i], jni_dataType);
    rmr_setName(jni_rmrs[i], jni_name);

    /* Release reference to the java objects */
    JNI_CALL(env,
        (*env)->ReleaseLongArrayElements(env, rmr_dataSizes, jni_dataSizes, 0));
    JNI_CALL(env, (*env)->ReleaseLongArrayElements(
                      env, rmr_dataStrides, jni_dataStrides, 0));
    JNI_CALL(env, (*env)->ReleaseStringUTFChars(env, rmr_name, jni_name));
  }

  /* Create RtMemRefList to be constructed and passed to the
   * model shared library
   */
  LIB_TYPE_VAR_CALL(RtMemRefList *, ormrd,
      rmr_list_create(jni_rmrs, rmr_list_nrmr), NULL, env, japi->ecpt_cls,
      "ormrd=null");

  return ormrd;
}

/* Convert native data structure to Java object */
jobject rmr_list_native_to_java(
    JNIEnv *env, jclass cls, RtMemRefList *dict, jniapi_t *japi) {

  /* Get the RtMemRef array in the RtMemRefList */
  LIB_TYPE_VAR_CALL(RtMemRef **, jni_rmrs, rmrListGetRmrs(dict), NULL, env,
      japi->ecpt_cls, "jni_rmrs=null");
  /* Get the number of RtMemRefs in the RtMemRefList */
  LIB_TYPE_VAR_CALL(int, jni_nrmr, rmrListGetNumOfRmrs(dict), 0, env,
      japi->ecpt_cls, "jni_nrmr=0");

  /* Create RtMemRef java object array */
  JNI_TYPE_VAR_CALL(env, jobjectArray, obj_rmrs,
      (*env)->NewObjectArray(env, jni_nrmr, japi->rmr_cls, NULL));

  /* Loop through the native RtMemRef structs */
  for (int i = 0; i < jni_nrmr; i++) {

    LIB_TYPE_VAR_CALL(void *, jni_data, rmrGetData(jni_rmrs[i]), NULL, env,
        japi->ecpt_cls, "rmr[%d]:data=null", i);
    LIB_TYPE_VAR_CALL(long *, jni_dataSizes, rmrGetDataSizes(jni_rmrs[i]), NULL,
        env, japi->ecpt_cls, "rmr[%d]:dataSizes=null", i);
    LIB_TYPE_VAR_CALL(long *, jni_dataStrides, rmrGetDataStrides(jni_rmrs[i]),
        NULL, env, japi->ecpt_cls, "rmr[%d]:dataStrides=null", i);
    LIB_TYPE_VAR_CALL(int, jni_dataType, rmrGetDataType(jni_rmrs[i]), 0, env,
        japi->ecpt_cls, "rmr[%d]:dataType=0", i);
    LIB_TYPE_VAR_CALL(long, jni_dataBufferSize,
        rmr_getDataBufferSize(jni_rmrs[i]), 0, env, japi->ecpt_cls,
        "rmr[%ld]:dataBufferSize=0", i);
    LIB_TYPE_VAR_CALL(int, jni_rank, rmrGetRank(jni_rmrs[i]), 0, env,
        japi->ecpt_cls, "rmr[%d]:rank=0", i);
    LIB_TYPE_VAR_CALL(char *, jni_name, rmrGetName(jni_rmrs[i]), NULL, env,
        japi->ecpt_cls, "rmr[%d]:name=null", i);
    LIB_TYPE_VAR_CALL(long, jni_numOfElems, rmrGetNumOfElems(jni_rmrs[i]), 0,
        env, japi->ecpt_cls, "rmr[%d]:numOfElems=0", i);

    /* Print debug info on what we got from the native side */
    RMR_DEBUG(i, jni_numOfElems, jni_data, jni_dataSizes, jni_dataStrides,
        jni_dataType, jni_dataBufferSize, jni_rank, jni_name);

    /* Create the RtMemRef Java object */
    JNI_TYPE_VAR_CALL(env, jobject, obj_rmr,
        (*env)->NewObject(env, japi->rmr_cls, japi->rmr_constructor, jni_rank));

    /* Create direct byte buffer Java object from native data buffer, and
     * call setData method
     */
    JNI_TYPE_VAR_CALL(env, jobject, rmr_data,
        (*env)->NewDirectByteBuffer(env, jni_data, jni_dataBufferSize));
    JNI_CALL(env,
        (*env)->CallObjectMethod(env, obj_rmr, japi->rmr_setData, rmr_data));

    /* Create data sizes array Java object, fill in from native array, and
     * call setDataSizes method
     */
    JNI_TYPE_VAR_CALL(
        env, jlongArray, rmr_dataSizes, (*env)->NewLongArray(env, jni_rank));
    JNI_CALL(env, (*env)->SetLongArrayRegion(
                      env, rmr_dataSizes, 0, jni_rank, jni_dataSizes));
    JNI_CALL(env, (*env)->CallObjectMethod(
                      env, obj_rmr, japi->rmr_setDataSizes, rmr_dataSizes));

    /* Create data strides array Java object, fill in from native array, and
     * call setStrides method
     */
    JNI_TYPE_VAR_CALL(
        env, jlongArray, rmr_dataStrides, (*env)->NewLongArray(env, jni_rank));
    JNI_CALL(env, (*env)->SetLongArrayRegion(
                      env, rmr_dataStrides, 0, jni_rank, jni_dataStrides));
    JNI_CALL(env, (*env)->CallObjectMethod(
                      env, obj_rmr, japi->rmr_setDataStrides, rmr_dataStrides));

    /* Primitive type int can be directly used. Call setDataType method */
    JNI_CALL(env, (*env)->CallIntMethod(
                      env, obj_rmr, japi->rmr_setDataType, (jint)jni_dataType));

    /* Create string Java object from native char * and call setName method */
    JNI_TYPE_VAR_CALL(
        env, jstring, rmr_name, (*env)->NewStringUTF(env, jni_name));
    JNI_CALL(env,
        (*env)->CallObjectMethod(env, obj_rmr, japi->rmr_setName, rmr_name));

    /* Set RtMemRef object in the object array */
    JNI_CALL(env, (*env)->SetObjectArrayElement(env, obj_rmrs, i, obj_rmr));
  }

  /* Create the RtMemRefList java object */
  JNI_TYPE_VAR_CALL(env, jobject, ormrd,
      (*env)->NewObject(
          env, japi->rmr_list_cls, japi->rmr_list_constructor, obj_rmrs));

  return ormrd;
}

JNIEXPORT jobject JNICALL Java_com_ibm_onnxmlir_DynEntryPoint_main_1graph_1jni(
    JNIEnv *env, jclass cls, jobject obj) {
  CHECK_CALL(jniapi_t *, japi, fill_jniapi(env, &jniapi), NULL);

  CHECK_CALL(RtMemRefList *, input_ormrd,
      rmr_list_java_to_native(env, cls, obj, japi), NULL);

  CHECK_CALL(
      RtMemRefList *, dict, _dyn_entry_point_main_graph(input_ormrd), NULL);

  CHECK_CALL(jobject, output_ormrd,
      rmr_list_native_to_java(env, cls, dict, japi), NULL);

  return output_ormrd;
}
