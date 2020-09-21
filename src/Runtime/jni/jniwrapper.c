#include <assert.h>
#ifdef __APPLE__
#include <stdlib.h>
#else
#include <malloc.h>
#endif
#include <string.h>

#include "OnnxMlir.h"
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
#define OMT_DEBUG(                                                             \
    i, n, data, dataSizes, dataStrides, dataType, dataBufferSize, rank, name)  \
  do {                                                                         \
    char tmp[1024];                                                            \
    LOG_TYPE_BUF(dataType, tmp, data, n);                                      \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:data=[%s]", i, tmp);                        \
    LOG_LONG_BUF(tmp, dataSizes, rank);                                        \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:dataSizes=[%s]", i, tmp);                   \
    LOG_LONG_BUF(tmp, dataStrides, rank);                                      \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:dataStrides=[%s]", i, tmp);                 \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:dataType=%d", i, dataType);                 \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:dataBufferSize=%ld", i, dataBufferSize);    \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:rank=%d", i, rank);                         \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:name=%s", i, name);                         \
    LOG_PRINTF(LOG_DEBUG, "omt[%d]:numOfElems=%ld", i, n);                     \
  } while (0)

/* Model shared library entry point */
extern OMTensorList *_dyn_entry_point_main_graph(OMTensorList *);

/* Java classes and methods needed for making various JNI API calls */
typedef struct {
  jclass ecpt_cls;     /* java/lang/Exception class                  */
  jclass long_cls;     /* java/lang/Long class                       */
  jclass string_cls;   /* java/lang/String class                     */
  jclass omt_cls;      /* com/ibm/onnxmlir/OMTensor class            */
  jclass omt_list_cls; /* com/ibm/onnxmlir/OMTensorList class */

  jmethodID omt_constructor;       /* OMTensor constructor              */
  jmethodID omt_getData;           /* OMTensor getData method           */
  jmethodID omt_setData;           /* OMTensor setData method           */
  jmethodID omt_getDataSizes;      /* OMTensor getDataSizes method      */
  jmethodID omt_setDataSizes;      /* OMTensor setDataSizes method      */
  jmethodID omt_getDataStrides;    /* OMTensor getDataStrides method    */
  jmethodID omt_setDataStrides;    /* OMTensor setDataStrides method    */
  jmethodID omt_getDataType;       /* OMTensor getDataType method       */
  jmethodID omt_setDataType;       /* OMTensor setDataType method       */
  jmethodID omt_getDataBufferSize; /* OMTensor getDataBufferSize method */
  jmethodID omt_getRank;           /* OMTensor getRank method           */
  jmethodID omt_getName;           /* OMTensor getName method           */
  jmethodID omt_setName;           /* OMTensor setName method           */
  jmethodID omt_getNumOfElems;     /* OMTensor getNumOfElems method     */

  jmethodID omt_list_constructor; /* OMTensorList constructor    */
  jmethodID omt_list_getOmts;     /* OMTensorList getOmts method */
} jniapi_t;

jniapi_t jniapi;

/* Fill in struct jniapi */
jniapi_t *fill_jniapi(JNIEnv *env, jniapi_t *japi) {
  /* Get Java Exception, Long, String, OMTensor, and OMTensorList classes
   */
  JNI_VAR_CALL(
      env, japi->ecpt_cls, (*env)->FindClass(env, "java/lang/Exception"));
  JNI_VAR_CALL(env, japi->long_cls, (*env)->FindClass(env, "java/lang/Long"));
  JNI_VAR_CALL(
      env, japi->string_cls, (*env)->FindClass(env, "java/lang/String"));
  JNI_VAR_CALL(
      env, japi->omt_cls, (*env)->FindClass(env, "com/ibm/onnxmlir/OMTensor"));
  JNI_VAR_CALL(env, japi->omt_list_cls,
      (*env)->FindClass(env, "com/ibm/onnxmlir/OMTensorList"));

  /* Get method ID of constructor and various methods in OMTensor */
  JNI_VAR_CALL(env, japi->omt_constructor,
      (*env)->GetMethodID(env, japi->omt_cls, "<init>", "(I)V"));
  JNI_VAR_CALL(env, japi->omt_getData,
      (*env)->GetMethodID(
          env, japi->omt_cls, "getData", "()Ljava/nio/ByteBuffer;"));
  JNI_VAR_CALL(env, japi->omt_setData,
      (*env)->GetMethodID(
          env, japi->omt_cls, "setData", "(Ljava/nio/ByteBuffer;)V"));
  JNI_VAR_CALL(env, japi->omt_getDataSizes,
      (*env)->GetMethodID(env, japi->omt_cls, "getDataSizes", "()[J"));
  JNI_VAR_CALL(env, japi->omt_setDataSizes,
      (*env)->GetMethodID(env, japi->omt_cls, "setDataSizes", "([J)V"));
  JNI_VAR_CALL(env, japi->omt_getDataStrides,
      (*env)->GetMethodID(env, japi->omt_cls, "getDataStrides", "()[J"));
  JNI_VAR_CALL(env, japi->omt_setDataStrides,
      (*env)->GetMethodID(env, japi->omt_cls, "setDataStrides", "([J)V"));
  JNI_VAR_CALL(env, japi->omt_getDataType,
      (*env)->GetMethodID(env, japi->omt_cls, "getDataType", "()I"));
  JNI_VAR_CALL(env, japi->omt_setDataType,
      (*env)->GetMethodID(env, japi->omt_cls, "setDataType", "(I)V"));
  JNI_VAR_CALL(env, japi->omt_getDataBufferSize,
      (*env)->GetMethodID(env, japi->omt_cls, "getDataBufferSize", "()J"));
  JNI_VAR_CALL(env, japi->omt_getRank,
      (*env)->GetMethodID(env, japi->omt_cls, "getRank", "()I"));
  JNI_VAR_CALL(env, japi->omt_getName,
      (*env)->GetMethodID(
          env, japi->omt_cls, "getName", "()Ljava/lang/String;"));
  JNI_VAR_CALL(env, japi->omt_setName,
      (*env)->GetMethodID(
          env, japi->omt_cls, "setName", "(Ljava/lang/String;)V"));
  JNI_VAR_CALL(env, japi->omt_getNumOfElems,
      (*env)->GetMethodID(env, japi->omt_cls, "getNumOfElems", "()J"));

  /* Get method ID of constructor and various methods in OMTensorList */
  JNI_VAR_CALL(env, japi->omt_list_constructor,
      (*env)->GetMethodID(env, japi->omt_list_cls, "<init>",
          "([Lcom/ibm/onnxmlir/OMTensor;)V"));
  JNI_VAR_CALL(env, japi->omt_list_getOmts,
      (*env)->GetMethodID(env, japi->omt_list_cls, "getOmts",
          "()[Lcom/ibm/onnxmlir/OMTensor;"));

  return japi;
}

/* Convert Java object to native data structure */
OMTensorList *omt_list_java_to_native(
    JNIEnv *env, jclass cls, jobject obj, jniapi_t *japi) {

  /* Get OMTensor array Java object in OMTensorList */
  JNI_TYPE_VAR_CALL(env, jobjectArray, omt_list_omts,
      (*env)->CallObjectMethod(env, obj, japi->omt_list_getOmts));

  /* Get the number of OMTensors in the array */
  JNI_TYPE_VAR_CALL(
      env, jsize, omt_list_nomt, (*env)->GetArrayLength(env, omt_list_omts));

  /* Allocate memory for holding each Java omt object and OMTensor pointers
   * for constructing native OMTensor array
   */
  LIB_TYPE_VAR_CALL(jobject *, obj_omts,
      malloc(omt_list_nomt * sizeof(jobject)), NULL, env, japi->ecpt_cls,
      "obj_omts=null");
  LIB_TYPE_VAR_CALL(OMTensor **, jni_omts,
      malloc(omt_list_nomt * sizeof(OMTensor *)), NULL, env, japi->ecpt_cls,
      "jni_omts=null");

  /* Loop through all the omt_list_omts  */
  for (int i = 0; i < omt_list_nomt; i++) {
    JNI_VAR_CALL(
        env, obj_omts[i], (*env)->GetObjectArrayElement(env, omt_list_omts, i));

    /* Get data, dataSizes, dataStrides, dataType, rank, name and
     * dataBufferSize by calling corresponding methods
     */
    JNI_TYPE_VAR_CALL(env, jobject, omt_data,
        (*env)->CallObjectMethod(env, obj_omts[i], japi->omt_getData));
    JNI_TYPE_VAR_CALL(env, jobject, omt_dataSizes,
        (*env)->CallObjectMethod(env, obj_omts[i], japi->omt_getDataSizes));
    JNI_TYPE_VAR_CALL(env, jobject, omt_dataStrides,
        (*env)->CallObjectMethod(env, obj_omts[i], japi->omt_getDataStrides));
    JNI_TYPE_VAR_CALL(env, jint, omt_dataType,
        (*env)->CallIntMethod(env, obj_omts[i], japi->omt_getDataType));
    JNI_TYPE_VAR_CALL(env, jlong, omt_dataBufferSize,
        (*env)->CallLongMethod(env, obj_omts[i], japi->omt_getDataBufferSize));
    JNI_TYPE_VAR_CALL(env, jint, omt_rank,
        (*env)->CallIntMethod(env, obj_omts[i], japi->omt_getRank));
    JNI_TYPE_VAR_CALL(env, jstring, omt_name,
        (*env)->CallObjectMethod(env, obj_omts[i], japi->omt_getName));
    JNI_TYPE_VAR_CALL(env, jlong, omt_numOfElems,
        (*env)->CallLongMethod(env, obj_omts[i], japi->omt_getNumOfElems));

    /* Get direct buffer associated with data */
    JNI_TYPE_VAR_CALL(
        env, void *, jni_data, (*env)->GetDirectBufferAddress(env, omt_data));

    /* Get long array associated with data sizes and strides */
    JNI_TYPE_VAR_CALL(env, long *, jni_dataSizes,
        (*env)->GetLongArrayElements(env, omt_dataSizes, NULL));
    JNI_TYPE_VAR_CALL(env, long *, jni_dataStrides,
        (*env)->GetLongArrayElements(env, omt_dataStrides, NULL));

    /* Primitive type int and long can be directly used */
    int jni_dataType = omt_dataType;
    long jni_dataBufferSize = omt_dataBufferSize;
    int jni_rank = omt_rank;
    long jni_numOfElems = omt_numOfElems;

    /* Get name string */
    JNI_TYPE_VAR_CALL(env, char *, jni_name,
        (char *)(*env)->GetStringUTFChars(env, omt_name, NULL));

    /* Print debug info on what we got from the Java side */
    OMT_DEBUG(i, jni_numOfElems, jni_data, jni_dataSizes, jni_dataStrides,
        jni_dataType, jni_dataBufferSize, jni_rank, jni_name);

    /* Create native OMTensor struct and fill in its fields */
    LIB_VAR_CALL(jni_omts[i], omt_create(jni_rank), NULL, env, japi->ecpt_cls,
        "jni_omts[%d]=null", i);
    omt_setData(jni_omts[i], jni_data);
    omt_setDataSizes(jni_omts[i], jni_dataSizes);
    omt_setDataStrides(jni_omts[i], jni_dataStrides);
    omt_setDataType(jni_omts[i], jni_dataType);
    omt_setName(jni_omts[i], jni_name);

    /* Release reference to the java objects */
    JNI_CALL(env,
        (*env)->ReleaseLongArrayElements(env, omt_dataSizes, jni_dataSizes, 0));
    JNI_CALL(env, (*env)->ReleaseLongArrayElements(
                      env, omt_dataStrides, jni_dataStrides, 0));
    JNI_CALL(env, (*env)->ReleaseStringUTFChars(env, omt_name, jni_name));
  }

  /* Create OMTensorList to be constructed and passed to the
   * model shared library
   */
  LIB_TYPE_VAR_CALL(OMTensorList *, list,
      omt_list_create(jni_omts, omt_list_nomt), NULL, env, japi->ecpt_cls,
      "list=null");

  return list;
}

/* Convert native data structure to Java object */
jobject omt_list_native_to_java(
    JNIEnv *env, jclass cls, OMTensorList *dict, jniapi_t *japi) {

  /* Get the OMTensor array in the OMTensorList */
  LIB_TYPE_VAR_CALL(OMTensor **, jni_omts, omTensorListGetOmts(dict), NULL, env,
      japi->ecpt_cls, "jni_omts=null");
  /* Get the number of OMTensors in the OMTensorList */
  LIB_TYPE_VAR_CALL(int, jni_nomt, omTensorListGetNumOfOmts(dict), 0, env,
      japi->ecpt_cls, "jni_nomt=0");

  /* Create OMTensor java object array */
  JNI_TYPE_VAR_CALL(env, jobjectArray, obj_omts,
      (*env)->NewObjectArray(env, jni_nomt, japi->omt_cls, NULL));

  /* Loop through the native OMTensor structs */
  for (int i = 0; i < jni_nomt; i++) {

    LIB_TYPE_VAR_CALL(void *, jni_data, omtGetData(jni_omts[i]), NULL, env,
        japi->ecpt_cls, "omt[%d]:data=null", i);
    LIB_TYPE_VAR_CALL(long *, jni_dataSizes, omtGetDataSizes(jni_omts[i]), NULL,
        env, japi->ecpt_cls, "omt[%d]:dataSizes=null", i);
    LIB_TYPE_VAR_CALL(long *, jni_dataStrides, omtGetDataStrides(jni_omts[i]),
        NULL, env, japi->ecpt_cls, "omt[%d]:dataStrides=null", i);
    LIB_TYPE_VAR_CALL(int, jni_dataType, omtGetDataType(jni_omts[i]), 0, env,
        japi->ecpt_cls, "omt[%d]:dataType=0", i);
    LIB_TYPE_VAR_CALL(long, jni_dataBufferSize,
        omt_getDataBufferSize(jni_omts[i]), 0, env, japi->ecpt_cls,
        "omt[%ld]:dataBufferSize=0", i);
    LIB_TYPE_VAR_CALL(int, jni_rank, omtGetRank(jni_omts[i]), 0, env,
        japi->ecpt_cls, "omt[%d]:rank=0", i);
    LIB_TYPE_VAR_CALL(char *, jni_name, omtGetName(jni_omts[i]), NULL, env,
        japi->ecpt_cls, "omt[%d]:name=null", i);
    LIB_TYPE_VAR_CALL(long, jni_numOfElems, omtGetNumOfElems(jni_omts[i]), 0,
        env, japi->ecpt_cls, "omt[%d]:numOfElems=0", i);

    /* Print debug info on what we got from the native side */
    OMT_DEBUG(i, jni_numOfElems, jni_data, jni_dataSizes, jni_dataStrides,
        jni_dataType, jni_dataBufferSize, jni_rank, jni_name);

    /* Create the OMTensor Java object */
    JNI_TYPE_VAR_CALL(env, jobject, obj_omt,
        (*env)->NewObject(env, japi->omt_cls, japi->omt_constructor, jni_rank));

    /* Create direct byte buffer Java object from native data buffer, and
     * call setData method
     */
    JNI_TYPE_VAR_CALL(env, jobject, omt_data,
        (*env)->NewDirectByteBuffer(env, jni_data, jni_dataBufferSize));
    JNI_CALL(env,
        (*env)->CallObjectMethod(env, obj_omt, japi->omt_setData, omt_data));

    /* Create data sizes array Java object, fill in from native array, and
     * call setDataSizes method
     */
    JNI_TYPE_VAR_CALL(
        env, jlongArray, omt_dataSizes, (*env)->NewLongArray(env, jni_rank));
    JNI_CALL(env, (*env)->SetLongArrayRegion(
                      env, omt_dataSizes, 0, jni_rank, jni_dataSizes));
    JNI_CALL(env, (*env)->CallObjectMethod(
                      env, obj_omt, japi->omt_setDataSizes, omt_dataSizes));

    /* Create data strides array Java object, fill in from native array, and
     * call setStrides method
     */
    JNI_TYPE_VAR_CALL(
        env, jlongArray, omt_dataStrides, (*env)->NewLongArray(env, jni_rank));
    JNI_CALL(env, (*env)->SetLongArrayRegion(
                      env, omt_dataStrides, 0, jni_rank, jni_dataStrides));
    JNI_CALL(env, (*env)->CallObjectMethod(
                      env, obj_omt, japi->omt_setDataStrides, omt_dataStrides));

    /* Primitive type int can be directly used. Call setDataType method */
    JNI_CALL(env, (*env)->CallIntMethod(
                      env, obj_omt, japi->omt_setDataType, (jint)jni_dataType));

    /* Create string Java object from native char * and call setName method */
    JNI_TYPE_VAR_CALL(
        env, jstring, omt_name, (*env)->NewStringUTF(env, jni_name));
    JNI_CALL(env,
        (*env)->CallObjectMethod(env, obj_omt, japi->omt_setName, omt_name));

    /* Set OMTensor object in the object array */
    JNI_CALL(env, (*env)->SetObjectArrayElement(env, obj_omts, i, obj_omt));
  }

  /* Create the OMTensorList java object */
  JNI_TYPE_VAR_CALL(env, jobject, list,
      (*env)->NewObject(
          env, japi->omt_list_cls, japi->omt_list_constructor, obj_omts));

  return list;
}

JNIEXPORT jobject JNICALL Java_com_ibm_onnxmlir_DynEntryPoint_main_1graph_1jni(
    JNIEnv *env, jclass cls, jobject obj) {
  CHECK_CALL(jniapi_t *, japi, fill_jniapi(env, &jniapi), NULL);

  CHECK_CALL(OMTensorList *, input_list,
      omt_list_java_to_native(env, cls, obj, japi), NULL);

  CHECK_CALL(
      OMTensorList *, dict, _dyn_entry_point_main_graph(input_list), NULL);

  CHECK_CALL(jobject, output_list,
      omt_list_native_to_java(env, cls, dict, japi), NULL);

  return output_list;
}
