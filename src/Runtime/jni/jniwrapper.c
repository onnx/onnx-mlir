#include <assert.h>
#include <malloc.h>
#include <string.h>

#include "com_ibm_onnxmlir_DynEntryPoint.h"
#include "RtMemRef.h"
#include "jnilog.h"

/* Declare type var, make call and assign to var, check against val */
#define CHECK_CALL(type, var, call, val) \
  type var = call;			 \
  if (var == val) return NULL

/* Make a JNI call and throw Java exception if the call failed */
#define JNI_CALL(env, stmt)				    \
  stmt;							    \
  do {							    \
    jthrowable e = (*env)->ExceptionOccurred(env);	    \
    if (e) {						    \
      LOG_PRINTF(LOG_ERROR, "JNI call exception occurred"); \
      (*env)->Throw(env, e);				    \
      return NULL;					    \
    }							    \
  } while(0)

/* Make a JNI call and assign return value to var,
 * throw Java exception if the call failed
 */
#define JNI_VAR_CALL(env, var, call) \
  JNI_CALL(env, var = call)

/* Declare type var, make a JNI call and assign return value to var,
 * throw Java exception if the call failed
 */
#define JNI_TYPE_VAR_CALL(env, type, var, call) \
  JNI_CALL(env, type var = call);

/* If cond is true (native code failed), log error and throw Java exception */
#define JNI_COND(type, var, call, val, env, cls, ...)  \
  type var = call;				       \
  do {						       \
    if (var == val) {				       \
      LOG_PRINTF(LOG_ERROR, __VA_ARGS__);	       \
      (*env)->ThrowNew(env, cls, "native code error"); \
      return NULL;				       \
    }						       \
  } while(0)

/* Debug output of RtMemRef fields */
#define RMR_DEBUG(i, type, rank, sizes, strides, data, datasize) \
  do {								 \
    char tmp[1024];						 \
    LOG_PRINTF(LOG_DEBUG, "rmr[%d]:type=%d", i, type);		 \
    LOG_PRINTF(LOG_DEBUG, "rmr[%d]:rank=%d", i, rank);		 \
    LOG_LONG_BUF(tmp, sizes, rank);				 \
    LOG_PRINTF(LOG_DEBUG, "rmr[%d]:sizes=[%s]", i, tmp);	 \
    LOG_LONG_BUF(tmp, strides, rank);				 \
    LOG_PRINTF(LOG_DEBUG, "rmr[%d]:strides=[%s]", i, tmp);	 \
    LOG_TYPE_BUF(type, tmp, data, datasize);			 \
    LOG_PRINTF(LOG_DEBUG, "rmr[%d]:data=[%s]", i, tmp);		 \
  } while(0)

/* Model shared library entry point */
extern OrderedRtMemRefDict *_dyn_entry_point_main_graph(OrderedRtMemRefDict *);

/* ONNX type to size (number of bytes) mapping */
int onnx_type_size[] =
  { 0,  /* UNDEFINED  = 0  */
    4,  /* FLOAT      = 1  */
    1,  /* UINT8      = 2  */
    1,  /* INT8       = 3  */
    2,  /* UINT16     = 4  */
    2,  /* INT16      = 5  */
    4,  /* INT32      = 6  */
    8,  /* INT64      = 7  */
    0,  /* STRING     = 8  */
    1,  /* BOOL       = 9  */
    2,  /* FLOAT16    = 10 */
    8,  /* DOUBLE     = 11 */
    4,  /* UINT32     = 12 */
    8,  /* UINT64     = 13 */
    8,  /* COMPLEX64  = 14 */
    16, /* COMPLEX128 = 15 */
    2,  /* BFLOAT16   = 16 */
  };

/* Java classes and methods needed for making various JNI API calls */
typedef struct {
  jclass ecpt_cls;             /* java/lang/Exception class                  */
  jclass long_cls;             /* java/lang/Long class                       */
  jclass string_cls;           /* java/lang/String class                     */
  jclass ormrd_cls;            /* com/ibm/onnxmlir/OrderedRtMemRefDict class */
  jclass rmr_cls;              /* com/ibm/onnxmlir/RtMemRef class            */

  jmethodID ormrd_constructor; /* OrderedRtMemRefDict constructor            */
  jmethodID ormrd_getRmrs;     /* OrderedRtMemRefDict getRmrs method         */
  jmethodID ormrd_getNames;    /* OrderedRtMemRefDict getNames method        */

  jmethodID rmr_constructor;   /* RtMemRef constructor                       */
  jmethodID rmr_getType;       /* RtMemRef getType method                    */
  jmethodID rmr_setType;       /* RtMemRef setType method                    */
  jmethodID rmr_getRank;       /* RtMemRef getRank method                    */
  jmethodID rmr_getData;       /* RtMemRef getData method                    */
  jmethodID rmr_setData;       /* RtMemRef setData method                    */
  jmethodID rmr_getSizes;      /* RtMemRef getSizes method                   */
  jmethodID rmr_setSizes;      /* RtMemRef setSizes method                   */
  jmethodID rmr_getStrides;    /* RtMemRef getStrides method                 */
  jmethodID rmr_setStrides;    /* RtMemRef setStrides method                 */
  jmethodID rmr_getDataSize;   /* RtMemRef getDataSize method                */
} jniapi_t;

jniapi_t jniapi;

/* Fill in struct jniapi */
jniapi_t *fill_jniapi(JNIEnv *env, jniapi_t *japi) {
  /* Get Java Exception, Long, String, OrderedRtMemRefDict, and RtMemRef classes */
  JNI_VAR_CALL(env, japi->ecpt_cls,
	       (*env)->FindClass(env, "java/lang/Exception"));
  JNI_VAR_CALL(env, japi->long_cls,
	       (*env)->FindClass(env, "java/lang/Long"));
  JNI_VAR_CALL(env, japi->string_cls,
	       (*env)->FindClass(env, "java/lang/String"));
  JNI_VAR_CALL(env, japi->ormrd_cls,
	       (*env)->FindClass(env, "com/ibm/onnxmlir/OrderedRtMemRefDict"));
  JNI_VAR_CALL(env, japi->rmr_cls,
	       (*env)->FindClass(env, "com/ibm/onnxmlir/RtMemRef"));
  
  /* Get method ID of constructor and various methods in OrderedRtMemRefDict */
  JNI_VAR_CALL(env, japi->ormrd_constructor,
	       (*env)->GetMethodID(env, japi->ormrd_cls,
				   "<init>", "([Lcom/ibm/onnxmlir/RtMemRef;)V"));
  JNI_VAR_CALL(env, japi->ormrd_getRmrs,
	       (*env)->GetMethodID(env, japi->ormrd_cls,
				   "getRmrs", "()[Lcom/ibm/onnxmlir/RtMemRef;"));
  JNI_VAR_CALL(env, japi->ormrd_getNames,
	       (*env)->GetMethodID(env, japi->ormrd_cls,
				   "getNames", "()[Ljava/lang/String;"));

  /* Get method ID of constructor and various methods in RtMemRef */
  JNI_VAR_CALL(env, japi->rmr_constructor,
	       (*env)->GetMethodID(env, japi->rmr_cls, "<init>", "(I)V"));
  JNI_VAR_CALL(env, japi->rmr_getType,
	       (*env)->GetMethodID(env, japi->rmr_cls, "getType", "()I"));
  JNI_VAR_CALL(env, japi->rmr_setType,
	       (*env)->GetMethodID(env, japi->rmr_cls, "setType", "(I)V"));
  JNI_VAR_CALL(env, japi->rmr_getRank,
	       (*env)->GetMethodID(env, japi->rmr_cls, "getRank", "()I"));
  JNI_VAR_CALL(env, japi->rmr_getData,
	       (*env)->GetMethodID(env, japi->rmr_cls,
				   "getData", "()Ljava/nio/ByteBuffer;"));
  JNI_VAR_CALL(env, japi->rmr_setData,
	       (*env)->GetMethodID(env, japi->rmr_cls,
				   "setData", "(Ljava/nio/ByteBuffer;)V"));
  JNI_VAR_CALL(env, japi->rmr_getSizes,
	       (*env)->GetMethodID(env, japi->rmr_cls, "getSizes", "()[J"));
  JNI_VAR_CALL(env, japi->rmr_setSizes,
	       (*env)->GetMethodID(env, japi->rmr_cls, "setSizes", "([J)V"));
  JNI_VAR_CALL(env, japi->rmr_getStrides,
	       (*env)->GetMethodID(env, japi->rmr_cls, "getStrides", "()[J"));
  JNI_VAR_CALL(env, japi->rmr_setStrides,
	       (*env)->GetMethodID(env, japi->rmr_cls, "setStrides", "([J)V"));
  JNI_VAR_CALL(env, japi->rmr_getDataSize,
	       (*env)->GetMethodID(env, japi->rmr_cls, "getDataSize", "()J"));

  return japi;
}

/* Convert Java object to native data structure */
OrderedRtMemRefDict *
ormrd_java_to_native(JNIEnv *env, jclass cls, jobject obj, jniapi_t *japi) {
  /* Get object array "rmrs" and "names" in OrderedRtMemRefDict */
  JNI_TYPE_VAR_CALL(env, jobjectArray, ormrd_rmrs,
		    (*env)->CallObjectMethod(env, obj, japi->ormrd_getRmrs));
  JNI_TYPE_VAR_CALL(env, jobjectArray, ormrd_names,
		    (*env)->CallObjectMethod(env, obj, japi->ormrd_getNames));

  /* Get length of object array "rmrs" and "names" in OrderedRtMemRefDict */
  JNI_TYPE_VAR_CALL(env, jsize, ormrd_rmrs_len, (*env)->GetArrayLength(env, ormrd_rmrs));
  JNI_TYPE_VAR_CALL(env, jsize, ormrd_names_len, (*env)->GetArrayLength(env, ormrd_names));

  /* Allocate memory for holding each Java rmr object and name string,
   * and RtMemRef and char pointers for constructing native RtMemRef and name array
   */
  JNI_COND(jobject *, obj_rmr, malloc(ormrd_rmrs_len  * sizeof(jobject)), NULL,
	   env, japi->ecpt_cls, "obj_rmr=null");
  JNI_COND(jstring *, obj_name, malloc(ormrd_names_len * sizeof(jstring)), NULL,
	   env, japi->ecpt_cls, "obj_name=null");
  JNI_COND(RtMemRef **, jni_rmr, malloc(ormrd_rmrs_len  * sizeof(RtMemRef *)), NULL,
	   env, japi->ecpt_cls, "jni_rmr=null");
  JNI_COND(const char **, jni_name, malloc(ormrd_names_len * sizeof(const char *)), NULL,
	   env, japi->ecpt_cls, "jni_name=null");

  /* Create OrderedRtMemRefDict to be constructed and passed to the model shared library */
  JNI_COND(OrderedRtMemRefDict *, ormrd, createOrderedRtMemRefDict(), NULL,
	   env, japi->ecpt_cls, "ormrd=null");

  /* Loop through all the ormrd_rmrs and ormrd_names */
  for (int i = 0; i < ormrd_rmrs_len; i++) {
    JNI_VAR_CALL(env, obj_rmr[i], (*env)->GetObjectArrayElement(env, ormrd_rmrs, i));
    JNI_VAR_CALL(env, obj_name[i], (*env)->GetObjectArrayElement(env, ormrd_names, i));

    /* Get type, rank, data, sizes, and strides by calling corresponding methods */
    JNI_TYPE_VAR_CALL(env, jint, rmr_type,
		      (*env)->CallIntMethod(env, obj_rmr[i], japi->rmr_getType));
    JNI_TYPE_VAR_CALL(env, jint, rmr_rank,
		      (*env)->CallIntMethod(env, obj_rmr[i], japi->rmr_getRank));
    JNI_TYPE_VAR_CALL(env, jlong, rmr_datasize,
		      (*env)->CallLongMethod(env, obj_rmr[i], japi->rmr_getDataSize));
    JNI_TYPE_VAR_CALL(env, jobject, rmr_data,
		      (*env)->CallObjectMethod(env, obj_rmr[i], japi->rmr_getData));
    JNI_TYPE_VAR_CALL(env, jobject, rmr_sizes,
		      (*env)->CallObjectMethod(env, obj_rmr[i], japi->rmr_getSizes));
    JNI_TYPE_VAR_CALL(env, jobject, rmr_strides,
		      (*env)->CallObjectMethod(env, obj_rmr[i], japi->rmr_getStrides));

    /* Primitive type int and long can be directly used */
    int jni_type = rmr_type, jni_rank = rmr_rank;
    long jni_datasize = rmr_datasize;
    
    /* Get direct buffer associated with data */
    JNI_TYPE_VAR_CALL(env, void *, jni_data, (*env)->GetDirectBufferAddress(env, rmr_data));
    
    /* Get long array associated with sizes and strides */
    JNI_TYPE_VAR_CALL(env, long *, jni_sizes,
		      (*env)->GetLongArrayElements(env, rmr_sizes, NULL));
    JNI_TYPE_VAR_CALL(env, long *, jni_strides,
		      (*env)->GetLongArrayElements(env, rmr_strides, NULL));

    /* Print debug info on what we got from the Java side */
    RMR_DEBUG(i, jni_type, jni_rank, jni_sizes, jni_strides, jni_data, jni_datasize);

    /* Create native RtMemRef struct and fill in its fields */
    jni_rmr[i] = createRtMemRef(jni_rank);
    setDType(jni_rmr[i], jni_type);
    setData(jni_rmr[i], jni_data);
    setSizes(jni_rmr[i], jni_sizes);
    setStrides(jni_rmr[i], jni_strides);
    
    /*jni_name[i] = (*env)->GetStringUTFChars(env, obj_name[i], NULL);
      printf("jni_name=%s\n", jni_name[i]);*/

    /* Install RtMemRef into OrderedRtMemRefDict */
    setRtMemRef(ormrd, i, jni_rmr[i]);

    /* Release reference to the java objects */
    JNI_CALL(env, (*env)->ReleaseLongArrayElements(env, rmr_sizes, jni_sizes, 0));
    JNI_CALL(env, (*env)->ReleaseLongArrayElements(env, rmr_strides, jni_strides, 0));
  }

  /* setRtMemRef(ormrd, jni_rmr, jni_name); */
  return ormrd;
}

/* Convert native data structure to Java object */
jobject
ormrd_native_to_java(JNIEnv *env, jclass cls, OrderedRtMemRefDict *dict, jniapi_t *japi) {
  JNI_COND(int, nrmr, numRtMemRefs(dict), 0, env, japi->ecpt_cls, "nrmr=0");

  /* Create RtMemRef java object array */
  JNI_TYPE_VAR_CALL(env, jobjectArray, rmrs,
		    (*env)->NewObjectArray(env, nrmr, japi->rmr_cls, NULL));

  /* Loop through the native RtMemRef structs */
  for (int i = 0; i < nrmr; i++) {
    JNI_COND(RtMemRef *, rmr, getRtMemRef(dict, i), NULL,
	     env, japi->ecpt_cls, "rmr[%d]=null", i);

    JNI_COND(int, jni_type, getDType(rmr), 0,
	     env, japi->ecpt_cls, "rmr[%d]:type=0", i);
    JNI_COND(int, jni_rank, getRank(rmr), 0,
	     env, japi->ecpt_cls, "rmr[%d]:rank=0", i);
    JNI_COND(long, jni_datasize, getDataSize(rmr), 0,
	     env, japi->ecpt_cls, "rmr[%d]:datasize=0", i);
    JNI_COND(void *, jni_data, getData(rmr), NULL,
	     env, japi->ecpt_cls, "rmr[%d]:data=null", i);
    JNI_COND(long *, jni_sizes, getSizes(rmr), NULL,
	     env, japi->ecpt_cls, "rmr[%d]:sizes=null", i);
    JNI_COND(long *, jni_strides, getStrides(rmr), NULL,
	     env, japi->ecpt_cls, "rmr[%d]:strides=null", i);

    /* Print debug info on what we got from the native side */
    RMR_DEBUG(i, jni_type, jni_rank, jni_sizes, jni_strides, jni_data, jni_datasize);

    /* create the following Java objects:
     *   - RtMemRef
     *   - DirectByteBuffer (from native buffers)
     *   - long array for sizes and strides
     */
    JNI_TYPE_VAR_CALL(env, jobject, obj_rmr,
		      (*env)->NewObject(env, japi->rmr_cls,
					japi->rmr_constructor, jni_rank));
    JNI_TYPE_VAR_CALL(env, jobject, rmr_data,
		      (*env)->NewDirectByteBuffer(env, jni_data,
						  jni_datasize * onnx_type_size[jni_type]));
    JNI_TYPE_VAR_CALL(env, jlongArray, rmr_sizes, (*env)->NewLongArray(env, jni_rank));
    JNI_TYPE_VAR_CALL(env, jlongArray, rmr_strides, (*env)->NewLongArray(env, jni_rank));

    /* Call setType method */
    JNI_CALL(env, (*env)->CallObjectMethod(env, obj_rmr, japi->rmr_setType, jni_type));

    /* Call setData method */
    JNI_CALL(env, (*env)->CallObjectMethod(env, obj_rmr, japi->rmr_setData, rmr_data));

    /* Fill in sizes array from native array and call setSizes method */
    JNI_CALL(env, (*env)->SetLongArrayRegion(env, rmr_sizes, 0, jni_rank, jni_sizes));
    JNI_CALL(env, (*env)->CallObjectMethod(env, obj_rmr, japi->rmr_setSizes, rmr_sizes));

    /* Fill in strides array from native array and call setStrides method */
    JNI_CALL(env, (*env)->SetLongArrayRegion(env, rmr_strides, 0, jni_rank, jni_strides));
    JNI_CALL(env, (*env)->CallObjectMethod(env, obj_rmr,
					   japi->rmr_setStrides, rmr_strides));

    /* Set DynMemRef object in the object array */
    JNI_CALL(env, (*env)->SetObjectArrayElement(env, rmrs, i, obj_rmr));
  }

  /* Create the OrderedRtMemRefDict java object */
  JNI_TYPE_VAR_CALL(env, jobject, ormrd,
	   (*env)->NewObject(env, japi->ormrd_cls, japi->ormrd_constructor, rmrs));

  return ormrd;
}

JNIEXPORT jobject JNICALL
Java_com_ibm_onnxmlir_DynEntryPoint_main_1graph_1jni(JNIEnv *env, jclass cls, jobject obj) {
  CHECK_CALL(jniapi_t *, japi,
	     fill_jniapi(env, &jniapi), NULL);
  
  CHECK_CALL(OrderedRtMemRefDict *, input_ormrd,
	     ormrd_java_to_native(env, cls, obj, japi), NULL);

  CHECK_CALL(OrderedRtMemRefDict *, dict,
	     _dyn_entry_point_main_graph(input_ormrd), NULL);

  CHECK_CALL(jobject, output_ormrd,
	     ormrd_native_to_java(env, cls, dict, japi), NULL);

  return output_ormrd;
}
