#include "com_ibm_onnxmlir_DynEntryPoint.h"

/* Dummy routine to force the link editor to embed code in libjniruntime.a
   into libmodel.so */
void __dummy_do_not_call__(JNIEnv *env, jclass cls, jobject obj) {
  Java_com_ibm_onnxmlir_DynEntryPoint_main_1graph_1jni(NULL, NULL, NULL);
}
