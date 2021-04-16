# SPDX-License-Identifier: Apache-2.0

# Don't require AWT header to allow headless JDK to reduce docker image size
set(JAVA_AWT_INCLUDE_PATH none)
find_package(Java COMPONENTS Development)
find_package(JNI)

if(Java_Development_FOUND AND JNI_FOUND)
  include(UseJava)

  # Target for Java runtime jar
  add_jar(javaruntime
          src/com/ibm/onnxmlir/OMModel.java
          src/com/ibm/onnxmlir/OMTensorList.java
          src/com/ibm/onnxmlir/OMTensor.java
          OUTPUT_DIR ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

  # Target for JNI runtime lib
  add_library(jniruntime STATIC
          jniwrapper.c jnilog.c jnidummy.c
          com_ibm_onnxmlir_OMModel.h jnilog.h)
  set_target_properties(jniruntime PROPERTIES
          POSITION_INDEPENDENT_CODE TRUE)
  target_include_directories(jniruntime PRIVATE
          ${ONNX_MLIR_SRC_ROOT}/include
          ${JAVA_INCLUDE_PATH}
          ${JAVA_INCLUDE_PATH2})

  install_jar(javaruntime DESTINATION lib)
  install(TARGETS jniruntime DESTINATION lib)

else()
  message(WARNING "Java Development component or JNI not found, JNI targets will not work")
endif()
