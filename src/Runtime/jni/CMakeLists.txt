# SPDX-License-Identifier: Apache-2.0

# Don't require AWT to allow headless JDK to reduce docker image size
set(JAVA_AWT_INCLUDE_PATH headless)
set(JAVA_AWT_LIBRARY headless)
find_package(Java COMPONENTS Development)
find_package(JNI)

if(Java_Development_FOUND AND JNI_FOUND)
  include(UseJava)

  # Target for Java runtime jar
  add_jar(javaruntime
          src/com/ibm/onnxmlir/OMModel.java
          src/com/ibm/onnxmlir/OMTensorList.java
          src/com/ibm/onnxmlir/OMTensor.java)

  # ONNX_MLIR_LIBRARY_PATH is a generator expression which is not supported by add_jar as the output
  # directory. Instead, we let add_jar place the jar file in the default location and copy it to the
  # library path after it is built.
  add_custom_command(TARGET javaruntime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different $<TARGET_PROPERTY:javaruntime,JAR_FILE> ${ONNX_MLIR_LIBRARY_PATH})

  # Target for JNI runtime lib
  add_onnx_mlir_library(jniruntime
          EXCLUDE_FROM_OM_LIBS
          STATIC
          jniwrapper.c jnilog.c jnidummy.c
          com_ibm_onnxmlir_OMModel.h jnilog.h)
  set_target_properties(jniruntime PROPERTIES
          POSITION_INDEPENDENT_CODE TRUE)
  target_include_directories(jniruntime PRIVATE
          ${ONNX_MLIR_SRC_ROOT}/include
          ${JAVA_INCLUDE_PATH}
          ${JAVA_INCLUDE_PATH2})

  install_jar(javaruntime DESTINATION lib)

else()
  message(WARNING "Java Development component or JNI not found, JNI targets will not work")
endif()
