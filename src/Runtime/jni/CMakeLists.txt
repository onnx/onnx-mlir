# SPDX-License-Identifier: Apache-2.0

if (ONNX_MLIR_ENABLE_JNI)
  option(ONNX_MLIR_BUILD_JSONITER "Build jsoniter locally" OFF)

  # Setup Jsoniter jar
  include(Jsoniter.cmake)
  setup_jsoniter(0.9.23)

  # Build javaruntime jar
  add_jar(javaruntime
    src/com/ibm/onnxmlir/OMModel.java
    src/com/ibm/onnxmlir/OMLogger.java
    src/com/ibm/onnxmlir/OMRunner.java
    src/com/ibm/onnxmlir/OMTensorList.java
    src/com/ibm/onnxmlir/OMTensor.java

    INCLUDE_JARS ${JSONITER_JAR}
    )
  if (TARGET jsoniter)
    add_dependencies(javaruntime jsoniter)
  endif()
  install_jar(javaruntime DESTINATION lib)

  # ONNX_MLIR_LIBRARY_PATH is a generator expression which is not supported by
  # add_jar as the output directory. Instead, we let add_jar place the jar file
  # in the default location and copy it to the library path after it is built.
  #
  # Note target must be ${ONNX_MLIR_LIBRARY_PATH}/javaruntime.jar instead of
  # just ${ONNX_MLIR_LIBRARY_PATH} for Windows CI.
  add_custom_command(TARGET javaruntime POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            $<TARGET_PROPERTY:javaruntime,JAR_FILE>
            ${ONNX_MLIR_LIBRARY_PATH}/javaruntime.jar)

  # Build jniruntime lib
  add_onnx_mlir_library(jniruntime STATIC
    jniwrapper.c
    jnilog.c
    jnidummy.c

    EXCLUDE_FROM_OM_LIBS

    INCLUDE_DIRS PUBLIC
    ${ONNX_MLIR_SRC_ROOT}/include
    ${JAVA_INCLUDE_PATH}
    ${JAVA_INCLUDE_PATH2}
    )

  set_target_properties(jniruntime
    PROPERTIES
    POSITION_INDEPENDENT_CODE TRUE
    )

endif()
