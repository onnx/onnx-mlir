# SPDX-License-Identifier: Apache-2.0

add_subdirectory(jni)

# Create static libcruntime.a to be embedded in model.so to make model.so self contained.
# However, by default object code for static library is not compiled with -fPIC. Embedding
# such static library in a shared library can cause runtime failure on some architectures,
# such as z. So we override the default and explicitly compile with -fPIC.
add_onnx_mlir_library(cruntime
        EXCLUDE_FROM_OM_LIBS
        STATIC
        OMTensor.c
        OMTensor.inc
        OMTensorList.c
        OMTensorList.inc
        OnnxDataType.cpp)
set_target_properties(cruntime PROPERTIES
        LANGUAGE C)
set_target_properties(cruntime PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE)
target_include_directories(cruntime PRIVATE
        ${ONNX_MLIR_SRC_ROOT}/include)

add_onnx_mlir_library(OMTensorUtils
        EXCLUDE_FROM_OM_LIBS
        OMTensor.cpp
        OMTensor.inc
        OMTensorList.cpp
        OMTensorList.inc
        OnnxDataType.cpp)
set_target_properties(OMTensorUtils PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE)
target_compile_definitions(OMTensorUtils PRIVATE RTMEMREF_INTERNAL_API)
target_include_directories(OMTensorUtils PRIVATE
        ${ONNX_MLIR_SRC_ROOT}/include)

add_onnx_mlir_library(ExecutionSession
        EXCLUDE_FROM_OM_LIBS
        ExecutionSession.cpp)
target_include_directories(ExecutionSession PRIVATE
        ${ONNX_MLIR_SRC_ROOT}/src/Runtime
        ${ONNX_MLIR_SRC_ROOT}/include)
target_link_libraries(ExecutionSession
        ${CMAKE_DL_LIBS})
set_target_properties(ExecutionSession PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE)

pybind11_add_module(PyRuntime
        PyExecutionSession.cpp
        PyExecutionSession.hpp)
target_include_directories(PyRuntime PRIVATE
        ${ONNX_MLIR_SRC_ROOT}/src/Runtime
        ${ONNX_MLIR_SRC_ROOT}/include)
target_link_libraries(PyRuntime PRIVATE
        ${CMAKE_DL_LIBS}
        ExecutionSession
        OMTensorUtils
        onnx)

add_dependencies(PyRuntime cruntime)
