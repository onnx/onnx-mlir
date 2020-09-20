add_subdirectory(jni)

# Create static libcruntime.a to be embedded in model.so to make model.so self contained.
# However, by default object code for static library is not compiled with -fPIC. Embedding
# such static library in a shared library can cause runtime failure on some architectures,
# such as z. So we override the default and explicitly compile with -fPIC.
add_library(cruntime STATIC
        include/OnnxMlirRuntime.h
        include/OnnxMlirRuntime/OnnxDataType.h
        RtMemRef.c)
set_target_properties(cruntime PROPERTIES
        LANGUAGE C)
set_target_properties(cruntime PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE)
target_include_directories(cruntime PRIVATE
        ${ONNX_MLIR_SRC_ROOT}/src/Runtime
        ${ONNX_MLIR_SRC_ROOT}/src/Runtime/include)

add_library(RtMemRefUtils
        RtMemRef.cpp
        include/OnnxMlirInternal.h
        include/OnnxMlirRuntime/OnnxDataType.h)
set_target_properties(RtMemRefUtils PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE)
target_compile_definitions(RtMemRefUtils PRIVATE RTMEMREF_INTERNAL_API)
target_include_directories(RtMemRefUtils PRIVATE
        ${ONNX_MLIR_SRC_ROOT}/src/Runtime
        ${ONNX_MLIR_SRC_ROOT}/src/Runtime/include)

add_library(ExecutionSession
        ExecutionSession.hpp
        ExecutionSession.cpp)
target_include_directories(ExecutionSession PRIVATE
        ${ONNX_MLIR_SRC_ROOT}/src/Runtime
        ${ONNX_MLIR_SRC_ROOT}/src/Runtime/include)
target_link_libraries(ExecutionSession
        ${CMAKE_DL_LIBS})
set_target_properties(ExecutionSession PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE)

pybind11_add_module(PyRuntime
        PyExecutionSession.cpp
        PyExecutionSession.hpp)
target_include_directories(PyRuntime PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_SRC_ROOT}/src/Runtime
        ${ONNX_MLIR_SRC_ROOT}/src/Runtime/include)
target_link_libraries(PyRuntime PRIVATE
        ${CMAKE_DL_LIBS}
        ExecutionSession
        RtMemRefUtils
        onnx)

# See comments above about libcruntime.a
add_library(EmbeddedDataLoader STATIC
        GetEmbeddedConstPool.h
        GetEmbeddedConstPool.cpp)
set_target_properties(EmbeddedDataLoader PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE)

add_dependencies(PyRuntime cruntime)
install(FILES include/OnnxMlirRuntime.h DESTINATION include)
install(FILES include/OnnxMlirRuntime/OnnxDataType.h DESTINATION include/OnnxMlirRuntime)
install(FILES include/OnnxMlirRuntime/OnnxDataTypeMetaData.inc DESTINATION include/OnnxMlirRuntime)

install(TARGETS cruntime DESTINATION lib)
install(TARGETS EmbeddedDataLoader DESTINATION lib)

add_subdirectory(include/Test)