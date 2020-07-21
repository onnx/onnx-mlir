add_subdirectory(jni)

# Create static libcruntime.a to be embedded in model.so to make model.so self contained.
# However, by default object code for static library is not compiled with -fPIC. Embedding
# such static library in a shared library can cause runtime failure on some architectures,
# such as z. So we override the default and explicitly compile with -fPIC.
add_library(cruntime STATIC
        RtMemRef.cpp
        RtMemRef.h
        DataType.h)
set_target_properties(cruntime PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE)

add_library(RtMemRefUtils
        RtMemRef.h
        RtMemRef.cpp
        DataType.h)
set_target_properties(RtMemRefUtils PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE)

add_library(ExecutionSession
        ExecusionSession.hpp
        ExecusionSession.cpp)
target_link_libraries(ExecutionSession
        ${CMAKE_DL_LIBS})
target_include_directories(ExecutionSession PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
set_target_properties(ExecutionSession PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE)

pybind11_add_module(PyRuntime
        PyExecutionSession.cpp
        PyExecutionSession.hpp)
target_link_libraries(PyRuntime PRIVATE
        ${CMAKE_DL_LIBS}
        ExecutionSession
        RtMemRefUtils
        onnx)
target_include_directories(PyRuntime PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})

# See comments above about libcruntime.a
add_library(EmbeddedDataLoader STATIC
        GetEmbeddedConstPool.h
        GetEmbeddedConstPool.cpp)
set_target_properties(EmbeddedDataLoader PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE)

add_dependencies(PyRuntime cruntime)
install(FILES RtMemRef.h DESTINATION include)
install(TARGETS cruntime DESTINATION lib)
install(TARGETS EmbeddedDataLoader DESTINATION lib)
