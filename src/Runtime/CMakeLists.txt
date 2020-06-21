# Create shared libcruntime.so since model.so linkage for backend tests
# will fail on x86 Linux if cruntime is statically linked.
add_library(cruntime STATIC
        DynMemRef.cpp
        DynMemRef.h
        DataType.h)

add_library(DynMemRefUtils
        DynMemRef.h
        DynMemRef.cpp
        DataType.h)

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
        DynMemRefUtils
        onnx)
target_include_directories(PyRuntime PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})

add_library(EmbeddedDataLoader STATIC
        GetEmbeddedConstPool.h
        GetEmbeddedConstPool.cpp)
set_target_properties(EmbeddedDataLoader PROPERTIES
        POSITION_INDEPENDENT_CODE TRUE)

add_dependencies(PyRuntime cruntime)
install(FILES DynMemRef.h DESTINATION include)
install(TARGETS cruntime DESTINATION lib)
install(TARGETS EmbeddedDataLoader DESTINATION lib)
