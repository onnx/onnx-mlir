# Create shared libcruntime.so since model.so linkage for backend tests
# will fail on x86 Linux if cruntime is statically linked.
add_library(cruntime
        DynMemRef.cpp
        DynMemRef.h
        DataType.h)
target_include_directories(cruntime
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})

pybind11_add_module(pyruntime
        DynMemRef.cpp
        DynMemRef.h
        Runtime.cpp
        Runtime.hpp)
target_link_libraries(pyruntime PRIVATE ${CMAKE_DL_LIBS})
target_include_directories(pyruntime
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(pyruntime cruntime)

add_library(EmbeddedDataLoader OBJECT
        EmbeddedDataLoader.h
        EmbeddedDataLoader.cpp)