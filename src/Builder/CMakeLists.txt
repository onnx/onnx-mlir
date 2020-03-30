add_library(OMBuilder
        FrontendDialectHelper.cpp
        FrontendDialectHelper.hpp
        FrontendDialectTransformer.cpp
        FrontendDialectTransformer.hpp
        OpBuildTable.inc)
add_dependencies(OMBuilder OMONNXOpsBuildTableIncGen)

target_include_directories(OMBuilder PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(OMBuilder PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(OMBuilder PRIVATE ${ONNX_MLIR_BIN_ROOT})

target_link_libraries(OMBuilder
        OMONNXOps
        onnx
        ${MLIRLibs}
        curses
        mpark_variant)
target_include_directories(OMBuilder
                           PUBLIC
                           ${ONNX_MLIR_SRC_ROOT}/third_party/onnx
                           ${ONNX_MLIR_SRC_ROOT}/third_party/variant
                           ${ONNX_MLIR_SRC_ROOT})
