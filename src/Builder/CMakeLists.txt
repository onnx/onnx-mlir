add_library(builder
        FrontendDialectHelper.cpp
        FrontendDialectHelper.hpp
        FrontendDialectTransformer.cpp
        FrontendDialectTransformer.hpp
        OpBuildTable.inc)

target_include_directories(builder PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(builder PRIVATE ${CMAKE_BINARY_DIR})

target_link_libraries(builder compiler onnx ${MLIRLibs} curses mpark_variant)
target_include_directories(builder
                           PUBLIC
                           ${ONNX_MLIR_SRC_ROOT}/third_party/onnx
                           ${ONNX_MLIR_SRC_ROOT}/third_party/variant
                           ${ONNX_MLIR_SRC_ROOT})
