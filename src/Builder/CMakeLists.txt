add_library(OMBuilder
        FrontendDialectHelper.cpp
        FrontendDialectHelper.hpp
        FrontendDialectTransformer.cpp
        FrontendDialectTransformer.hpp)

target_include_directories(OMBuilder PRIVATE ${ONNX_MLIR_SRC_ROOT})
target_include_directories(OMBuilder PRIVATE ${CMAKE_BINARY_DIR})
target_include_directories(OMBuilder PRIVATE ${ONNX_MLIR_BIN_ROOT})

if (INCLUDE_ONNX_ML)
  add_definitions(-DINCLUDE_ONNX__ML=1)
endif()

# This will cause onnx to be built. More importantly, some variable definitions
# when building onnx such as -DONNX_ML=1 -DONNX_NAMESPACE=onnx will be carried over
# when compiling FrontendDialectHelper.cpp, etc.
target_link_libraries(OMBuilder
        onnx
        mpark_variant)
target_include_directories(OMBuilder
                           PUBLIC
                           ${ONNX_MLIR_SRC_ROOT}/third_party/onnx
                           ${ONNX_MLIR_SRC_ROOT}/third_party/variant
                           ${ONNX_MLIR_SRC_ROOT})

# If you add onnx here, it will also cause onnx to be built. However, some
# variable definitions when building onnx such as -DONNX_ML=1 -DONNX_NAMESPACE=onnx
# will NOT be carried over when compiling FrontendDialectHelper.cpp, etc. so
# the compilation will fail.
add_dependencies(OMBuilder OMONNXOpsInc)
add_dependencies(OMBuilder OMONNXOps OMResultTypeInferenceOpInterface)
