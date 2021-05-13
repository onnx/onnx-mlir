# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMBuilder
        FrontendDialectHelper.cpp
        FrontendDialectTransformer.cpp)

if (INCLUDE_ONNX_ML)
  add_definitions(-DINCLUDE_ONNX__ML=1)
endif()

# This will cause onnx to be built. More importantly, some variable definitions
# when building onnx such as -DONNX_ML=1 -DONNX_NAMESPACE=onnx will be carried over
# when compiling FrontendDialectHelper.cpp, etc.
target_link_libraries(OMBuilder
        onnx)

# If you add onnx here, it will also cause onnx to be built. However, some
# variable definitions when building onnx such as -DONNX_ML=1 -DONNX_NAMESPACE=onnx
# will NOT be carried over when compiling FrontendDialectHelper.cpp, etc. so
# the compilation will fail.
add_dependencies(OMBuilder OMONNXOpsInc)
add_dependencies(OMBuilder OMONNXOps OMResultTypeInferenceOpInterface)
add_dependencies(OMBuilder OMONNXOps OMHasOnnxSubgraphOpInterface)
