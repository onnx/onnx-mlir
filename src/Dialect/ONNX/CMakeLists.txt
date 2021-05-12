# SPDX-License-Identifier: Apache-2.0

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "s390x")
  # currently needed for a bug on Z for the ONNXOps.cpp.inc file
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O2" "-O1")
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O3" "-O1")
endif()

add_onnx_mlir_dialect(ONNXOps)

# Header dependencies target for ONNXOps.hpp
add_custom_target(OMONNXOpsInc
        DEPENDS OMONNXOpsIncGen
                OMResultTypeInferenceOpInterfaceIncGen
                OMShapeInferenceOpInterfaceIncGen
                OMHasOnnxSubgraphOpInterfaceIncGen)

add_library(OMONNXOps
        ONNXOps.cpp
        ONNXOps.hpp
        ONNXOpsHelper.cpp
        ONNXOpsHelper.hpp
        ONNXShapeHelper.cpp
        ONNXShapeHelper.hpp
        IndexExpr.hpp
        IndexExpr.cpp
        IndexExprDetail.hpp
        IndexExprDetail.cpp)

target_include_directories(OMONNXOps
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})

add_dependencies(OMONNXOps OMONNXOpsInc)

# Linking dependencies:
add_dependencies(OMONNXOps
        OMResultTypeInferenceOpInterface
        OMShapeInferenceOpInterface)

add_onnx_mlir_dialect_doc(onnx ONNXOps.td)

install(TARGETS OMONNXOps DESTINATION lib)
