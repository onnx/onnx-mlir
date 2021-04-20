# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_interface(ShapeInferenceOpInterface)

add_library(OMShapeInferenceOpInterface
        ShapeInferenceOpInterface.hpp
        ShapeInferenceOpInterface.cpp)
target_include_directories(OMShapeInferenceOpInterface
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMShapeInferenceOpInterface OMShapeInferenceOpInterfaceIncGen)

add_onnx_mlir_interface(ResultTypeInferenceOpInterface)

add_library(OMResultTypeInferenceOpInterface
        ResultTypeInferenceOpInterface.hpp
        ResultTypeInferenceOpInterface.cpp)
target_include_directories(OMResultTypeInferenceOpInterface
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMResultTypeInferenceOpInterface
        OMResultTypeInferenceOpInterfaceIncGen)

add_onnx_mlir_interface(HasOnnxSubgraphOpInterface)

add_library(OMHasOnnxSubgraphOpInterface
        HasOnnxSubgraphOpInterface.hpp
        HasOnnxSubgraphOpInterface.cpp
        )
target_include_directories(OMHasOnnxSubgraphOpInterface
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMHasOnnxSubgraphOpInterface
        OMHasOnnxSubgraphOpInterfaceIncGen)

add_onnx_mlir_interface(SpecializedKernelOpInterface)

add_library(OMSpecializedKernelOpInterface
        SpecializedKernelOpInterface.hpp
        SpecializedKernelOpInterface.cpp
        )
target_include_directories(OMSpecializedKernelOpInterface
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMSpecializedKernelOpInterface
        OMSpecializedKernelOpInterfaceIncGen)

install(TARGETS OMShapeInferenceOpInterface          DESTINATION lib)
install(TARGETS OMResultTypeInferenceOpInterface     DESTINATION lib)
install(TARGETS OMHasOnnxSubgraphOpInterface         DESTINATION lib)
install(TARGETS OMSpecializedKernelOpInterface       DESTINATION lib)