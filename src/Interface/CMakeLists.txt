# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_interface(ShapeInferenceOpInterface)

add_onnx_mlir_library(OMShapeInferenceOpInterface
        ShapeInferenceOpInterface.cpp)
add_dependencies(OMShapeInferenceOpInterface OMShapeInferenceOpInterfaceIncGen)

add_onnx_mlir_interface(ResultTypeInferenceOpInterface)

add_onnx_mlir_library(OMResultTypeInferenceOpInterface
        ResultTypeInferenceOpInterface.cpp)
add_dependencies(OMResultTypeInferenceOpInterface
        OMResultTypeInferenceOpInterfaceIncGen)

add_onnx_mlir_interface(HasOnnxSubgraphOpInterface)

add_onnx_mlir_library(OMHasOnnxSubgraphOpInterface
        HasOnnxSubgraphOpInterface.cpp
        )
add_dependencies(OMHasOnnxSubgraphOpInterface
        OMHasOnnxSubgraphOpInterfaceIncGen)

add_onnx_mlir_interface(SpecializedKernelOpInterface)

add_onnx_mlir_library(OMSpecializedKernelOpInterface
        SpecializedKernelOpInterface.cpp
        )
add_dependencies(OMSpecializedKernelOpInterface
        OMSpecializedKernelOpInterfaceIncGen)
