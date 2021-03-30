# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS ShapeInferenceInterface.td)
onnx_mlir_tablegen(ShapeInference.hpp.inc -gen-op-interface-decls)
onnx_mlir_tablegen(ShapeInference.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ShapeInferenceOpInterfaceIncGen)

add_library(OMShapeInferenceOpInterface
        ShapeInferenceInterface.hpp
        ShapeInferenceInterface.cpp)
target_include_directories(OMShapeInferenceOpInterface
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMShapeInferenceOpInterface ShapeInferenceOpInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS ResultTypeInferenceOpInterface.td)
onnx_mlir_tablegen(ResultTypeInferenceOpInterface.hpp.inc -gen-op-interface-decls)
onnx_mlir_tablegen(ResultTypeInferenceOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(OMResultTypeInferenceOpInterfaceIncGen)

add_library(OMResultTypeInferenceOpInterface
        ResultTypeInferenceOpInterface.hpp
        ResultTypeInferenceOpInterface.cpp)
target_include_directories(OMResultTypeInferenceOpInterface
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMResultTypeInferenceOpInterface
        OMResultTypeInferenceOpInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS HasOnnxSubgraphOpInterface.td)
onnx_mlir_tablegen(HasOnnxSubgraphOpInterface.hpp.inc -gen-op-interface-decls)
onnx_mlir_tablegen(HasOnnxSubgraphOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(OMHasOnnxSubgraphOpInterfaceIncGen)

add_library(OMHasOnnxSubgraphOpInterface
        HasOnnxSubgraphOpInterface.hpp
        HasOnnxSubgraphOpInterface.cpp
        )
target_include_directories(OMHasOnnxSubgraphOpInterface
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMHasOnnxSubgraphOpInterface
        OMHasOnnxSubgraphOpInterfaceIncGen)

set(LLVM_TARGET_DEFINITIONS SpecializedKernelOpInterface.td)
onnx_mlir_tablegen(SpecializedKernelOpInterface.hpp.inc -gen-op-interface-decls)
onnx_mlir_tablegen(SpecializedKernelOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(OMSpecializedKernelOpInterfaceIncGen)

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