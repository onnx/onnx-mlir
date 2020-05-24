set(LLVM_TARGET_DEFINITIONS PromotableConstOperandsOpInterface.td)
onnx_mlir_tablegen(PromotableConstOperandsOpInterface.hpp.inc -gen-op-interface-decls)
onnx_mlir_tablegen(PromotableConstOperandsOpInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(OMPromotableConstOperandsOpInterfaceIncGen)

add_library(OMPromotableConstOperandsOpInterface
        PromotableConstOperandsOpInterface.hpp
        PromotableConstOperandsOpInterface.cpp)
target_include_directories(OMPromotableConstOperandsOpInterface
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMPromotableConstOperandsOpInterface
        OMPromotableConstOperandsOpInterfaceIncGen)


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

set(LLVM_TARGET_DEFINITIONS ResultTypeInferenceInterface.td)
onnx_mlir_tablegen(ResultTypeInferenceInterface.hpp.inc -gen-op-interface-decls)
onnx_mlir_tablegen(ResultTypeInferenceInterface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(OMResultTypeInferenceInterfaceIncGen)

add_library(OMResultTypeInferenceInterface
        ResultTypeInferenceInterface.hpp
        ResultTypeInferenceInterface.cpp)
target_include_directories(OMResultTypeInferenceInterface
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_dependencies(OMResultTypeInferenceInterface
        OMResultTypeInferenceInterfaceIncGen)
