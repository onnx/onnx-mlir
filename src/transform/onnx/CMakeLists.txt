add_library(onnx_mlir_attribute_promotion
        attribute_promotion.cpp)
target_include_directories(onnx_mlir_attribute_promotion
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNF_MLIR_SRC_ROOT})
target_link_libraries(onnx_mlir_attribute_promotion
        promotable_const_operands_op_interface)