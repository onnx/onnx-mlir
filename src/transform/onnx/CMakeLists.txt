add_library(attribute_promotion
        attribute_promotion.cpp)
target_include_directories(attribute_promotion
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNF_MLIR_SRC_ROOT})
target_link_libraries(attribute_promotion
        promotable_const_operands_interface)