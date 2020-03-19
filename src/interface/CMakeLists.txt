set(LLVM_TARGET_DEFINITIONS promotable_const_operands_op_interface.td)
onnx_mlir_tablegen(promotable_const_operands_op_interface.hpp.inc -gen-op-interface-decls)
onnx_mlir_tablegen(promotable_const_operands_op_interface.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(onnx_mlir_gen_promotable_const_operands_op_interface)

add_library(onnx_mlir_promotable_const_operands_op_interface
        promotable_const_operands_op_interface.hpp
        promotable_const_operands_op_interface.cpp)
target_include_directories(onnx_mlir_promotable_const_operands_op_interface
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})

add_dependencies(onnx_mlir_promotable_const_operands_op_interface
        onnx_mlir_gen_promotable_const_operands_op_interface)