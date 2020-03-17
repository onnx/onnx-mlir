set(LLVM_TARGET_DEFINITIONS promotable_const_operands.td)
onnx_mlir_tablegen(promotable_const_operands.hpp.inc -gen-op-interface-decls)
onnx_mlir_tablegen(promotable_const_operands.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(gen_promotable_const_operands_interface)

add_library(promotable_const_operands_interface
        promotable_const_operands.hpp
        promotable_const_operands.cpp)
target_include_directories(promotable_const_operands_interface
        PRIVATE ${ONNX_MLIR_SRC_ROOT} ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})

add_dependencies(promotable_const_operands_interface
        gen_promotable_const_operands_interface)