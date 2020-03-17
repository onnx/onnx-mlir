#add_library(onnf_fold_const_operand_to_attribute fold_const_operand_to_attribute.hpp)
#target_include_directories(onnf_fold_const_operand_to_attribute
#        PRIVATE ${ONNF_SRC_ROOT} ${ONNF_BIN_ROOT})
#target_link_libraries(onnf_fold_const_operand_to_attribute ${MLIRLibs})

set(LLVM_TARGET_DEFINITIONS const_operands_interface.td)
onnf_tablegen(const_operands_interface.hpp.inc -gen-op-interface-decls)
onnf_tablegen(const_operands_interface.cpp.inc -gen-op-interface-defs)
message("TBLGENOUT" ${TABLEGEN_OUTPUT})
add_public_tablegen_target(gen_const_operands_interface)
add_dependencies(compiler gen_const_operands_interface)