
set(LLVM_TARGET_DEFINITIONS ONNXOpInterfaces.td)
mlir_tablegen(ONNXOpInterfaces.hpp.inc -gen-op-interface-decls)
mlir_tablegen(ONNXOpInterfaces.cpp.inc -gen-op-interface-defs)
add_public_tablegen_target(ONNXOnnxInterfacesIncGen)
add_dependencies(ONNXOnnxIncGen ONNXOnnxInterfacesIncGen)

