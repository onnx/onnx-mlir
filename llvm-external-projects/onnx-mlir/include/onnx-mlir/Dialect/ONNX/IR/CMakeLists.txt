

function(_add_dialect_td_gen)
  set(LLVM_TARGET_DEFINITIONS ONNX.td)
  # mlir_tablegen(OnnxAttrs.h.inc -gen-attrdef-decls)
  # mlir_tablegen(OnnxAttrs.cpp.inc -gen-attrdef-defs)
  # mlir_tablegen(OnnxEnums.h.inc -gen-enum-decls)
  # mlir_tablegen(OnnxEnums.cpp.inc -gen-enum-defs)
  mlir_tablegen(ONNXOps.hpp.inc -gen-op-decls)
  mlir_tablegen(ONNXOps.cpp.inc -gen-op-defs)
  mlir_tablegen(ONNXTypes.hpp.inc -gen-typedef-decls)
  mlir_tablegen(ONNXTypes.cpp.inc -gen-typedef-defs)
  mlir_tablegen(ONNXDialect.hpp.inc --gen-dialect-decls
                --dialect=onnx)
  mlir_tablegen(ONNXDialect.cpp.inc --gen-dialect-defs
                --dialect=onnx)
  add_public_tablegen_target(ONNXOnnxIncGen)
  add_dependencies(mlir-headers ONNXOnnxIncGen)
endfunction()

# You need these functions because the LLVM macros depend on the scoping in
# order to create dependencies, and without the function scopes you'll end up
# with a dependency cycle.
_add_dialect_td_gen()
# _add_dialect_td_gen_interfaces()
