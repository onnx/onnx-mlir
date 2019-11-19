set(LLVM_LIT ${LLVM_SRC}/utils/lit/lit.py)
set(LLVM_DEFAULT_EXTERNAL_LIT ${LLVM_BUILD}/bin/llvm-lit)

configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
                       ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
                       MAIN_CONFIG
                       ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(ONNF_MLIR_TEST_DEPENDS onnf-opt)

add_lit_testsuite(check-mlir-lit
                  "Running the ONNF MLIR regression tests"
                  ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS
                  ${ONNF_MLIR_TEST_DEPENDS})
set_target_properties(check-mlir-lit PROPERTIES FOLDER "Tests")

add_lit_testsuites(ONNF_MLIR
                   ${CMAKE_CURRENT_SOURCE_DIR}
                   DEPENDS
                   ${ONNF_MLIR_TEST_DEPS})
