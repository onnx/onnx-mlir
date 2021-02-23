# SPDX-License-Identifier: Apache-2.0

find_package(Python3 REQUIRED COMPONENTS Interpreter)

set(LLVM_LIT ${LLVM_PROJ_SRC}/utils/lit/lit.py)
set(LLVM_DEFAULT_EXTERNAL_LIT ${LLVM_PROJ_BUILD}/bin/llvm-lit)

configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
                       ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
                       MAIN_CONFIG
                       ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(ONNX_MLIR_TEST_DEPENDS onnx-mlir-opt binary-decoder)

add_lit_testsuite(check-onnx-lit
                  "Running the ONNX MLIR regression tests"
                  ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS
                  ${ONNX_MLIR_TEST_DEPENDS})
set_target_properties(check-onnx-lit PROPERTIES FOLDER "Tests")

add_lit_testsuites(ONNX_MLIR
                   ${CMAKE_CURRENT_SOURCE_DIR}
                   DEPENDS
                   ${ONNX_MLIR_TEST_DEPS})
