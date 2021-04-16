# SPDX-License-Identifier: Apache-2.0

# Set LLVM_DEFAULT_EXTERNAL_LIT to an empty string to avoid warnings about the path
# when using multi-config generators such as VS or Xcode
set(LLVM_DEFAULT_EXTERNAL_LIT "")
set(LLVM_LIT_OUTPUT_DIR ${LLVM_PROJ_BIN})

# For multi-config generators, this will replace $(Configuration) (etc.) with
# %(build_mode)s which will in turn be replaced with the correct build mode at
# run time.
set_llvm_build_mode()
string(REPLACE "${CMAKE_CFG_INTDIR}" "${LLVM_BUILD_MODE}" ONNX_MLIR_TOOLS_DIR "${ONNX_MLIR_RUNTIME_PATH}")

configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
                       ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
                       MAIN_CONFIG
                       ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(ONNX_MLIR_TEST_DEPENDS
  binary-decoder
  onnx-mlir
  onnx-mlir-opt
  )

add_lit_testsuite(check-onnx-lit
                  "Running the ONNX MLIR regression tests"
                  ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS
                  ${ONNX_MLIR_TEST_DEPENDS})
set_target_properties(check-onnx-lit PROPERTIES FOLDER "Tests")

add_lit_testsuites(ONNX_MLIR
                   ${CMAKE_CURRENT_SOURCE_DIR}
                   DEPENDS
                   ${ONNX_MLIR_TEST_DEPENDS})
