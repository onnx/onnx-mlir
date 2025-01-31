# SPDX-License-Identifier: Apache-2.0

# Turn on lit test for specified accelerators even if the
# accelerator code itself cannot be built.
if (ONNX_MLIR_ACCELERATORS)
  foreach(t ${ONNX_MLIR_ACCELERATORS})
    string(TOUPPER ${t} T)
    set(${T}_LIT_ENABLED 1)
    list(APPEND ACCEL_LIT_LIST "${t}")
  endforeach(t)
endif(ONNX_MLIR_ACCELERATORS)
if (ACCEL_LIT_LIST)
  message(STATUS "Accelerator lit tests    : ${ACCEL_LIT_LIST}")
else()
  message(STATUS "Accelerator lit tests    : NONE")
endif()

if (ONNX_MLIR_ENABLE_STABLEHLO)
  set(ONNX_MLIR_STABLEHLO_ENABLED 1)
else()
  set(ONNX_MLIR_STABLEHLO_ENABLED 0)
endif()

# Set LLVM_DEFAULT_EXTERNAL_LIT to an empty string to avoid warnings about the path
# when using multi-config generators such as VS or Xcode
set(LLVM_DEFAULT_EXTERNAL_LIT "")
set(LLVM_LIT_OUTPUT_DIR ${LLVM_TOOLS_BINARY_DIR})

# CMAKE_CFG_INTDIR is . for single-config generators such as make, and
# it has a value (e.g. $(Configuration)) otherwise, so we can use it to
# determine whether we are dealing with a multi-config generator.
# For multi-config generators, this will replace $<CONFIG> with %(build_mode)s
# which will in turn be replaced with the correct build mode at run time.
# For single-config generators, this will replace $<CONFIG> with the correct
# build mode at configuration time.
# Both need to be done because configure_lit_site_cfg does not support generator
# expressions.
if (NOT "${CMAKE_CFG_INTDIR}" STREQUAL ".")
  set_llvm_build_mode()
  string(REPLACE "$<CONFIG>" "${LLVM_BUILD_MODE}" ONNX_MLIR_TOOLS_DIR "${ONNX_MLIR_RUNTIME_PATH}")
else()
  string(REPLACE "$<CONFIG>" "${CMAKE_BUILD_TYPE}" ONNX_MLIR_TOOLS_DIR "${ONNX_MLIR_RUNTIME_PATH}")
endif()

configure_lit_site_cfg(${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
                       ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
                       MAIN_CONFIG
                       ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py)

set(ONNX_MLIR_TEST_DEPENDS
  onnx-mlir
  onnx-mlir-opt
  FileCheck
  count
  not
  split-file
  llvm-dis
  binary-decoder
  )

add_lit_testsuite(check-onnx-lit
                  "Running the ONNX-MLIR regression tests"
                  ${CMAKE_CURRENT_BINARY_DIR}
                  DEPENDS
                  ${ONNX_MLIR_TEST_DEPENDS})
set_target_properties(check-onnx-lit PROPERTIES FOLDER "Tests")

add_lit_testsuites(ONNX_MLIR
                   ${CMAKE_CURRENT_SOURCE_DIR}
                   DEPENDS
                   ${ONNX_MLIR_TEST_DEPENDS})
