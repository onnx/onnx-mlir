# SPDX-License-Identifier: Apache-2.0

# Unit tests for NNPA runtime and compiler fixes.
# These tests exercise pure logic (no NNPA hardware required) and are run as
# part of ctest under the "unittest" label alongside the rest of onnx-mlir's
# unit tests.

add_custom_target(unittest-nnpa)
set_target_properties(unittest-nnpa PROPERTIES FOLDER "Tests")

add_custom_target(check-onnx-unittest-nnpa
  COMMENT "Running the NNPA unit tests"
  COMMAND "${CMAKE_CTEST_COMMAND}" -L unittest-nnpa --output-on-failure -C $<CONFIG> --force-new-ctest-process
  USES_TERMINAL
  DEPENDS unittest-nnpa
  )
set_target_properties(check-onnx-unittest-nnpa PROPERTIES FOLDER "Tests")

function(add_nnpa_unittest test_name)
  add_onnx_mlir_executable(${test_name} NO_INSTALL ${ARGN})
  add_dependencies(unittest-nnpa ${test_name})
  add_test(NAME ${test_name} COMMAND ${test_name})
  set_tests_properties(${test_name} PROPERTIES LABELS unittest-nnpa)
endfunction()

# ---------------------------------------------------------------------------
# f032: zdnnx_is_full_tile() EQUAL_SPLIT flag inversion.
# Exercises pure flag logic against hand-crafted structs; no hardware needed.
# ---------------------------------------------------------------------------
add_nnpa_unittest(TestZdnnxUtils
  TestZdnnxUtils.c

  INCLUDE_DIRS PRIVATE
  ${ONNX_MLIR_SRC_ROOT}
  ${NNPA_INCLUDE_PATH}

  LINK_LIBS PRIVATE
  RuntimeNNPA
  )

# ---------------------------------------------------------------------------
# f031: get_rnn_concatenated_dim1() uint32 overflow guard.
# Tested via the public generate_transformed_desc_concatenated() API.
# ---------------------------------------------------------------------------
add_nnpa_unittest(TestStickifyRNN
  TestStickifyRNN.cpp

  INCLUDE_DIRS PRIVATE
  ${ONNX_MLIR_SRC_ROOT}
  ${NNPA_INCLUDE_PATH}

  LINK_LIBS PRIVATE
  OMStickify
  )

# ---------------------------------------------------------------------------
# f036: zdnnx_create_view() deep-copy / no-alias invariant.
# Verifies the view's descriptor pointers point to caller-supplied storage
# and that the original tensor's descriptors are not mutated.
# ---------------------------------------------------------------------------
add_nnpa_unittest(TestZdnnxCreateView
  TestZdnnxCreateView.c

  INCLUDE_DIRS PRIVATE
  ${ONNX_MLIR_SRC_ROOT}
  ${NNPA_INCLUDE_PATH}

  LINK_LIBS PRIVATE
  RuntimeNNPA
  )
