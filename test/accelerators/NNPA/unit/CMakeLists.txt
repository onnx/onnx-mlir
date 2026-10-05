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

# Common include paths used by NNPA unit tests.
# Both variables are set in the top-level CMakeLists.txt and are globally visible:
#   CMAKE_INCLUDE_OUTPUT_DIRECTORY — the per-config build include dir containing
#                                    zdnn.h (same as NNPA_INCLUDE_PATH, but that
#                                    var is scoped to the NNPA src subtree only)
#   ONNX_MLIR_SRC_ROOT             — the repo root (onnx-mlir/)
set(NNPA_ZDNNX_SRC_DIR
    ${ONNX_MLIR_SRC_ROOT}/src/Accelerators/NNPA/Runtime/zdnnx)

# ---------------------------------------------------------------------------
# f032: zdnnx_is_full_tile() EQUAL_SPLIT flag inversion.
# ---------------------------------------------------------------------------
add_nnpa_unittest(TestZdnnxUtils
  TestZdnnxUtils.c

  INCLUDE_DIRS PRIVATE
  ${CMAKE_INCLUDE_OUTPUT_DIRECTORY}
  ${NNPA_ZDNNX_SRC_DIR}

  LINK_LIBS PRIVATE
  RuntimeNNPA
  )

# ---------------------------------------------------------------------------
# f031: get_rnn_concatenated_dim1() uint32 overflow guard.
# f036: zdnnx_create_view() deep-copy / no-alias invariant.
#
# These tests call zdnn API functions (zdnn_init_pre_transformed_desc,
# zdnn_generate_transformed_desc, zdnn_getsize_ztensor, etc.) which are
# only compiled into libzdnn.a on s390x.  On other arches the libzdnn
# custom target is a no-op (header-copy only), so libzdnn.a does not exist
# and these tests cannot be linked.  Guard them s390x-only.
# ---------------------------------------------------------------------------
if (CMAKE_SYSTEM_PROCESSOR STREQUAL "s390x")

  add_nnpa_unittest(TestStickifyRNN
    TestStickifyRNN.cpp

    INCLUDE_DIRS PRIVATE
    ${CMAKE_INCLUDE_OUTPUT_DIRECTORY}
    ${ONNX_MLIR_SRC_ROOT}/src

    DEPENDS
    libzdnn

    LINK_LIBS PRIVATE
    OMStickify
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libzdnn.a
    )

  add_nnpa_unittest(TestZdnnxCreateView
    TestZdnnxCreateView.c

    INCLUDE_DIRS PRIVATE
    ${CMAKE_INCLUDE_OUTPUT_DIRECTORY}
    ${NNPA_ZDNNX_SRC_DIR}

    DEPENDS
    libzdnn

    LINK_LIBS PRIVATE
    RuntimeNNPA
    ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libzdnn.a
    )

endif()
