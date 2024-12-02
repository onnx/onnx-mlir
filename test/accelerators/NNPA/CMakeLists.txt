# SPDX-License-Identifier: Apache-2.0

if (NOT ONNX_MLIR_BUILD_TESTS)
  set(EXCLUDE_FROM_ALL ON)
endif()

add_custom_target(check-onnx-backend-numerical-nnpa)

add_subdirectory(backend)
add_subdirectory(numerical)
