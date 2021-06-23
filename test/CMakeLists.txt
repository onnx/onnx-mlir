# SPDX-License-Identifier: Apache-2.0

add_subdirectory(mlir)
add_subdirectory(backend)
add_subdirectory(numerical)
add_subdirectory(unit)
add_subdirectory(onnx2mlir)
add_subdirectory(models)
if (MAIA_ENABLE_SDK_TESTS)
  add_subdirectory(maia)
endif()
