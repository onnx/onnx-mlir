# SPDX-License-Identifier: Apache-2.0

# Valid optimization levels are 0, 1, 2, and 3
# Default to 3 if ONNX_MLIR_TEST_OPTLEVEL not set or cached
set(OPTLEVELS 0 1 2 3)
if (NOT DEFINED ONNX_MLIR_TEST_OPTLEVEL)
  set(ONNX_MLIR_TEST_OPTLEVEL 3 CACHE STRING "" FORCE)
endif()
# Fail if specified/cached ONNX_MLIR_TEST_OPTLEVEL is not valid
if (NOT ${ONNX_MLIR_TEST_OPTLEVEL} IN_LIST OPTLEVELS)
  unset(ONNX_MLIR_TEST_OPTLEVEL CACHE)
  message(FATAL_ERROR "ONNX_MLIR_TEST_OPTLEVEL must be one of ${OPTLEVELS}")
endif()
message(STATUS "Tests optimization level : ${ONNX_MLIR_TEST_OPTLEVEL}")

# The backend tests require ONNX package installation.
add_subdirectory(backend)
add_subdirectory(accelerators)

# The following do NOT require ONNX package installation.
add_subdirectory(mlir)
add_subdirectory(modellib)
add_subdirectory(numerical)
add_subdirectory(backend-cpp)
add_subdirectory(onnx2mlir)
add_subdirectory(unit)
add_subdirectory(compilerlib)
add_subdirectory(perf)

