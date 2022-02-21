# SPDX-License-Identifier: Apache-2.0

# The backend tests require ONNX package installation.
add_subdirectory(backend)

# The following do NOT require ONNX package installation.
add_subdirectory(mlir)
add_subdirectory(modellib)
add_subdirectory(numerical)
add_subdirectory(backend-cpp)
add_subdirectory(onnx2mlir)
add_subdirectory(unit)
add_subdirectory(compilerlib)
add_subdirectory(perf)

