# SPDX-License-Identifier: Apache-2.0

if (NOT ONNX_MLIR_BUILD_TESTS)
  set(EXCLUDE_FROM_ALL ON)
endif()

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/add_sub_onnx.py
  ${CMAKE_CURRENT_BINARY_DIR}/add_sub_onnx.py
  COPYONLY
)

set(OMRuntimeTestMultiModelONNXFiles ${CMAKE_CURRENT_BINARY_DIR}/add.onnx)
list(APPEND OMRuntimeTestMultiModelONNXFiles ${CMAKE_CURRENT_BINARY_DIR}/sub.onnx)

add_custom_command(
  OUTPUT ${OMRuntimeTestMultiModelONNXFiles}
  COMMAND
    MODELS_PATH=${CMAKE_CURRENT_BINARY_DIR}
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/add_sub_onnx.py --create-models
)

add_custom_target(check-multiple-models
  COMMAND
    PY_LIB=${CMAKE_BINARY_DIR}/$<CONFIG>/lib
    MODELS_PATH=${CMAKE_CURRENT_BINARY_DIR}
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/add_sub_onnx.py --run-models
  DEPENDS ${OMRuntimeTestMultiModelONNXFiles}
)

add_dependencies(check-onnx-backend onnx-mlir)
add_dependencies(check-multiple-models PyRuntimeC)
add_dependencies(check-multiple-models PyCompileAndRuntimeC)
