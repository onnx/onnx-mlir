# SPDX-License-Identifier: Apache-2.0

set(check_cmd ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/check.py)

add_custom_target(check-doc
  COMMAND ${check_cmd}
    ${ONNX_MLIR_SRC_ROOT}
    --exclude_dirs
      third_party
      docs/doc_check/test
      docs/_site
  COMMAND ${check_cmd} ${CMAKE_BINARY_DIR}
  )

set_target_properties(check-doc PROPERTIES FOLDER "Docs")
# Exclude the target from the default VS build
set_target_properties(check-doc PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)
