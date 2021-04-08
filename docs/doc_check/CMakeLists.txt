# SPDX-License-Identifier: Apache-2.0

add_custom_target(check-doc
        COMMAND ${Python3_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/check.py
                        ${ONNX_MLIR_SRC_ROOT}
                        --exclude_dirs
                            third_party
                            docs/doc_check/test
                            docs/_site)

