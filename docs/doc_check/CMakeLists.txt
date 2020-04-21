find_package(PythonInterp 3 REQUIRED)

add_custom_target(check-doc
        COMMAND ${PYTHON_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/check.py
                        ${ONNX_MLIR_SRC_ROOT}
                        --exclude_dirs
                            third_party
                            docs/doc_check/test
                            docs/_site)

