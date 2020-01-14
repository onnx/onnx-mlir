find_package(PythonInterp 3 REQUIRED)

add_custom_target(check-doc
        COMMAND ${PYTHON_EXECUTABLE}
                    ${CMAKE_CURRENT_SOURCE_DIR}/check.py
                        ${CMAKE_SOURCE_DIR}
                        --exclude_dirs third_party doc/doc_check/test)

