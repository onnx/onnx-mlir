find_package(Python3 REQUIRED
             COMPONENTS Interpreter)

add_custom_target(check-doc
        COMMAND Python3::Interpreter
                    ${CMAKE_CURRENT_SOURCE_DIR}/check.py
                        ${CMAKE_SOURCE_DIR}
                        --exclude_dirs third_party doc/doc_check/test)

