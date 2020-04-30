configure_file(test.py test.py COPYONLY)
configure_file(test_config.py.in test_config.py)

find_package(PythonInterp 3 REQUIRED)
add_custom_target(check-onnx-backend
        COMMAND LD_PRELOAD=$<TARGET_FILE:cruntime> ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_BINARY_DIR}/test.py)

add_dependencies(check-onnx-backend onnx-mlir)
add_dependencies(check-onnx-backend pyruntime)
