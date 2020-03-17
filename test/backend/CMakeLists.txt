configure_file(test.py test.py COPYONLY)
configure_file(test_config.py.in test_config.py)

find_package(PythonInterp 3 REQUIRED)
add_custom_target(run-onnx-backend-test
        COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_BINARY_DIR}/test.py)

add_dependencies(run-onnx-backend-test onnx-mlir)
add_dependencies(run-onnx-backend-test pyruntime)
