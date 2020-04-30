configure_file(test.py test.py COPYONLY)
configure_file(test_config.py.in test_config.py)

find_package(PythonInterp 3 REQUIRED)

get_filename_component(CXX_BIN_DIR ${CMAKE_CXX_COMPILER} PATH)
set(LIBSTDCXX_FILE "${CXX_BIN_DIR}/../lib64/libstdc++.so.6")

add_custom_target(check-onnx-backend
        COMMAND LD_PRELOAD=${LIBSTDCXX_FILE}:$<TARGET_FILE:cruntime> ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_BINARY_DIR}/test.py)

add_dependencies(check-onnx-backend onnx-mlir)
add_dependencies(check-onnx-backend pyruntime)
