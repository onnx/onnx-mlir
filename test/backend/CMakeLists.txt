configure_file(test.py test.py COPYONLY)
configure_file(test_config.py.in test_config.py)

find_package(PythonInterp 3 REQUIRED)

set(LD_PRELOADS $<TARGET_FILE:cruntime>)

get_filename_component(CXX_BIN_DIR ${CMAKE_CXX_COMPILER} PATH)
set(LIBSTDCXX_LIB "${CXX_BIN_DIR}/../lib64/libstdc++.so")
if(EXISTS ${LIBSTDCXX_LIB})
  set(LD_PRELOADS ${LIBSTDCXX_LIB}:${LD_PRELOADS})
endif()

add_custom_target(check-onnx-backend
        COMMAND ${PYTHON_EXECUTABLE}
        ${CMAKE_CURRENT_BINARY_DIR}/test.py)

add_dependencies(check-onnx-backend onnx-mlir)
add_dependencies(check-onnx-backend pyruntime)
