# SPDX-License-Identifier: Apache-2.0

add_executable(TestONNXEinsumOp TestONNXEinsumOp.cpp)
target_include_directories(TestONNXEinsumOp PRIVATE
        ${ONNX_MLIR_SRC_ROOT}/include)

add_test(NAME TestONNXEinsumOp COMMAND TestONNXEinsumOp)

target_link_libraries(TestONNXEinsumOp
        OMONNXOps)
