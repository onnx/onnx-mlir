add_executable(TestConv TestConv.cpp)
target_compile_definitions(TestConv PRIVATE RTMEMREF_INTERNAL_API)
target_include_directories(TestConv
        PRIVATE
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT}/include)
target_link_libraries(TestConv
        ${OMLibs}
        ${MLIRLibs}
        ${CMAKE_DL_LIBS}
        rapidcheck
        MainUtils
        ExecutionSession
        OMTensorUtils)

add_executable(TestLSTM TestLSTM.cpp)
target_compile_definitions(TestLSTM PRIVATE RTMEMREF_INTERNAL_API)
target_include_directories(TestLSTM
        PRIVATE
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT}/include)
target_link_libraries(TestLSTM
        ${OMLibs}
        ${MLIRLibs}
        ${CMAKE_DL_LIBS}
        rapidcheck
        MainUtils
        ExecutionSession
        OMTensorUtils)

add_executable(TestRNN TestRNN.cpp)
target_compile_definitions(TestRNN PRIVATE RTMEMREF_INTERNAL_API)
target_include_directories(TestRNN
        PRIVATE
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT}/include)
target_link_libraries(TestRNN
        ${OMLibs}
        ${MLIRLibs}
        ${CMAKE_DL_LIBS}
        rapidcheck
        MainUtils
        ExecutionSession
        OMTensorUtils)

add_executable(TestGRU TestGRU.cpp)
target_compile_definitions(TestGRU PRIVATE RTMEMREF_INTERNAL_API)
target_include_directories(TestGRU
        PRIVATE
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT}/include)
target_link_libraries(TestGRU
        ${OMLibs}
        ${MLIRLibs}
        ${CMAKE_DL_LIBS}
        rapidcheck
        MainUtils
        ExecutionSession
        OMTensorUtils)

add_executable(TestLoop TestLoop.cpp)
target_compile_definitions(TestLoop PRIVATE RTMEMREF_INTERNAL_API)
target_include_directories(TestLoop
        PRIVATE
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT}/include)
target_link_libraries(TestLoop
        ${OMLibs}
        ${MLIRLibs}
        ${CMAKE_DL_LIBS}
        rapidcheck
        MainUtils
        ExecutionSession
        OMTensorUtils)
        

add_test(NAME OMTestConv COMMAND TestConv)
add_test(NAME OMTestLoop COMMAND TestLoop)
add_test(NAME OMTestGRU COMMAND gdb -return-child-result -ex run -ex "thread apply all bt" -ex "quit" --args ${CMAKE_BINARY_DIR}/bin/TestGRU)
add_test(NAME OMTestLSTM COMMAND gdb -return-child-result -ex run -ex "thread apply all bt" -ex "quit" --args ${CMAKE_BINARY_DIR}/bin/TestLSTM)
add_test(NAME OMTestRNN COMMAND gdb -return-child-result -ex run -ex "thread apply all bt" -ex "quit" --args ${CMAKE_BINARY_DIR}/bin/TestRNN)
