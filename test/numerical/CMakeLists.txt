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


add_test(NAME OMTestConv COMMAND TestConv)
add_test(NAME OMTestLSTM COMMAND TestLSTM)
add_test(NAME OMTestRNN COMMAND TestRNN)
