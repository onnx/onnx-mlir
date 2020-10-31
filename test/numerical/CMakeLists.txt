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
add_test(NAME OMTestConv COMMAND TestConv)

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
add_test(NAME OMTestLoop COMMAND TestLoop)
