add_executable(OMTensorTest OMTensorTest.c)
target_include_directories(OMTensorTest PRIVATE
        ${ONNX_MLIR_SRC_ROOT}/include)

add_test(NAME OMTensorTest COMMAND OMTensorTest)

target_link_libraries(OMTensorTest
        cruntime)
