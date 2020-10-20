add_subdirectory(DocExampleTest)

add_c_unit_test(OMTensorTest OMTensorTest.c)
target_include_directories(OMTensorTest PRIVATE
        ${ONNX_MLIR_SRC_ROOT}/include)
target_link_libraries(OMTensorTest
        cruntime)