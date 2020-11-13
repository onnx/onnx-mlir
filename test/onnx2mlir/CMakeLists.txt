add_executable(CustomFnTest CustomFnTest.cpp)

target_link_libraries(CustomFnTest
        OMBuilder OMONNXOps OMPromotableConstOperandsOpInterface OMResultTypeInferenceOpInterface OMShapeInferenceOpInterface
        OMAttributePromotion OMONNXRewrite ${MLIRLibs} onnx)

target_include_directories(CustomFnTest
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}/third_party/onnx
        ${ONNX_MLIR_BIN_ROOT})
