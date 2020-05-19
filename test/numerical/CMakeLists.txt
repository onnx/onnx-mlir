add_executable(TestConv test_conv.cpp)
target_link_libraries(TestConv
        OMBuilder
        OMKrnlOps
        OMONNXOps
        OMShapeInference
        OMShapeInferenceOpInterface
        OMAttributePromotion
        OMPromotableConstOperandsOpInterface
        OMElideConstants
        OMElideKrnlGlobalConstants
        OMKrnlToAffine
        OMKrnlToLLVM
        OMONNXToKrnl
        OMONNXRewrite
        ${MLIRLibs}
        ${CMAKE_DL_LIBS}
        rapidcheck
        MainUtils
        ExecutionSession)

target_include_directories(TestConv
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})