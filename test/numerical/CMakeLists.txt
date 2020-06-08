add_executable(TestConv TestConv.cpp)
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
        ExecutionSession
        DynMemRefUtils)
whole_archive_link_mlir(TestConv ${MLIRWholeArchiveLibs})
target_include_directories(TestConv
        PRIVATE
        ${ONNX_MLIR_SRC_ROOT}
        ${ONNX_MLIR_BIN_ROOT}
        ${ONNX_MLIR_SRC_ROOT})
add_test(NAME OMTestConv COMMAND TestConv)