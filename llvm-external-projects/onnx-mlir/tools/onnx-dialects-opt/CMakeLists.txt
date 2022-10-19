set(LIBS
  # Local dialects.  
  ONNXOnnxDialect
  
    # Core dialects.
  MLIRAffineDialect
  MLIRArithDialect
  MLIRAsyncDialect
  MLIRControlFlowDialect
  MLIRDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRLinalgTransformOps
  MLIRMemRefDialect
  MLIRMemRefTransforms
  MLIROptLib
  MLIRPDLDialect
  MLIRPDLInterpDialect
  MLIRSCFDialect
  MLIRSCFTransformOps
  MLIRSCFTransforms
  MLIRTensorDialect
  MLIRTransforms
)

add_llvm_tool(onnx-dialects-opt
  onnx-dialects-opt.cpp

  DEPENDS
  ${LIBS}
)
target_link_libraries(onnx-dialects-opt PRIVATE ${LIBS})
