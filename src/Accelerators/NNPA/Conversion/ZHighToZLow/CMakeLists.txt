add_onnx_mlir_library(OMZHighToZLow
  ZHighToZLow.cpp

  LINK_LIBS PUBLIC
  MLIRMemRefTransforms
  OMONNXToKrnl
  OMZHighOps
  OMZLowOps
  )

