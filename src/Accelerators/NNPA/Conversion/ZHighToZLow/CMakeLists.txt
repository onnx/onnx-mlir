add_onnx_mlir_library(OMZHighToZLow
  ZHighToZLow.cpp

  LINK_LIBS PUBLIC
  MLIRMemRefTransforms
  OMLayoutHelper
  OMONNXToKrnl
  OMStickify
  OMZHighOps
  OMZLowOps
  MLIRAsyncDialect

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

