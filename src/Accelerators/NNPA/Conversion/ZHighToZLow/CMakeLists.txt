add_onnx_mlir_library(OMZHighToZLow
  ZHighToZLow.cpp

  DEPENDS
  libzdnn

  LINK_LIBS PUBLIC
  MLIRMemRefTransforms
  OMLayoutHelper
  OMONNXToKrnl
  OMZHighOps
  OMZLowOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

