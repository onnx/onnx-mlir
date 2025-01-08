add_onnx_mlir_library(OMZHighToZLow
  ProcessStickData.cpp
  ZHighToZLow.cpp

  LINK_LIBS PUBLIC
  MLIRMemRefTransforms
  OMKrnlToLLVM
  OMLayoutHelper
  OMONNXToKrnl
  OMStickify
  OMZHighOps
  OMZLowOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

