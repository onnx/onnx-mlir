add_onnx_mlir_library(OMZLowToLLVM
  ZLowToLLVM.cpp
  ZLowToLLVMCommon.cpp

  LINK_LIBS PUBLIC
  MLIRLLVMCommonConversion
  OMLayoutHelper
  OMZLowOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
