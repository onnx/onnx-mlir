add_onnx_mlir_library(OMZLowToLLVM
  ZLowToLLVM.cpp
  ZLowToLLVMCommon.cpp

  LINK_LIBS PUBLIC
  MLIRLLVMCommonConversion
  OMKrnlToLLVM
  OMLayoutHelper
  OMZLowOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_SRC_ROOT}
  ${NNPA_BIN_ROOT}
  )
