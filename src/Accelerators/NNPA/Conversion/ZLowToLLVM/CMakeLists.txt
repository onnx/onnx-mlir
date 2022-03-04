add_onnx_mlir_library(OMZLowToLLVM
  ZLowToLLVM.cpp
  ZLowToLLVMCommon.cpp

  LINK_LIBS PUBLIC
  MLIRLLVMCommonConversion
  OMKrnlToLLVM
  OMLayoutHelper
  OMZLowOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${DLC_SRC_ROOT}
  ${DLC_BIN_ROOT}
  )
