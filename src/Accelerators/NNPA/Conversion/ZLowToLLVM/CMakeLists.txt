add_onnx_mlir_library(OMZLowToLLVM
  ZLowToLLVM.cpp
  ZLowToLLVMCommon.cpp

  LINK_LIBS PUBLIC
  MLIRLLVMCommonConversion
  OMKrnlToLLVM
  OMLayoutHelper
  OMZLowOps
  )
