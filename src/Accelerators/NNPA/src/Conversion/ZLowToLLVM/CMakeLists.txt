add_dlc_library(OMZLowToLLVM
  ZLowToLLVM.cpp
  ZLowToLLVMCommon.cpp

  LINK_LIBS PUBLIC
  MLIRLLVMCommonConversion
  OMKrnlToLLVM
  OMLayoutHelper
  OMZLowOps
  )
