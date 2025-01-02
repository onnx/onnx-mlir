add_onnx_mlir_library(OMZLowToLLVM
  ZLowToLLVM.cpp
  ZLowToLLVMCommon.cpp

  DEPENDS
  libzdnn

  LINK_LIBS PUBLIC
  OMCompilerOptions
  MLIRLLVMCommonConversion
  OMKrnlToLLVM
  OMLayoutHelper
  OMZLowOps
  OMMlirDialects

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
