add_onnx_mlir_library(OMStickify
  Convert.cpp
  Stickify.cpp

  DEPENDS
  libzdnn

  LINK_LIBS PUBLIC
  LLVMSupport
  OMLayoutHelper

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
