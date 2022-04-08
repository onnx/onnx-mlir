add_onnx_mlir_library(OMStickify
  Convert.cpp
  Stickify.cpp

  LINK_LIBS PUBLIC
  LLVMSupport

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
