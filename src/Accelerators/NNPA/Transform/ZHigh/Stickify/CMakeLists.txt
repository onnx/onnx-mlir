add_onnx_mlir_library(OMStickify
  Convert.cpp
  Stickify.cpp

  LINK_LIBS PUBLIC
  LLVMSupport
  )
