add_onnx_mlir_dialect(ZLow zlow)
add_onnx_mlir_dialect_doc(zlow ZLow.td)

add_onnx_mlir_library(OMZLowOps
  ZLowOps.cpp

  DEPENDS
  OMZLowIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
