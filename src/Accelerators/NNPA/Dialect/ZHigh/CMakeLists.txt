add_onnx_mlir_dialect(ZHigh zhigh)
add_onnx_mlir_dialect_doc(zhigh ZHigh.td)

add_onnx_mlir_rewriter(ZHighCombine)

add_onnx_mlir_library(OMZHighOps
  ZHighHelper.cpp
  ZHighOps.cpp
  ZHighCombine.cpp
  ZHighShapeHelper.cpp

  DEPENDS
  OMLayoutHelper
  OMZHighIncGen
  OMONNXZHighCombineIncGen
  OMShapeInferenceOpInterfaceIncGen
  libzdnn

  LINK_LIBS PUBLIC
  OMONNXOps # Use ONNXShapeHelper 
  OMLayoutHelper
  MLIRIR

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
