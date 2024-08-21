add_onnx_mlir_dialect(ZLow zlow)
add_onnx_mlir_dialect_doc(zlow ZLow.td)

add_onnx_mlir_rewriter(ZLowCombine)

add_onnx_mlir_library(OMZLowOps
  ZLowOps.cpp
  ZLowCombine.cpp
  DialectBuilder.cpp

  DEPENDS
  OMZLowIncGen
  OMONNXZLowCombineIncGen
  OMConstantOpInterface

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRMemRefTransforms
  OMMlirDialects
  OMLayoutHelper
  OMStickify

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
