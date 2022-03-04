add_onnx_mlir_library(OMZLowRewrite
  ZLowRewrite.cpp

  LINK_LIBS PUBLIC
  MLIRRewrite
  MLIRTransformUtils
  MLIRViewLikeInterface
  OMZLowOps
  )

# add_onnx_mlir_library(OMZHighLayoutPropagation
#   ZHighLayoutPropagation.cpp
# 
#   DEPENDS
#   OMZHighLayoutPropagationIncGen
# 
#   LINK_LIBS PUBLIC
#   MLIRRewrite
#   MLIRTransformUtils
#   OMZHighOps
#   OMONNXOps
# 
#   ACCEL_INCLUDE_DIRS PRIVATE
#   ${DLC_SRC_ROOT}
#   ${DLC_BIN_ROOT}
#    )
