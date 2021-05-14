add_onnx_mlir_library(OMKrnlToStandard
  KrnlToStandard.cpp
  )

# Header dependencies
add_dependencies(OMKrnlToStandard OMKrnlOpsInc)
# Linking dependencies
add_dependencies(OMKrnlToStandard OMKrnlOps)
