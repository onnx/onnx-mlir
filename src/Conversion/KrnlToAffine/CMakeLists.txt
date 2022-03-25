# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMKrnlToAffine
  KrnlToAffine.cpp

  LINK_LIBS PUBLIC
  OMSpecializedKernelOpInterface
  OMONNXOps
  OMSupport
  MLIRTransforms
  MLIRAffineUtils
  )
