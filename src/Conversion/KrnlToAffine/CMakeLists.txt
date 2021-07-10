# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMKrnlToAffine
  KrnlToAffine.cpp
  KrnlLoopsLowering.cpp

  LINK_LIBS PUBLIC
  OMSpecializedKernelOpInterface
  OMSupport
  MLIRTransforms
  )
