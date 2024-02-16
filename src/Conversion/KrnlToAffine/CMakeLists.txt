# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMKrnlToAffine
  ConvertKrnlToAffine.cpp
  KrnlCopyFromBuffer.cpp
  KrnlCopyToBuffer.cpp
  KrnlLoad.cpp
  KrnlMatmul.cpp
  KrnlMemset.cpp
  KrnlStore.cpp
  KrnlTerminator.cpp
  KrnlToAffineHelper.cpp

  LINK_LIBS PUBLIC
  OMSpecializedKernelOpInterface
  OMONNXOps
  OMSupport
  MLIRTransforms
  MLIRAffineUtils
  )
