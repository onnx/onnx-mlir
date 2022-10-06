# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMSeqToMemref
  ConvertSeqToMemref.cpp
  KrnlSeqAlloc.cpp
  KrnlSeqDealloc.cpp
  KrnlSeqExtract.cpp
  KrnlSeqStore.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  OMSupport
  MLIRTransforms
  MLIRAffineUtils
  OMONNXToKrnl
  )
