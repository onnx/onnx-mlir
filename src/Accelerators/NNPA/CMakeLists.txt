# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_dialect(KrnlOps)
add_onnx_mlir_dialect_doc(krnl KrnlOps.td)

add_onnx_mlir_library(OMKrnlOps
  KrnlOps.cpp
  KrnlTypes.cpp
  KrnlHelper.cpp

  DEPENDS
  OMKrnlOpsIncGen

  LINK_LIBS PUBLIC
  OMONNXOps
  )