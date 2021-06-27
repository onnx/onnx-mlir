# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMSimplifyKrnl
  SimplifyKrnl.cpp

  DEPENDS
  OMKrnlOpsIncGen
  OMSpecializedKernelOpInterfaceIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  )
