# SPDX-License-Identifier: Apache-2.0

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "s390x")
  # currently needed for a bug on Z for the KrnlOps.cpp.inc file
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O2" "-O1")
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O3" "-O1")
endif()

add_onnx_mlir_dialect(KrnlOps)

add_onnx_mlir_library(OMKrnlOps
  KrnlOps.cpp
  KrnlTypes.cpp
  KrnlHelper.cpp
  SimplifyKrnl.cpp

  DEPENDS
  OMKrnlOpsIncGen
  OMSpecializedKernelOpInterfaceIncGen

  LINK_LIBS PUBLIC
  OMONNXOps
  )
