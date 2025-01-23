# SPDX-License-Identifier: Apache-2.0

if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "s390x")
  # currently needed for a bug on Z for the KrnlOps.cpp.inc file
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O2" "-O1")
  llvm_replace_compiler_option(CMAKE_CXX_FLAGS_RELEASE "-O3" "-O1")
endif()

add_onnx_mlir_dialect(Krnl krnl)
add_onnx_mlir_dialect_doc(krnl Krnl.td)

add_onnx_mlir_library(OMKrnlOps
  DialectBuilder.cpp
  KrnlOps.cpp
  KrnlTypes.cpp
  KrnlHelper.cpp

  DEPENDS
  OMKrnlIncGen
  OMSpecializedKernelOpInterface

  LINK_LIBS PUBLIC
  OMCompilerOptions
  OMONNXOps
  MLIRLLVMCommonConversion
  MLIRAffineDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRShapeDialect
  )
