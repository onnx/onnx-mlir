# SPDX-License-Identifier: Apache-2.0

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(Passes.h.inc -gen-pass-decls -name Transforms)
mlir_tablegen(Transforms.capi.h.inc -gen-pass-capi-header --prefix Transforms)
mlir_tablegen(Transforms.capi.cpp.inc -gen-pass-capi-impl --prefix Transforms)
add_public_tablegen_target(OMTransformsPassIncGen)

add_onnx_mlir_library(OMLowerKrnlRegion
  LowerKrnlRegion.cpp

  LINK_LIBS PUBLIC
  OMSupport
  MLIRTransformUtils
  )

add_onnx_mlir_library(OMScfParallelPrivateRegion
  ProcessScfParallelPrivate.cpp
  ProcessKrnlParallelClause.cpp

  LINK_LIBS PUBLIC
  OMSupport
  MLIRTransformUtils
  MLIROpenMPToLLVM
  )

add_onnx_mlir_library(OMInstrument
  InstrumentPass.cpp
  InstrumentCleanupPass.cpp

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  OMONNXOps
  OMKrnlOps
  MLIRPass
  OMOptionUtils
  )

add_onnx_mlir_library(BufferOMPLoopHoist
  BufferOMPLoopHoist.cpp
  
  DEPENDS
  OMTransformsPassIncGen

  LINK_LIBS PUBLIC
  MLIRPass
  MLIROpenMPDialect
  MLIRFuncDialect
  MLIRMemRefDialect
  OMOptionUtils
  )
