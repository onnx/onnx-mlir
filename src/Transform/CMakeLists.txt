# SPDX-License-Identifier: Apache-2.0

#separate krnl pass into different .td
add_onnx_mlir_pass(TransformsKrnl PassesKrnl TransformsKrnl)
add_onnx_mlir_pass(Transforms Passes Transforms)

add_onnx_mlir_library(OMLowerKrnlRegion
  LowerKrnlRegion.cpp

  DEPENDS
  OMTransformsKrnlPassIncGen

  LINK_LIBS PUBLIC
  OMSupport
  MLIRTransformUtils
  )

add_onnx_mlir_library(OMScfParallelPrivateRegion
  ProcessScfParallelPrivate.cpp
  ProcessKrnlParallelClause.cpp

  DEPENDS
  OMTransformsPassIncGen

  LINK_LIBS PUBLIC
  OMSupport
  MLIRTransformUtils
  MLIROpenMPToLLVM
  )

add_onnx_mlir_library(OMInstrument
  InstrumentPass.cpp
  InstrumentCleanupPass.cpp

  DEPENDS
  OMTransformsPassIncGen

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

add_onnx_mlir_library(OMWriteOpStatsToModuleAttribute
  WriteOpStatsToModuleAttribute.cpp

  DEPENDS
  OMTransformsPassIncGen

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  OMONNXOps
  MLIRPass
  MLIRFuncDialect
  MLIRTransforms
  )
