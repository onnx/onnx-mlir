# SPDX-License-Identifier: Apache-2.0

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
