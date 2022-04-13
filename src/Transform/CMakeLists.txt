# SPDX-License-Identifier: Apache-2.0

add_subdirectory(ONNX)

add_onnx_mlir_library(OMElideKrnlGlobalConstants
  ElideKrnlGlobalConstants.cpp

  LINK_LIBS PUBLIC
  OMKrnlOps
  MLIRTransformUtils
  )

add_onnx_mlir_library(OMEnableMemoryPool
  EnableMemoryPool.cpp

  LINK_LIBS PUBLIC
  OMSupport
  MLIRTransformUtils
  )

add_onnx_mlir_library(OMOptimizeMemoryPools
  OptimizeMemoryPools.cpp

  LINK_LIBS PUBLIC
  OMSupport
  MLIRTransformUtils
  )


add_onnx_mlir_library(OMBundleMemoryPools
  BundleMemoryPools.cpp

  LINK_LIBS PUBLIC
  OMSupport
  MLIRTransformUtils
  )

add_onnx_mlir_library(OMDisconnectKrnlDimFromAlloc
  DisconnectKrnlDimFromAlloc.cpp

  LINK_LIBS PUBLIC
  OMSupport
  MLIRTransformUtils
  )

add_onnx_mlir_library(OMLowerKrnlShape
  LowerKrnlShape.cpp

  LINK_LIBS PUBLIC
  OMSupport
  MLIRTransformUtils
  )

add_onnx_mlir_library(OMLowerKrnlRegion
  LowerKrnlRegion.cpp

  LINK_LIBS PUBLIC
  OMSupport
  MLIRTransformUtils
  )
