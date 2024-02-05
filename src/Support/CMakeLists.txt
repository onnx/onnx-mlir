# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMDiagnostic
  Diagnostic.cpp

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC  
  MLIRIR
  )

add_onnx_mlir_library(OMSupport
  KrnlSupport.cpp

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  OMKrnlOps
  )

add_onnx_mlir_library(OMMlirUtilities
  SmallFP.cpp
  TypeUtilities.cpp
  # Expect to have other utilities for Attribute and Value in the future.
  # AttributeUtilities.cpp
  # ValueUtilities.cpp

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  OMSmallFPConversion
  MLIRIR
  )

add_onnx_mlir_library(OMSmallFPConversion
  SmallFPConversion.c
  )
set_target_properties(OMSmallFPConversion
  PROPERTIES
  LANGUAGE C
  )
