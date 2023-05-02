# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMONNXElementsAttr
  BType.cpp
  DisposableElementsAttr.cpp
  DisposablePool.cpp
  ElementsAttrBuilder.cpp
  ElementsAttrHelper.cpp
  Strides.cpp
  WideNum.cpp

  INCLUDE_DIRS PUBLIC
  ${ONNX_MLIR_SRC_ROOT}/include

  LINK_LIBS PUBLIC
  OMMlirUtilities
  MLIRIR
  MLIRDialect # Dialect/Traits
  )
