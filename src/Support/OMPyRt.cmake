# SPDX-License-Identifier: Apache-2.0

add_onnx_mlir_library(OMSmallFPConversion
  SmallFPConversion.c
  )
set_target_properties(OMSmallFPConversion
  PROPERTIES
  LANGUAGE C
  )
