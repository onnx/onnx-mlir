# SPDX-License-Identifier: Apache-2.0

# TODO: statically link to libcruntime.a

add_onnx_mlir_library(RuntimeNNPA STATIC
  OMRuntimeNNPA.c
  zDNNExtension/zDNNExtension.c
  zDNNExtension/Elementwise.c
  zDNNExtension/MatMul.c
  zDNNExtension/Softmax.c

  EXCLUDE_FROM_OM_LIBS

  DEPENDS
  libzdnn

  INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
set_target_properties(RuntimeNNPA
  PROPERTIES
  LANGUAGE C
  POSITION_INDEPENDENT_CODE TRUE
  COMPILE_OPTIONS "-O3"
  )

