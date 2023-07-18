# SPDX-License-Identifier: Apache-2.0

# TODO: statically link to libcruntime.a

add_onnx_mlir_library(RuntimeNNPA STATIC
  OMRuntimeNNPA.c
  OMMatMulAsyncNNPA.c

  EXCLUDE_FROM_OM_LIBS

  DEPENDS
  libzdnn

  INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  ${ONNX_MLIR_SRC_ROOT}/include
  )
set_target_properties(RuntimeNNPA
  PROPERTIES
  LANGUAGE C
  POSITION_INDEPENDENT_CODE TRUE
  )

