# SPDX-License-Identifier: Apache-2.0

# TODO: statically link to libcruntime.a

add_onnx_mlir_library(OMRuntimeNNPA STATIC
  OMNNPA.c

  EXCLUDE_FROM_OM_LIBS

  DEPENDS
  libzdnn

  INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
set_target_properties(OMRuntimeNNPA
  PROPERTIES
  LANGUAGE C
  POSITION_INDEPENDENT_CODE TRUE
  )

