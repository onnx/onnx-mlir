# SPDX-License-Identifier: Apache-2.0

set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# TODO: statically link to libcruntime.a

add_onnx_mlir_library(RuntimeNNPA STATIC
  OMRuntimeNNPA.c
  zDNNExtension/zDNNExtension.c
  zDNNExtension/MatMul.c

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
  )

