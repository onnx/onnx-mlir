# SPDX-License-Identifier: Apache-2.0

# TODO: statically link to libcruntime.a

# We want to enable zdnn status messages when a user
# manually specifies the environment variable.
# The default value for zdnn status message is "OFF"
option(ZDNN_STATUS_MESSAGE "Enable ZDNN Status Messages" OFF)
if (ZDNN_STATUS_MESSAGE)
  add_compile_definitions(ZDNN_STATUS_MESSAGE)
endif()

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

