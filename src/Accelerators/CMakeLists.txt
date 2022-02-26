# SPDX-License-Identifier: Apache-2.0

if (ACCELERATORS_TO_BUILD)
  foreach(t ${ACCELERATORS_TO_BUILD})
    message(DEBUG "Targeting ${t}")
    add_subdirectory(${t})
  endforeach(t)
endif (ACCELERATORS_TO_BUILD)
add_onnx_mlir_library(NoAccel
  NoAccelerator.cpp
  INCLUDE_DIRS PUBLIC
    ${ONNX_MLIR_SRC_ROOT}/include
  LINK_LIBS PUBLIC
    LLVMSupport
)
