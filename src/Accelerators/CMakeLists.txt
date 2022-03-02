# SPDX-License-Identifier: Apache-2.0

if (ACCELERATORS_TO_BUILD)
  foreach(t ${ACCELERATORS_TO_BUILD})
    message(DEBUG "Targeting ${t}")
    add_subdirectory(${t})
  endforeach(t)
else (ACCELERATORS_TO_BUILD)
endif (ACCELERATORS_TO_BUILD)

add_onnx_mlir_library(Accelerator
    Accelerator.cpp
    INCLUDE_DIRS PUBLIC
      ${ONNX_MLIR_SRC_ROOT}/include
    LINK_LIBS PUBLIC
      LLVMSupport
  ) 