# SPDX-License-Identifier: Apache-2.0
set(ACCEL_LIST "")
if (ACCELERATORS_TO_BUILD)
  foreach(t ${ACCELERATORS_TO_BUILD})
    message(DEBUG "Targeting ${t}")
    add_subdirectory(${t})
    list(APPEND ACCEL_LIST "${t}Accel")
  endforeach(t)
else (ACCELERATORS_TO_BUILD)
endif (ACCELERATORS_TO_BUILD)

add_onnx_mlir_library(Accelerator STATIC
    Accelerator.cpp
    DEPENDS
      MLIRIR
    INCLUDE_DIRS PUBLIC
      ${ONNX_MLIR_SRC_ROOT}/include
    LINK_LIBS PUBLIC
      LLVMSupport
      ${ACCEL_LIST}
  ) 