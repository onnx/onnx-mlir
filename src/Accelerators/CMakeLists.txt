# SPDX-License-Identifier: Apache-2.0

if (ACCELERATORS_TO_BUILD)
  set (ACCEL_LIST "") 
  foreach(t ${ACCELERATORS_TO_BUILD})
    message(DEBUG "Targeting ${t}")
    add_subdirectory(${t})
    list(APPEND ACCEL_LIST ${t}"Accel")
  endforeach(t)
else (ACCELERATORS_TO_BUILD)
  set (ACCEL_LIST "NoAccelerator")    
  #add_onnx_mlir_library(NoAccelerator
  #  NoAccelerator.cpp
  #  EXCLUDE_FROM_OM_LIBS
  #  INCLUDE_DIRS PUBLIC
  #    ${ONNX_MLIR_SRC_ROOT}/include
  #  LINK_LIBS PUBLIC
  #    LLVMSupport
  #) 
endif (ACCELERATORS_TO_BUILD)

add_onnx_mlir_library(Accelerator
    Accelerator.cpp
    INCLUDE_DIRS PUBLIC
      ${ONNX_MLIR_SRC_ROOT}/include
    LINK_LIBS PUBLIC
      LLVMSupport
  ) 