# SPDX-License-Identifier: Apache-2.0

if( ACCELERATORS_TO_BUILD )
  foreach(t ${ACCELERATORS_TO_BUILD})
    message(STATUS "Targeting ${t}")
    add_subdirectory(${t})
    endforeach(t)
endif( ACCELERATORS_TO_BUILD )




  add_onnx_mlir_library(NoAccel
  NoAccelerator.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  )

