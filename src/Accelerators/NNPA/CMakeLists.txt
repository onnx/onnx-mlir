# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
add_onnx_mlir_library(NNPAAccel
  NNPAAccelerator.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  )

add_dependencies(NNPAAccel Accel)  

