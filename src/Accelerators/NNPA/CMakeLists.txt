# SPDX-License-Identifier: Apache-2.0

# SPDX-License-Identifier: Apache-2.0
add_onnx_mlir_library(OMNNPAAccel
  PrepareAccelerator.cpp
  ../OMAccelerator.cpp

  DEPENDS

  LINK_LIBS PUBLIC
  )

