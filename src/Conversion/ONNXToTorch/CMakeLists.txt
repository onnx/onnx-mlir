# SPDX-License-Identifier: Apache-2.0

# Please keep in alphabetical order.

install(TARGETS
  TorchMLIRTorchDialect
  TorchMLIRTorchUtils
  )

add_onnx_mlir_library(OMONNXToTorch
  ConvertONNXToTorch.cpp
  ConvertONNXToTorchPipeline.cpp
  EraseONNXEntryPoint.cpp
  ONNXToTorchCommon.cpp
  TypeConversion/TorchTypeConversion.cpp
  TypeConversion/TorchTypeConversionPasses.cpp

  Math/Elementwise.cpp

  Tensor/Constant.cpp

  LINK_LIBS PUBLIC
  TorchMLIRTorchDialect
  TorchMLIRTorchUtils
  OMONNXOps
  )
