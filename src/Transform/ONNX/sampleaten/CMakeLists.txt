# SPDX-License-Identifier: Apache-2.0



#add_onnx_mlir_library(OMSaplOpTransform
#  SampleOpTransformPass.cpp

#  LINK_LIBS PUBLIC
#  OMONNXOps
#  MLIRPass
#  OMSupport
#  OMONNXRewrite
#  OMShapeInference
#  MLIRTransforms
#  )

#add_onnx_mlir_library(OMLeakyReluOpTransform
#  LeakyReluOpTransformPass.cpp

#  LINK_LIBS PUBLIC
#  OMONNXOps
#  MLIRPass
#  OMSupport
#  OMONNXRewrite
#  OMShapeInference
#  MLIRTransforms
#  )

add_onnx_mlir_library(ONNXToAtenLeakyReluOpTransform
  LeakyReluOpTransformToTorchPass.cpp
  
  LINK_LIBS PUBLIC
  OMONNXOps
  OMSupport
  OMONNXRewrite
  OMShapeInference
  MLIRIR
  MLIRPass
  MLIRTransforms
  TorchMLIRTorchDialect
  TorchMLIRTorchUtils
  TorchMLIRInitAll
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchPasses
  MLIRMemRefTransforms
  )

add_onnx_mlir_library(ONNXToAtenMaxPoolSingleOutOpTransform
  MaxPoolSingleOutOpTransformToTorchPass.cpp
 
  LINK_LIBS PUBLIC
  OMONNXOps
  OMSupport
  OMONNXRewrite
  OMShapeInference
  MLIRIR
  MLIRPass
  MLIRTransforms
  TorchMLIRTorchDialect
  TorchMLIRTorchUtils
  TorchMLIRInitAll
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchPasses
  MLIRMemRefTransforms
  )



add_onnx_mlir_library(ONNXToAtenConv2DOpTransform
  Conv2DOpTransformToTorchPass.cpp 

  LINK_LIBS PUBLIC
  OMONNXOps
  OMSupport
  OMONNXRewrite
  OMShapeInference
  MLIRIR
  MLIRPass
  MLIRTransforms
  TorchMLIRTorchDialect
  TorchMLIRTorchUtils
  TorchMLIRInitAll
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchPasses
  MLIRMemRefTransforms
  )


add_onnx_mlir_library(ONNXToPaddingOpTransform
  PaddingOpTransformToTorchPass.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  OMSupport
  OMONNXRewrite
  OMShapeInference
  MLIRIR
  MLIRPass
  MLIRTransforms
  TorchMLIRTorchDialect
  TorchMLIRTorchUtils
  TorchMLIRInitAll
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchPasses
  MLIRMemRefTransforms
  )


add_onnx_mlir_library(ONNXToConstantOpTransform
  ConstOpTransformToTorchPass.cpp 

  LINK_LIBS PUBLIC
  OMONNXOps
  OMSupport
  OMONNXRewrite
  OMShapeInference
  MLIRIR
  MLIRPass
  MLIRTransforms
  TorchMLIRTorchDialect
  TorchMLIRTorchUtils
  TorchMLIRInitAll
  TorchMLIRTorchConversionDialect
  TorchMLIRTorchPasses
  MLIRMemRefTransforms
  )
