add_onnx_mlir_rewriter(ONNXToZHigh)
add_onnx_mlir_rewriter(ZHighToONNX)

add_onnx_mlir_library(OMONNXToZHigh
  ONNXLegalityCheck.cpp
  ONNXToZHigh.cpp
  ONNXToZHighCommon.cpp

  DEPENDS
  OMONNXONNXToZHighIncGen
  libzdnn

  LINK_LIBS PUBLIC
  OMNNPACompilerOptions
  OMONNXOps
  OMONNXToKrnl
  OMShapeInferencePass
  OMZHighOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_rewriter(RewriteONNXForZHigh)

add_onnx_mlir_library(OMRewriteONNXForZHigh
  ONNXLegalityCheck.cpp
  RewriteONNXForZHigh.cpp
  ONNXToZHighCommon.cpp

  DEPENDS
  OMONNXRewriteONNXForZHighIncGen
  libzdnn
  
  LINK_LIBS PUBLIC
  OMLayoutHelper
  OMNNPACompilerOptions
  OMONNXOps
  OMONNXToKrnl
  OMZHighOps


  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_library(OMZHighToONNX
  ZHighToONNX.cpp

  DEPENDS
  OMONNXZHighToONNXIncGen
  libzdnn

  LINK_LIBS PUBLIC
  OMONNXOps
  OMZHighOps

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_library(OMDevicePlacement
  DevicePlacement.cpp
  DevicePlacementHeuristic.cpp
  PerfModel.cpp

  DEPENDS
  libzdnn

  LINK_LIBS PUBLIC
  OMONNXOps
  OMONNXToZHigh
  OMRewriteONNXForZHigh
  OMLayoutHelper
  OMNNPAJSONConfig

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_library(OMNNPAGenerateConfigFile
  GenerateConfigFile.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  OMNNPAJSONConfig

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_library(OMNNPAQuantOpSelection
  QuantOpSelection.cpp

  LINK_LIBS PUBLIC
  OMONNXOps
  OMONNXToZHigh
  OMNNPAJSONConfig

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

add_onnx_mlir_library(OMNNPAJSONConfig
  JsonConfigFile.cpp
  JsonConfigObject.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  LLVMSupport

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )

