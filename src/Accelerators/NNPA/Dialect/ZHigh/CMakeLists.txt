add_onnx_mlir_dialect(ZHigh zhigh)
add_onnx_mlir_dialect_doc(zhigh ZHigh.td)

add_onnx_mlir_rewriter(ZHighCombine)

add_onnx_mlir_library(OMZHighOps
  # Top files for ZHigh dialect
  ZHighHelper.cpp
  ZHighOps.cpp
  ZHighCombine.cpp

  # Support for shape inference and verifiers
  # Please list in the alphabetical order.
  ZHighOps/ShapeHelper.cpp
  ZHighOps/BatchNorm.cpp
  ZHighOps/Conv2D.cpp
  ZHighOps/Elementwise.cpp
  ZHighOps/GRU.cpp
  ZHighOps/LSTM.cpp
  ZHighOps/MatMul.cpp
  ZHighOps/MeanReduce2D.cpp
  ZHighOps/Pooling.cpp
  ZHighOps/Softmax.cpp
  ZHighOps/Stick.cpp
  ZHighOps/StickForGRU.cpp
  ZHighOps/StickForLSTM.cpp
  ZHighOps/Unstick.cpp

  DEPENDS
  OMLayoutHelper
  OMZHighIncGen
  OMONNXZHighCombineIncGen
  OMShapeInferenceOpInterfaceIncGen
  libzdnn

  LINK_LIBS PRIVATE
  OMDiagnostic

  LINK_LIBS PUBLIC
  OMONNXOps # Use ONNXShapeHelper 
  OMLayoutHelper
  MLIRIR

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
