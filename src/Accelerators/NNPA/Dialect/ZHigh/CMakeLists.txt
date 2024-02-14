add_subdirectory(ZHighOps)

add_onnx_mlir_dialect(ZHigh zhigh)
add_onnx_mlir_dialect_doc(zhigh ZHigh.td)

add_onnx_mlir_library(OMZHighOps
  # Top files for ZHigh dialect.
  ZHighOps.cpp
  ZHighOps/OpHelper.cpp

  # Support for shape inference, verifier, and canonicalization.
  # Please list in the alphabetical order.
  ZHighOps/BatchNorm/BatchNorm.cpp
  ZHighOps/Conv2D/Conv2D.cpp
  ZHighOps/Elementwise/Elementwise.cpp
  ZHighOps/GRU/FixGRUY.cpp
  ZHighOps/GRU/FixGRUYh.cpp
  ZHighOps/GRU/GRU.cpp
  ZHighOps/GRU/GRU.cpp
  ZHighOps/LSTM/LSTM.cpp
  ZHighOps/MatMul/MatMul.cpp
  ZHighOps/MeanReduce2D/MeanReduce2D.cpp
  ZHighOps/Pooling/Pooling.cpp
  ZHighOps/Softmax/Softmax.cpp
  ZHighOps/Stick/Stick.cpp
  ZHighOps/StickForGRU/StickForGRU.cpp
  ZHighOps/StickForLSTM/StickForLSTM.cpp
  ZHighOps/StickifiedConstantOfShape/StickifiedConstantOfShape.cpp
  ZHighOps/Unstick/Unstick.cpp

  DEPENDS
  OMLayoutHelper
  OMZHighIncGen
  OMONNXZHighStickIncGen
  OMONNXZHighUnstickIncGen
  OMShapeInferenceOpInterfaceIncGen
  libzdnn

  LINK_LIBS PRIVATE
  OMDiagnostic

  LINK_LIBS PUBLIC
  OMONNXOps # Use ONNXShapeHelper 
  OMLayoutHelper
  OMShapeHelperOpInterface
  MLIRIR

  ACCEL_INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )
