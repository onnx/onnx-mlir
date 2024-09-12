# SPDX-License-Identifier: Apache-2.0

if(ONNX_MLIR_INSTALL_HEADERS)
  install(FILES OMEntryPoint.h DESTINATION include/onnx-mlir/Runtime)
  install(FILES OMInstrument.h DESTINATION include/onnx-mlir/Runtime)
  install(FILES OMSignature.h DESTINATION include/onnx-mlir/Runtime)
  install(FILES OMTensor.h DESTINATION include/onnx-mlir/Runtime)
  install(FILES OMTensorList.h DESTINATION include/onnx-mlir/Runtime)
  install(FILES OnnxDataType.h DESTINATION include/onnx-mlir/Runtime)
  install(FILES OnnxDataTypeMetaData.inc DESTINATION include/onnx-mlir/Runtime)
endif()
