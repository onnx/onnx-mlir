
// This file contains hand-written definitions for ONNX-T operators useful for BERT.
// These manual definitions will be deleted once we add support for these operators
// in the onnx-mlir scripts.
 
import_handler_map_["Gelu"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXGeluOp>;
