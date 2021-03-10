
// This file contains hand-written definitions for ONNX-T operators useful for BERT.
// These manual definitions will be deleted once we add support for these operators
// in the onnx-mlir scripts.
 
import_handler_map_["Gelu"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXGeluOp>;
import_handler_map_["LayerNormalization"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXLayerNormalizationOp>;
import_handler_map_["SoftmaxCrossEntropy"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXSoftmaxCrossEntropyOp>;
import_handler_map_["SparseSoftmaxCrossEntropy"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXSparseSoftmaxCrossEntropyOp>;
import_handler_map_["TrainableDropout"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXTrainableDropoutOp>;
import_handler_map_["AdamOptimizer"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXAdamOptimizerOp>;
import_handler_map_["SparseSoftmaxCrossEntropyGrad"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXSparseSoftmaxCrossEntropyGradOp>;
import_handler_map_["GatherNDGrad"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXGatherNDGradOp>;
import_handler_map_["GatherGrad"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXGatherGradOp>;
import_handler_map_["GeluGrad"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXGeluGradOp>;
import_handler_map_["SoftmaxGrad"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXSoftmaxGradOp>;
import_handler_map_["DropoutGrad"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXDropoutGradOp>;
import_handler_map_["LayerNormalizationGrad"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXLayerNormalizationGradOp>;
import_handler_map_["SoftmaxCrossEntropyGrad"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXSoftmaxCrossEntropyGradOp>;
import_handler_map_["ReluGrad"] =
   &onnx_mlir::detail::FrontendGenImpl::buildOperation<mlir::ONNXReluGradOp>;

