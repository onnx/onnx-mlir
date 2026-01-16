
#include "XCOMPILERVerify.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

using namespace mlir;
using namespace onnx_mlir;

namespace mlir {

LogicalResult XCOMPILERFusedEltwiseOpVerify(Operation *op) {

  auto fusedEltwiseOp = dyn_cast<XCOMPILERFusedEltwiseOp>(op);
  if (!fusedEltwiseOp)
    return failure();

  StringRef nonlinear = fusedEltwiseOp.getNonlinear();
  StringRef type = fusedEltwiseOp.getType();

  // leakyrelu_alpha, prelu_in, prelu_shift exist only if nonlinear == LEAKYRELU
  bool isLeakyRelu = (nonlinear == "LEAKYRELU");
  if (fusedEltwiseOp.getLeakyreluAlphaAttr() && !isLeakyRelu)
    return op->emitOpError("'leakyrelu_alpha' is only valid when nonlinear is LEAKYRELU");
  if (fusedEltwiseOp.getPreluInAttr() && !isLeakyRelu)
    return op->emitOpError("'prelu_in' is only valid when nonlinear is LEAKYRELU");
  if (fusedEltwiseOp.getPreluShiftAttr() && !isLeakyRelu)
    return op->emitOpError("'prelu_shift' is only valid when nonlinear is LEAKYRELU");

  // clip_min, clip_max exist only if type == CLIP
  bool isClip = (type == "CLIP");
  if (fusedEltwiseOp.getClipMinAttr() && !isClip)
    return op->emitOpError("'clip_min' is only valid when type is CLIP");
  if (fusedEltwiseOp.getClipMaxAttr() && !isClip)
    return op->emitOpError("'clip_max' is only valid when type is CLIP");

  // nonlinear_in_scales and nonlinear_in_zeropoints exist only if nonlinear is not NONE
  bool nonlinearIsNone = (nonlinear == "NONE");
  if (fusedEltwiseOp.getNonlinearInScalesAttr() && nonlinearIsNone)
    return op->emitOpError("'nonlinear_in_scales' is only valid when nonlinear is not NONE");
  if (fusedEltwiseOp.getNonlinearInZeropointsAttr() && nonlinearIsNone)
    return op->emitOpError("'nonlinear_in_zeropoints' is only valid when nonlinear is not NONE");

  return success();
}

} // namespace mlir
