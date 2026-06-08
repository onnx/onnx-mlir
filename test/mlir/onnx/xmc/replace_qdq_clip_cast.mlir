// RUN: onnx-mlir-opt --replace-qdq-clip-cast %s | FileCheck %s

// Pattern: onnx.Clip(quantized, f32 min/max) -> f32 -> onnx.Cast(f32 -> ui8)
// becomes onnx.XCOMPILERFusedEltwise(type="CLAMP") with y_scale=1.0, zp=0.

// CHECK-LABEL: func.func @test_quantized_clip_cast_ui8
func.func @test_quantized_clip_cast_ui8(
    %arg0: tensor<1x256x256x3x!quant.uniform<u8:f32, 2.000000e+00:128>>)
    -> tensor<1x256x256x3xui8> {
  %min = "onnx.Constant"() {value = dense<0.0> : tensor<f32>} : () -> tensor<f32>
  %max = "onnx.Constant"() {value = dense<255.0> : tensor<f32>} : () -> tensor<f32>
  %clip = "onnx.Clip"(%arg0, %min, %max) :
      (tensor<1x256x256x3x!quant.uniform<u8:f32, 2.000000e+00:128>>, tensor<f32>, tensor<f32>)
      -> tensor<1x256x256x3xf32>
  %cast = "onnx.Cast"(%clip) {saturate = 1 : si64, to = 2 : si64} :
      (tensor<1x256x256x3xf32>) -> tensor<1x256x256x3xui8>
  return %cast : tensor<1x256x256x3xui8>

  // CHECK-NOT: "onnx.Clip"
  // CHECK-NOT: "onnx.Cast"
  // CHECK: %[[NOVAL:.*]] = "onnx.NoValue"()
  // CHECK: %[[FUSED:.*]] = "onnx.XCOMPILERFusedEltwise"(%arg0, %[[NOVAL]])
  // CHECK-SAME: max = 255 : i32
  // CHECK-SAME: min = 0 : i32
  // CHECK-SAME: nonlinear = "NONE"
  // CHECK-SAME: type = "CLAMP"
  // CHECK: return %[[FUSED]] : tensor<1x256x256x3x!quant.uniform<u8:f32, 1.000000e+00:0>>
}
