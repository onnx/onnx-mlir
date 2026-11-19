// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --rewrite-onnx-for-zhigh %s | FileCheck --check-prefix=CLAMP %s
// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --nnpa-saturation=mask --rewrite-onnx-for-zhigh %s | FileCheck --check-prefix=CLAMP %s
// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --nnpa-saturation=max --rewrite-onnx-for-zhigh %s | FileCheck --check-prefix=KEEP %s
// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --nnpa-saturation=limit --rewrite-onnx-for-zhigh %s | FileCheck --check-prefix=KEEP %s
// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --nnpa-saturation=limit1e9 --rewrite-onnx-for-zhigh %s | FileCheck --check-prefix=KEEP %s
// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --nnpa-disable-saturation --rewrite-onnx-for-zhigh %s | FileCheck --check-prefix=KEEP %s
// RUN: not onnx-mlir-opt --march=z16 --maccel=NNPA --nnpa-saturation=foo --rewrite-onnx-for-zhigh %s 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: not onnx-mlir-opt --march=z16 --maccel=NNPA --nnpa-saturation=limit0.5 --rewrite-onnx-for-zhigh %s 2>&1 | FileCheck --check-prefix=INVALID %s
// RUN: not onnx-mlir-opt --march=z16 --maccel=NNPA --nnpa-saturation=limit1e12 --rewrite-onnx-for-zhigh %s 2>&1 | FileCheck --check-prefix=INVALID %s

// COM: --nnpa-saturation selects how values are kept in the dlfloat16 range.
// COM: Only `mask` (the default) clamps mask fill constants feeding a Softmax;
// COM: `max`, `limit`, `limit<value>` and --nnpa-disable-saturation leave
// COM: them unchanged. The clamping itself is tested in
// COM: rewrite-onnx-for-zhigh-clamp-mask.mlir, and the stick limits in
// COM: zlow-stick-expansion-saturation-limit.mlir and
// COM: onnx-to-krnl/mul-add-stick-saturation-limit.mlir.

func.func @test_saturation_modes(%pad: tensor<1x4xi1>, %s: tensor<3x4xf32>) -> tensor<3x4xf32> {
  %min = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
  %zero = onnx.Constant dense<0.000000e+00> : tensor<f32>
  %m = "onnx.Where"(%pad, %min, %zero) : (tensor<1x4xi1>, tensor<f32>, tensor<f32>) -> tensor<1x4xf32>
  %a = "onnx.Add"(%s, %m) : (tensor<3x4xf32>, tensor<1x4xf32>) -> tensor<3x4xf32>
  %y = "onnx.Softmax"(%a) {axis = -1 : si64} : (tensor<3x4xf32>) -> tensor<3x4xf32>
  return %y : tensor<3x4xf32>

// CLAMP-LABEL:  func.func @test_saturation_modes
// CLAMP:          [[MIN:%.+]] = onnx.Constant dense<-1.000000e+09> : tensor<f32>
// CLAMP-NOT:      -3.40282347E+38
// CLAMP:          "onnx.Where"({{.*}}, [[MIN]], {{.*}})

// KEEP-LABEL:   func.func @test_saturation_modes
// KEEP:           [[MIN:%.+]] = onnx.Constant dense<-3.40282347E+38> : tensor<f32>
// KEEP-NOT:       -1.000000e+09
// KEEP:           "onnx.Where"({{.*}}, [[MIN]], {{.*}})

// INVALID:      for the --nnpa-saturation option: invalid value
// INVALID-SAME: expected mask, max, limit, or limit<value>
}
