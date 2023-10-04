// RUN: onnx-mlir-opt --mcpu=z16 --maccel=NNPA --zhigh-clip-to-dlfloat -split-input-file %s || FileCheck %s

func.func @should_clip_stick(%arg0: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> { 
  %0 = "zhigh.Stick"(%arg0) {layout = "3DS"} : (tensor<3x4x5xf32>) -> tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>
  %1 = "zhigh.Softmax"(%0) {act_func = "ACT_NONE"} : (tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Unstick"(%1) : (tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32>
  return %2 : tensor<3x4x5xf32>

// CHECK-LABEL:  func.func @should_clip_stick
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<-8.57315738E+9> : tensor<1xf32>
// CHECK:           [[VAR_1_:%.+]] = "onnx.Max"([[PARAM_0_]], [[VAR_0_]]) : (tensor<3x4x5xf32>, tensor<1xf32>) -> tensor<3x4x5xf32>
// CHECK:           [[VAR_2_:%.+]] = "zhigh.Stick"([[VAR_1_]]) {layout = "3DS"} : (tensor<3x4x5xf32>) -> tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Softmax"([[VAR_2_]]) {act_func = "ACT_NONE"} : (tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Unstick"([[VAR_3_]]) : (tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32>
// CHECK:           return [[VAR_4_]] : tensor<3x4x5xf32>
// CHECK:         }
}

// -----

func.func @should_clip_transpose(%arg0: tensor<3x5x4xf32>) -> tensor<3x4x5xf32> {
  %1 = "onnx.Transpose"(%arg0) { perm = [0, 2, 1]} : (tensor<3x5x4xf32>) -> tensor<3x4x5xf32>
  %2 = "zhigh.Stick"(%1) {layout = "3DS"} : (tensor<3x4x5xf32>) -> tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>
  %3 = "zhigh.Softmax"(%2) {act_func = "ACT_NONE"} : (tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>
  %4 = "zhigh.Unstick"(%3) : (tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32>
  return %4 : tensor<3x4x5xf32>

// CHECK-LABEL:  func.func @should_clip_transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<3x5x4xf32>) -> tensor<3x4x5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<-8.57315738E+9> : tensor<1xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[PARAM_0_]]) {perm = [0, 2, 1]} : (tensor<3x5x4xf32>) -> tensor<3x4x5xf32>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Max"([[VAR_1_]], [[VAR_0_]]) : (tensor<3x4x5xf32>, tensor<1xf32>) -> tensor<3x4x5xf32>
// CHECK:           [[VAR_3_:%.+]] = "zhigh.Stick"([[VAR_2_]]) {layout = "3DS"} : (tensor<3x4x5xf32>) -> tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_4_:%.+]] = "zhigh.Softmax"([[VAR_3_]]) {act_func = "ACT_NONE"} : (tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>
// CHECK:           [[VAR_5_:%.+]] = "zhigh.Unstick"([[VAR_4_]]) : (tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32>
// CHECK:           return [[VAR_5_]] : tensor<3x4x5xf32>
// CHECK:         }
}

// -----

// Do not clip because the input comes from a zTensor via Unstick.
func.func @donot_clip_stick(%arg0: tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32> { 
  %0 = "zhigh.Unstick"(%arg0) : (tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32>
  %1 = "zhigh.Stick"(%0) {layout = "3DS"} : (tensor<3x4x5xf32>) -> tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>
  %2 = "zhigh.Softmax"(%1) {act_func = "ACT_NONE"} : (tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>
  %3 = "zhigh.Unstick"(%2) : (tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32>
  return %3 : tensor<3x4x5xf32>

// CHECK-LABEL: donot_clip_stick
// CHECK: zhigh.Unstick
// CHECK: zhigh.Stick
// CHECK: zhigh.Softmax
// CHECK: zhigh.Unstick
}

// -----

// Do not clip because transpose does not change the zTensor.
func.func @donot_clip_stick_transpose(%arg0: tensor<3x5x4xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<3x5x4xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<3x5x4xf32>
  %1 = "onnx.Transpose"(%0) { perm = [0, 2, 1]} : (tensor<3x5x4xf32>) -> tensor<3x4x5xf32>
  %2 = "zhigh.Stick"(%1) {layout = "3DS"} : (tensor<3x4x5xf32>) -> tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>
  %3 = "zhigh.Softmax"(%2) {act_func = "ACT_NONE"} : (tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>
  %4 = "zhigh.Unstick"(%3) : (tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32>
  return %4 : tensor<3x4x5xf32>

// CHECK-LABEL: donot_clip_stick_transpose
// CHECK: zhigh.Unstick
// CHECK: onnx.Transpose.
// CHECK: zhigh.Stick
// CHECK: zhigh.Softmax
// CHECK: zhigh.Unstick
}

// -----

// Do not clip because concat does not change the zTensor.
func.func @donot_clip_stick_concat(%arg0: tensor<3x2x5xf32, #zhigh.layout<{dataLayout = "3D"}>>, %arg1: tensor<3x2x5xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<3x4x5xf32> {
  %0 = "zhigh.Unstick"(%arg0) : (tensor<3x2x5xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<3x2x5xf32>
  %1 = "zhigh.Unstick"(%arg1) : (tensor<3x2x5xf32, #zhigh.layout<{dataLayout = "3D"}>>) -> tensor<3x2x5xf32>
  %2 = "onnx.Concat"(%0, %1) { axis = 1 : si64} : (tensor<3x2x5xf32>, tensor<3x2x5xf32>) -> tensor<3x4x5xf32>
  %3 = "zhigh.Stick"(%2) {layout = "3DS"} : (tensor<3x4x5xf32>) -> tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>
  %4 = "zhigh.Softmax"(%3) {act_func = "ACT_NONE"} : (tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>
  %5 = "zhigh.Unstick"(%4) : (tensor<3x4x5xf32, #zhigh.layout<{dataLayout = "3DS"}>>) -> tensor<3x4x5xf32>
  return %5 : tensor<3x4x5xf32>

// CHECK-LABEL: donot_clip_stick_concat
// CHECK: zhigh.Unstick
// CHECK: zhigh.Unstick
// CHECK: onnx.Concat.
// CHECK: zhigh.Stick
// CHECK: zhigh.Softmax
// CHECK: zhigh.Unstick
}
