// RUN: onnx-mlir-opt --recompose-onnx --canonicalize %s --mlir-print-debuginfo -split-input-file | FileCheck %s


func.func @test_combine_conv_split(%arg0: tensor<1x1x512x512xf32>) -> tensor<1x96x512x512xf32> {
  %0 = onnx.Constant dense<0.00999999976> : tensor<32x1x3x3xf32>
  %1 = onnx.Constant dense<0.00999999976> : tensor<32xf32>
  %2 = onnx.Constant dense<0.00999999976> : tensor<32x1x3x3xf32>
  %3 = onnx.Constant dense<0.00999999976> : tensor<32xf32>
  %4 = onnx.Constant dense<0.00999999976> : tensor<32x1x3x3xf32>
  %5 = onnx.Constant dense<0.00999999976> : tensor<32xf32> 
  %6 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32> loc("conv1")
  %7 = "onnx.Conv"(%arg0, %2, %3) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32> loc("conv2")
  %8 = "onnx.Conv"(%arg0, %4, %5) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32> loc("conv3")
  %9 = "onnx.Relu"(%6)  : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32> loc("relu")
  %10 = "onnx.Sigmoid"(%7) : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32> loc("sigmoid")
  %11 = "onnx.Tanh"(%8) : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32> loc("tanh")
  %12 = "onnx.Concat"(%9, %10, %11) {axis = 1 : si64} : (tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>) -> tensor<1x96x512x512xf32> loc("concat")
  return %12 : tensor<1x96x512x512xf32>

// CHECK-LABEL:  func.func @test_combine_conv_split
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x512x512xf32>
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<32> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.00999999977> : tensor<32x1x3x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0.00999999977> : tensor<32xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Concat"([[VAR_1_]], [[VAR_1_]], [[VAR_1_]]) <{axis = 0 : si64}> : (tensor<32x1x3x3xf32>, tensor<32x1x3x3xf32>, tensor<32x1x3x3xf32>) -> tensor<96x1x3x3xf32> loc([[LOC_FUSED:#.+]])
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.Concat"([[VAR_2_]], [[VAR_2_]], [[VAR_2_]]) <{axis = 0 : si64}> : (tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) -> tensor<96xf32> loc([[LOC_FUSED:#.+]])
// CHECK:           [[VAR_5_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_3_]], [[VAR_4_]]) <{auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]}> : (tensor<1x1x512x512xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>) -> tensor<1x96x512x512xf32> loc([[LOC_FUSED:#.+]])
// CHECK:           [[VAR_6_:%.+]]:3 = "onnx.Split"([[VAR_5_]], [[VAR_0_]]) <{axis = 1 : si64}> : (tensor<1x96x512x512xf32>, tensor<3xi64>) -> (tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>) loc([[LOC_FUSED:#.+]])
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Relu"([[VAR_6_]]#2) : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32> loc([[LOC_RELU:#.+]])
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Sigmoid"([[VAR_6_]]#1) : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32> loc([[LOC_SIGMOID:#.+]])
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Tanh"([[VAR_6_]]#0) : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32> loc([[LOC_TANH:#.+]])
// CHECK:           [[VAR_10_:%.+]] = "onnx.Concat"([[VAR_7_]], [[VAR_8_]], [[VAR_9_]]) <{axis = 1 : si64}> : (tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>) -> tensor<1x96x512x512xf32> loc([[LOC_ORIGINAL_CONCAT:#.+]])
// CHECK:           return [[VAR_10_]] : tensor<1x96x512x512xf32>
// CHECK:         }

// CHECK-DAG:       [[LOC_RELU:#.+]] = loc("relu")
// CHECK-DAG:       [[LOC_SIGMOID:#.+]] = loc("sigmoid")
// CHECK-DAG:       [[LOC_TANH:#.+]] = loc("tanh")
// CHECK-DAG:       [[LOC_ORIGINAL_CONCAT:#.+]] = loc("concat")
// CHECK-DAG:       [[LOC_CONV1:#.+]] = loc("conv1")
// CHECK-DAG:       [[LOC_CONV2:#.+]] = loc("conv2")
// CHECK-DAG:       [[LOC_CONV3:#.+]] = loc("conv3")
// CHECK-DAG:       [[LOC_FUSED]] = loc(fused[[[LOC_CONV1]], [[LOC_CONV3]], [[LOC_CONV2]]])
}
