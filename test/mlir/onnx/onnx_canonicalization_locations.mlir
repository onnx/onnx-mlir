// RUN: onnx-mlir-opt --shape-inference --canonicalize="test-convergence=true" --shape-inference --cse %s -split-input-file --mlir-print-debuginfo | FileCheck %s


func.func @test_reorder_relu_maxpool(%arg0: tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<1x64x32x32xf32>) -> tensor<1x64x32x32xf32> loc("Relu")
  %1 = "onnx.MaxPoolSingleOut"(%0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], onnx_node_name = "onnx.MaxPoolSingleOut_1", storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32> loc("MaxPool")
  return %1 : tensor<1x64x16x16xf32>

  // CHECK-LABEL: func @test_reorder_relu_maxpool
  // CHECK:           [[VAR_0_:%.+]] = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", ceil_mode = 0 : si64, kernel_shape = [2, 2], storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x64x32x32xf32>) -> tensor<1x64x16x16xf32> loc([[LOC_MAX_POOL:#.+]])
  // CHECK:           [[VAR_1_:%.+]] = "onnx.Relu"([[VAR_0_]]) : (tensor<1x64x16x16xf32>) -> tensor<1x64x16x16xf32> loc([[LOC_RELU:#.+]])
  // CHECK-DAG:       [[LOC_MAX_POOL:#.+]] = loc("MaxPool")
  // CHECK-DAG:       [[LOC_RELU:#.+]] = loc("Relu")
}

// -----

func.func @test_recompose_concat(%arg0: tensor<1x3x4xf32>, %arg1: tensor<1x3x4xf32> ) -> tensor<1x12x4xf32> {
  %0 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} : (tensor<1x3x4xf32>, tensor<1x3x4xf32>) -> tensor<1x6x4xf32> loc("Concat1")
  %1 = "onnx.Concat"(%0, %arg0) {axis = 1 : si64} : (tensor<1x6x4xf32>, tensor<1x3x4xf32>) -> tensor<1x9x4xf32> loc("Concat2")
  %2 = "onnx.Concat"(%1, %arg1) {axis = 1 : si64} : (tensor<1x9x4xf32>, tensor<1x3x4xf32>) -> tensor<1x12x4xf32> loc("Concat3")
  return %2 : tensor<1x12x4xf32>

  // CHECK-LABEL: func @test_recompose_concat
  // CHECK: "onnx.Concat"
  // CHECK-SAME: loc([[LOC_FUSED:#.+]])
  // CHECK-DAG:       [[LOC_C1:#.+]] = loc("Concat1")
  // CHECK-DAG:       [[LOC_C2:#.+]] = loc("Concat2")
  // CHECK-DAG:       [[LOC_C3:#.+]] = loc("Concat3")
  // CHECK:           [[LOC_FUSED]] = loc(fused[[[LOC_C3]], [[LOC_C2]], [[LOC_C1]]]) 
}

// -----

func.func @consecutive_clips(%arg0: tensor<3x1024x1024xf32>) -> (tensor<3x1024x1024xf32> {onnx.name = "output"}) {
  %0 = onnx.Constant dense<-5.000000e-01> : tensor<f32>
  %1 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %2 = onnx.Constant dense<-3.000000e-01> : tensor<f32>
  %3 = onnx.Constant dense<3.000000e-01> : tensor<f32>
  %4 = "onnx.Clip"(%arg0, %0, %1) : (tensor<3x1024x1024xf32>, tensor<f32>, tensor<f32>) -> tensor<3x1024x1024xf32> loc("Clip1")
  %5 = "onnx.Clip"(%4, %2, %3) : (tensor<3x1024x1024xf32>, tensor<f32>, tensor<f32>) -> tensor<3x1024x1024xf32> loc("Clip2")
  onnx.Return %5 : tensor<3x1024x1024xf32>

  // CHECK-LABEL: func.func @consecutive_clips
  // CHECK: onnx.Max  
  // CHECK-SAME: loc([[FUSED_LOC:#.+]])
  // CHECK: onnx.Min
  // CHECK-SAME: loc([[FUSED_LOC]])

  // CHECK: onnx.Clip
  // CHECK-SAME: loc([[FUSED_LOC]])

  // CHECK-DAG: [[LOC_CLIP1:#.+]] = loc("Clip1")
  // CHECK-DAG: [[LOC_CLIP2:#.+]] = loc("Clip2")
  // CHECK: [[FUSED_LOC]] = loc(fused[[[LOC_CLIP2]], [[LOC_CLIP1]]])
}
