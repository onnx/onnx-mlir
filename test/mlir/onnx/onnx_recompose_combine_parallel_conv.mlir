// RUN: onnx-mlir  --useOnnxModelTypes=false --EmitONNXIR --printIR %s | FileCheck %s

func.func @test_conv_concat_simple(%arg0: tensor<1x1x512x512xf32>) -> tensor<1x64x512x512xf32> {
  %0 = onnx.Constant dense<0.00999999977> : tensor<32x1x3x3xf32>
  %1 = onnx.Constant dense<0.00999999977> : tensor<32xf32>
  %2 = onnx.Constant dense<0.00999999977> : tensor<32x1x3x3xf32>
  %3 = onnx.Constant dense<0.00999999977> : tensor<32xf32>
  %4 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", group = 1 : si64, onnx_node_name = "onnx.Conv_0", pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
  %5 = "onnx.Conv"(%arg0, %2, %3) {auto_pad = "NOTSET", group = 1 : si64, onnx_node_name = "onnx.Conv_1", pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
  %6 = "onnx.Concat"(%4, %5) {axis = 1 : si64, onnx_node_name = "onnx.Concat_2"} : (tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>) -> tensor<1x64x512x512xf32>
  return %6 : tensor<1x64x512x512xf32>

  // CHECK-LABEL: func @test_conv_concat_simple
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x512x512xf32>) -> tensor<1x64x512x512xf32> {
  // CHECK:      [[VAR_0_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<64x1x3x3xf32>
  
  // CHECK:      [[VAR_1_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<64xf32>

  // CHECK:     [[VAR_2_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]])
  // CHECK-SAME:     : (tensor<1x1x512x512xf32>, tensor<64x1x3x3xf32>, tensor<64xf32>) -> tensor<1x64x512x512xf32>
  // CHECK-NEXT:     return [[VAR_2_]] : tensor<1x64x512x512xf32>

}

func.func @test_conv_concat_complex(%arg0: tensor<1x1x512x512xf32>) -> tensor<1x192x512x512xf32> {
  %0 = onnx.Constant dense<0.00999999977> : tensor<32x1x3x3xf32>
  %1 = onnx.Constant dense<0.00999999977> : tensor<32xf32>
  %2 = onnx.Constant dense<0.00999999977> : tensor<32x1x3x3xf32>
  %3 = onnx.Constant dense<0.00999999977> : tensor<32xf32>
  %4 = onnx.Constant dense<0.00999999977> : tensor<32x1x3x3xf32>
  %5 = onnx.Constant dense<0.00999999977> : tensor<32xf32>
  %6 = onnx.Constant dense<0.00999999977> : tensor<32x1x3x3xf32>
  %7 = onnx.Constant dense<0.00999999977> : tensor<32xf32>
  %8 = onnx.Constant dense<0.00999999977> : tensor<32x1x3x3xf32>
  %9 = onnx.Constant dense<0.00999999977> : tensor<32xf32>
  %10 = onnx.Constant dense<0.00999999977> : tensor<32x1x3x3xf32>
  %11 = onnx.Constant dense<0.00999999977> : tensor<32xf32>
  %12 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", group = 1 : si64, onnx_node_name = "onnx.Conv_0", pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
  %13 = "onnx.Conv"(%arg0, %2, %3) {auto_pad = "NOTSET", group = 1 : si64, onnx_node_name = "onnx.Conv_1", pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
  %14 = "onnx.Conv"(%arg0, %4, %5) {auto_pad = "NOTSET", group = 1 : si64, onnx_node_name = "onnx.Conv_2", pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
  %15 = "onnx.Conv"(%arg0, %6, %7) {auto_pad = "NOTSET", group = 1 : si64, onnx_node_name = "onnx.Conv_3", pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
  %16 = "onnx.Conv"(%arg0, %8, %9) {auto_pad = "NOTSET", group = 1 : si64, onnx_node_name = "onnx.Conv_4", pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
  %17 = "onnx.Conv"(%arg0, %10, %11) {auto_pad = "NOTSET", group = 1 : si64, onnx_node_name = "onnx.Conv_5", pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
  %18 = "onnx.Concat"(%12, %13, %14, %15, %16, %17) {axis = 1 : si64, onnx_node_name = "onnx.Concat_6"} : (tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>) -> tensor<1x192x512x512xf32>
  return %18 : tensor<1x192x512x512xf32>

  // CHECK-LABEL: func @test_conv_concat_complex
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x512x512xf32>) -> tensor<1x192x512x512xf32> {
  // CHECK:      [[VAR_0_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<192x1x3x3xf32>
  
  // CHECK:      [[VAR_1_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<192xf32>

  // CHECK:     [[VAR_2_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]])
  // CHECK-SAME:     : (tensor<1x1x512x512xf32>, tensor<192x1x3x3xf32>, tensor<192xf32>) -> tensor<1x192x512x512xf32>
  // CHECK-NEXT:     return [[VAR_2_]] : tensor<1x192x512x512xf32>

}


func.func @test_conv_concat_fail(%arg0: tensor<1x3x64x64xf32>) -> tensor<1x64x66x66xf32> {
  %0 = onnx.Constant dense<0.00999999977> : tensor<16x3x1x1xf32>
  %1 = onnx.Constant dense<0.00999999977> : tensor<16x3x3x3xf32>
  %2 = onnx.Constant dense<0.00999999977> : tensor<32x3x5x5xf32>
  %3 = "onnx.NoValue"() {onnx_node_name = "onnx.NoValue_0", value} : () -> none
  %4 = "onnx.Conv"(%arg0, %0, %3) {auto_pad = "NOTSET", group = 1 : si64, onnx_node_name = "onnx.Conv_1", pads = [1, 1, 1, 1]} : (tensor<1x3x64x64xf32>, tensor<16x3x1x1xf32>, none) -> tensor<1x16x66x66xf32>
  %5 = "onnx.Conv"(%arg0, %1, %3) {auto_pad = "NOTSET", group = 1 : si64, onnx_node_name = "onnx.Conv_2", pads = [2, 2, 2, 2]} : (tensor<1x3x64x64xf32>, tensor<16x3x3x3xf32>, none) -> tensor<1x16x66x66xf32>
  %6 = "onnx.Conv"(%arg0, %2, %3) {auto_pad = "NOTSET", group = 1 : si64, onnx_node_name = "onnx.Conv_3", pads = [3, 3, 3, 3]} : (tensor<1x3x64x64xf32>, tensor<32x3x5x5xf32>, none) -> tensor<1x32x66x66xf32>
  %7 = "onnx.Concat"(%4, %5, %6) {axis = 1 : si64, onnx_node_name = "onnx.Concat_4"} : (tensor<1x16x66x66xf32>, tensor<1x16x66x66xf32>, tensor<1x32x66x66xf32>) -> tensor<1x64x66x66xf32>
  return %7 : tensor<1x64x66x66xf32>

  // CHECK-LABEL: func @test_conv_concat_fail
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x3x64x64xf32>) -> tensor<1x64x66x66xf32> {
  // CHECK:      [[VAR_0_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<16x3x1x1xf32>
  // CHECK:      [[VAR_1_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<16x3x3x3xf32>
  // CHECK:      [[VAR_2_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<32x3x5x5xf32>
  // CHECK: [[VAR_NO_VALUE:%.+]] = "onnx.NoValue"()
  // CHECK:      [[VAR_3_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_NO_VALUE]]) <{auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]}> {onnx_node_name = "onnx.Conv_1"}
  // CHECK-SAME:     : (tensor<1x3x64x64xf32>, tensor<16x3x1x1xf32>, none) -> tensor<1x16x66x66xf32>
  // CHECK:     [[VAR_4_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_1_]], [[VAR_NO_VALUE]]) <{auto_pad = "NOTSET", group = 1 : si64, pads = [2, 2, 2, 2]}> {onnx_node_name = "onnx.Conv_2"}
  // CHECK-SAME:     : (tensor<1x3x64x64xf32>, tensor<16x3x3x3xf32>, none) -> tensor<1x16x66x66xf32>
  // CHECK:     [[VAR_5_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_2_]], [[VAR_NO_VALUE]]) <{auto_pad = "NOTSET", group = 1 : si64, pads = [3, 3, 3, 3]}> {onnx_node_name = "onnx.Conv_3"}
  // CHECK-SAME:     : (tensor<1x3x64x64xf32>, tensor<32x3x5x5xf32>, none) -> tensor<1x32x66x66xf32>
  // CHECK:     [[VAR_6_:%.+]] = "onnx.Concat"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]])
  // CHECK-NEXT:     return [[VAR_6_]] : tensor<1x64x66x66xf32>

}

func.func @test_combine_conv_split(%arg0: tensor<1x1x512x512xf32>) -> tensor<1x96x512x512xf32> {
  %0 = onnx.Constant dense<0.00999999976> : tensor<32x1x3x3xf32>
  %1 = onnx.Constant dense<0.00999999976> : tensor<32xf32>
  %2 = onnx.Constant dense<0.00999999976> : tensor<32x1x3x3xf32>
  %3 = onnx.Constant dense<0.00999999976> : tensor<32xf32>
  %4 = onnx.Constant dense<0.00999999976> : tensor<32x1x3x3xf32>
  %5 = onnx.Constant dense<0.00999999976> : tensor<32xf32>
  %6 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", group = 1 : si64, onnx_node_name = "onnx.Conv_0", pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
  %7 = "onnx.Conv"(%arg0, %2, %3) {auto_pad = "NOTSET", group = 1 : si64, onnx_node_name = "onnx.Conv_1", pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
  %8 = "onnx.Conv"(%arg0, %4, %5) {auto_pad = "NOTSET", group = 1 : si64, onnx_node_name = "onnx.Conv_2", pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
  %9 = "onnx.Relu"(%6) {onnx_node_name = "ReLU_1"} : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
  %10 = "onnx.Sigmoid"(%7) {onnx_node_name = "Sigmoid_2"} : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
  %11 = "onnx.Tanh"(%8) {onnx_node_name = "Tanh_3"} : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
  %12 = "onnx.Concat"(%9, %10, %11) {axis = 1 : si64, onnx_node_name = "onnx.Concat_4"} : (tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>) -> tensor<1x96x512x512xf32>
  return %12 : tensor<1x96x512x512xf32>

// CHECK-LABEL: func @test_combine_conv_split
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x512x512xf32>) -> tensor<1x96x512x512xf32> {
// CHECK:      [[CONST_SPLIT_:%.+]] = onnx.Constant dense<32> : tensor<3xi64>
// CHECK:      [[VAR_0_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<96x1x3x3xf32>
// CHECK:      [[VAR_1_:%.+]] = onnx.Constant dense<{{.*}}> : tensor<96xf32>
// CHECK:      [[CONV_OUT_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_0_]], [[VAR_1_]])
// CHECK-SAME:     : (tensor<1x1x512x512xf32>, tensor<96x1x3x3xf32>, tensor<96xf32>) -> tensor<1x96x512x512xf32>
// CHECK: [[VAR_2_:[^ ]+]]:3 = "onnx.Split"([[CONV_OUT_]], [[CONST_SPLIT_]]) <{axis = 1 : si64}> {onnx_node_name = "onnx.Split_6"} : (tensor<1x96x512x512xf32>, tensor<3xi64>) -> (tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>)
// CHECK: [[VAR_3_:%.+]] = "onnx.Relu"([[VAR_2_]]#2) {onnx_node_name = "ReLU_1"} : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
// CHECK: [[VAR_4_:%.+]] = "onnx.Sigmoid"([[VAR_2_]]#1) {onnx_node_name = "Sigmoid_2"} : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
// CHECK: [[VAR_5_:%.+]] = "onnx.Tanh"([[VAR_2_]]#0) {onnx_node_name = "Tanh_3"} : (tensor<1x32x512x512xf32>) -> tensor<1x32x512x512xf32>
// CHECK: [[FINAL_OUT:%.+]] = "onnx.Concat"([[VAR_3_]], [[VAR_4_]], [[VAR_5_]]) <{axis = 1 : si64}> {onnx_node_name = "onnx.Concat_4_7"} : (tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>) -> tensor<1x96x512x512xf32>
// CHECK: return [[FINAL_OUT]] : tensor<1x96x512x512xf32>

}

func.func @test_conv_concat_dependency(%arg0: tensor<1x1x512x512xf32>) -> tensor<1x64x512x512xf32> {
  %0 = onnx.Constant dense<0.00999999977> : tensor<32x1x3x3xf32>
  %1 = onnx.Constant dense<0.00999999977> : tensor<32xf32>
  %2 = onnx.Constant dense<0.00999999977> : tensor<32x1x3x3xf32>
  %3 = onnx.Constant dense<0.00999999977> : tensor<32xf32>
  %4 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
  %reduceAxes = onnx.Constant dense<[0, 2, 3]> : tensor<3xi64>
  %reduced = "onnx.ReduceMean"(%4, %reduceAxes) {keepdims = 0 : si64} : (tensor<1x32x512x512xf32>, tensor<3xi64>) -> tensor<32xf32>
  %5 = "onnx.Conv"(%arg0, %2, %reduced) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
  %6 = "onnx.Concat"(%4, %5) {axis = 1 : si64} : (tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>) -> tensor<1x64x512x512xf32>
  return %6 : tensor<1x64x512x512xf32>

// COM: Can not be rewritten as there is a def-use chain between the Convs
// CHECK-LABEL:  func.func @test_conv_concat_dependency
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x512x512xf32>) -> tensor<1x64x512x512xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<[0, 2, 3]> : tensor<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.00999999977> : tensor<32x1x3x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<0.00999999977> : tensor<32xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_1_]], [[VAR_2_]]) <{auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]}> {onnx_node_name = "onnx.Conv_8"} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.ReduceMean"([[VAR_3_]], [[VAR_0_]]) <{keepdims = 0 : si64, noop_with_empty_axes = 0 : si64}> {onnx_node_name = "onnx.ReduceMean_9"} : (tensor<1x32x512x512xf32>, tensor<3xi64>) -> tensor<32xf32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_1_]], [[VAR_4_]]) <{auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]}> {onnx_node_name = "onnx.Conv_10"} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Concat"([[VAR_3_]], [[VAR_5_]]) <{axis = 1 : si64}> {onnx_node_name = "onnx.Concat_11"} : (tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>) -> tensor<1x64x512x512xf32>
// CHECK:           return [[VAR_6_]] : tensor<1x64x512x512xf32>
// CHECK:         }
}

func.func @test_conv_concat_not_static_shape(%arg0: tensor<1x1x512x512xf32>, %0: tensor<*xf32>) -> tensor<1x64x512x512xf32> {
  %1 = onnx.Constant dense<0.00999999977> : tensor<32xf32>
  %2 = onnx.Constant dense<0.00999999977> : tensor<32x1x3x3xf32>
  %3 = onnx.Constant dense<0.00999999977> : tensor<32xf32>
  %4 = "onnx.Conv"(%arg0, %0, %1) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<*xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
  %5 = "onnx.Conv"(%arg0, %2, %3) {auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x?x512x512xf32>
  %6 = "onnx.Concat"(%4, %5) {axis = 1 : si64} : (tensor<1x32x512x512xf32>, tensor<1x?x512x512xf32>) -> tensor<1x64x512x512xf32>
  return %6 : tensor<1x64x512x512xf32>

// CHECK-LABEL:  func.func @test_conv_concat_not_static_shape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x512x512xf32>, [[PARAM_1_:%.+]]: tensor<*xf32>) -> tensor<1x64x512x512xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<0.00999999977> : tensor<32xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<0.00999999977> : tensor<32x1x3x3xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[PARAM_1_]], [[VAR_0_]]) <{auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]}> {onnx_node_name = "onnx.Conv_12"} : (tensor<1x1x512x512xf32>, tensor<*xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_1_]], [[VAR_0_]]) <{auto_pad = "NOTSET", group = 1 : si64, pads = [1, 1, 1, 1]}> {onnx_node_name = "onnx.Conv_13"} : (tensor<1x1x512x512xf32>, tensor<32x1x3x3xf32>, tensor<32xf32>) -> tensor<1x32x512x512xf32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Concat"([[VAR_2_]], [[VAR_3_]]) <{axis = 1 : si64}> {onnx_node_name = "onnx.Concat_14"} : (tensor<1x32x512x512xf32>, tensor<1x32x512x512xf32>) -> tensor<1x64x512x512xf32>
// CHECK:           return [[VAR_4_]] : tensor<1x64x512x512xf32>
// CHECK:         }
}


func.func @complex_and_bias_none(%arg0: tensor<1x16x160x256xf32>, %wts0: tensor<7x16x3x3xf32>, %wts1: tensor<7x16x3x3xf32>, %wts2: tensor<7x16x3x3xf32>, %wts3: tensor<7x16x3x3xf32>) -> (tensor<1x7x320x512xf32>) {
    %0 = onnx.Constant dense<[1, 7, 320, 512]> : tensor<4xi64>
    %1 = onnx.Constant dense<[1, 7, 160, 1, 512]> : tensor<5xi64>
    %2 = onnx.Constant dense<[1, 7, 160, 256, 1]> : tensor<5xi64>
    %15 = "onnx.NoValue"() {value} : () -> none
    %25 = "onnx.Conv"(%arg0, %wts0, %15) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x16x160x256xf32>, tensor<7x16x3x3xf32>, none) -> tensor<1x7x160x256xf32>
    %26 = "onnx.Conv"(%arg0, %wts1, %15) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x16x160x256xf32>, tensor<7x16x3x3xf32>, none) -> tensor<1x7x160x256xf32>
    %27 = "onnx.Conv"(%arg0, %wts2, %15) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x16x160x256xf32>, tensor<7x16x3x3xf32>, none) -> tensor<1x7x160x256xf32>
    %28 = "onnx.Conv"(%arg0, %wts3, %15) {auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]} : (tensor<1x16x160x256xf32>, tensor<7x16x3x3xf32>, none) -> tensor<1x7x160x256xf32>
    %29 = "onnx.Reshape"(%25, %2) {allowzero = 0 : si64} : (tensor<1x7x160x256xf32>, tensor<5xi64>) -> tensor<1x7x160x256x1xf32>
    %30 = "onnx.Reshape"(%26, %2) {allowzero = 0 : si64} : (tensor<1x7x160x256xf32>, tensor<5xi64>) -> tensor<1x7x160x256x1xf32>
    %31 = "onnx.Reshape"(%27, %2) {allowzero = 0 : si64} : (tensor<1x7x160x256xf32>, tensor<5xi64>) -> tensor<1x7x160x256x1xf32>
    %32 = "onnx.Reshape"(%28, %2) {allowzero = 0 : si64} : (tensor<1x7x160x256xf32>, tensor<5xi64>) -> tensor<1x7x160x256x1xf32>
    %33 = "onnx.Concat"(%29, %31) {axis = -1 : si64} : (tensor<1x7x160x256x1xf32>, tensor<1x7x160x256x1xf32>) -> tensor<1x7x160x256x2xf32>
    %34 = "onnx.Concat"(%32, %30) {axis = -1 : si64} : (tensor<1x7x160x256x1xf32>, tensor<1x7x160x256x1xf32>) -> tensor<1x7x160x256x2xf32>
    %35 = "onnx.Reshape"(%33, %1) {allowzero = 0 : si64} : (tensor<1x7x160x256x2xf32>, tensor<5xi64>) -> tensor<1x7x160x1x512xf32>
    %36 = "onnx.Reshape"(%34, %1) {allowzero = 0 : si64} : (tensor<1x7x160x256x2xf32>, tensor<5xi64>) -> tensor<1x7x160x1x512xf32>
    %37 = "onnx.Concat"(%35, %36) {axis = -2 : si64} : (tensor<1x7x160x1x512xf32>, tensor<1x7x160x1x512xf32>) -> tensor<1x7x160x2x512xf32>
    %38 = "onnx.Reshape"(%37, %0) {allowzero = 0 : si64} : (tensor<1x7x160x2x512xf32>, tensor<4xi64>) -> tensor<1x7x320x512xf32>
    onnx.Return %38 : tensor<1x7x320x512xf32>
    
// CHECK-LABEL:  func.func @complex_and_bias_none
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x16x160x256xf32>, [[PARAM_1_:%.+]]: tensor<7x16x3x3xf32>, [[PARAM_2_:%.+]]: tensor<7x16x3x3xf32>, [[PARAM_3_:%.+]]: tensor<7x16x3x3xf32>, [[PARAM_4_:%.+]]: tensor<7x16x3x3xf32>) -> tensor<1x7x320x512xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<7> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<[1, 7, 320, 512]> : tensor<4xi64>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense<[1, 7, 160, 1, 512]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[1, 7, 160, 256, 1]> : tensor<5xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = "onnx.NoValue"() <{value}> {onnx_node_name = "onnx.NoValue_15"} : () -> none
// CHECK-DAG:       [[VAR_5_:%.+]] = "onnx.Concat"([[PARAM_4_]], [[PARAM_3_]], [[PARAM_2_]], [[PARAM_1_]]) <{axis = 0 : si64}> {onnx_node_name = "onnx.Concat_16"} : (tensor<7x16x3x3xf32>, tensor<7x16x3x3xf32>, tensor<7x16x3x3xf32>, tensor<7x16x3x3xf32>) -> tensor<28x16x3x3xf32>
// CHECK:           [[VAR_6_:%.+]] = "onnx.Conv"([[PARAM_0_]], [[VAR_5_]], [[VAR_4_]]) <{auto_pad = "NOTSET", group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1]}> {onnx_node_name = "onnx.Conv_17"} : (tensor<1x16x160x256xf32>, tensor<28x16x3x3xf32>, none) -> tensor<1x28x160x256xf32>
// CHECK:           [[VAR_7_:%.+]]:4 = "onnx.Split"([[VAR_6_]], [[VAR_0_]]) <{axis = 1 : si64}> {onnx_node_name = "onnx.Split_18"} : (tensor<1x28x160x256xf32>, tensor<4xi64>) -> (tensor<1x7x160x256xf32>, tensor<1x7x160x256xf32>, tensor<1x7x160x256xf32>, tensor<1x7x160x256xf32>)
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Reshape"([[VAR_7_]]#3, [[VAR_3_]]) <{allowzero = 0 : si64}> {onnx_node_name = "onnx.Reshape_19"} : (tensor<1x7x160x256xf32>, tensor<5xi64>) -> tensor<1x7x160x256x1xf32>
// CHECK-DAG:       [[VAR_9_:%.+]] = "onnx.Reshape"([[VAR_7_]]#2, [[VAR_3_]]) <{allowzero = 0 : si64}> {onnx_node_name = "onnx.Reshape_20"} : (tensor<1x7x160x256xf32>, tensor<5xi64>) -> tensor<1x7x160x256x1xf32>
// CHECK-DAG:       [[VAR_10_:%.+]] = "onnx.Reshape"([[VAR_7_]]#1, [[VAR_3_]]) <{allowzero = 0 : si64}> {onnx_node_name = "onnx.Reshape_21"} : (tensor<1x7x160x256xf32>, tensor<5xi64>) -> tensor<1x7x160x256x1xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = "onnx.Reshape"([[VAR_7_]]#0, [[VAR_3_]]) <{allowzero = 0 : si64}> {onnx_node_name = "onnx.Reshape_22"} : (tensor<1x7x160x256xf32>, tensor<5xi64>) -> tensor<1x7x160x256x1xf32>
// CHECK-DAG:       [[VAR_12_:%.+]] = "onnx.Concat"([[VAR_8_]], [[VAR_10_]]) <{axis = 4 : si64}> {onnx_node_name = "onnx.Concat_23"} : (tensor<1x7x160x256x1xf32>, tensor<1x7x160x256x1xf32>) -> tensor<1x7x160x256x2xf32>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Concat"([[VAR_11_]], [[VAR_9_]]) <{axis = 4 : si64}> {onnx_node_name = "onnx.Concat_24"} : (tensor<1x7x160x256x1xf32>, tensor<1x7x160x256x1xf32>) -> tensor<1x7x160x256x2xf32>
// CHECK-DAG:       [[VAR_14_:%.+]] = "onnx.Reshape"([[VAR_12_]], [[VAR_2_]]) <{allowzero = 0 : si64}> {onnx_node_name = "onnx.Reshape_25"} : (tensor<1x7x160x256x2xf32>, tensor<5xi64>) -> tensor<1x7x160x1x512xf32>
// CHECK-DAG:       [[VAR_15_:%.+]] = "onnx.Reshape"([[VAR_13_]], [[VAR_2_]]) <{allowzero = 0 : si64}> {onnx_node_name = "onnx.Reshape_26"} : (tensor<1x7x160x256x2xf32>, tensor<5xi64>) -> tensor<1x7x160x1x512xf32>
// CHECK:           [[VAR_16_:%.+]] = "onnx.Concat"([[VAR_14_]], [[VAR_15_]]) <{axis = 3 : si64}> {onnx_node_name = "onnx.Concat_27"} : (tensor<1x7x160x1x512xf32>, tensor<1x7x160x1x512xf32>) -> tensor<1x7x160x2x512xf32>
// CHECK:           [[VAR_17_:%.+]] = "onnx.Reshape"([[VAR_16_]], [[VAR_1_]]) <{allowzero = 0 : si64}> {onnx_node_name = "onnx.Reshape_28"} : (tensor<1x7x160x2x512xf32>, tensor<4xi64>) -> tensor<1x7x320x512xf32>
// CHECK:           return [[VAR_17_]] : tensor<1x7x320x512xf32>
// CHECK:         }

}

