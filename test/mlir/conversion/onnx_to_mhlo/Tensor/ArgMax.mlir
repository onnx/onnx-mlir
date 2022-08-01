// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo %s --canonicalize -split-input-file | FileCheck %s

func.func @test_argmax_verifier_1(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xi64> {
  %1 = "onnx.ArgMax"(%arg0) { axis = -1 : si64} : (tensor<5x5x1x32xf32>)  -> tensor<*xi64>
  "func.return"(%1) : (tensor<*xi64>) -> ()
// CHECK-LABEL: func @test_argmax_verifier_1(%arg0: tensor<5x5x1x32xf32>)    
// CHECK-DAG:    [[VAR_0_:%.+]] = mhlo.constant dense<0> : tensor<i64>
// CHECK-DAG:    [[VAR_1_:%.+]] = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG:    [[VAR_2_:%.+]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<32xi64>
// CHECK-NEXT:    [[VAR_3_:%.+]] = "mhlo.broadcast_in_dim"([[VAR_2_]]) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<32xi64>) -> tensor<5x5x1x32xi64>
// CHECK-NEXT:    [[VAR_4_:%.+]]:2 = mhlo.reduce(%arg0 init: [[VAR_1_]]), ([[VAR_3_]] init: [[VAR_0_]]) across dimensions = [3] : (tensor<5x5x1x32xf32>, tensor<5x5x1x32xi64>, tensor<f32>, tensor<i64>) -> (tensor<5x5x1xf32>, tensor<5x5x1xi64>)
// CHECK-NEXT:     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i64>, %arg4: tensor<i64>)  {
// CHECK-NEXT:      [[VAR_6_:%.+]] = "mhlo.compare"(%arg1, %arg3) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-NEXT:      [[VAR_7_:%.+]] = "mhlo.select"([[VAR_6_]], %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:      [[VAR_8_:%.+]] = "mhlo.compare"(%arg1, %arg3) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-NEXT:      [[VAR_9_:%.+]] = mhlo.minimum %arg2, %arg4 : tensor<i64>
// CHECK-NEXT:      [[VAR_10_:%.+]] = "mhlo.select"([[VAR_6_]], %arg2, %arg4) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-NEXT:      [[VAR_11_:%.+]] = "mhlo.select"([[VAR_8_]], [[VAR_9_]], [[VAR_10_]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-NEXT:      "mhlo.return"([[VAR_7_]], [[VAR_11_]]) : (tensor<f32>, tensor<i64>) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    [[VAR_5_:%.+]] = "mhlo.reshape"([[VAR_4_]]#1) : (tensor<5x5x1xi64>) -> tensor<5x5x1x1xi64>
}

func.func @test_argmax_verifier_2(%arg0 : tensor<5x?x1x32xf32>) -> tensor<*xi64> {
  %1 = "onnx.ArgMax"(%arg0) { axis = 3 : si64} : (tensor<5x?x1x32xf32>)  -> tensor<*xi64>
  "func.return"(%1) : (tensor<*xi64>) -> ()
// CHECK-LABEL: func @test_argmax_verifier_2(%arg0: tensor<5x?x1x32xf32>)
// CHECK-DAG:    [[C2:%.+]] = arith.constant 2 : index
// CHECK-DAG:    [[C1:%.+]] = arith.constant 1 : index
// CHECK-DAG:    [[C0:%.+]] = arith.constant 0 : index
// CHECK-DAG:    [[VAR_0_:%.+]] = mhlo.constant dense<0> : tensor<i64>
// CHECK-DAG:    [[VAR_1_:%.+]] = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG:    [[VAR_2_:%.+]] = shape.shape_of %arg0 : tensor<5x?x1x32xf32> -> tensor<4xindex>
// CHECK-NEXT:    [[VAR_3_:%.+]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<32xi64>
// CHECK-NEXT:    [[VAR_4_:%.+]] = "mhlo.dynamic_broadcast_in_dim"([[VAR_3_]], [[VAR_2_]]) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<32xi64>, tensor<4xindex>) -> tensor<5x?x1x32xi64>
// CHECK-NEXT:    [[VAR_5_:%.+]]:2 = mhlo.reduce(%arg0 init: [[VAR_1_]]), ([[VAR_4_]] init: [[VAR_0_]]) across dimensions = [3] : (tensor<5x?x1x32xf32>, tensor<5x?x1x32xi64>, tensor<f32>, tensor<i64>) -> (tensor<5x?x1xf32>, tensor<5x?x1xi64>)
// CHECK-NEXT:     reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i64>, %arg4: tensor<i64>)  {
// CHECK-NEXT:      [[VAR_12_:%.+]] = "mhlo.compare"(%arg1, %arg3) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction GE>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-NEXT:      [[VAR_13_:%.+]] = "mhlo.select"([[VAR_12_]], %arg1, %arg3) : (tensor<i1>, tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK-NEXT:      [[VAR_14_:%.+]] = "mhlo.compare"(%arg1, %arg3) {compare_type = #mhlo<comparison_type NOTYPE>, comparison_direction = #mhlo<comparison_direction EQ>} : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-NEXT:      [[VAR_15_:%.+]] = mhlo.minimum %arg2, %arg4 : tensor<i64>
// CHECK-NEXT:      [[VAR_16_:%.+]] = "mhlo.select"([[VAR_12_]], %arg2, %arg4) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-NEXT:      [[VAR_17_:%.+]] = "mhlo.select"([[VAR_14_]], [[VAR_15_]], [[VAR_16_]]) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
// CHECK-NEXT:      "mhlo.return"([[VAR_13_]], [[VAR_17_]]) : (tensor<f32>, tensor<i64>) -> ()
// CHECK-NEXT:    }
// CHECK-NEXT:    [[VAR_6_:%.+]] = shape.get_extent [[VAR_2_]], [[C0]] : tensor<4xindex>, index -> index
// CHECK-NEXT:    [[VAR_7_:%.+]] = shape.get_extent [[VAR_2_]], [[C1]] : tensor<4xindex>, index -> index
// CHECK-NEXT:    [[VAR_8_:%.+]] = shape.get_extent [[VAR_2_]], [[C2]] : tensor<4xindex>, index -> index
// CHECK-NEXT:    [[VAR_9_:%.+]] = shape.from_extents [[VAR_6_]], [[VAR_7_]], [[VAR_8_]], [[C1]] : index, index, index, index
// CHECK-NEXT:    [[VAR_10_:%.+]] = shape.to_extent_tensor [[VAR_9_]] : !shape.shape -> tensor<4xindex>
// CHECK-NEXT:    [[VAR_11_:%.+]] = "mhlo.dynamic_reshape"([[VAR_5_]]#1, [[VAR_10_]]) : (tensor<5x?x1xi64>, tensor<4xindex>) -> tensor<5x?x1x1xi64>
}
