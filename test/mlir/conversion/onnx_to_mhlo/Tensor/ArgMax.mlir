// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-mhlo --lower-affine %s --canonicalize -split-input-file | FileCheck %s

func.func @test_argmax_verifier_1(%arg0 : tensor<5x5x1x32xf32>) -> tensor<*xi64> {
  %1 = "onnx.ArgMax"(%arg0) { axis = -1 : si64} : (tensor<5x5x1x32xf32>)  -> tensor<*xi64>
  "func.return"(%1) : (tensor<*xi64>) -> ()
// CHECK-LABEL: func @test_argmax_verifier_1(%arg0: tensor<5x5x1x32xf32>)  
// CHECK-DAG:     %[[V0:.*]] = mhlo.constant dense<0> : tensor<i64>
// CHECK-DAG:     %[[V1:.*]] = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG:     %[[V2:.*]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<32xi64>
// CHECK-DAG:     %[[V3:.*]] = "mhlo.broadcast_in_dim"(%[[V2]]) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<32xi64>) -> tensor<5x5x1x32xi64>
// CHECK-DAG:     %[[V4:.*]]:2 = mhlo.reduce(%arg0 init: %[[V1]]), (%[[V3]] init: %[[V0]]) across dimensions = [3] : (tensor<5x5x1x32xf32>, tensor<5x5x1x32xi64>, tensor<f32>, tensor<i64>) -> (tensor<5x5x1xf32>, tensor<5x5x1xi64>)
// CHECK-DAG:      reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i64>, %arg4: tensor<i64>)  {
// CHECK-DAG:       %[[V6:.*]] = mhlo.compare  GE, %arg1, %arg3,  NOTYPE : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-DAG:       %[[V7:.*]] = mhlo.select %[[V6]], %arg1, %arg3 : tensor<i1>, tensor<f32>
// CHECK-DAG:       %[[V8:.*]] = mhlo.compare  EQ, %arg1, %arg3,  NOTYPE : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-DAG:       %[[V9:.*]] = mhlo.minimum %arg2, %arg4 : tensor<i64>
// CHECK-DAG:       %[[V10:.*]] = mhlo.select %[[V6]], %arg2, %arg4 : tensor<i1>, tensor<i64>
// CHECK-DAG:       %[[V11:.*]] = mhlo.select %[[V8]], %[[V9]], %[[V10]] : tensor<i1>, tensor<i64>
// CHECK-DAG:       mhlo.return %[[V7]], %[[V11]] : tensor<f32>, tensor<i64>
// CHECK-DAG:     }
// CHECK-DAG:     %[[V5:.*]] = mhlo.reshape %[[V4]]#1 : (tensor<5x5x1xi64>) -> tensor<5x5x1x1xi64>
// CHECK-DAG:     return %[[V5]] : tensor<5x5x1x1xi64>  
}

func.func @test_argmax_verifier_2(%arg0 : tensor<5x?x1x32xf32>) -> tensor<*xi64> {
  %1 = "onnx.ArgMax"(%arg0) { axis = 3 : si64} : (tensor<5x?x1x32xf32>)  -> tensor<*xi64>
  "func.return"(%1) : (tensor<*xi64>) -> ()
// CHECK-LABEL: func.func @test_argmax_verifier_2(%arg0: tensor<5x?x1x32xf32>) -> tensor<5x?x1x1xi64> {
// CHECK-DAG:     %[[C1:.*]] = arith.constant 1 : index
// CHECK-DAG:     %[[C5:.*]] = arith.constant 5 : index
// CHECK-DAG:     %[[V0:.*]] = mhlo.constant dense<0> : tensor<i64>
// CHECK-DAG:     %[[V1:.*]] = mhlo.constant dense<0xFF800000> : tensor<f32>
// CHECK-DAG:     %[[V6:.*]] = shape.shape_of %arg0 : tensor<5x?x1x32xf32> -> tensor<4xindex>
// CHECK-DAG:     %[[V2:.*]] = shape.shape_of %arg0 : tensor<5x?x1x32xf32> -> tensor<4xindex>
// CHECK-DAG:     %[[V3:.*]] = "mhlo.iota"() {iota_dimension = 0 : i64} : () -> tensor<32xi64>
// CHECK-DAG:     %[[V4:.*]] = "mhlo.dynamic_broadcast_in_dim"(%[[V3]], %[[V2]]) {broadcast_dimensions = dense<3> : tensor<1xi64>} : (tensor<32xi64>, tensor<4xindex>) -> tensor<5x?x1x32xi64>
// CHECK-DAG:     %[[V5:.*]]:2 = mhlo.reduce(%arg0 init: %[[V1]]), (%[[V4]] init: %[[V0]]) across dimensions = [3] : (tensor<5x?x1x32xf32>, tensor<5x?x1x32xi64>, tensor<f32>, tensor<i64>) -> (tensor<5x?x1xf32>, tensor<5x?x1xi64>)
// CHECK-DAG:      reducer(%arg1: tensor<f32>, %arg3: tensor<f32>) (%arg2: tensor<i64>, %arg4: tensor<i64>)  {
// CHECK-DAG:       %[[V12:.*]] = mhlo.compare  GE, %arg1, %arg3,  NOTYPE : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-DAG:       %[[V13:.*]] = mhlo.select %[[V12]], %arg1, %arg3 : tensor<i1>, tensor<f32>
// CHECK-DAG:       %[[V14:.*]] = mhlo.compare  EQ, %arg1, %arg3,  NOTYPE : (tensor<f32>, tensor<f32>) -> tensor<i1>
// CHECK-DAG:       %[[V15:.*]] = mhlo.minimum %arg2, %arg4 : tensor<i64>
// CHECK-DAG:       %[[V16:.*]] = mhlo.select %[[V12]], %arg2, %arg4 : tensor<i1>, tensor<i64>
// CHECK-DAG:       %[[V17:.*]] = mhlo.select %[[V14]], %[[V15]], %[[V16]] : tensor<i1>, tensor<i64>
// CHECK-DAG:       mhlo.return %[[V13]], %[[V17]] : tensor<f32>, tensor<i64>
// CHECK-DAG:     }
// CHECK-DAG:     %[[V7:.*]] = shape.get_extent %[[V6]], %[[C1]] : tensor<4xindex>, index -> index
// CHECK-DAG:     %[[V8:.*]] = shape.from_extents %[[C5]], %[[V7]], %[[C1]], %[[C1]] : index, index, index, index
// CHECK-DAG:     %[[V9:.*]] = shape.to_extent_tensor %[[V8]] : !shape.shape -> tensor<4xindex>
// CHECK-DAG:     %[[V10:.*]] = mhlo.dynamic_reshape %[[V5]]#1, %[[V9]] : (tensor<5x?x1xi64>, tensor<4xindex>) -> tensor<5x?x1x1xi64>
// CHECK-DAG:     return %[[V10]] : tensor<5x?x1x1xi64>
}
