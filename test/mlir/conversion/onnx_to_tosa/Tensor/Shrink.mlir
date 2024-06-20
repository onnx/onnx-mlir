// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @test_shrink_float(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %2 = "onnx.Shrink"(%arg0) {lambd = -7.500000e-01 : f32, bias = 5.000000e-01 : f32} : (tensor<4x4xf32>) -> tensor<4x4xf32>
    return %2 : tensor<4x4xf32>
// CHECK-LABEL:  func.func @test_shrink_float(
//       CHECK:    %0 = "tosa.const"() <{value = dense<-7.500000e-01> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
//       CHECK:    %1 = "tosa.const"() <{value = dense<7.500000e-01> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
//       CHECK:    %2 = "tosa.const"() <{value = dense<5.000000e-01> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
//       CHECK:    %3 = "tosa.const"() <{value = dense<0.000000e+00> : tensor<1x1xf32>}> : () -> tensor<1x1xf32>
//       CHECK:    %4 = tosa.greater %1, %arg0 : (tensor<1x1xf32>, tensor<4x4xf32>) -> tensor<4x4xi1>
//       CHECK:    %5 = tosa.add %arg0, %2 : (tensor<4x4xf32>, tensor<1x1xf32>) -> tensor<4x4xf32>
//       CHECK:    %6 = tosa.select %4, %5, %3 : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<1x1xf32>) -> tensor<4x4xf32>
//       CHECK:    %7 = tosa.greater %arg0, %0 : (tensor<4x4xf32>, tensor<1x1xf32>) -> tensor<4x4xi1>
//       CHECK:    %8 = tosa.sub %arg0, %2 : (tensor<4x4xf32>, tensor<1x1xf32>) -> tensor<4x4xf32>
//       CHECK:    %9 = tosa.select %7, %8, %6 : (tensor<4x4xi1>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
//       CHECK:    return %9 : tensor<4x4xf32>
}

func.func @test_shrink_int(%arg0: tensor<4x4xi8>) -> tensor<4x4xi8> {
    %2 = "onnx.Shrink"(%arg0) {lambd = -7.500000e-01 : f32, bias = 5.000000e-01 : f32} : (tensor<4x4xi8>) -> tensor<4x4xi8>
    return %2 : tensor<4x4xi8>
// CHECK-LABEL:   func.func @test_shrink_int(
// CHECK-SAME:                               %[[VAL_0:.*]]: tensor<4x4xi8>) -> tensor<4x4xi8> {
// CHECK:           %[[VAL_1:.*]] = "tosa.const"() <{value = dense<-1> : tensor<1x1xi8>}> : () -> tensor<1x1xi8>
// CHECK:           %[[VAL_2:.*]] = "tosa.const"() <{value = dense<1> : tensor<1x1xi8>}> : () -> tensor<1x1xi8>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{value = dense<0> : tensor<1x1xi8>}> : () -> tensor<1x1xi8>
// CHECK:           %[[VAL_4:.*]] = tosa.greater %[[VAL_2]], %[[VAL_0]] : (tensor<1x1xi8>, tensor<4x4xi8>) -> tensor<4x4xi1>
// CHECK:           %[[VAL_5:.*]] = tosa.add %[[VAL_0]], %[[VAL_3]] : (tensor<4x4xi8>, tensor<1x1xi8>) -> tensor<4x4xi8>
// CHECK:           %[[VAL_6:.*]] = tosa.select %[[VAL_4]], %[[VAL_5]], %[[VAL_3]] : (tensor<4x4xi1>, tensor<4x4xi8>, tensor<1x1xi8>) -> tensor<4x4xi8>
// CHECK:           %[[VAL_7:.*]] = tosa.greater %[[VAL_0]], %[[VAL_1]] : (tensor<4x4xi8>, tensor<1x1xi8>) -> tensor<4x4xi1>
// CHECK:           %[[VAL_8:.*]] = tosa.sub %[[VAL_0]], %[[VAL_3]] : (tensor<4x4xi8>, tensor<1x1xi8>) -> tensor<4x4xi8>
// CHECK:           %[[VAL_9:.*]] = tosa.select %[[VAL_7]], %[[VAL_8]], %[[VAL_6]] : (tensor<4x4xi1>, tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi8>
// CHECK:           return %[[VAL_9]] : tensor<4x4xi8>
// CHECK:         }
}

func.func @test_shrink_int_constants_are_one(%arg0: tensor<4x4xi8>) -> tensor<4x4xi8> {
    %2 = "onnx.Shrink"(%arg0) {lambd = 1.000000e00 : f32, bias = 1.000000e00 : f32} : (tensor<4x4xi8>) -> tensor<4x4xi8>
    return %2 : tensor<4x4xi8>
// CHECK-LABEL:  func.func @test_shrink_int_constants_are_one(
//       CHECK:    %0 = "tosa.const"() <{value = dense<1> : tensor<1x1xi8>}> : () -> tensor<1x1xi8>
//       CHECK:    %1 = "tosa.const"() <{value = dense<-1> : tensor<1x1xi8>}> : () -> tensor<1x1xi8>
//       CHECK:    %2 = "tosa.const"() <{value = dense<0> : tensor<1x1xi8>}> : () -> tensor<1x1xi8>
//       CHECK:    %3 = tosa.greater %1, %arg0 : (tensor<1x1xi8>, tensor<4x4xi8>) -> tensor<4x4xi1>
//       CHECK:    %4 = tosa.add %arg0, %0 : (tensor<4x4xi8>, tensor<1x1xi8>) -> tensor<4x4xi8>
//       CHECK:    %5 = tosa.select %3, %4, %2 : (tensor<4x4xi1>, tensor<4x4xi8>, tensor<1x1xi8>) -> tensor<4x4xi8>
//       CHECK:    %6 = tosa.greater %arg0, %0 : (tensor<4x4xi8>, tensor<1x1xi8>) -> tensor<4x4xi1>
//       CHECK:    %7 = tosa.sub %arg0, %0 : (tensor<4x4xi8>, tensor<1x1xi8>) -> tensor<4x4xi8>
//       CHECK:    %8 = tosa.select %6, %7, %5 : (tensor<4x4xi1>, tensor<4x4xi8>, tensor<4x4xi8>) -> tensor<4x4xi8>
//       CHECK:    return %8 : tensor<4x4xi8>
} 
