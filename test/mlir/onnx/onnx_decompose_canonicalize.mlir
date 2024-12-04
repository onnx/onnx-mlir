
// RUN: onnx-mlir-opt --decompose-onnx --canonicalize %s -split-input-file | FileCheck %s

// -----

// Test one pattern in lstm_no_data.onnx.
// The type of output of SequenceAt is not the same as the element type
// of the input sequence
func.func @sequence_at_squeezed(%arg0 : tensor<1x1x100xf32>) -> tensor<1x100xf32> {
  %26 = onnx.Constant dense<0> : tensor<i64>
  %27 = onnx.Constant dense<1> : tensor<i64>
  %32 = "onnx.SplitToSequence"(%arg0, %27) {axis = 0 : si64, keepdims = 0 : si64} : (tensor<1x1x100xf32>, tensor<i64>) -> !onnx.Seq<tensor<1x1x100xf32>>
  %33 = "onnx.SequenceAt"(%32, %26) : (!onnx.Seq<tensor<1x1x100xf32>>, tensor<i64>) -> tensor<1x100xf32>
  return %33: tensor<1x100xf32>
// CHECK-LABEL:  func.func @sequence_at_squeezed
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x100xf32>) -> tensor<1x100xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense<0> : tensor<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense<1> : tensor<1xi64>
// CHECK:           [[VAR_2_:%.+]] = "onnx.Split"([[PARAM_0_]], [[VAR_1_]]) {axis = 0 : si64} : (tensor<1x1x100xf32>, tensor<1xi64>) -> tensor<1x1x100xf32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Squeeze"([[VAR_2_]], [[VAR_0_]]) : (tensor<1x1x100xf32>, tensor<1xi64>) -> tensor<1x100xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x100xf32>
// CHECK:         }
}

func.func @sequence_at_multi(%arg0 : tensor<1x1x400xf32>) -> tensor<1x1x100xf32> {
  %15 = onnx.Constant dense<0> : tensor<i64>
  %38 = onnx.Constant dense<1> : tensor<i64>
  %65 = onnx.Constant dense<100> : tensor<i64> 
  %66 = "onnx.SplitToSequence"(%arg0, %65) {axis = 2 : si64, keepdims = 1 : si64} : (tensor<1x1x400xf32>, tensor<i64>) -> !onnx.Seq<tensor<1x1x100xf32>>
  %67 = "onnx.SequenceAt"(%66, %15) : (!onnx.Seq<tensor<1x1x100xf32>>, tensor<i64>) -> tensor<1x1x100xf32>
  %68 = "onnx.SequenceAt"(%66, %38) : (!onnx.Seq<tensor<1x1x100xf32>>, tensor<i64>) -> tensor<1x1x100xf32>
  %40 = "onnx.Add"(%67, %68) : (tensor<1x1x100xf32>, tensor<1x1x100xf32>) -> tensor<1x1x100xf32>
  return %40: tensor<1x1x100xf32>
// CHECK-LABEL:  func.func @sequence_at_multi
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x400xf32>) -> tensor<1x1x100xf32> {
// CHECK:           [[VAR_0_:%.+]] = onnx.Constant dense<100> : tensor<4xi64>
// CHECK-DAG:       [[VAR_1_:%.+]]:4 = "onnx.Split"([[PARAM_0_]], [[VAR_0_]]) {axis = 2 : si64} : (tensor<1x1x400xf32>, tensor<4xi64>) -> (tensor<1x1x100xf32>, tensor<1x1x100xf32>, tensor<1x1x100xf32>, tensor<1x1x100xf32>)
// CHECK-DAG:       [[VAR_2_:%.+]]:4 = "onnx.Split"([[PARAM_0_]], [[VAR_0_]]) {axis = 2 : si64} : (tensor<1x1x400xf32>, tensor<4xi64>) -> (tensor<1x1x100xf32>, tensor<1x1x100xf32>, tensor<1x1x100xf32>, tensor<1x1x100xf32>)
// CHECK:           [[VAR_3_:%.+]] = "onnx.Add"([[VAR_1_]]#0, [[VAR_2_]]#1) : (tensor<1x1x100xf32>, tensor<1x1x100xf32>) -> tensor<1x1x100xf32>
// CHECK:           return [[VAR_3_]] : tensor<1x1x100xf32>
// CHECK:         }
}

