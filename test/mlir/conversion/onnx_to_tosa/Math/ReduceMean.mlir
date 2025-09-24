// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-tosa -cse %s -split-input-file | FileCheck %s

func.func @reduce_mean(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMean"(%arg0, %0) : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5x1x1xf32>
return %1 : tensor<2x5x1x1xf32>
// CHECK-LABEL:   func.func @reduce_mean(
// CHECK-SAME:                           %[[VAL_0:.*]]: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
// CHECK:           %[[VAL_1:.*]] = tosa.reduce_sum %[[VAL_0]] {axis = 2 : i32} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x11xf32>
// CHECK:           %[[VAL_2:.*]] = tosa.reduce_sum %[[VAL_1]] {axis = 3 : i32} : (tensor<2x5x1x11xf32>) -> tensor<2x5x1x1xf32>
// CHECK:           %[[VAL_3:.*]] = "tosa.const"() <{values = dense<0.0101010101> : tensor<f32>}> : () -> tensor<f32>
// CHECK:           %[[SHAPE:.*]] = tosa.const_shape {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK:           %[[VAL_4:.*]] = tosa.reshape %[[VAL_3]], %[[SHAPE]] : (tensor<f32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
// CHECK:           %[[ZERO:.*]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           %[[VAL_5:.*]] = tosa.mul %[[VAL_2]], %[[VAL_4]], %[[ZERO]] : (tensor<2x5x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<2x5x1x1xf32>
// CHECK:           return %[[VAL_5]] : tensor<2x5x1x1xf32>
}

// -----

func.func @reduce_mean_no_axes_attr(%arg0: tensor<2x5x9x11xf32>) -> tensor<1x1x1x1xf32> {
%none = "onnx.NoValue"() {value} : () -> none
%0 = "onnx.ReduceMean"(%arg0, %none) : (tensor<2x5x9x11xf32>, none) -> tensor<1x1x1x1xf32>
return %0 : tensor<1x1x1x1xf32>
// CHECK-LABEL:  func.func @reduce_mean_no_axes_attr
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5x9x11xf32>) -> tensor<1x1x1x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.reduce_sum [[PARAM_0_]] {axis = 0 : i32} : (tensor<2x5x9x11xf32>) -> tensor<1x5x9x11xf32>
// CHECK:           [[VAR_1_:%.+]] = tosa.reduce_sum [[VAR_0_]] {axis = 1 : i32} : (tensor<1x5x9x11xf32>) -> tensor<1x1x9x11xf32>
// CHECK:           [[VAR_2_:%.+]] = tosa.reduce_sum [[VAR_1_]] {axis = 2 : i32} : (tensor<1x1x9x11xf32>) -> tensor<1x1x1x11xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.reduce_sum [[VAR_2_]] {axis = 3 : i32} : (tensor<1x1x1x11xf32>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tosa.const"() <{values = dense<0.00101010106> : tensor<f32>}> : () -> tensor<f32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.reshape [[VAR_4_]], [[VAR_5_]] : (tensor<f32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           [[VAR_8_:%.+]] = tosa.mul [[VAR_3_]], [[VAR_6_]], [[VAR_7_]] : (tensor<1x1x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<1x1x1x1xf32>
// CHECK:           return [[VAR_8_]] : tensor<1x1x1x1xf32>
// CHECK:         }
}

// -----

func.func @reduce_mean_keepdims_false(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMean"(%arg0, %0) {keepdims = 0 : si64} : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5xf32>
return %1 : tensor<2x5xf32>
// CHECK-LABEL:  func.func @reduce_mean_keepdims_false
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5x9x11xf32>) -> tensor<2x5xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.reduce_sum [[PARAM_0_]] {axis = 2 : i32} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x11xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reduce_sum [[VAR_0_]] {axis = 3 : i32} : (tensor<2x5x1x11xf32>) -> tensor<2x5x1x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {values = dense<[2, 5]> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.reshape [[VAR_1_]], [[VAR_2_]] : (tensor<2x5x1x1xf32>, !tosa.shape<2>) -> tensor<2x5xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tosa.const"() <{values = dense<0.0101010101> : tensor<f32>}> : () -> tensor<f32>
// CHECK-DAG:       [[VAR_5_:%.+]] = tosa.const_shape  {values = dense<1> : tensor<2xindex>} : () -> !tosa.shape<2>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = tosa.reshape [[VAR_4_]], [[VAR_5_]] : (tensor<f32>, !tosa.shape<2>) -> tensor<1x1xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           [[VAR_8_:%.+]] = tosa.mul [[VAR_3_]], [[VAR_6_]], [[VAR_7_]] : (tensor<2x5xf32>, tensor<1x1xf32>, tensor<1xi8>) -> tensor<2x5xf32>
// CHECK:           return [[VAR_8_]] : tensor<2x5xf32>
// CHECK:         }
}

// -----

func.func @reduce_mean_noop_with_emtpy_axes_one(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
%0 = "onnx.Constant"() {value = dense<[2, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
%1 = "onnx.ReduceMean"(%arg0, %0) {noop_with_empty_axes = 1 : si64} : (tensor<2x5x9x11xf32>, tensor<2xi64>) -> tensor<2x5x1x1xf32>
return %1 : tensor<2x5x1x1xf32>
// CHECK-LABEL:  func.func @reduce_mean_noop_with_emtpy_axes_one
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5x9x11xf32>) -> tensor<2x5x1x1xf32> {
// CHECK:           [[VAR_0_:%.+]] = tosa.reduce_sum [[PARAM_0_]] {axis = 2 : i32} : (tensor<2x5x9x11xf32>) -> tensor<2x5x1x11xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = tosa.reduce_sum [[VAR_0_]] {axis = 3 : i32} : (tensor<2x5x1x11xf32>) -> tensor<2x5x1x1xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "tosa.const"() <{values = dense<0.0101010101> : tensor<f32>}> : () -> tensor<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = tosa.reshape [[VAR_2_]], [[VAR_3_]] : (tensor<f32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           [[VAR_6_:%.+]] = tosa.mul [[VAR_1_]], [[VAR_4_]], [[VAR_5_]] : (tensor<2x5x1x1xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<2x5x1x1xf32>
// CHECK:           return [[VAR_6_]] : tensor<2x5x1x1xf32>
// CHECK:         }
}

// -----

func.func @reduce_mean_noop_with_emtpy_axes_one_none_input(%arg0: tensor<2x5x9x11xf32>) -> tensor<2x5x9x11xf32> {
%none = "onnx.NoValue"() {value} : () -> none
%0 = "onnx.ReduceMean"(%arg0, %none) {noop_with_empty_axes = 1 : si64} : (tensor<2x5x9x11xf32>, none) ->  tensor<2x5x9x11xf32>
return %0 : tensor<2x5x9x11xf32>
// CHECK-LABEL:  func.func @reduce_mean_noop_with_emtpy_axes_one_none_input
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<2x5x9x11xf32>) -> tensor<2x5x9x11xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = tosa.identity [[PARAM_0_]] : (tensor<2x5x9x11xf32>) -> tensor<2x5x9x11xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "tosa.const"() <{values = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = tosa.const_shape  {values = dense<1> : tensor<4xindex>} : () -> !tosa.shape<4>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = tosa.reshape [[VAR_1_]], [[VAR_2_]] : (tensor<f32>, !tosa.shape<4>) -> tensor<1x1x1x1xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = "tosa.const"() <{values = dense<0> : tensor<1xi8>}> : () -> tensor<1xi8>
// CHECK:           [[VAR_5_:%.+]] = tosa.mul [[VAR_0_]], [[VAR_3_]], [[VAR_4_]] : (tensor<2x5x9x11xf32>, tensor<1x1x1x1xf32>, tensor<1xi8>) -> tensor<2x5x9x11xf32>
// CHECK:           return [[VAR_5_]] : tensor<2x5x9x11xf32>
// CHECK:         }
}
