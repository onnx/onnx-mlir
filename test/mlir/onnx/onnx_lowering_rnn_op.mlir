// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='test-rnn-ops-lowering' %s -split-input-file | FileCheck %s

func private @test_rnn_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x4x3xf32>, %arg2: tensor<1x4x4xf32>, %arg3: tensor<1x8xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_rnn_forward_mode(
// CHECK-SAME:                                        %[[VAL_0:.*]]: memref<7x2x3xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: memref<1x4x3xf32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: memref<1x4x4xf32>,
// CHECK-SAME:                                        %[[VAL_3:.*]]: memref<1x8xf32>,
// CHECK-SAME:                                        %[[VAL_4:.*]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<1x2x4xf32>
// CHECK:           %[[VAL_7:.*]] = constant unit
// CHECK:           %[[VAL_8:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_9:.*]] = constant 0 : index
// CHECK:           %[[VAL_10:.*]] = constant 1 : index
// CHECK:           %[[VAL_11:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_11]]#0, %[[VAL_11]]#1) with (%[[VAL_11]]#0 -> %[[VAL_12:.*]] = 0 to 2, %[[VAL_11]]#1 -> %[[VAL_13:.*]] = 0 to 4) {
// CHECK:             %[[VAL_14:.*]] = krnl.load %[[VAL_4]]{{\[}}%[[VAL_9]], %[[VAL_12]], %[[VAL_13]]] : memref<1x2x4xf32>
// CHECK:             krnl.store %[[VAL_14]], %[[VAL_5]]{{\[}}%[[VAL_12]], %[[VAL_13]]] : memref<2x4xf32>
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x4x3xf32>) -> memref<4x3xf32>
// CHECK:           %[[VAL_16:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_15]]) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_16]]) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Squeeze"(%[[VAL_3]]) {axes = [0]} : (memref<1x8xf32>) -> memref<8xf32>
// CHECK:           %[[VAL_20:.*]]:2 = "onnx.Split"(%[[VAL_19]]) {axis = 0 : si64} : (memref<8xf32>) -> (memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_21:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_21]]) with (%[[VAL_21]] -> %[[VAL_22:.*]] = 0 to 7) {
// CHECK:             %[[VAL_23:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_24:.*]] = constant 0 : index
// CHECK:             %[[VAL_25:.*]] = constant 2 : index
// CHECK:             %[[VAL_26:.*]] = constant 3 : index
// CHECK:             %[[VAL_27:.*]] = constant 0 : index
// CHECK:             %[[VAL_28:.*]] = constant 0 : index
// CHECK:             %[[VAL_29:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_29]]#0, %[[VAL_29]]#1) with (%[[VAL_29]]#0 -> %[[VAL_30:.*]] = %[[VAL_27]] to %[[VAL_25]], %[[VAL_29]]#1 -> %[[VAL_31:.*]] = %[[VAL_28]] to %[[VAL_26]]) {
// CHECK:               %[[VAL_32:.*]]:2 = krnl.get_induction_var_value(%[[VAL_29]]#0, %[[VAL_29]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_33:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_22]], %[[VAL_32]]#0, %[[VAL_32]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_33]], %[[VAL_23]]{{\[}}%[[VAL_32]]#0, %[[VAL_32]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_34:.*]] = "onnx.MatMul"(%[[VAL_23]], %[[VAL_17]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_35:.*]] = "onnx.MatMul"(%[[VAL_5]], %[[VAL_18]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_36:.*]] = constant 2 : index
// CHECK:             %[[VAL_37:.*]] = constant 4 : index
// CHECK:             %[[VAL_38:.*]] = constant 0 : index
// CHECK:             %[[VAL_39:.*]] = constant 0 : index
// CHECK:             %[[VAL_40:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_40]]#0, %[[VAL_40]]#1) with (%[[VAL_40]]#0 -> %[[VAL_41:.*]] = %[[VAL_38]] to %[[VAL_36]], %[[VAL_40]]#1 -> %[[VAL_42:.*]] = %[[VAL_39]] to %[[VAL_37]]) {
// CHECK:               %[[VAL_43:.*]]:2 = krnl.get_induction_var_value(%[[VAL_40]]#0, %[[VAL_40]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_44:.*]] = krnl.load %[[VAL_34]]{{\[}}%[[VAL_43]]#0, %[[VAL_43]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_45:.*]] = krnl.load %[[VAL_35]]{{\[}}%[[VAL_43]]#0, %[[VAL_43]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_46:.*]] = addf %[[VAL_44]], %[[VAL_45]] : f32
// CHECK:               %[[VAL_47:.*]] = krnl.load %[[VAL_20]]#0{{\[}}%[[VAL_43]]#1] : memref<4xf32>
// CHECK:               %[[VAL_48:.*]] = krnl.load %[[VAL_20]]#1{{\[}}%[[VAL_43]]#1] : memref<4xf32>
// CHECK:               %[[VAL_49:.*]] = addf %[[VAL_46]], %[[VAL_47]] : f32
// CHECK:               %[[VAL_50:.*]] = addf %[[VAL_49]], %[[VAL_48]] : f32
// CHECK:               %[[VAL_51:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_50]], %[[VAL_51]][] : memref<f32>
// CHECK:               %[[VAL_52:.*]] = "onnx.Tanh"(%[[VAL_51]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_53:.*]] = krnl.load %[[VAL_52]][] : memref<f32>
// CHECK:               krnl.store %[[VAL_53]], %[[VAL_5]]{{\[}}%[[VAL_43]]#0, %[[VAL_43]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_23]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = constant 32 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_6]], %[[VAL_5]], %[[VAL_54]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_5]] : memref<2x4xf32>
// CHECK:           return %[[VAL_6]] : memref<1x2x4xf32>
// CHECK:         }

}

// -----

func private @test_rnn_forward_mode_constant_weight_and_bias(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %w = "onnx.Constant"() {value = dense<[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]> : tensor<1x4x3xf32>} : () -> tensor<1x4x3xf32> 
  %r = "onnx.Constant"() {value = dense<[[[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]]]> : tensor<1x4x4xf32>} : () -> tensor<1x4x4xf32> 
  %b = "onnx.Constant"() {value = dense<[[1., 2., 3., 4., 5., 6., 7., 8.]]> : tensor<1x8xf32>} : () -> tensor<1x8xf32> 

  %Y, %Y_h = "onnx.RNN"(%arg0, %w, %r, %b, %cst, %arg1) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_rnn_forward_mode_constant_weight_and_bias(
// CHECK-SAME:                                                                 %[[VAL_0:.*]]: memref<7x2x3xf32>,
// CHECK-SAME:                                                                 %[[VAL_1:.*]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<1x2x4xf32>
// CHECK:           %[[VAL_4:.*]] = constant unit
// CHECK:           %[[VAL_5:.*]] = "krnl.global"() {name = "constant_0", shape = [1, 4, 3], value = dense<1.000000e+00> : tensor<1x4x3xf32>} : () -> memref<1x4x3xf32>
// CHECK:           %[[VAL_6:.*]] = "krnl.global"() {name = "constant_1", shape = [1, 4, 4], value = dense<2.000000e+00> : tensor<1x4x4xf32>} : () -> memref<1x4x4xf32>
// CHECK:           %[[VAL_7:.*]] = "krnl.global"() {name = "constant_2", shape = [1, 8], value = dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]]> : tensor<1x8xf32>} : () -> memref<1x8xf32>
// CHECK:           %[[VAL_8:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_9:.*]] = constant 0 : index
// CHECK:           %[[VAL_10:.*]] = constant 1 : index
// CHECK:           %[[VAL_11:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_11]]#0, %[[VAL_11]]#1) with (%[[VAL_11]]#0 -> %[[VAL_12:.*]] = 0 to 2, %[[VAL_11]]#1 -> %[[VAL_13:.*]] = 0 to 4) {
// CHECK:             %[[VAL_14:.*]] = krnl.load %[[VAL_1]]{{\[}}%[[VAL_9]], %[[VAL_12]], %[[VAL_13]]] : memref<1x2x4xf32>
// CHECK:             krnl.store %[[VAL_14]], %[[VAL_2]]{{\[}}%[[VAL_12]], %[[VAL_13]]] : memref<2x4xf32>
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = "krnl.global"() {name = "constant_3", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK:           %[[VAL_16:.*]] = "krnl.global"() {name = "constant_4", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_17:.*]] = "krnl.global"() {name = "constant_5", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK:           %[[VAL_18:.*]] = "krnl.global"() {name = "constant_6", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_19:.*]] = "krnl.global"() {name = "constant_7", shape = [8], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<8xf32>} : () -> memref<8xf32>
// CHECK:           %[[VAL_20:.*]] = "krnl.global"() {name = "constant_8", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_21:.*]] = "krnl.global"() {name = "constant_9", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_22:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_22]]) with (%[[VAL_22]] -> %[[VAL_23:.*]] = 0 to 7) {
// CHECK:             %[[VAL_24:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_25:.*]] = constant 0 : index
// CHECK:             %[[VAL_26:.*]] = constant 2 : index
// CHECK:             %[[VAL_27:.*]] = constant 3 : index
// CHECK:             %[[VAL_28:.*]] = constant 0 : index
// CHECK:             %[[VAL_29:.*]] = constant 0 : index
// CHECK:             %[[VAL_30:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_30]]#0, %[[VAL_30]]#1) with (%[[VAL_30]]#0 -> %[[VAL_31:.*]] = %[[VAL_28]] to %[[VAL_26]], %[[VAL_30]]#1 -> %[[VAL_32:.*]] = %[[VAL_29]] to %[[VAL_27]]) {
// CHECK:               %[[VAL_33:.*]]:2 = krnl.get_induction_var_value(%[[VAL_30]]#0, %[[VAL_30]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_34:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_23]], %[[VAL_33]]#0, %[[VAL_33]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_34]], %[[VAL_24]]{{\[}}%[[VAL_33]]#0, %[[VAL_33]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_35:.*]] = "onnx.MatMul"(%[[VAL_24]], %[[VAL_17]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_36:.*]] = "onnx.MatMul"(%[[VAL_2]], %[[VAL_18]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_37:.*]] = constant 2 : index
// CHECK:             %[[VAL_38:.*]] = constant 4 : index
// CHECK:             %[[VAL_39:.*]] = constant 0 : index
// CHECK:             %[[VAL_40:.*]] = constant 0 : index
// CHECK:             %[[VAL_41:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_41]]#0, %[[VAL_41]]#1) with (%[[VAL_41]]#0 -> %[[VAL_42:.*]] = %[[VAL_39]] to %[[VAL_37]], %[[VAL_41]]#1 -> %[[VAL_43:.*]] = %[[VAL_40]] to %[[VAL_38]]) {
// CHECK:               %[[VAL_44:.*]]:2 = krnl.get_induction_var_value(%[[VAL_41]]#0, %[[VAL_41]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_45:.*]] = krnl.load %[[VAL_35]]{{\[}}%[[VAL_44]]#0, %[[VAL_44]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_46:.*]] = krnl.load %[[VAL_36]]{{\[}}%[[VAL_44]]#0, %[[VAL_44]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_47:.*]] = addf %[[VAL_45]], %[[VAL_46]] : f32
// CHECK:               %[[VAL_48:.*]] = krnl.load %[[VAL_20]]{{\[}}%[[VAL_44]]#1] : memref<4xf32>
// CHECK:               %[[VAL_49:.*]] = krnl.load %[[VAL_21]]{{\[}}%[[VAL_44]]#1] : memref<4xf32>
// CHECK:               %[[VAL_50:.*]] = addf %[[VAL_47]], %[[VAL_48]] : f32
// CHECK:               %[[VAL_51:.*]] = addf %[[VAL_50]], %[[VAL_49]] : f32
// CHECK:               %[[VAL_52:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_51]], %[[VAL_52]][] : memref<f32>
// CHECK:               %[[VAL_53:.*]] = "onnx.Tanh"(%[[VAL_52]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_54:.*]] = krnl.load %[[VAL_53]][] : memref<f32>
// CHECK:               krnl.store %[[VAL_54]], %[[VAL_2]]{{\[}}%[[VAL_44]]#0, %[[VAL_44]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_24]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_55:.*]] = constant 32 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_3]], %[[VAL_2]], %[[VAL_55]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_2]] : memref<2x4xf32>
// CHECK:           return %[[VAL_3]] : memref<1x2x4xf32>
// CHECK:         }

}

// -----

func private @test_rnn_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x4x3xf32>, %arg2: tensor<1x4x4xf32>, %arg3: tensor<1x8xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x4x3xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_rnn_reverse_mode(
// CHECK-SAME:                                        %[[VAL_0:.*]]: memref<7x2x3xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: memref<1x4x3xf32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: memref<1x4x4xf32>,
// CHECK-SAME:                                        %[[VAL_3:.*]]: memref<1x8xf32>,
// CHECK-SAME:                                        %[[VAL_4:.*]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<1x2x4xf32>
// CHECK:           %[[VAL_7:.*]] = constant unit
// CHECK:           %[[VAL_8:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_9:.*]] = constant 0 : index
// CHECK:           %[[VAL_10:.*]] = constant 1 : index
// CHECK:           %[[VAL_11:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_11]]#0, %[[VAL_11]]#1) with (%[[VAL_11]]#0 -> %[[VAL_12:.*]] = 0 to 2, %[[VAL_11]]#1 -> %[[VAL_13:.*]] = 0 to 4) {
// CHECK:             %[[VAL_14:.*]] = krnl.load %[[VAL_4]]{{\[}}%[[VAL_9]], %[[VAL_12]], %[[VAL_13]]] : memref<1x2x4xf32>
// CHECK:             krnl.store %[[VAL_14]], %[[VAL_5]]{{\[}}%[[VAL_12]], %[[VAL_13]]] : memref<2x4xf32>
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x4x3xf32>) -> memref<4x3xf32>
// CHECK:           %[[VAL_16:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_15]]) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_16]]) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Squeeze"(%[[VAL_3]]) {axes = [0]} : (memref<1x8xf32>) -> memref<8xf32>
// CHECK:           %[[VAL_20:.*]]:2 = "onnx.Split"(%[[VAL_19]]) {axis = 0 : si64} : (memref<8xf32>) -> (memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_21:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_21]]) with (%[[VAL_21]] -> %[[VAL_22:.*]] = 0 to 7) {
// CHECK:             %[[VAL_23:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_24:.*]] = constant 0 : index
// CHECK:             %[[VAL_25:.*]] = constant 7 : index
// CHECK:             %[[VAL_26:.*]] = affine.apply #map(%[[VAL_22]]){{\[}}%[[VAL_25]]]
// CHECK:             %[[VAL_27:.*]] = constant 2 : index
// CHECK:             %[[VAL_28:.*]] = constant 3 : index
// CHECK:             %[[VAL_29:.*]] = constant 0 : index
// CHECK:             %[[VAL_30:.*]] = constant 0 : index
// CHECK:             %[[VAL_31:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_31]]#0, %[[VAL_31]]#1) with (%[[VAL_31]]#0 -> %[[VAL_32:.*]] = %[[VAL_29]] to %[[VAL_27]], %[[VAL_31]]#1 -> %[[VAL_33:.*]] = %[[VAL_30]] to %[[VAL_28]]) {
// CHECK:               %[[VAL_34:.*]]:2 = krnl.get_induction_var_value(%[[VAL_31]]#0, %[[VAL_31]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_35:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_26]], %[[VAL_34]]#0, %[[VAL_34]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_35]], %[[VAL_23]]{{\[}}%[[VAL_34]]#0, %[[VAL_34]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_36:.*]] = "onnx.MatMul"(%[[VAL_23]], %[[VAL_17]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_37:.*]] = "onnx.MatMul"(%[[VAL_5]], %[[VAL_18]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_38:.*]] = constant 2 : index
// CHECK:             %[[VAL_39:.*]] = constant 4 : index
// CHECK:             %[[VAL_40:.*]] = constant 0 : index
// CHECK:             %[[VAL_41:.*]] = constant 0 : index
// CHECK:             %[[VAL_42:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_42]]#0, %[[VAL_42]]#1) with (%[[VAL_42]]#0 -> %[[VAL_43:.*]] = %[[VAL_40]] to %[[VAL_38]], %[[VAL_42]]#1 -> %[[VAL_44:.*]] = %[[VAL_41]] to %[[VAL_39]]) {
// CHECK:               %[[VAL_45:.*]]:2 = krnl.get_induction_var_value(%[[VAL_42]]#0, %[[VAL_42]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_46:.*]] = krnl.load %[[VAL_36]]{{\[}}%[[VAL_45]]#0, %[[VAL_45]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_47:.*]] = krnl.load %[[VAL_37]]{{\[}}%[[VAL_45]]#0, %[[VAL_45]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_48:.*]] = addf %[[VAL_46]], %[[VAL_47]] : f32
// CHECK:               %[[VAL_49:.*]] = krnl.load %[[VAL_20]]#0{{\[}}%[[VAL_45]]#1] : memref<4xf32>
// CHECK:               %[[VAL_50:.*]] = krnl.load %[[VAL_20]]#1{{\[}}%[[VAL_45]]#1] : memref<4xf32>
// CHECK:               %[[VAL_51:.*]] = addf %[[VAL_48]], %[[VAL_49]] : f32
// CHECK:               %[[VAL_52:.*]] = addf %[[VAL_51]], %[[VAL_50]] : f32
// CHECK:               %[[VAL_53:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_52]], %[[VAL_53]][] : memref<f32>
// CHECK:               %[[VAL_54:.*]] = "onnx.Tanh"(%[[VAL_53]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_55:.*]] = krnl.load %[[VAL_54]][] : memref<f32>
// CHECK:               krnl.store %[[VAL_55]], %[[VAL_5]]{{\[}}%[[VAL_45]]#0, %[[VAL_45]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_23]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_56:.*]] = constant 32 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_6]], %[[VAL_5]], %[[VAL_56]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_5]] : memref<2x4xf32>
// CHECK:           return %[[VAL_6]] : memref<1x2x4xf32>
// CHECK:         }

}

// -----

func private @test_rnn_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x4x3xf32>, %arg2: tensor<2x4x4xf32>, %arg3: tensor<2x8xf32>, %arg4: tensor<2x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x4x3xf32>, tensor<2x4x4xf32>, tensor<2x8xf32>, none, tensor<2x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_rnn_bidirectional_mode(
// CHECK-SAME:                                              %[[VAL_0:.*]]: memref<7x2x3xf32>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: memref<2x4x3xf32>,
// CHECK-SAME:                                              %[[VAL_2:.*]]: memref<2x4x4xf32>,
// CHECK-SAME:                                              %[[VAL_3:.*]]: memref<2x8xf32>,
// CHECK-SAME:                                              %[[VAL_4:.*]]: memref<2x2x4xf32>) -> memref<2x2x4xf32> {
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<2x2x4xf32>
// CHECK:           %[[VAL_8:.*]] = constant unit
// CHECK:           %[[VAL_9:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_10:.*]] = constant 0 : index
// CHECK:           %[[VAL_11:.*]] = constant 1 : index
// CHECK:           %[[VAL_12:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_12]]#0, %[[VAL_12]]#1) with (%[[VAL_12]]#0 -> %[[VAL_13:.*]] = 0 to 2, %[[VAL_12]]#1 -> %[[VAL_14:.*]] = 0 to 4) {
// CHECK:             %[[VAL_15:.*]] = krnl.load %[[VAL_4]]{{\[}}%[[VAL_10]], %[[VAL_13]], %[[VAL_14]]] : memref<2x2x4xf32>
// CHECK:             krnl.store %[[VAL_15]], %[[VAL_6]]{{\[}}%[[VAL_13]], %[[VAL_14]]] : memref<2x4xf32>
// CHECK:             %[[VAL_16:.*]] = krnl.load %[[VAL_4]]{{\[}}%[[VAL_11]], %[[VAL_13]], %[[VAL_14]]] : memref<2x2x4xf32>
// CHECK:             krnl.store %[[VAL_16]], %[[VAL_5]]{{\[}}%[[VAL_13]], %[[VAL_14]]] : memref<2x4xf32>
// CHECK:           }
// CHECK:           %[[VAL_17:.*]]:2 = "onnx.Split"(%[[VAL_1]]) {axis = 0 : si64} : (memref<2x4x3xf32>) -> (memref<1x4x3xf32>, memref<1x4x3xf32>)
// CHECK:           %[[VAL_18:.*]] = "onnx.Squeeze"(%[[VAL_17]]#0) {axes = [0]} : (memref<1x4x3xf32>) -> memref<4x3xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Squeeze"(%[[VAL_17]]#1) {axes = [0]} : (memref<1x4x3xf32>) -> memref<4x3xf32>
// CHECK:           %[[VAL_20:.*]]:2 = "onnx.Split"(%[[VAL_2]]) {axis = 0 : si64} : (memref<2x4x4xf32>) -> (memref<1x4x4xf32>, memref<1x4x4xf32>)
// CHECK:           %[[VAL_21:.*]] = "onnx.Squeeze"(%[[VAL_20]]#0) {axes = [0]} : (memref<1x4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Squeeze"(%[[VAL_20]]#1) {axes = [0]} : (memref<1x4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Transpose"(%[[VAL_18]]) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Transpose"(%[[VAL_21]]) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Transpose"(%[[VAL_19]]) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.Transpose"(%[[VAL_22]]) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_27:.*]]:2 = "onnx.Split"(%[[VAL_3]]) {axis = 0 : si64} : (memref<2x8xf32>) -> (memref<1x8xf32>, memref<1x8xf32>)
// CHECK:           %[[VAL_28:.*]] = "onnx.Squeeze"(%[[VAL_27]]#0) {axes = [0]} : (memref<1x8xf32>) -> memref<8xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Squeeze"(%[[VAL_27]]#1) {axes = [0]} : (memref<1x8xf32>) -> memref<8xf32>
// CHECK:           %[[VAL_30:.*]]:2 = "onnx.Split"(%[[VAL_28]]) {axis = 0 : si64} : (memref<8xf32>) -> (memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_31:.*]]:2 = "onnx.Split"(%[[VAL_29]]) {axis = 0 : si64} : (memref<8xf32>) -> (memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_32:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_32]]) with (%[[VAL_32]] -> %[[VAL_33:.*]] = 0 to 7) {
// CHECK:             %[[VAL_34:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_35:.*]] = constant 0 : index
// CHECK:             %[[VAL_36:.*]] = constant 2 : index
// CHECK:             %[[VAL_37:.*]] = constant 3 : index
// CHECK:             %[[VAL_38:.*]] = constant 0 : index
// CHECK:             %[[VAL_39:.*]] = constant 0 : index
// CHECK:             %[[VAL_40:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_40]]#0, %[[VAL_40]]#1) with (%[[VAL_40]]#0 -> %[[VAL_41:.*]] = %[[VAL_38]] to %[[VAL_36]], %[[VAL_40]]#1 -> %[[VAL_42:.*]] = %[[VAL_39]] to %[[VAL_37]]) {
// CHECK:               %[[VAL_43:.*]]:2 = krnl.get_induction_var_value(%[[VAL_40]]#0, %[[VAL_40]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_44:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_33]], %[[VAL_43]]#0, %[[VAL_43]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_44]], %[[VAL_34]]{{\[}}%[[VAL_43]]#0, %[[VAL_43]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_45:.*]] = "onnx.MatMul"(%[[VAL_34]], %[[VAL_23]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_46:.*]] = "onnx.MatMul"(%[[VAL_6]], %[[VAL_24]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_47:.*]] = constant 2 : index
// CHECK:             %[[VAL_48:.*]] = constant 4 : index
// CHECK:             %[[VAL_49:.*]] = constant 0 : index
// CHECK:             %[[VAL_50:.*]] = constant 0 : index
// CHECK:             %[[VAL_51:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_51]]#0, %[[VAL_51]]#1) with (%[[VAL_51]]#0 -> %[[VAL_52:.*]] = %[[VAL_49]] to %[[VAL_47]], %[[VAL_51]]#1 -> %[[VAL_53:.*]] = %[[VAL_50]] to %[[VAL_48]]) {
// CHECK:               %[[VAL_54:.*]]:2 = krnl.get_induction_var_value(%[[VAL_51]]#0, %[[VAL_51]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_55:.*]] = krnl.load %[[VAL_45]]{{\[}}%[[VAL_54]]#0, %[[VAL_54]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_56:.*]] = krnl.load %[[VAL_46]]{{\[}}%[[VAL_54]]#0, %[[VAL_54]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_57:.*]] = addf %[[VAL_55]], %[[VAL_56]] : f32
// CHECK:               %[[VAL_58:.*]] = krnl.load %[[VAL_30]]#0{{\[}}%[[VAL_54]]#1] : memref<4xf32>
// CHECK:               %[[VAL_59:.*]] = krnl.load %[[VAL_30]]#1{{\[}}%[[VAL_54]]#1] : memref<4xf32>
// CHECK:               %[[VAL_60:.*]] = addf %[[VAL_57]], %[[VAL_58]] : f32
// CHECK:               %[[VAL_61:.*]] = addf %[[VAL_60]], %[[VAL_59]] : f32
// CHECK:               %[[VAL_62:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_61]], %[[VAL_62]][] : memref<f32>
// CHECK:               %[[VAL_63:.*]] = "onnx.Tanh"(%[[VAL_62]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_64:.*]] = krnl.load %[[VAL_63]][] : memref<f32>
// CHECK:               krnl.store %[[VAL_64]], %[[VAL_6]]{{\[}}%[[VAL_54]]#0, %[[VAL_54]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_34]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_65:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_65]]) with (%[[VAL_65]] -> %[[VAL_66:.*]] = 0 to 7) {
// CHECK:             %[[VAL_67:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_68:.*]] = constant 1 : index
// CHECK:             %[[VAL_69:.*]] = constant 7 : index
// CHECK:             %[[VAL_70:.*]] = affine.apply #map(%[[VAL_66]]){{\[}}%[[VAL_69]]]
// CHECK:             %[[VAL_71:.*]] = constant 2 : index
// CHECK:             %[[VAL_72:.*]] = constant 3 : index
// CHECK:             %[[VAL_73:.*]] = constant 0 : index
// CHECK:             %[[VAL_74:.*]] = constant 0 : index
// CHECK:             %[[VAL_75:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_75]]#0, %[[VAL_75]]#1) with (%[[VAL_75]]#0 -> %[[VAL_76:.*]] = %[[VAL_73]] to %[[VAL_71]], %[[VAL_75]]#1 -> %[[VAL_77:.*]] = %[[VAL_74]] to %[[VAL_72]]) {
// CHECK:               %[[VAL_78:.*]]:2 = krnl.get_induction_var_value(%[[VAL_75]]#0, %[[VAL_75]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_79:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_70]], %[[VAL_78]]#0, %[[VAL_78]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_79]], %[[VAL_67]]{{\[}}%[[VAL_78]]#0, %[[VAL_78]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_80:.*]] = "onnx.MatMul"(%[[VAL_67]], %[[VAL_25]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_81:.*]] = "onnx.MatMul"(%[[VAL_5]], %[[VAL_26]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_82:.*]] = constant 2 : index
// CHECK:             %[[VAL_83:.*]] = constant 4 : index
// CHECK:             %[[VAL_84:.*]] = constant 0 : index
// CHECK:             %[[VAL_85:.*]] = constant 0 : index
// CHECK:             %[[VAL_86:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_86]]#0, %[[VAL_86]]#1) with (%[[VAL_86]]#0 -> %[[VAL_87:.*]] = %[[VAL_84]] to %[[VAL_82]], %[[VAL_86]]#1 -> %[[VAL_88:.*]] = %[[VAL_85]] to %[[VAL_83]]) {
// CHECK:               %[[VAL_89:.*]]:2 = krnl.get_induction_var_value(%[[VAL_86]]#0, %[[VAL_86]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_90:.*]] = krnl.load %[[VAL_80]]{{\[}}%[[VAL_89]]#0, %[[VAL_89]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_91:.*]] = krnl.load %[[VAL_81]]{{\[}}%[[VAL_89]]#0, %[[VAL_89]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_92:.*]] = addf %[[VAL_90]], %[[VAL_91]] : f32
// CHECK:               %[[VAL_93:.*]] = krnl.load %[[VAL_31]]#0{{\[}}%[[VAL_89]]#1] : memref<4xf32>
// CHECK:               %[[VAL_94:.*]] = krnl.load %[[VAL_31]]#1{{\[}}%[[VAL_89]]#1] : memref<4xf32>
// CHECK:               %[[VAL_95:.*]] = addf %[[VAL_92]], %[[VAL_93]] : f32
// CHECK:               %[[VAL_96:.*]] = addf %[[VAL_95]], %[[VAL_94]] : f32
// CHECK:               %[[VAL_97:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_96]], %[[VAL_97]][] : memref<f32>
// CHECK:               %[[VAL_98:.*]] = "onnx.Tanh"(%[[VAL_97]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_99:.*]] = krnl.load %[[VAL_98]][] : memref<f32>
// CHECK:               krnl.store %[[VAL_99]], %[[VAL_5]]{{\[}}%[[VAL_89]]#0, %[[VAL_89]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_67]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_100:.*]] = constant 2 : index
// CHECK:           %[[VAL_101:.*]] = constant 4 : index
// CHECK:           %[[VAL_102:.*]] = constant 0 : index
// CHECK:           %[[VAL_103:.*]] = constant 0 : index
// CHECK:           %[[VAL_104:.*]] = constant 0 : index
// CHECK:           %[[VAL_105:.*]] = constant 1 : index
// CHECK:           %[[VAL_106:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_106]]#0, %[[VAL_106]]#1) with (%[[VAL_106]]#0 -> %[[VAL_107:.*]] = %[[VAL_102]] to %[[VAL_100]], %[[VAL_106]]#1 -> %[[VAL_108:.*]] = %[[VAL_103]] to %[[VAL_101]]) {
// CHECK:             %[[VAL_109:.*]]:2 = krnl.get_induction_var_value(%[[VAL_106]]#0, %[[VAL_106]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             %[[VAL_110:.*]] = krnl.load %[[VAL_6]]{{\[}}%[[VAL_109]]#0, %[[VAL_109]]#1] : memref<2x4xf32>
// CHECK:             krnl.store %[[VAL_110]], %[[VAL_7]]{{\[}}%[[VAL_104]], %[[VAL_109]]#0, %[[VAL_109]]#1] : memref<2x2x4xf32>
// CHECK:             %[[VAL_111:.*]] = krnl.load %[[VAL_5]]{{\[}}%[[VAL_109]]#0, %[[VAL_109]]#1] : memref<2x4xf32>
// CHECK:             krnl.store %[[VAL_111]], %[[VAL_7]]{{\[}}%[[VAL_105]], %[[VAL_109]]#0, %[[VAL_109]]#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           memref.dealloc %[[VAL_6]] : memref<2x4xf32>
// CHECK:           memref.dealloc %[[VAL_5]] : memref<2x4xf32>
// CHECK:           return %[[VAL_7]] : memref<2x2x4xf32>
// CHECK:         }

}

// -----

func private @test_rnn_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x4x?xf32>, %arg2: tensor<1x4x4xf32>, %arg3: tensor<1x8xf32>, %arg4: tensor<1x?x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<?x?x?xf32>, tensor<1x4x?xf32>, tensor<1x4x4xf32>, tensor<1x8xf32>, none, tensor<1x?x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_rnn_unknown_dims(
// CHECK-SAME:                                        %[[VAL_0:.*]]: memref<?x?x?xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: memref<1x4x?xf32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: memref<1x4x4xf32>,
// CHECK-SAME:                                        %[[VAL_3:.*]]: memref<1x8xf32>,
// CHECK-SAME:                                        %[[VAL_4:.*]]: memref<1x?x4xf32>) -> memref<1x?x4xf32> {
// CHECK:           %[[VAL_5:.*]] = constant unit
// CHECK:           %[[VAL_6:.*]] = constant 1 : index
// CHECK:           %[[VAL_7:.*]] = memref.dim %[[VAL_0]], %[[VAL_6]] : memref<?x?x?xf32>
// CHECK:           %[[VAL_8:.*]] = memref.alloc(%[[VAL_7]]) : memref<1x?x4xf32>
// CHECK:           %[[VAL_9:.*]] = constant 1 : index
// CHECK:           %[[VAL_10:.*]] = memref.dim %[[VAL_0]], %[[VAL_9]] : memref<?x?x?xf32>
// CHECK:           %[[VAL_11:.*]] = memref.alloc(%[[VAL_10]]) : memref<?x4xf32>
// CHECK:           %[[VAL_12:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_13:.*]] = constant 0 : index
// CHECK:           %[[VAL_14:.*]] = constant 1 : index
// CHECK:           %[[VAL_15:.*]]:2 = krnl.define_loops 2
// CHECK:           %[[VAL_16:.*]] = constant 0 : index
// CHECK:           %[[VAL_17:.*]] = memref.dim %[[VAL_11]], %[[VAL_16]] : memref<?x4xf32>
// CHECK:           krnl.iterate(%[[VAL_15]]#0, %[[VAL_15]]#1) with (%[[VAL_15]]#0 -> %[[VAL_18:.*]] = 0 to %[[VAL_17]], %[[VAL_15]]#1 -> %[[VAL_19:.*]] = 0 to 4) {
// CHECK:             %[[VAL_20:.*]] = krnl.load %[[VAL_4]]{{\[}}%[[VAL_13]], %[[VAL_18]], %[[VAL_19]]] : memref<1x?x4xf32>
// CHECK:             krnl.store %[[VAL_20]], %[[VAL_11]]{{\[}}%[[VAL_18]], %[[VAL_19]]] : memref<?x4xf32>
// CHECK:           }
// CHECK:           %[[VAL_21:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x4x?xf32>) -> memref<4x?xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Transpose"(%[[VAL_21]]) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Transpose"(%[[VAL_22]]) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Squeeze"(%[[VAL_3]]) {axes = [0]} : (memref<1x8xf32>) -> memref<8xf32>
// CHECK:           %[[VAL_26:.*]]:2 = "onnx.Split"(%[[VAL_25]]) {axis = 0 : si64} : (memref<8xf32>) -> (memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_27:.*]] = krnl.define_loops 1
// CHECK:           %[[VAL_28:.*]] = constant 0 : index
// CHECK:           %[[VAL_29:.*]] = memref.dim %[[VAL_0]], %[[VAL_28]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate(%[[VAL_27]]) with (%[[VAL_27]] -> %[[VAL_30:.*]] = 0 to %[[VAL_29]]) {
// CHECK:             %[[VAL_31:.*]] = constant 0 : index
// CHECK:             %[[VAL_32:.*]] = constant 1 : index
// CHECK:             %[[VAL_33:.*]] = memref.dim %[[VAL_0]], %[[VAL_32]] : memref<?x?x?xf32>
// CHECK:             %[[VAL_34:.*]] = constant 2 : index
// CHECK:             %[[VAL_35:.*]] = memref.dim %[[VAL_0]], %[[VAL_34]] : memref<?x?x?xf32>
// CHECK:             %[[VAL_36:.*]] = memref.alloc(%[[VAL_33]], %[[VAL_35]]) : memref<?x?xf32>
// CHECK:             %[[VAL_37:.*]] = constant 0 : index
// CHECK:             %[[VAL_38:.*]] = memref.dim %[[VAL_36]], %[[VAL_37]] : memref<?x?xf32>
// CHECK:             %[[VAL_39:.*]] = constant 1 : index
// CHECK:             %[[VAL_40:.*]] = memref.dim %[[VAL_36]], %[[VAL_39]] : memref<?x?xf32>
// CHECK:             %[[VAL_41:.*]] = constant 0 : index
// CHECK:             %[[VAL_42:.*]] = constant 0 : index
// CHECK:             %[[VAL_43:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_43]]#0, %[[VAL_43]]#1) with (%[[VAL_43]]#0 -> %[[VAL_44:.*]] = %[[VAL_41]] to %[[VAL_38]], %[[VAL_43]]#1 -> %[[VAL_45:.*]] = %[[VAL_42]] to %[[VAL_40]]) {
// CHECK:               %[[VAL_46:.*]]:2 = krnl.get_induction_var_value(%[[VAL_43]]#0, %[[VAL_43]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_47:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_30]], %[[VAL_46]]#0, %[[VAL_46]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store %[[VAL_47]], %[[VAL_36]]{{\[}}%[[VAL_46]]#0, %[[VAL_46]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK:             %[[VAL_48:.*]] = "onnx.MatMul"(%[[VAL_36]], %[[VAL_23]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_49:.*]] = "onnx.MatMul"(%[[VAL_11]], %[[VAL_24]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_50:.*]] = constant 0 : index
// CHECK:             %[[VAL_51:.*]] = memref.dim %[[VAL_11]], %[[VAL_50]] : memref<?x4xf32>
// CHECK:             %[[VAL_52:.*]] = constant 4 : index
// CHECK:             %[[VAL_53:.*]] = constant 0 : index
// CHECK:             %[[VAL_54:.*]] = constant 0 : index
// CHECK:             %[[VAL_55:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_55]]#0, %[[VAL_55]]#1) with (%[[VAL_55]]#0 -> %[[VAL_56:.*]] = %[[VAL_53]] to %[[VAL_51]], %[[VAL_55]]#1 -> %[[VAL_57:.*]] = %[[VAL_54]] to %[[VAL_52]]) {
// CHECK:               %[[VAL_58:.*]]:2 = krnl.get_induction_var_value(%[[VAL_55]]#0, %[[VAL_55]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_59:.*]] = krnl.load %[[VAL_48]]{{\[}}%[[VAL_58]]#0, %[[VAL_58]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_60:.*]] = krnl.load %[[VAL_49]]{{\[}}%[[VAL_58]]#0, %[[VAL_58]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_61:.*]] = addf %[[VAL_59]], %[[VAL_60]] : f32
// CHECK:               %[[VAL_62:.*]] = krnl.load %[[VAL_26]]#0{{\[}}%[[VAL_58]]#1] : memref<4xf32>
// CHECK:               %[[VAL_63:.*]] = krnl.load %[[VAL_26]]#1{{\[}}%[[VAL_58]]#1] : memref<4xf32>
// CHECK:               %[[VAL_64:.*]] = addf %[[VAL_61]], %[[VAL_62]] : f32
// CHECK:               %[[VAL_65:.*]] = addf %[[VAL_64]], %[[VAL_63]] : f32
// CHECK:               %[[VAL_66:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_65]], %[[VAL_66]][] : memref<f32>
// CHECK:               %[[VAL_67:.*]] = "onnx.Tanh"(%[[VAL_66]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_68:.*]] = krnl.load %[[VAL_67]][] : memref<f32>
// CHECK:               krnl.store %[[VAL_68]], %[[VAL_11]]{{\[}}%[[VAL_58]]#0, %[[VAL_58]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_36]] : memref<?x?xf32>
// CHECK:           }
// CHECK:           %[[VAL_69:.*]] = constant 16 : i64
// CHECK:           %[[VAL_70:.*]] = constant 0 : index
// CHECK:           %[[VAL_71:.*]] = memref.dim %[[VAL_11]], %[[VAL_70]] : memref<?x4xf32>
// CHECK:           %[[VAL_72:.*]] = index_cast %[[VAL_71]] : index to i64
// CHECK:           %[[VAL_73:.*]] = muli %[[VAL_69]], %[[VAL_72]] : i64
// CHECK:           "krnl.memcpy"(%[[VAL_8]], %[[VAL_11]], %[[VAL_73]]) : (memref<1x?x4xf32>, memref<?x4xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_11]] : memref<?x4xf32>
// CHECK:           return %[[VAL_8]] : memref<1x?x4xf32>
// CHECK:         }

}
