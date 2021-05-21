// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='test-rnn-ops-lowering' %s -split-input-file | FileCheck %s

/// Check GRU with three required inputs (X, W, R). The optional inputs are default.
/// Also check the equation for 'ht' when linear_before_reset = 0 (default)
func private @test_gru_general_computation(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_gru_general_computation(
// CHECK-SAME:                                               %[[VAL_0:.*]]: memref<4x3x2xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: memref<1x9x2xf32>,
// CHECK-SAME:                                               %[[VAL_2:.*]]: memref<1x9x3xf32>) -> memref<1x3x3xf32> {
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<1x3x3xf32>
// CHECK:           %[[VAL_5:.*]] = constant unit
// CHECK:           %[[VAL_6:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_7:.*]] = constant 0 : index
// CHECK:           %[[VAL_8:.*]] = constant 1 : index
// Initialize the intermediate states Ht.
// CHECK:           %[[VAL_9:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_9]]#0, %[[VAL_9]]#1) with (%[[VAL_9]]#0 -> %[[VAL_10:.*]] = 0 to 3, %[[VAL_9]]#1 -> %[[VAL_11:.*]] = 0 to 3) {
// CHECK:             krnl.store %[[VAL_6]], %[[VAL_3]]{{\[}}%[[VAL_10]], %[[VAL_11]]] : memref<3x3xf32>
// CHECK:           }
// Prepare weights and biases.
// CHECK:           %[[VAL_12:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x9x2xf32>) -> memref<9x2xf32>
// CHECK:           %[[VAL_13:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x9x3xf32>) -> memref<9x3xf32>
// CHECK:           %[[VAL_14:.*]]:3 = "onnx.Split"(%[[VAL_12]]) {axis = 0 : si64} : (memref<9x2xf32>) -> (memref<3x2xf32>, memref<3x2xf32>, memref<3x2xf32>)
// CHECK:           %[[VAL_15:.*]] = "onnx.Transpose"(%[[VAL_14]]#0) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_16:.*]] = "onnx.Transpose"(%[[VAL_14]]#1) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_14]]#2) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_18:.*]]:3 = "onnx.Split"(%[[VAL_13]]) {axis = 0 : si64} : (memref<9x3xf32>) -> (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>)
// CHECK:           %[[VAL_19:.*]] = "onnx.Transpose"(%[[VAL_18]]#0) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Transpose"(%[[VAL_18]]#1) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Transpose"(%[[VAL_18]]#2) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// GRU computation. Iterate over the sequence.
// CHECK:           %[[VAL_22:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_22]]) with (%[[VAL_22]] -> %[[VAL_23:.*]] = 0 to 4) {
// Get a slice of X for the current timestep.
// CHECK:             %[[VAL_24:.*]] = memref.alloc() : memref<3x2xf32>
// CHECK:             %[[VAL_25:.*]] = constant 0 : index
// CHECK:             %[[VAL_26:.*]] = constant 3 : index
// CHECK:             %[[VAL_27:.*]] = constant 2 : index
// CHECK:             %[[VAL_28:.*]] = constant 0 : index
// CHECK:             %[[VAL_29:.*]] = constant 0 : index
// CHECK:             %[[VAL_30:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_30]]#0, %[[VAL_30]]#1) with (%[[VAL_30]]#0 -> %[[VAL_31:.*]] = %[[VAL_28]] to %[[VAL_26]], %[[VAL_30]]#1 -> %[[VAL_32:.*]] = %[[VAL_29]] to %[[VAL_27]]) {
// CHECK:               %[[VAL_33:.*]]:2 = krnl.get_induction_var_value(%[[VAL_30]]#0, %[[VAL_30]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_34:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_23]], %[[VAL_33]]#0, %[[VAL_33]]#1] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_34]], %[[VAL_24]]{{\[}}%[[VAL_33]]#0, %[[VAL_33]]#1] : memref<3x2xf32>
// CHECK:             }
// CHECK:             %[[VAL_35:.*]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> memref<1xf32>
// zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
// CHECK:             %[[VAL_36:.*]] = "onnx.MatMul"(%[[VAL_24]], %[[VAL_15]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_37:.*]] = "onnx.MatMul"(%[[VAL_3]], %[[VAL_19]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_38:.*]] = "onnx.Add"(%[[VAL_36]], %[[VAL_37]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_39:.*]] = "onnx.Sigmoid"(%[[VAL_38]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
// CHECK:             %[[VAL_40:.*]] = "onnx.MatMul"(%[[VAL_24]], %[[VAL_16]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_41:.*]] = "onnx.MatMul"(%[[VAL_3]], %[[VAL_20]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_42:.*]] = "onnx.Add"(%[[VAL_40]], %[[VAL_41]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_43:.*]] = "onnx.Sigmoid"(%[[VAL_42]]) : (memref<3x3xf32>) -> memref<3x3xf32>
//   ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
// CHECK:             %[[VAL_44:.*]] = "onnx.MatMul"(%[[VAL_24]], %[[VAL_17]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_45:.*]] = "onnx.Mul"(%[[VAL_43]], %[[VAL_3]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_46:.*]] = "onnx.MatMul"(%[[VAL_45]], %[[VAL_21]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_47:.*]] = "onnx.Add"(%[[VAL_44]], %[[VAL_46]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_48:.*]] = "onnx.Tanh"(%[[VAL_47]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// Ht = (1 - zt) (.) ht + zt (.) Ht-1
// CHECK:             %[[VAL_49:.*]] = "onnx.Sub"(%[[VAL_35]], %[[VAL_39]]) : (memref<1xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_50:.*]] = "onnx.Mul"(%[[VAL_49]], %[[VAL_48]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_51:.*]] = "onnx.Mul"(%[[VAL_39]], %[[VAL_3]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_52:.*]] = "onnx.Add"(%[[VAL_50]], %[[VAL_51]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// Store the current Ht.
// CHECK:             %[[VAL_53:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_3]], %[[VAL_52]], %[[VAL_53]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             memref.dealloc %[[VAL_24]] : memref<3x2xf32>
// CHECK:           }
// Store the intermediate states to the returned states.
// CHECK:           %[[VAL_54:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_4]], %[[VAL_3]], %[[VAL_54]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_3]] : memref<3x3xf32>
// CHECK:           return %[[VAL_4]] : memref<1x3x3xf32>
// CHECK:         }

}

// -----

/// GRU with three required inputs (X, W, R). The optional inputs are default.
/// Check the equation for 'ht' when linear_before_reset !=0.
func private @test_gru_linear_before_reset(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64, linear_before_reset = 1 : si64} : (tensor<4x3x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_gru_linear_before_reset(
// CHECK-SAME:                                               %[[VAL_0:.*]]: memref<4x3x2xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: memref<1x9x2xf32>,
// CHECK-SAME:                                               %[[VAL_2:.*]]: memref<1x9x3xf32>) -> memref<1x3x3xf32> {
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<1x3x3xf32>
// CHECK:           %[[VAL_5:.*]] = constant unit
// CHECK:           %[[VAL_6:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_7:.*]] = constant 0 : index
// CHECK:           %[[VAL_8:.*]] = constant 1 : index
// CHECK:           %[[VAL_9:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_9]]#0, %[[VAL_9]]#1) with (%[[VAL_9]]#0 -> %[[VAL_10:.*]] = 0 to 3, %[[VAL_9]]#1 -> %[[VAL_11:.*]] = 0 to 3) {
// CHECK:             krnl.store %[[VAL_6]], %[[VAL_3]]{{\[}}%[[VAL_10]], %[[VAL_11]]] : memref<3x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x9x2xf32>) -> memref<9x2xf32>
// CHECK:           %[[VAL_13:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x9x3xf32>) -> memref<9x3xf32>
// CHECK:           %[[VAL_14:.*]]:3 = "onnx.Split"(%[[VAL_12]]) {axis = 0 : si64} : (memref<9x2xf32>) -> (memref<3x2xf32>, memref<3x2xf32>, memref<3x2xf32>)
// CHECK:           %[[VAL_15:.*]] = "onnx.Transpose"(%[[VAL_14]]#0) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_16:.*]] = "onnx.Transpose"(%[[VAL_14]]#1) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_14]]#2) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_18:.*]]:3 = "onnx.Split"(%[[VAL_13]]) {axis = 0 : si64} : (memref<9x3xf32>) -> (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>)
// CHECK:           %[[VAL_19:.*]] = "onnx.Transpose"(%[[VAL_18]]#0) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Transpose"(%[[VAL_18]]#1) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Transpose"(%[[VAL_18]]#2) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_22:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_22]]) with (%[[VAL_22]] -> %[[VAL_23:.*]] = 0 to 4) {
// CHECK:             %[[VAL_24:.*]] = memref.alloc() : memref<3x2xf32>
// CHECK:             %[[VAL_25:.*]] = constant 0 : index
// CHECK:             %[[VAL_26:.*]] = constant 3 : index
// CHECK:             %[[VAL_27:.*]] = constant 2 : index
// CHECK:             %[[VAL_28:.*]] = constant 0 : index
// CHECK:             %[[VAL_29:.*]] = constant 0 : index
// CHECK:             %[[VAL_30:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_30]]#0, %[[VAL_30]]#1) with (%[[VAL_30]]#0 -> %[[VAL_31:.*]] = %[[VAL_28]] to %[[VAL_26]], %[[VAL_30]]#1 -> %[[VAL_32:.*]] = %[[VAL_29]] to %[[VAL_27]]) {
// CHECK:               %[[VAL_33:.*]]:2 = krnl.get_induction_var_value(%[[VAL_30]]#0, %[[VAL_30]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_34:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_23]], %[[VAL_33]]#0, %[[VAL_33]]#1] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_34]], %[[VAL_24]]{{\[}}%[[VAL_33]]#0, %[[VAL_33]]#1] : memref<3x2xf32>
// CHECK:             }
// CHECK:             %[[VAL_35:.*]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> memref<1xf32>
// CHECK:             %[[VAL_36:.*]] = "onnx.MatMul"(%[[VAL_24]], %[[VAL_15]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_37:.*]] = "onnx.MatMul"(%[[VAL_3]], %[[VAL_19]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_38:.*]] = "onnx.Add"(%[[VAL_36]], %[[VAL_37]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_39:.*]] = "onnx.Sigmoid"(%[[VAL_38]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_40:.*]] = "onnx.MatMul"(%[[VAL_24]], %[[VAL_16]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_41:.*]] = "onnx.MatMul"(%[[VAL_3]], %[[VAL_20]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_42:.*]] = "onnx.Add"(%[[VAL_40]], %[[VAL_41]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_43:.*]] = "onnx.Sigmoid"(%[[VAL_42]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_44:.*]] = "onnx.MatMul"(%[[VAL_24]], %[[VAL_17]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_45:.*]] = "onnx.MatMul"(%[[VAL_3]], %[[VAL_21]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_46:.*]] = "onnx.Mul"(%[[VAL_43]], %[[VAL_45]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_47:.*]] = "onnx.Add"(%[[VAL_44]], %[[VAL_46]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_48:.*]] = "onnx.Tanh"(%[[VAL_47]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_49:.*]] = "onnx.Sub"(%[[VAL_35]], %[[VAL_39]]) : (memref<1xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_50:.*]] = "onnx.Mul"(%[[VAL_49]], %[[VAL_48]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_51:.*]] = "onnx.Mul"(%[[VAL_39]], %[[VAL_3]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_52:.*]] = "onnx.Add"(%[[VAL_50]], %[[VAL_51]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_53:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_3]], %[[VAL_52]], %[[VAL_53]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             memref.dealloc %[[VAL_24]] : memref<3x2xf32>
// CHECK:           }
// CHECK:           %[[VAL_54:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_4]], %[[VAL_3]], %[[VAL_54]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_3]] : memref<3x3xf32>
// CHECK:           return %[[VAL_4]] : memref<1x3x3xf32>
// CHECK:         }

}

// -----

/// Check GRU with three required inputs (X, W, R), and bias input.
func private @test_gru_with_bias(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>, %arg3: tensor<1x18xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, tensor<1x18xf32>, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_gru_with_bias(
// CHECK-SAME:                                     %[[VAL_0:.*]]: memref<4x3x2xf32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: memref<1x9x2xf32>,
// CHECK-SAME:                                     %[[VAL_2:.*]]: memref<1x9x3xf32>,
// CHECK-SAME:                                     %[[VAL_3:.*]]: memref<1x18xf32>) -> memref<1x3x3xf32> {
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<1x3x3xf32>
// CHECK:           %[[VAL_6:.*]] = constant unit
// CHECK:           %[[VAL_7:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_8:.*]] = constant 0 : index
// CHECK:           %[[VAL_9:.*]] = constant 1 : index
// CHECK:           %[[VAL_10:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_10]]#0, %[[VAL_10]]#1) with (%[[VAL_10]]#0 -> %[[VAL_11:.*]] = 0 to 3, %[[VAL_10]]#1 -> %[[VAL_12:.*]] = 0 to 3) {
// CHECK:             krnl.store %[[VAL_7]], %[[VAL_4]]{{\[}}%[[VAL_11]], %[[VAL_12]]] : memref<3x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x9x2xf32>) -> memref<9x2xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x9x3xf32>) -> memref<9x3xf32>
// CHECK:           %[[VAL_15:.*]]:3 = "onnx.Split"(%[[VAL_13]]) {axis = 0 : si64} : (memref<9x2xf32>) -> (memref<3x2xf32>, memref<3x2xf32>, memref<3x2xf32>)
// CHECK:           %[[VAL_16:.*]] = "onnx.Transpose"(%[[VAL_15]]#0) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_15]]#1) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_15]]#2) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_19:.*]]:3 = "onnx.Split"(%[[VAL_14]]) {axis = 0 : si64} : (memref<9x3xf32>) -> (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>)
// CHECK:           %[[VAL_20:.*]] = "onnx.Transpose"(%[[VAL_19]]#0) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Transpose"(%[[VAL_19]]#1) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Transpose"(%[[VAL_19]]#2) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Squeeze"(%[[VAL_3]]) {axes = [0]} : (memref<1x18xf32>) -> memref<18xf32>
// CHECK:           %[[VAL_24:.*]]:6 = "onnx.Split"(%[[VAL_23]]) {axis = 0 : si64} : (memref<18xf32>) -> (memref<3xf32>, memref<3xf32>, memref<3xf32>, memref<3xf32>, memref<3xf32>, memref<3xf32>)
// CHECK:           %[[VAL_25:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_25]]) with (%[[VAL_25]] -> %[[VAL_26:.*]] = 0 to 4) {
// CHECK:             %[[VAL_27:.*]] = memref.alloc() : memref<3x2xf32>
// CHECK:             %[[VAL_28:.*]] = constant 0 : index
// CHECK:             %[[VAL_29:.*]] = constant 3 : index
// CHECK:             %[[VAL_30:.*]] = constant 2 : index
// CHECK:             %[[VAL_31:.*]] = constant 0 : index
// CHECK:             %[[VAL_32:.*]] = constant 0 : index
// CHECK:             %[[VAL_33:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_33]]#0, %[[VAL_33]]#1) with (%[[VAL_33]]#0 -> %[[VAL_34:.*]] = %[[VAL_31]] to %[[VAL_29]], %[[VAL_33]]#1 -> %[[VAL_35:.*]] = %[[VAL_32]] to %[[VAL_30]]) {
// CHECK:               %[[VAL_36:.*]]:2 = krnl.get_induction_var_value(%[[VAL_33]]#0, %[[VAL_33]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_37:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_26]], %[[VAL_36]]#0, %[[VAL_36]]#1] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_37]], %[[VAL_27]]{{\[}}%[[VAL_36]]#0, %[[VAL_36]]#1] : memref<3x2xf32>
// CHECK:             }
// CHECK:             %[[VAL_38:.*]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> memref<1xf32>
// CHECK:             %[[VAL_39:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_16]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_40:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_20]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_41:.*]] = "onnx.Add"(%[[VAL_39]], %[[VAL_40]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_42:.*]] = "onnx.Add"(%[[VAL_41]], %[[VAL_24]]#0) : (memref<3x3xf32>, memref<3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_43:.*]] = "onnx.Add"(%[[VAL_42]], %[[VAL_24]]#3) : (memref<3x3xf32>, memref<3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_44:.*]] = "onnx.Sigmoid"(%[[VAL_43]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_45:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_17]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_46:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_21]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_47:.*]] = "onnx.Add"(%[[VAL_45]], %[[VAL_46]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_48:.*]] = "onnx.Add"(%[[VAL_47]], %[[VAL_24]]#1) : (memref<3x3xf32>, memref<3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_49:.*]] = "onnx.Add"(%[[VAL_48]], %[[VAL_24]]#4) : (memref<3x3xf32>, memref<3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_50:.*]] = "onnx.Sigmoid"(%[[VAL_49]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_51:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_18]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_52:.*]] = "onnx.Mul"(%[[VAL_50]], %[[VAL_4]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_53:.*]] = "onnx.MatMul"(%[[VAL_52]], %[[VAL_22]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_54:.*]] = "onnx.Add"(%[[VAL_51]], %[[VAL_53]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.Add"(%[[VAL_54]], %[[VAL_24]]#5) : (memref<3x3xf32>, memref<3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_56:.*]] = "onnx.Add"(%[[VAL_55]], %[[VAL_24]]#2) : (memref<3x3xf32>, memref<3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_57:.*]] = "onnx.Tanh"(%[[VAL_56]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_58:.*]] = "onnx.Sub"(%[[VAL_38]], %[[VAL_44]]) : (memref<1xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_59:.*]] = "onnx.Mul"(%[[VAL_58]], %[[VAL_57]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_60:.*]] = "onnx.Mul"(%[[VAL_44]], %[[VAL_4]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_61:.*]] = "onnx.Add"(%[[VAL_59]], %[[VAL_60]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_62:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_4]], %[[VAL_61]], %[[VAL_62]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             memref.dealloc %[[VAL_27]] : memref<3x2xf32>
// CHECK:           }
// CHECK:           %[[VAL_63:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_5]], %[[VAL_4]], %[[VAL_63]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_4]] : memref<3x3xf32>
// CHECK:           return %[[VAL_5]] : memref<1x3x3xf32>
// CHECK:         }

}

// -----

// Check handling unknown dimensions for GRU by checking the
// correctness of allocating and deallocating memory.
func private @test_gru_unknown_dims_allocation(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x9x?xf32>, %arg2: tensor<1x9x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<?x?x?xf32>, tensor<1x9x?xf32>, tensor<1x9x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_gru_unknown_dims_allocation(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: memref<?x?x?xf32>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: memref<1x9x?xf32>,
// CHECK-SAME:                                                   %[[VAL_2:.*]]: memref<1x9x3xf32>) -> memref<1x?x3xf32> {
// CHECK:           %[[VAL_3:.*]] = constant unit
// CHECK:           %[[VAL_4:.*]] = constant 1 : index
// CHECK:           %[[VAL_5:.*]] = memref.dim %[[VAL_0]], %[[VAL_4]] : memref<?x?x?xf32>
// CHECK:           %[[VAL_6:.*]] = memref.alloc(%[[VAL_5]]) : memref<1x?x3xf32>
// CHECK:           %[[VAL_7:.*]] = constant 1 : index
// CHECK:           %[[VAL_8:.*]] = memref.dim %[[VAL_0]], %[[VAL_7]] : memref<?x?x?xf32>
// CHECK:           %[[VAL_9:.*]] = memref.alloc(%[[VAL_8]]) : memref<?x3xf32>
// CHECK:           %[[VAL_10:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_11:.*]] = constant 0 : index
// CHECK:           %[[VAL_12:.*]] = constant 1 : index
// CHECK:           %[[VAL_13:.*]]:2 = krnl.define_loops 2
// CHECK:           %[[VAL_14:.*]] = constant 0 : index
// CHECK:           %[[VAL_15:.*]] = memref.dim %[[VAL_9]], %[[VAL_14]] : memref<?x3xf32>
// CHECK:           krnl.iterate(%[[VAL_13]]#0, %[[VAL_13]]#1) with (%[[VAL_13]]#0 -> %[[VAL_16:.*]] = 0 to %[[VAL_15]], %[[VAL_13]]#1 -> %[[VAL_17:.*]] = 0 to 3) {
// CHECK:             krnl.store %[[VAL_10]], %[[VAL_9]]{{\[}}%[[VAL_16]], %[[VAL_17]]] : memref<?x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_18:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x9x?xf32>) -> memref<9x?xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x9x3xf32>) -> memref<9x3xf32>
// CHECK:           %[[VAL_20:.*]]:3 = "onnx.Split"(%[[VAL_18]]) {axis = 0 : si64} : (memref<9x?xf32>) -> (memref<3x?xf32>, memref<3x?xf32>, memref<3x?xf32>)
// CHECK:           %[[VAL_21:.*]] = "onnx.Transpose"(%[[VAL_20]]#0) {perm = [1, 0]} : (memref<3x?xf32>) -> memref<?x3xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Transpose"(%[[VAL_20]]#1) {perm = [1, 0]} : (memref<3x?xf32>) -> memref<?x3xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Transpose"(%[[VAL_20]]#2) {perm = [1, 0]} : (memref<3x?xf32>) -> memref<?x3xf32>
// CHECK:           %[[VAL_24:.*]]:3 = "onnx.Split"(%[[VAL_19]]) {axis = 0 : si64} : (memref<9x3xf32>) -> (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>)
// CHECK:           %[[VAL_25:.*]] = "onnx.Transpose"(%[[VAL_24]]#0) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.Transpose"(%[[VAL_24]]#1) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_27:.*]] = "onnx.Transpose"(%[[VAL_24]]#2) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_28:.*]] = krnl.define_loops 1
// CHECK:           %[[VAL_29:.*]] = constant 0 : index
// CHECK:           %[[VAL_30:.*]] = memref.dim %[[VAL_0]], %[[VAL_29]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate(%[[VAL_28]]) with (%[[VAL_28]] -> %[[VAL_31:.*]] = 0 to %[[VAL_30]]) {
// CHECK:             %[[VAL_32:.*]] = constant 0 : index
// CHECK:             %[[VAL_33:.*]] = constant 1 : index
// CHECK:             %[[VAL_34:.*]] = memref.dim %[[VAL_0]], %[[VAL_33]] : memref<?x?x?xf32>
// CHECK:             %[[VAL_35:.*]] = constant 2 : index
// CHECK:             %[[VAL_36:.*]] = memref.dim %[[VAL_0]], %[[VAL_35]] : memref<?x?x?xf32>
// CHECK:             %[[VAL_37:.*]] = memref.alloc(%[[VAL_34]], %[[VAL_36]]) : memref<?x?xf32>
// CHECK:             %[[VAL_38:.*]] = constant 0 : index
// CHECK:             %[[VAL_39:.*]] = memref.dim %[[VAL_37]], %[[VAL_38]] : memref<?x?xf32>
// CHECK:             %[[VAL_40:.*]] = constant 1 : index
// CHECK:             %[[VAL_41:.*]] = memref.dim %[[VAL_37]], %[[VAL_40]] : memref<?x?xf32>
// CHECK:             %[[VAL_42:.*]] = constant 0 : index
// CHECK:             %[[VAL_43:.*]] = constant 0 : index
// CHECK:             %[[VAL_44:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_44]]#0, %[[VAL_44]]#1) with (%[[VAL_44]]#0 -> %[[VAL_45:.*]] = %[[VAL_42]] to %[[VAL_39]], %[[VAL_44]]#1 -> %[[VAL_46:.*]] = %[[VAL_43]] to %[[VAL_41]]) {
// CHECK:               %[[VAL_47:.*]]:2 = krnl.get_induction_var_value(%[[VAL_44]]#0, %[[VAL_44]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_48:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_31]], %[[VAL_47]]#0, %[[VAL_47]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store %[[VAL_48]], %[[VAL_37]]{{\[}}%[[VAL_47]]#0, %[[VAL_47]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK:             %[[VAL_49:.*]] = "onnx.Constant"() {value = dense<1.000000e+00> : tensor<1xf32>} : () -> memref<1xf32>
// CHECK:             %[[VAL_50:.*]] = "onnx.MatMul"(%[[VAL_37]], %[[VAL_21]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_51:.*]] = "onnx.MatMul"(%[[VAL_9]], %[[VAL_25]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_52:.*]] = "onnx.Add"(%[[VAL_50]], %[[VAL_51]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_53:.*]] = "onnx.Sigmoid"(%[[VAL_52]]) : (memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_54:.*]] = "onnx.MatMul"(%[[VAL_37]], %[[VAL_22]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.MatMul"(%[[VAL_9]], %[[VAL_26]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_56:.*]] = "onnx.Add"(%[[VAL_54]], %[[VAL_55]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_57:.*]] = "onnx.Sigmoid"(%[[VAL_56]]) : (memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_58:.*]] = "onnx.MatMul"(%[[VAL_37]], %[[VAL_23]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_59:.*]] = "onnx.Mul"(%[[VAL_57]], %[[VAL_9]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_60:.*]] = "onnx.MatMul"(%[[VAL_59]], %[[VAL_27]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_61:.*]] = "onnx.Add"(%[[VAL_58]], %[[VAL_60]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_62:.*]] = "onnx.Tanh"(%[[VAL_61]]) : (memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_63:.*]] = "onnx.Sub"(%[[VAL_49]], %[[VAL_53]]) : (memref<1xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_64:.*]] = "onnx.Mul"(%[[VAL_63]], %[[VAL_62]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_65:.*]] = "onnx.Mul"(%[[VAL_53]], %[[VAL_9]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_66:.*]] = "onnx.Add"(%[[VAL_64]], %[[VAL_65]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_67:.*]] = constant 12 : i64
// CHECK:             %[[VAL_68:.*]] = constant 0 : index
// CHECK:             %[[VAL_69:.*]] = memref.dim %[[VAL_66]], %[[VAL_68]] : memref<?x3xf32>
// CHECK:             %[[VAL_70:.*]] = index_cast %[[VAL_69]] : index to i64
// CHECK:             %[[VAL_71:.*]] = muli %[[VAL_67]], %[[VAL_70]] : i64
// CHECK:             "krnl.memcpy"(%[[VAL_9]], %[[VAL_66]], %[[VAL_71]]) : (memref<?x3xf32>, memref<?x3xf32>, i64) -> ()
// CHECK:             memref.dealloc %[[VAL_37]] : memref<?x?xf32>
// CHECK:           }
// CHECK:           %[[VAL_72:.*]] = constant 12 : i64
// CHECK:           %[[VAL_73:.*]] = constant 0 : index
// CHECK:           %[[VAL_74:.*]] = memref.dim %[[VAL_9]], %[[VAL_73]] : memref<?x3xf32>
// CHECK:           %[[VAL_75:.*]] = index_cast %[[VAL_74]] : index to i64
// CHECK:           %[[VAL_76:.*]] = muli %[[VAL_72]], %[[VAL_75]] : i64
// CHECK:           "krnl.memcpy"(%[[VAL_6]], %[[VAL_9]], %[[VAL_76]]) : (memref<1x?x3xf32>, memref<?x3xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_9]] : memref<?x3xf32>
// CHECK:           return %[[VAL_6]] : memref<1x?x3xf32>
// CHECK:         }

}
