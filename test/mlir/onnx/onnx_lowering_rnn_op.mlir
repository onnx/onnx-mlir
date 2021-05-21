// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='test-rnn-ops-lowering' %s -split-input-file

// FIXME: enable this.
// | FileCheck %s

/// Check RNN with three required inputs (X, W, R). The optional inputs are default.
func private @test_rnn_general_computation(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x3x2xf32>, %arg2: tensor<1x3x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x3x2xf32>, tensor<1x3x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_rnn_general_computation(
// CHECK-SAME:                                               %[[VAL_0:.*]]: memref<4x3x2xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: memref<1x3x2xf32>,
// CHECK-SAME:                                               %[[VAL_2:.*]]: memref<1x3x3xf32>) -> memref<1x3x3xf32> {
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
// CHECK:           %[[VAL_12:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x3x2xf32>) -> memref<3x2xf32>
// CHECK:           %[[VAL_13:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Transpose"(%[[VAL_12]]) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_15:.*]] = "onnx.Transpose"(%[[VAL_13]]) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// RNN computation. Iterate over the sequence.
// CHECK:           %[[VAL_16:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_16]]) with (%[[VAL_16]] -> %[[VAL_17:.*]] = 0 to 4) {
// Get a slice of X for the current timestep.
// CHECK:             %[[VAL_18:.*]] = memref.alloc() : memref<3x2xf32>
// CHECK:             %[[VAL_19:.*]] = constant 0 : index
// CHECK:             %[[VAL_20:.*]] = constant 3 : index
// CHECK:             %[[VAL_21:.*]] = constant 2 : index
// CHECK:             %[[VAL_22:.*]] = constant 0 : index
// CHECK:             %[[VAL_23:.*]] = constant 0 : index
// CHECK:             %[[VAL_24:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_24]]#0, %[[VAL_24]]#1) with (%[[VAL_24]]#0 -> %[[VAL_25:.*]] = %[[VAL_22]] to %[[VAL_20]], %[[VAL_24]]#1 -> %[[VAL_26:.*]] = %[[VAL_23]] to %[[VAL_21]]) {
// CHECK:               %[[VAL_27:.*]]:2 = krnl.get_induction_var_value(%[[VAL_24]]#0, %[[VAL_24]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_28:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_17]], %[[VAL_27]]#0, %[[VAL_27]]#1] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_28]], %[[VAL_18]]{{\[}}%[[VAL_27]]#0, %[[VAL_27]]#1] : memref<3x2xf32>
// CHECK:             }
  // Ht = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Wbi + Rbi)
// CHECK:             %[[VAL_29:.*]] = "onnx.MatMul"(%[[VAL_18]], %[[VAL_14]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_30:.*]] = "onnx.MatMul"(%[[VAL_3]], %[[VAL_15]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_31:.*]] = "onnx.Add"(%[[VAL_29]], %[[VAL_30]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_32:.*]] = "onnx.Tanh"(%[[VAL_31]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// Store the current Ht.
// CHECK:             %[[VAL_33:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_3]], %[[VAL_32]], %[[VAL_33]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             memref.dealloc %[[VAL_18]] : memref<3x2xf32>
// CHECK:           }
// Store the intermediate states to the returned states.
// CHECK:           %[[VAL_34:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_4]], %[[VAL_3]], %[[VAL_34]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_3]] : memref<3x3xf32>
// CHECK:           return %[[VAL_4]] : memref<1x3x3xf32>
// CHECK:         }

}

// -----

/// Check RNN with three required inputs (X, W, R), and bias input.
func private @test_rnn_with_bias(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x3x2xf32>, %arg2: tensor<1x3x3xf32>, %arg3: tensor<1x6xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x3x2xf32>, tensor<1x3x3xf32>, tensor<1x6xf32>, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_rnn_with_bias(
// CHECK-SAME:                                     %[[VAL_0:.*]]: memref<4x3x2xf32>,
// CHECK-SAME:                                     %[[VAL_1:.*]]: memref<1x3x2xf32>,
// CHECK-SAME:                                     %[[VAL_2:.*]]: memref<1x3x3xf32>,
// CHECK-SAME:                                     %[[VAL_3:.*]]: memref<1x6xf32>) -> memref<1x3x3xf32> {
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
// CHECK:           %[[VAL_13:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x3x2xf32>) -> memref<3x2xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_15:.*]] = "onnx.Transpose"(%[[VAL_13]]) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_16:.*]] = "onnx.Transpose"(%[[VAL_14]]) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Squeeze"(%[[VAL_3]]) {axes = [0]} : (memref<1x6xf32>) -> memref<6xf32>
// CHECK:           %[[VAL_18:.*]]:2 = "onnx.Split"(%[[VAL_17]]) {axis = 0 : si64} : (memref<6xf32>) -> (memref<3xf32>, memref<3xf32>)
// CHECK:           %[[VAL_19:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_19]]) with (%[[VAL_19]] -> %[[VAL_20:.*]] = 0 to 4) {
// CHECK:             %[[VAL_21:.*]] = memref.alloc() : memref<3x2xf32>
// CHECK:             %[[VAL_22:.*]] = constant 0 : index
// CHECK:             %[[VAL_23:.*]] = constant 3 : index
// CHECK:             %[[VAL_24:.*]] = constant 2 : index
// CHECK:             %[[VAL_25:.*]] = constant 0 : index
// CHECK:             %[[VAL_26:.*]] = constant 0 : index
// CHECK:             %[[VAL_27:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_27]]#0, %[[VAL_27]]#1) with (%[[VAL_27]]#0 -> %[[VAL_28:.*]] = %[[VAL_25]] to %[[VAL_23]], %[[VAL_27]]#1 -> %[[VAL_29:.*]] = %[[VAL_26]] to %[[VAL_24]]) {
// CHECK:               %[[VAL_30:.*]]:2 = krnl.get_induction_var_value(%[[VAL_27]]#0, %[[VAL_27]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_31:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_20]], %[[VAL_30]]#0, %[[VAL_30]]#1] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_31]], %[[VAL_21]]{{\[}}%[[VAL_30]]#0, %[[VAL_30]]#1] : memref<3x2xf32>
// CHECK:             }
// CHECK:             %[[VAL_32:.*]] = "onnx.MatMul"(%[[VAL_21]], %[[VAL_15]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_33:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_16]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_34:.*]] = "onnx.Add"(%[[VAL_32]], %[[VAL_33]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_35:.*]] = "onnx.Add"(%[[VAL_34]], %[[VAL_18]]#0) : (memref<3x3xf32>, memref<3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_36:.*]] = "onnx.Add"(%[[VAL_35]], %[[VAL_18]]#1) : (memref<3x3xf32>, memref<3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_37:.*]] = "onnx.Tanh"(%[[VAL_36]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_38:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_4]], %[[VAL_37]], %[[VAL_38]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             memref.dealloc %[[VAL_21]] : memref<3x2xf32>
// CHECK:           }
// CHECK:           %[[VAL_39:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_5]], %[[VAL_4]], %[[VAL_39]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_4]] : memref<3x3xf32>
// CHECK:           return %[[VAL_5]] : memref<1x3x3xf32>
// CHECK:         }

}

// -----

// Check handling unknown dimensions for RNN by checking the
// correctness of allocating and deallocating memory.
func private @test_rnn_unknown_dims_allocation(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x3x?xf32>, %arg2: tensor<1x3x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<?x?x?xf32>, tensor<1x3x?xf32>, tensor<1x3x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_rnn_unknown_dims_allocation(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: memref<?x?x?xf32>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: memref<1x3x?xf32>,
// CHECK-SAME:                                                   %[[VAL_2:.*]]: memref<1x3x3xf32>) -> memref<1x?x3xf32> {
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
// CHECK:           %[[VAL_18:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x3x?xf32>) -> memref<3x?xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Transpose"(%[[VAL_18]]) {perm = [1, 0]} : (memref<3x?xf32>) -> memref<?x3xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Transpose"(%[[VAL_19]]) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_22:.*]] = krnl.define_loops 1
// CHECK:           %[[VAL_23:.*]] = constant 0 : index
// CHECK:           %[[VAL_24:.*]] = memref.dim %[[VAL_0]], %[[VAL_23]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate(%[[VAL_22]]) with (%[[VAL_22]] -> %[[VAL_25:.*]] = 0 to %[[VAL_24]]) {
// CHECK:             %[[VAL_26:.*]] = constant 0 : index
// CHECK:             %[[VAL_27:.*]] = constant 1 : index
// CHECK:             %[[VAL_28:.*]] = memref.dim %[[VAL_0]], %[[VAL_27]] : memref<?x?x?xf32>
// CHECK:             %[[VAL_29:.*]] = constant 2 : index
// CHECK:             %[[VAL_30:.*]] = memref.dim %[[VAL_0]], %[[VAL_29]] : memref<?x?x?xf32>
// CHECK:             %[[VAL_31:.*]] = memref.alloc(%[[VAL_28]], %[[VAL_30]]) : memref<?x?xf32>
// CHECK:             %[[VAL_32:.*]] = constant 0 : index
// CHECK:             %[[VAL_33:.*]] = memref.dim %[[VAL_31]], %[[VAL_32]] : memref<?x?xf32>
// CHECK:             %[[VAL_34:.*]] = constant 1 : index
// CHECK:             %[[VAL_35:.*]] = memref.dim %[[VAL_31]], %[[VAL_34]] : memref<?x?xf32>
// CHECK:             %[[VAL_36:.*]] = constant 0 : index
// CHECK:             %[[VAL_37:.*]] = constant 0 : index
// CHECK:             %[[VAL_38:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_38]]#0, %[[VAL_38]]#1) with (%[[VAL_38]]#0 -> %[[VAL_39:.*]] = %[[VAL_36]] to %[[VAL_33]], %[[VAL_38]]#1 -> %[[VAL_40:.*]] = %[[VAL_37]] to %[[VAL_35]]) {
// CHECK:               %[[VAL_41:.*]]:2 = krnl.get_induction_var_value(%[[VAL_38]]#0, %[[VAL_38]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_42:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_25]], %[[VAL_41]]#0, %[[VAL_41]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store %[[VAL_42]], %[[VAL_31]]{{\[}}%[[VAL_41]]#0, %[[VAL_41]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK:             %[[VAL_43:.*]] = "onnx.MatMul"(%[[VAL_31]], %[[VAL_20]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_44:.*]] = "onnx.MatMul"(%[[VAL_9]], %[[VAL_21]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_45:.*]] = "onnx.Add"(%[[VAL_43]], %[[VAL_44]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_46:.*]] = "onnx.Tanh"(%[[VAL_45]]) : (memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_47:.*]] = constant 12 : i64
// CHECK:             %[[VAL_48:.*]] = constant 0 : index
// CHECK:             %[[VAL_49:.*]] = memref.dim %[[VAL_46]], %[[VAL_48]] : memref<?x3xf32>
// CHECK:             %[[VAL_50:.*]] = index_cast %[[VAL_49]] : index to i64
// CHECK:             %[[VAL_51:.*]] = muli %[[VAL_47]], %[[VAL_50]] : i64
// CHECK:             "krnl.memcpy"(%[[VAL_9]], %[[VAL_46]], %[[VAL_51]]) : (memref<?x3xf32>, memref<?x3xf32>, i64) -> ()
// CHECK:             memref.dealloc %[[VAL_31]] : memref<?x?xf32>
// CHECK:           }
// CHECK:           %[[VAL_52:.*]] = constant 12 : i64
// CHECK:           %[[VAL_53:.*]] = constant 0 : index
// CHECK:           %[[VAL_54:.*]] = memref.dim %[[VAL_9]], %[[VAL_53]] : memref<?x3xf32>
// CHECK:           %[[VAL_55:.*]] = index_cast %[[VAL_54]] : index to i64
// CHECK:           %[[VAL_56:.*]] = muli %[[VAL_52]], %[[VAL_55]] : i64
// CHECK:           "krnl.memcpy"(%[[VAL_6]], %[[VAL_9]], %[[VAL_56]]) : (memref<1x?x3xf32>, memref<?x3xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_9]] : memref<?x3xf32>
// CHECK:           return %[[VAL_6]] : memref<1x?x3xf32>
// CHECK:         }

}
