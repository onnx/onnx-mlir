// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='test-rnn-ops-lowering' %s -split-input-file | FileCheck %s

/// Check GRU with three required inputs (X, W, R). The optional inputs are default.
/// Also check the equation for 'ht' when linear_before_reset = 0 (default)
func private @test_gru_general_computation(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_general_computation
}

// -----

/// GRU with three required inputs (X, W, R). The optional inputs are default.
/// Check the equation for 'ht' when linear_before_reset !=0.
func private @test_gru_linear_before_reset(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64, linear_before_reset = 1 : si64} : (tensor<4x3x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

}

// -----

/// Check GRU with three required inputs (X, W, R), and bias input.
func private @test_gru_with_bias(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>, %arg3: tensor<1x18xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, tensor<1x18xf32>, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>
}

// -----

// Check handling unknown dimensions for GRU by checking the
// correctness of allocating and deallocating memory.
func private @test_gru_unknown_dims_allocation(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x9x?xf32>, %arg2: tensor<1x9x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<?x?x?xf32>, tensor<1x9x?xf32>, tensor<1x9x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: @test_gru_unknown_dims_allocation
}

// -----

func private @test_lstm_general_computation(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_lstm_general_computation(
// CHECK-SAME:                                                %[[VAL_0:.*]]: memref<4x3x2xf32>,
// CHECK-SAME:                                                %[[VAL_1:.*]]: memref<1x12x2xf32>,
// CHECK-SAME:                                                %[[VAL_2:.*]]: memref<1x12x3xf32>) -> memref<1x3x3xf32> {
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<1x3x3xf32>
// CHECK:           %[[VAL_6:.*]] = constant unit
// CHECK:           %[[VAL_7:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_8:.*]] = constant 0 : index
// CHECK:           %[[VAL_9:.*]] = constant 1 : index
// CHECK:           %[[VAL_10:.*]]:2 = krnl.define_loops 2
// Initialize the intermediate states Ht and Ct.
// CHECK:           krnl.iterate(%[[VAL_10]]#0, %[[VAL_10]]#1) with (%[[VAL_10]]#0 -> %[[VAL_11:.*]] = 0 to 3, %[[VAL_10]]#1 -> %[[VAL_12:.*]] = 0 to 3) {
// CHECK:             krnl.store %[[VAL_7]], %[[VAL_4]]{{\[}}%[[VAL_11]], %[[VAL_12]]] : memref<3x3xf32>
// CHECK:             krnl.store %[[VAL_7]], %[[VAL_3]]{{\[}}%[[VAL_11]], %[[VAL_12]]] : memref<3x3xf32>
// CHECK:           }
// Prepare weights and biases.
// CHECK:           %[[VAL_13:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x12x2xf32>) -> memref<12x2xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK:           %[[VAL_15:.*]]:4 = "onnx.Split"(%[[VAL_13]]) {axis = 0 : si64} : (memref<12x2xf32>) -> (memref<3x2xf32>, memref<3x2xf32>, memref<3x2xf32>, memref<3x2xf32>)
// CHECK:           %[[VAL_16:.*]] = "onnx.Transpose"(%[[VAL_15]]#0) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_15]]#1) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_15]]#2) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Transpose"(%[[VAL_15]]#3) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_20:.*]]:4 = "onnx.Split"(%[[VAL_14]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>)
// CHECK:           %[[VAL_21:.*]] = "onnx.Transpose"(%[[VAL_20]]#0) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Transpose"(%[[VAL_20]]#1) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Transpose"(%[[VAL_20]]#2) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Transpose"(%[[VAL_20]]#3) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// LSTM computation.
// CHECK:           %[[VAL_25:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_25]]) with (%[[VAL_25]] -> %[[VAL_26:.*]] = 0 to 4) {
// CHECK:             %[[VAL_27:.*]] = memref.alloc() : memref<3x2xf32>
// CHECK:             %[[VAL_28:.*]] = constant 0 : index
// CHECK:             %[[VAL_29:.*]] = constant 3 : index
// CHECK:             %[[VAL_30:.*]] = constant 2 : index
// Get a slice of X for the current timestep.
// CHECK:             %[[VAL_31:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_31]]#0, %[[VAL_31]]#1) with (%[[VAL_31]]#0 -> %[[VAL_32:.*]] = 0 to %[[VAL_29]], %[[VAL_31]]#1 -> %[[VAL_33:.*]] = 0 to %[[VAL_30]]) {
// CHECK:               %[[VAL_34:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_26]], %[[VAL_32]], %[[VAL_33]]] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_34]], %[[VAL_27]]{{\[}}%[[VAL_32]], %[[VAL_33]]] : memref<3x2xf32>
// CHECK:             }
// it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
// CHECK:             %[[VAL_35:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_16]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_36:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_21]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_37:.*]] = "onnx.Add"(%[[VAL_35]], %[[VAL_36]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_38:.*]] = "onnx.Sigmoid"(%[[VAL_37]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
// CHECK:             %[[VAL_39:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_18]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_40:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_23]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_41:.*]] = "onnx.Add"(%[[VAL_39]], %[[VAL_40]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_42:.*]] = "onnx.Sigmoid"(%[[VAL_41]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
// CHECK:             %[[VAL_43:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_19]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_44:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_24]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_45:.*]] = "onnx.Add"(%[[VAL_43]], %[[VAL_44]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_46:.*]] = "onnx.Tanh"(%[[VAL_45]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// Ct = ft (.) Ct-1 + it (.) ct
// CHECK:             %[[VAL_47:.*]] = "onnx.Mul"(%[[VAL_42]], %[[VAL_3]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_48:.*]] = "onnx.Mul"(%[[VAL_38]], %[[VAL_46]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_49:.*]] = "onnx.Add"(%[[VAL_47]], %[[VAL_48]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
// CHECK:             %[[VAL_50:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_17]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_51:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_22]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_52:.*]] = "onnx.Add"(%[[VAL_50]], %[[VAL_51]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_53:.*]] = "onnx.Sigmoid"(%[[VAL_52]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// Ht = ot (.) h(Ct)
// CHECK:             %[[VAL_54:.*]] = "onnx.Tanh"(%[[VAL_49]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.Mul"(%[[VAL_53]], %[[VAL_54]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// Store the current Ht.
// CHECK:             %[[VAL_56:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_4]], %[[VAL_55]], %[[VAL_56]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// Store the current Ct.
// CHECK:             %[[VAL_57:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_3]], %[[VAL_49]], %[[VAL_57]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:           }
// Store the intermediate states to the returned states.
// CHECK:           %[[VAL_58:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_5]], %[[VAL_4]], %[[VAL_58]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_4]] : memref<3x3xf32>
// CHECK:           memref.dealloc %[[VAL_3]] : memref<3x3xf32>
// CHECK:           return %[[VAL_5]] : memref<1x3x3xf32>
// CHECK:         }
}

// -----

func private @test_lstm_reverse_mode(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64, direction = "reverse"} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_lstm_reverse_mode(
// CHECK-SAME:                                         %[[VAL_0:.*]]: memref<4x3x2xf32>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: memref<1x12x2xf32>,
// CHECK-SAME:                                         %[[VAL_2:.*]]: memref<1x12x3xf32>) -> memref<1x3x3xf32> {
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<1x3x3xf32>
// CHECK:           %[[VAL_6:.*]] = constant unit
// CHECK:           %[[VAL_7:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_8:.*]] = constant 0 : index
// CHECK:           %[[VAL_9:.*]] = constant 1 : index
// CHECK:           %[[VAL_10:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_10]]#0, %[[VAL_10]]#1) with (%[[VAL_10]]#0 -> %[[VAL_11:.*]] = 0 to 3, %[[VAL_10]]#1 -> %[[VAL_12:.*]] = 0 to 3) {
// CHECK:             krnl.store %[[VAL_7]], %[[VAL_4]]{{\[}}%[[VAL_11]], %[[VAL_12]]] : memref<3x3xf32>
// CHECK:             krnl.store %[[VAL_7]], %[[VAL_3]]{{\[}}%[[VAL_11]], %[[VAL_12]]] : memref<3x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_13:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x12x2xf32>) -> memref<12x2xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK:           %[[VAL_15:.*]]:4 = "onnx.Split"(%[[VAL_13]]) {axis = 0 : si64} : (memref<12x2xf32>) -> (memref<3x2xf32>, memref<3x2xf32>, memref<3x2xf32>, memref<3x2xf32>)
// CHECK:           %[[VAL_16:.*]] = "onnx.Transpose"(%[[VAL_15]]#0) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Transpose"(%[[VAL_15]]#1) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_15]]#2) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Transpose"(%[[VAL_15]]#3) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_20:.*]]:4 = "onnx.Split"(%[[VAL_14]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>)
// CHECK:           %[[VAL_21:.*]] = "onnx.Transpose"(%[[VAL_20]]#0) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Transpose"(%[[VAL_20]]#1) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Transpose"(%[[VAL_20]]#2) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Transpose"(%[[VAL_20]]#3) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_25:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_25]]) with (%[[VAL_25]] -> %[[VAL_26:.*]] = 0 to 4) {
// CHECK:             %[[VAL_27:.*]] = memref.alloc() : memref<3x2xf32>
// CHECK:             %[[VAL_28:.*]] = constant 0 : index
// CHECK:             %[[VAL_29:.*]] = constant 4 : index
// CHECK:             %[[VAL_30:.*]] = affine.apply #map(%[[VAL_26]]){{\[}}%[[VAL_29]]]
// CHECK:             %[[VAL_31:.*]] = constant 3 : index
// CHECK:             %[[VAL_32:.*]] = constant 2 : index
// CHECK:             %[[VAL_33:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_33]]#0, %[[VAL_33]]#1) with (%[[VAL_33]]#0 -> %[[VAL_34:.*]] = 0 to %[[VAL_31]], %[[VAL_33]]#1 -> %[[VAL_35:.*]] = 0 to %[[VAL_32]]) {
// CHECK:               %[[VAL_36:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_30]], %[[VAL_34]], %[[VAL_35]]] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_36]], %[[VAL_27]]{{\[}}%[[VAL_34]], %[[VAL_35]]] : memref<3x2xf32>
// CHECK:             }
// CHECK:             %[[VAL_37:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_16]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_38:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_21]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_39:.*]] = "onnx.Add"(%[[VAL_37]], %[[VAL_38]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_40:.*]] = "onnx.Sigmoid"(%[[VAL_39]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_41:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_18]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_42:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_23]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_43:.*]] = "onnx.Add"(%[[VAL_41]], %[[VAL_42]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_44:.*]] = "onnx.Sigmoid"(%[[VAL_43]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_45:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_19]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_46:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_24]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_47:.*]] = "onnx.Add"(%[[VAL_45]], %[[VAL_46]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_48:.*]] = "onnx.Tanh"(%[[VAL_47]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_49:.*]] = "onnx.Mul"(%[[VAL_44]], %[[VAL_3]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_50:.*]] = "onnx.Mul"(%[[VAL_40]], %[[VAL_48]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_51:.*]] = "onnx.Add"(%[[VAL_49]], %[[VAL_50]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_52:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_17]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_53:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_22]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_54:.*]] = "onnx.Add"(%[[VAL_52]], %[[VAL_53]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.Sigmoid"(%[[VAL_54]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_56:.*]] = "onnx.Tanh"(%[[VAL_51]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_57:.*]] = "onnx.Mul"(%[[VAL_55]], %[[VAL_56]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_58:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_4]], %[[VAL_57]], %[[VAL_58]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             %[[VAL_59:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_3]], %[[VAL_51]], %[[VAL_59]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:           }
// CHECK:           %[[VAL_60:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_5]], %[[VAL_4]], %[[VAL_60]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_4]] : memref<3x3xf32>
// CHECK:           memref.dealloc %[[VAL_3]] : memref<3x3xf32>
// CHECK:           return %[[VAL_5]] : memref<1x3x3xf32>
// CHECK:         }
}

// -----

func private @test_lstm_bidirectional_mode(%arg0: tensor<4x3x2xf32>, %arg1: tensor<2x12x2xf32>, %arg2: tensor<2x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64, direction = "bidirectional"} : (tensor<4x3x2xf32>, tensor<2x12x2xf32>, tensor<2x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_lstm_bidirectional_mode(
// CHECK-SAME:                                               %[[VAL_0:.*]]: memref<4x3x2xf32>,
// CHECK-SAME:                                               %[[VAL_1:.*]]: memref<2x12x2xf32>,
// CHECK-SAME:                                               %[[VAL_2:.*]]: memref<2x12x3xf32>) -> memref<2x3x3xf32> {
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:           %[[VAL_6:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<2x3x3xf32>
// CHECK:           %[[VAL_8:.*]] = constant unit
// CHECK:           %[[VAL_9:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_10:.*]] = constant 0 : index
// CHECK:           %[[VAL_11:.*]] = constant 1 : index
// CHECK:           %[[VAL_12:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_12]]#0, %[[VAL_12]]#1) with (%[[VAL_12]]#0 -> %[[VAL_13:.*]] = 0 to 3, %[[VAL_12]]#1 -> %[[VAL_14:.*]] = 0 to 3) {
// CHECK:             krnl.store %[[VAL_9]], %[[VAL_6]]{{\[}}%[[VAL_13]], %[[VAL_14]]] : memref<3x3xf32>
// CHECK:             krnl.store %[[VAL_9]], %[[VAL_5]]{{\[}}%[[VAL_13]], %[[VAL_14]]] : memref<3x3xf32>
// CHECK:             krnl.store %[[VAL_9]], %[[VAL_4]]{{\[}}%[[VAL_13]], %[[VAL_14]]] : memref<3x3xf32>
// CHECK:             krnl.store %[[VAL_9]], %[[VAL_3]]{{\[}}%[[VAL_13]], %[[VAL_14]]] : memref<3x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_15:.*]]:2 = "onnx.Split"(%[[VAL_1]]) {axis = 0 : si64} : (memref<2x12x2xf32>) -> (memref<1x12x2xf32>, memref<1x12x2xf32>)
// CHECK:           %[[VAL_16:.*]] = "onnx.Squeeze"(%[[VAL_15]]#0) {axes = [0]} : (memref<1x12x2xf32>) -> memref<12x2xf32>
// CHECK:           %[[VAL_17:.*]] = "onnx.Squeeze"(%[[VAL_15]]#1) {axes = [0]} : (memref<1x12x2xf32>) -> memref<12x2xf32>
// CHECK:           %[[VAL_18:.*]]:2 = "onnx.Split"(%[[VAL_2]]) {axis = 0 : si64} : (memref<2x12x3xf32>) -> (memref<1x12x3xf32>, memref<1x12x3xf32>)
// CHECK:           %[[VAL_19:.*]] = "onnx.Squeeze"(%[[VAL_18]]#0) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Squeeze"(%[[VAL_18]]#1) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK:           %[[VAL_21:.*]]:4 = "onnx.Split"(%[[VAL_16]]) {axis = 0 : si64} : (memref<12x2xf32>) -> (memref<3x2xf32>, memref<3x2xf32>, memref<3x2xf32>, memref<3x2xf32>)
// CHECK:           %[[VAL_22:.*]] = "onnx.Transpose"(%[[VAL_21]]#0) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Transpose"(%[[VAL_21]]#1) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Transpose"(%[[VAL_21]]#2) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Transpose"(%[[VAL_21]]#3) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_26:.*]]:4 = "onnx.Split"(%[[VAL_19]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>)
// CHECK:           %[[VAL_27:.*]] = "onnx.Transpose"(%[[VAL_26]]#0) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.Transpose"(%[[VAL_26]]#1) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Transpose"(%[[VAL_26]]#2) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.Transpose"(%[[VAL_26]]#3) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_31:.*]]:4 = "onnx.Split"(%[[VAL_17]]) {axis = 0 : si64} : (memref<12x2xf32>) -> (memref<3x2xf32>, memref<3x2xf32>, memref<3x2xf32>, memref<3x2xf32>)
// CHECK:           %[[VAL_32:.*]] = "onnx.Transpose"(%[[VAL_31]]#0) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_33:.*]] = "onnx.Transpose"(%[[VAL_31]]#1) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Transpose"(%[[VAL_31]]#2) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Transpose"(%[[VAL_31]]#3) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_36:.*]]:4 = "onnx.Split"(%[[VAL_20]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>)
// CHECK:           %[[VAL_37:.*]] = "onnx.Transpose"(%[[VAL_36]]#0) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Transpose"(%[[VAL_36]]#1) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_39:.*]] = "onnx.Transpose"(%[[VAL_36]]#2) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_40:.*]] = "onnx.Transpose"(%[[VAL_36]]#3) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_41:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_41]]) with (%[[VAL_41]] -> %[[VAL_42:.*]] = 0 to 4) {
// CHECK:             %[[VAL_43:.*]] = memref.alloc() : memref<3x2xf32>
// CHECK:             %[[VAL_44:.*]] = constant 0 : index
// CHECK:             %[[VAL_45:.*]] = constant 3 : index
// CHECK:             %[[VAL_46:.*]] = constant 2 : index
// CHECK:             %[[VAL_47:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_47]]#0, %[[VAL_47]]#1) with (%[[VAL_47]]#0 -> %[[VAL_48:.*]] = 0 to %[[VAL_45]], %[[VAL_47]]#1 -> %[[VAL_49:.*]] = 0 to %[[VAL_46]]) {
// CHECK:               %[[VAL_50:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_42]], %[[VAL_48]], %[[VAL_49]]] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_50]], %[[VAL_43]]{{\[}}%[[VAL_48]], %[[VAL_49]]] : memref<3x2xf32>
// CHECK:             }
// CHECK:             %[[VAL_51:.*]] = "onnx.MatMul"(%[[VAL_43]], %[[VAL_22]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_52:.*]] = "onnx.MatMul"(%[[VAL_6]], %[[VAL_27]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_53:.*]] = "onnx.Add"(%[[VAL_51]], %[[VAL_52]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_54:.*]] = "onnx.Sigmoid"(%[[VAL_53]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.MatMul"(%[[VAL_43]], %[[VAL_24]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_56:.*]] = "onnx.MatMul"(%[[VAL_6]], %[[VAL_29]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_57:.*]] = "onnx.Add"(%[[VAL_55]], %[[VAL_56]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_58:.*]] = "onnx.Sigmoid"(%[[VAL_57]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_59:.*]] = "onnx.MatMul"(%[[VAL_43]], %[[VAL_25]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_60:.*]] = "onnx.MatMul"(%[[VAL_6]], %[[VAL_30]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_61:.*]] = "onnx.Add"(%[[VAL_59]], %[[VAL_60]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_62:.*]] = "onnx.Tanh"(%[[VAL_61]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_63:.*]] = "onnx.Mul"(%[[VAL_58]], %[[VAL_5]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_64:.*]] = "onnx.Mul"(%[[VAL_54]], %[[VAL_62]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_65:.*]] = "onnx.Add"(%[[VAL_63]], %[[VAL_64]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_66:.*]] = "onnx.MatMul"(%[[VAL_43]], %[[VAL_23]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_67:.*]] = "onnx.MatMul"(%[[VAL_6]], %[[VAL_28]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_68:.*]] = "onnx.Add"(%[[VAL_66]], %[[VAL_67]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_69:.*]] = "onnx.Sigmoid"(%[[VAL_68]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_70:.*]] = "onnx.Tanh"(%[[VAL_65]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_71:.*]] = "onnx.Mul"(%[[VAL_69]], %[[VAL_70]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_72:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_6]], %[[VAL_71]], %[[VAL_72]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             %[[VAL_73:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_5]], %[[VAL_65]], %[[VAL_73]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:           }
// CHECK:           %[[VAL_74:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_74]]) with (%[[VAL_74]] -> %[[VAL_75:.*]] = 0 to 4) {
// CHECK:             %[[VAL_76:.*]] = memref.alloc() : memref<3x2xf32>
// CHECK:             %[[VAL_77:.*]] = constant 1 : index
// CHECK:             %[[VAL_78:.*]] = constant 4 : index
// CHECK:             %[[VAL_79:.*]] = affine.apply #map(%[[VAL_75]]){{\[}}%[[VAL_78]]]
// CHECK:             %[[VAL_80:.*]] = constant 3 : index
// CHECK:             %[[VAL_81:.*]] = constant 2 : index
// CHECK:             %[[VAL_82:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_82]]#0, %[[VAL_82]]#1) with (%[[VAL_82]]#0 -> %[[VAL_83:.*]] = 0 to %[[VAL_80]], %[[VAL_82]]#1 -> %[[VAL_84:.*]] = 0 to %[[VAL_81]]) {
// CHECK:               %[[VAL_85:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_79]], %[[VAL_83]], %[[VAL_84]]] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_85]], %[[VAL_76]]{{\[}}%[[VAL_83]], %[[VAL_84]]] : memref<3x2xf32>
// CHECK:             }
// CHECK:             %[[VAL_86:.*]] = "onnx.MatMul"(%[[VAL_76]], %[[VAL_32]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_87:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_37]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_88:.*]] = "onnx.Add"(%[[VAL_86]], %[[VAL_87]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_89:.*]] = "onnx.Sigmoid"(%[[VAL_88]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_90:.*]] = "onnx.MatMul"(%[[VAL_76]], %[[VAL_34]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_91:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_39]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_92:.*]] = "onnx.Add"(%[[VAL_90]], %[[VAL_91]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_93:.*]] = "onnx.Sigmoid"(%[[VAL_92]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_94:.*]] = "onnx.MatMul"(%[[VAL_76]], %[[VAL_35]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_95:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_40]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_96:.*]] = "onnx.Add"(%[[VAL_94]], %[[VAL_95]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_97:.*]] = "onnx.Tanh"(%[[VAL_96]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_98:.*]] = "onnx.Mul"(%[[VAL_93]], %[[VAL_3]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_99:.*]] = "onnx.Mul"(%[[VAL_89]], %[[VAL_97]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_100:.*]] = "onnx.Add"(%[[VAL_98]], %[[VAL_99]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_101:.*]] = "onnx.MatMul"(%[[VAL_76]], %[[VAL_33]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_102:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_38]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_103:.*]] = "onnx.Add"(%[[VAL_101]], %[[VAL_102]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_104:.*]] = "onnx.Sigmoid"(%[[VAL_103]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_105:.*]] = "onnx.Tanh"(%[[VAL_100]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_106:.*]] = "onnx.Mul"(%[[VAL_104]], %[[VAL_105]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_107:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_4]], %[[VAL_106]], %[[VAL_107]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             %[[VAL_108:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_3]], %[[VAL_100]], %[[VAL_108]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:           }
// CHECK:           %[[VAL_109:.*]] = constant 3 : index
// CHECK:           %[[VAL_110:.*]] = constant 3 : index
// CHECK:           %[[VAL_111:.*]] = constant 0 : index
// CHECK:           %[[VAL_112:.*]] = constant 0 : index
// CHECK:           %[[VAL_113:.*]] = constant 0 : index
// CHECK:           %[[VAL_114:.*]] = constant 1 : index
// CHECK:           %[[VAL_115:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_115]]#0, %[[VAL_115]]#1) with (%[[VAL_115]]#0 -> %[[VAL_116:.*]] = %[[VAL_111]] to %[[VAL_109]], %[[VAL_115]]#1 -> %[[VAL_117:.*]] = %[[VAL_112]] to %[[VAL_110]]) {
// CHECK:             %[[VAL_118:.*]]:2 = krnl.get_induction_var_value(%[[VAL_115]]#0, %[[VAL_115]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             %[[VAL_119:.*]] = krnl.load %[[VAL_6]]{{\[}}%[[VAL_118]]#0, %[[VAL_118]]#1] : memref<3x3xf32>
// CHECK:             krnl.store %[[VAL_119]], %[[VAL_7]]{{\[}}%[[VAL_113]], %[[VAL_118]]#0, %[[VAL_118]]#1] : memref<2x3x3xf32>
// CHECK:             %[[VAL_120:.*]] = krnl.load %[[VAL_4]]{{\[}}%[[VAL_118]]#0, %[[VAL_118]]#1] : memref<3x3xf32>
// CHECK:             krnl.store %[[VAL_120]], %[[VAL_7]]{{\[}}%[[VAL_114]], %[[VAL_118]]#0, %[[VAL_118]]#1] : memref<2x3x3xf32>
// CHECK:           }
// CHECK:           memref.dealloc %[[VAL_6]] : memref<3x3xf32>
// CHECK:           memref.dealloc %[[VAL_5]] : memref<3x3xf32>
// CHECK:           memref.dealloc %[[VAL_4]] : memref<3x3xf32>
// CHECK:           memref.dealloc %[[VAL_3]] : memref<3x3xf32>
// CHECK:           return %[[VAL_7]] : memref<2x3x3xf32>
// CHECK:         }

}

// -----

// Check handling unknown dimensions for LSTM by checking the
// correctness of allocating and deallocating memory.
func private @test_lstm_unknown_dims_allocation(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x12x?xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<?x?x?xf32>, tensor<1x12x?xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_lstm_unknown_dims_allocation(
// CHECK-SAME:                                                   %[[VAL_0:.*]]: memref<?x?x?xf32>,
// CHECK-SAME:                                                   %[[VAL_1:.*]]: memref<1x12x?xf32>,
// CHECK-SAME:                                                   %[[VAL_2:.*]]: memref<1x12x3xf32>) -> memref<1x?x3xf32> {
// CHECK:           %[[VAL_3:.*]] = constant unit
// CHECK:           %[[VAL_4:.*]] = constant 0 : index
// CHECK:           %[[VAL_5:.*]] = memref.dim %[[VAL_0]], %[[VAL_4]] : memref<?x?x?xf32>
// CHECK:           %[[VAL_6:.*]] = constant 1 : index
// CHECK:           %[[VAL_7:.*]] = memref.dim %[[VAL_0]], %[[VAL_6]] : memref<?x?x?xf32>
// CHECK:           %[[VAL_8:.*]] = memref.alloc(%[[VAL_5]], %[[VAL_7]]) : memref<?x1x?x3xf32>
// CHECK:           %[[VAL_9:.*]] = constant 1 : index
// CHECK:           %[[VAL_10:.*]] = memref.dim %[[VAL_0]], %[[VAL_9]] : memref<?x?x?xf32>
// CHECK:           %[[VAL_11:.*]] = memref.alloc(%[[VAL_10]]) : memref<1x?x3xf32>
// CHECK:           %[[VAL_12:.*]] = constant 1 : index
// CHECK:           %[[VAL_13:.*]] = memref.dim %[[VAL_0]], %[[VAL_12]] : memref<?x?x?xf32>
// CHECK:           %[[VAL_14:.*]] = memref.alloc(%[[VAL_13]]) : memref<?x3xf32>
// CHECK:           %[[VAL_15:.*]] = constant 1 : index
// CHECK:           %[[VAL_16:.*]] = memref.dim %[[VAL_0]], %[[VAL_15]] : memref<?x?x?xf32>
// CHECK:           %[[VAL_17:.*]] = memref.alloc(%[[VAL_16]]) : memref<?x3xf32>
// CHECK:           %[[VAL_18:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_19:.*]] = constant 0 : index
// CHECK:           %[[VAL_20:.*]] = constant 1 : index
// CHECK:           %[[VAL_21:.*]]:2 = krnl.define_loops 2
// CHECK:           %[[VAL_22:.*]] = constant 0 : index
// CHECK:           %[[VAL_23:.*]] = memref.dim %[[VAL_14]], %[[VAL_22]] : memref<?x3xf32>
// CHECK:           krnl.iterate(%[[VAL_21]]#0, %[[VAL_21]]#1) with (%[[VAL_21]]#0 -> %[[VAL_24:.*]] = 0 to %[[VAL_23]], %[[VAL_21]]#1 -> %[[VAL_25:.*]] = 0 to 3) {
// CHECK:             krnl.store %[[VAL_18]], %[[VAL_14]]{{\[}}%[[VAL_24]], %[[VAL_25]]] : memref<?x3xf32>
// CHECK:             krnl.store %[[VAL_18]], %[[VAL_17]]{{\[}}%[[VAL_24]], %[[VAL_25]]] : memref<?x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_26:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x12x?xf32>) -> memref<12x?xf32>
// CHECK:           %[[VAL_27:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK:           %[[VAL_28:.*]]:4 = "onnx.Split"(%[[VAL_26]]) {axis = 0 : si64} : (memref<12x?xf32>) -> (memref<3x?xf32>, memref<3x?xf32>, memref<3x?xf32>, memref<3x?xf32>)
// CHECK:           %[[VAL_29:.*]] = "onnx.Transpose"(%[[VAL_28]]#0) {perm = [1, 0]} : (memref<3x?xf32>) -> memref<?x3xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.Transpose"(%[[VAL_28]]#1) {perm = [1, 0]} : (memref<3x?xf32>) -> memref<?x3xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Transpose"(%[[VAL_28]]#2) {perm = [1, 0]} : (memref<3x?xf32>) -> memref<?x3xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.Transpose"(%[[VAL_28]]#3) {perm = [1, 0]} : (memref<3x?xf32>) -> memref<?x3xf32>
// CHECK:           %[[VAL_33:.*]]:4 = "onnx.Split"(%[[VAL_27]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>)
// CHECK:           %[[VAL_34:.*]] = "onnx.Transpose"(%[[VAL_33]]#0) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_35:.*]] = "onnx.Transpose"(%[[VAL_33]]#1) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Transpose"(%[[VAL_33]]#2) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.Transpose"(%[[VAL_33]]#3) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_38:.*]] = krnl.define_loops 1
// CHECK:           %[[VAL_39:.*]] = constant 0 : index
// CHECK:           %[[VAL_40:.*]] = memref.dim %[[VAL_0]], %[[VAL_39]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate(%[[VAL_38]]) with (%[[VAL_38]] -> %[[VAL_41:.*]] = 0 to %[[VAL_40]]) {
// CHECK:             %[[VAL_42:.*]] = constant 0 : index
// CHECK:             %[[VAL_43:.*]] = constant 1 : index
// CHECK:             %[[VAL_44:.*]] = memref.dim %[[VAL_0]], %[[VAL_43]] : memref<?x?x?xf32>
// CHECK:             %[[VAL_45:.*]] = constant 2 : index
// CHECK:             %[[VAL_46:.*]] = memref.dim %[[VAL_0]], %[[VAL_45]] : memref<?x?x?xf32>
// CHECK:             %[[VAL_47:.*]] = memref.alloc(%[[VAL_44]], %[[VAL_46]]) : memref<?x?xf32>
// CHECK:             %[[VAL_48:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_48]]#0, %[[VAL_48]]#1) with (%[[VAL_48]]#0 -> %[[VAL_49:.*]] = 0 to %[[VAL_44]], %[[VAL_48]]#1 -> %[[VAL_50:.*]] = 0 to %[[VAL_46]]) {
// CHECK:               %[[VAL_51:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_41]], %[[VAL_49]], %[[VAL_50]]] : memref<?x?x?xf32>
// CHECK:               krnl.store %[[VAL_51]], %[[VAL_47]]{{\[}}%[[VAL_49]], %[[VAL_50]]] : memref<?x?xf32>
// CHECK:             }
// CHECK:             %[[VAL_52:.*]] = "onnx.MatMul"(%[[VAL_47]], %[[VAL_29]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_53:.*]] = "onnx.MatMul"(%[[VAL_14]], %[[VAL_34]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_54:.*]] = "onnx.Add"(%[[VAL_52]], %[[VAL_53]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.Sigmoid"(%[[VAL_54]]) : (memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_56:.*]] = "onnx.MatMul"(%[[VAL_47]], %[[VAL_31]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_57:.*]] = "onnx.MatMul"(%[[VAL_14]], %[[VAL_36]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_58:.*]] = "onnx.Add"(%[[VAL_56]], %[[VAL_57]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_59:.*]] = "onnx.Sigmoid"(%[[VAL_58]]) : (memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_60:.*]] = "onnx.MatMul"(%[[VAL_47]], %[[VAL_32]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_61:.*]] = "onnx.MatMul"(%[[VAL_14]], %[[VAL_37]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_62:.*]] = "onnx.Add"(%[[VAL_60]], %[[VAL_61]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_63:.*]] = "onnx.Tanh"(%[[VAL_62]]) : (memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_64:.*]] = "onnx.Mul"(%[[VAL_59]], %[[VAL_17]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_65:.*]] = "onnx.Mul"(%[[VAL_55]], %[[VAL_63]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_66:.*]] = "onnx.Add"(%[[VAL_64]], %[[VAL_65]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_67:.*]] = "onnx.MatMul"(%[[VAL_47]], %[[VAL_30]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_68:.*]] = "onnx.MatMul"(%[[VAL_14]], %[[VAL_35]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_69:.*]] = "onnx.Add"(%[[VAL_67]], %[[VAL_68]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_70:.*]] = "onnx.Sigmoid"(%[[VAL_69]]) : (memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_71:.*]] = "onnx.Tanh"(%[[VAL_66]]) : (memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_72:.*]] = "onnx.Mul"(%[[VAL_70]], %[[VAL_71]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_73:.*]] = constant 12 : i64
// CHECK:             %[[VAL_74:.*]] = constant 0 : index
// CHECK:             %[[VAL_75:.*]] = memref.dim %[[VAL_72]], %[[VAL_74]] : memref<?x3xf32>
// CHECK:             %[[VAL_76:.*]] = index_cast %[[VAL_75]] : index to i64
// CHECK:             %[[VAL_77:.*]] = muli %[[VAL_73]], %[[VAL_76]] : i64
// CHECK:             "krnl.memcpy"(%[[VAL_14]], %[[VAL_72]], %[[VAL_77]]) : (memref<?x3xf32>, memref<?x3xf32>, i64) -> ()
// CHECK:             %[[VAL_78:.*]] = constant 12 : i64
// CHECK:             %[[VAL_79:.*]] = constant 0 : index
// CHECK:             %[[VAL_80:.*]] = memref.dim %[[VAL_66]], %[[VAL_79]] : memref<?x3xf32>
// CHECK:             %[[VAL_81:.*]] = index_cast %[[VAL_80]] : index to i64
// CHECK:             %[[VAL_82:.*]] = muli %[[VAL_78]], %[[VAL_81]] : i64
// CHECK:             "krnl.memcpy"(%[[VAL_17]], %[[VAL_66]], %[[VAL_82]]) : (memref<?x3xf32>, memref<?x3xf32>, i64) -> ()
// CHECK:             %[[VAL_83:.*]] = constant 0 : index
// CHECK:             %[[VAL_84:.*]] = memref.dim %[[VAL_72]], %[[VAL_83]] : memref<?x3xf32>
// CHECK:             %[[VAL_85:.*]] = constant 3 : index
// CHECK:             %[[VAL_86:.*]] = constant 0 : index
// CHECK:             %[[VAL_87:.*]] = constant 0 : index
// CHECK:             %[[VAL_88:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_88]]#0, %[[VAL_88]]#1) with (%[[VAL_88]]#0 -> %[[VAL_89:.*]] = %[[VAL_86]] to %[[VAL_84]], %[[VAL_88]]#1 -> %[[VAL_90:.*]] = %[[VAL_87]] to %[[VAL_85]]) {
// CHECK:               %[[VAL_91:.*]]:2 = krnl.get_induction_var_value(%[[VAL_88]]#0, %[[VAL_88]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_92:.*]] = krnl.load %[[VAL_72]]{{\[}}%[[VAL_91]]#0, %[[VAL_91]]#1] : memref<?x3xf32>
// CHECK:               krnl.store %[[VAL_92]], %[[VAL_8]]{{\[}}%[[VAL_41]], %[[VAL_42]], %[[VAL_91]]#0, %[[VAL_91]]#1] : memref<?x1x?x3xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           %[[VAL_93:.*]] = constant 12 : i64
// CHECK:           %[[VAL_94:.*]] = constant 0 : index
// CHECK:           %[[VAL_95:.*]] = memref.dim %[[VAL_14]], %[[VAL_94]] : memref<?x3xf32>
// CHECK:           %[[VAL_96:.*]] = index_cast %[[VAL_95]] : index to i64
// CHECK:           %[[VAL_97:.*]] = muli %[[VAL_93]], %[[VAL_96]] : i64
// CHECK:           "krnl.memcpy"(%[[VAL_11]], %[[VAL_14]], %[[VAL_97]]) : (memref<1x?x3xf32>, memref<?x3xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_8]] : memref<?x1x?x3xf32>
// CHECK:           memref.dealloc %[[VAL_14]] : memref<?x3xf32>
// CHECK:           memref.dealloc %[[VAL_17]] : memref<?x3xf32>
// CHECK:           return %[[VAL_11]] : memref<1x?x3xf32>
// CHECK:         }
}

// -----

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
// CHECK:           %[[VAL_9:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_9]]#0, %[[VAL_9]]#1) with (%[[VAL_9]]#0 -> %[[VAL_10:.*]] = 0 to 3, %[[VAL_9]]#1 -> %[[VAL_11:.*]] = 0 to 3) {
// CHECK:             krnl.store %[[VAL_6]], %[[VAL_3]]{{\[}}%[[VAL_10]], %[[VAL_11]]] : memref<3x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_12:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x3x2xf32>) -> memref<3x2xf32>
// CHECK:           %[[VAL_13:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Transpose"(%[[VAL_12]]) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[VAL_15:.*]] = "onnx.Transpose"(%[[VAL_13]]) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[VAL_16:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_16]]) with (%[[VAL_16]] -> %[[VAL_17:.*]] = 0 to 4) {
// CHECK:             %[[VAL_18:.*]] = memref.alloc() : memref<3x2xf32>
// CHECK:             %[[VAL_19:.*]] = constant 0 : index
// CHECK:             %[[VAL_20:.*]] = constant 3 : index
// CHECK:             %[[VAL_21:.*]] = constant 2 : index
// CHECK:             %[[VAL_22:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_22]]#0, %[[VAL_22]]#1) with (%[[VAL_22]]#0 -> %[[VAL_23:.*]] = 0 to %[[VAL_20]], %[[VAL_22]]#1 -> %[[VAL_24:.*]] = 0 to %[[VAL_21]]) {
// CHECK:               %[[VAL_25:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_17]], %[[VAL_23]], %[[VAL_24]]] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_25]], %[[VAL_18]]{{\[}}%[[VAL_23]], %[[VAL_24]]] : memref<3x2xf32>
// CHECK:             }
// CHECK:             %[[VAL_26:.*]] = "onnx.MatMul"(%[[VAL_18]], %[[VAL_14]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_27:.*]] = "onnx.MatMul"(%[[VAL_3]], %[[VAL_15]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_28:.*]] = "onnx.Add"(%[[VAL_26]], %[[VAL_27]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_29:.*]] = "onnx.Tanh"(%[[VAL_28]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_30:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_3]], %[[VAL_29]], %[[VAL_30]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:           }
// CHECK:           %[[VAL_31:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_4]], %[[VAL_3]], %[[VAL_31]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
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
// CHECK:             %[[VAL_25:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_25]]#0, %[[VAL_25]]#1) with (%[[VAL_25]]#0 -> %[[VAL_26:.*]] = 0 to %[[VAL_23]], %[[VAL_25]]#1 -> %[[VAL_27:.*]] = 0 to %[[VAL_24]]) {
// CHECK:               %[[VAL_28:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_20]], %[[VAL_26]], %[[VAL_27]]] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_28]], %[[VAL_21]]{{\[}}%[[VAL_26]], %[[VAL_27]]] : memref<3x2xf32>
// CHECK:             }
// CHECK:             %[[VAL_29:.*]] = "onnx.MatMul"(%[[VAL_21]], %[[VAL_15]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_30:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_16]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_31:.*]] = "onnx.Add"(%[[VAL_29]], %[[VAL_30]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_32:.*]] = "onnx.Add"(%[[VAL_31]], %[[VAL_18]]#0) : (memref<3x3xf32>, memref<3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_33:.*]] = "onnx.Add"(%[[VAL_32]], %[[VAL_18]]#1) : (memref<3x3xf32>, memref<3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_34:.*]] = "onnx.Tanh"(%[[VAL_33]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_35:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_4]], %[[VAL_34]], %[[VAL_35]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:           }
// CHECK:           %[[VAL_36:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_5]], %[[VAL_4]], %[[VAL_36]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
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
// CHECK-SAME:                                                  %[[VAL_0:.*]]: memref<?x?x?xf32>,
// CHECK-SAME:                                                  %[[VAL_1:.*]]: memref<1x3x?xf32>,
// CHECK-SAME:                                                  %[[VAL_2:.*]]: memref<1x3x3xf32>) -> memref<1x?x3xf32> {
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
// CHECK:             %[[VAL_32:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_32]]#0, %[[VAL_32]]#1) with (%[[VAL_32]]#0 -> %[[VAL_33:.*]] = 0 to %[[VAL_28]], %[[VAL_32]]#1 -> %[[VAL_34:.*]] = 0 to %[[VAL_30]]) {
// CHECK:               %[[VAL_35:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_25]], %[[VAL_33]], %[[VAL_34]]] : memref<?x?x?xf32>
// CHECK:               krnl.store %[[VAL_35]], %[[VAL_31]]{{\[}}%[[VAL_33]], %[[VAL_34]]] : memref<?x?xf32>
// CHECK:             }
// CHECK:             %[[VAL_36:.*]] = "onnx.MatMul"(%[[VAL_31]], %[[VAL_20]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_37:.*]] = "onnx.MatMul"(%[[VAL_9]], %[[VAL_21]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_38:.*]] = "onnx.Add"(%[[VAL_36]], %[[VAL_37]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_39:.*]] = "onnx.Tanh"(%[[VAL_38]]) : (memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_40:.*]] = constant 12 : i64
// CHECK:             %[[VAL_41:.*]] = constant 0 : index
// CHECK:             %[[VAL_42:.*]] = memref.dim %[[VAL_39]], %[[VAL_41]] : memref<?x3xf32>
// CHECK:             %[[VAL_43:.*]] = index_cast %[[VAL_42]] : index to i64
// CHECK:             %[[VAL_44:.*]] = muli %[[VAL_40]], %[[VAL_43]] : i64
// CHECK:             "krnl.memcpy"(%[[VAL_9]], %[[VAL_39]], %[[VAL_44]]) : (memref<?x3xf32>, memref<?x3xf32>, i64) -> ()
// CHECK:           }
// CHECK:           %[[VAL_45:.*]] = constant 12 : i64
// CHECK:           %[[VAL_46:.*]] = constant 0 : index
// CHECK:           %[[VAL_47:.*]] = memref.dim %[[VAL_9]], %[[VAL_46]] : memref<?x3xf32>
// CHECK:           %[[VAL_48:.*]] = index_cast %[[VAL_47]] : index to i64
// CHECK:           %[[VAL_49:.*]] = muli %[[VAL_45]], %[[VAL_48]] : i64
// CHECK:           "krnl.memcpy"(%[[VAL_6]], %[[VAL_9]], %[[VAL_49]]) : (memref<1x?x3xf32>, memref<?x3xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_9]] : memref<?x3xf32>
// CHECK:           return %[[VAL_6]] : memref<1x?x3xf32>
// CHECK:         }

}

