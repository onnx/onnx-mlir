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
// LSTM computation. Iterate over the sequence.
// CHECK:           %[[VAL_25:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_25]]) with (%[[VAL_25]] -> %[[VAL_26:.*]] = 0 to 4) {
// Get a slice of X for the current timestep.
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
// ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
// CHECK:             %[[VAL_38:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_16]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_39:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_21]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_40:.*]] = "onnx.Add"(%[[VAL_38]], %[[VAL_39]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_41:.*]] = "onnx.Sigmoid"(%[[VAL_40]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
// CHECK:             %[[VAL_42:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_18]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_43:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_23]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_44:.*]] = "onnx.Add"(%[[VAL_42]], %[[VAL_43]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_45:.*]] = "onnx.Sigmoid"(%[[VAL_44]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
// CHECK:             %[[VAL_46:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_19]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_47:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_24]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_48:.*]] = "onnx.Add"(%[[VAL_46]], %[[VAL_47]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_49:.*]] = "onnx.Tanh"(%[[VAL_48]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// Ct = ft (.) Ct-1 + it (.) ct
// CHECK:             %[[VAL_50:.*]] = "onnx.Mul"(%[[VAL_45]], %[[VAL_3]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_51:.*]] = "onnx.Mul"(%[[VAL_41]], %[[VAL_49]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_52:.*]] = "onnx.Add"(%[[VAL_50]], %[[VAL_51]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
// CHECK:             %[[VAL_53:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_17]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_54:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_22]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.Add"(%[[VAL_53]], %[[VAL_54]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_56:.*]] = "onnx.Sigmoid"(%[[VAL_55]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// Ht = ot (.) h(Ct)
// CHECK:             %[[VAL_57:.*]] = "onnx.Tanh"(%[[VAL_52]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_58:.*]] = "onnx.Mul"(%[[VAL_56]], %[[VAL_57]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// Store the current Ht.
// CHECK:             %[[VAL_59:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_4]], %[[VAL_58]], %[[VAL_59]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// Store the current Ct.
// CHECK:             %[[VAL_60:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_3]], %[[VAL_52]], %[[VAL_60]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             memref.dealloc %[[VAL_27]] : memref<3x2xf32>
// CHECK:           }
// Store the intermediate states to the returned states.
// CHECK:           %[[VAL_61:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_5]], %[[VAL_4]], %[[VAL_61]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
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
// CHECK:             %[[VAL_33:.*]] = constant 0 : index
// CHECK:             %[[VAL_34:.*]] = constant 0 : index
// CHECK:             %[[VAL_35:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_35]]#0, %[[VAL_35]]#1) with (%[[VAL_35]]#0 -> %[[VAL_36:.*]] = %[[VAL_33]] to %[[VAL_31]], %[[VAL_35]]#1 -> %[[VAL_37:.*]] = %[[VAL_34]] to %[[VAL_32]]) {
// CHECK:               %[[VAL_38:.*]]:2 = krnl.get_induction_var_value(%[[VAL_35]]#0, %[[VAL_35]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_39:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_30]], %[[VAL_38]]#0, %[[VAL_38]]#1] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_39]], %[[VAL_27]]{{\[}}%[[VAL_38]]#0, %[[VAL_38]]#1] : memref<3x2xf32>
// CHECK:             }
// CHECK:             %[[VAL_40:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_16]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_41:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_21]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_42:.*]] = "onnx.Add"(%[[VAL_40]], %[[VAL_41]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_43:.*]] = "onnx.Sigmoid"(%[[VAL_42]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_44:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_18]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_45:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_23]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_46:.*]] = "onnx.Add"(%[[VAL_44]], %[[VAL_45]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_47:.*]] = "onnx.Sigmoid"(%[[VAL_46]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_48:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_19]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_49:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_24]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_50:.*]] = "onnx.Add"(%[[VAL_48]], %[[VAL_49]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_51:.*]] = "onnx.Tanh"(%[[VAL_50]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_52:.*]] = "onnx.Mul"(%[[VAL_47]], %[[VAL_3]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_53:.*]] = "onnx.Mul"(%[[VAL_43]], %[[VAL_51]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_54:.*]] = "onnx.Add"(%[[VAL_52]], %[[VAL_53]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_17]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_56:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_22]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_57:.*]] = "onnx.Add"(%[[VAL_55]], %[[VAL_56]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_58:.*]] = "onnx.Sigmoid"(%[[VAL_57]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_59:.*]] = "onnx.Tanh"(%[[VAL_54]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_60:.*]] = "onnx.Mul"(%[[VAL_58]], %[[VAL_59]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_61:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_4]], %[[VAL_60]], %[[VAL_61]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             %[[VAL_62:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_3]], %[[VAL_54]], %[[VAL_62]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             memref.dealloc %[[VAL_27]] : memref<3x2xf32>
// CHECK:           }
// CHECK:           %[[VAL_63:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_5]], %[[VAL_4]], %[[VAL_63]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
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
// CHECK:             %[[VAL_47:.*]] = constant 0 : index
// CHECK:             %[[VAL_48:.*]] = constant 0 : index
// CHECK:             %[[VAL_49:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_49]]#0, %[[VAL_49]]#1) with (%[[VAL_49]]#0 -> %[[VAL_50:.*]] = %[[VAL_47]] to %[[VAL_45]], %[[VAL_49]]#1 -> %[[VAL_51:.*]] = %[[VAL_48]] to %[[VAL_46]]) {
// CHECK:               %[[VAL_52:.*]]:2 = krnl.get_induction_var_value(%[[VAL_49]]#0, %[[VAL_49]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_53:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_42]], %[[VAL_52]]#0, %[[VAL_52]]#1] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_53]], %[[VAL_43]]{{\[}}%[[VAL_52]]#0, %[[VAL_52]]#1] : memref<3x2xf32>
// CHECK:             }
// CHECK:             %[[VAL_54:.*]] = "onnx.MatMul"(%[[VAL_43]], %[[VAL_22]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.MatMul"(%[[VAL_6]], %[[VAL_27]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_56:.*]] = "onnx.Add"(%[[VAL_54]], %[[VAL_55]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_57:.*]] = "onnx.Sigmoid"(%[[VAL_56]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_58:.*]] = "onnx.MatMul"(%[[VAL_43]], %[[VAL_24]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_59:.*]] = "onnx.MatMul"(%[[VAL_6]], %[[VAL_29]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_60:.*]] = "onnx.Add"(%[[VAL_58]], %[[VAL_59]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_61:.*]] = "onnx.Sigmoid"(%[[VAL_60]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_62:.*]] = "onnx.MatMul"(%[[VAL_43]], %[[VAL_25]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_63:.*]] = "onnx.MatMul"(%[[VAL_6]], %[[VAL_30]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_64:.*]] = "onnx.Add"(%[[VAL_62]], %[[VAL_63]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_65:.*]] = "onnx.Tanh"(%[[VAL_64]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_66:.*]] = "onnx.Mul"(%[[VAL_61]], %[[VAL_5]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_67:.*]] = "onnx.Mul"(%[[VAL_57]], %[[VAL_65]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_68:.*]] = "onnx.Add"(%[[VAL_66]], %[[VAL_67]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_69:.*]] = "onnx.MatMul"(%[[VAL_43]], %[[VAL_23]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_70:.*]] = "onnx.MatMul"(%[[VAL_6]], %[[VAL_28]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_71:.*]] = "onnx.Add"(%[[VAL_69]], %[[VAL_70]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_72:.*]] = "onnx.Sigmoid"(%[[VAL_71]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_73:.*]] = "onnx.Tanh"(%[[VAL_68]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_74:.*]] = "onnx.Mul"(%[[VAL_72]], %[[VAL_73]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_75:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_6]], %[[VAL_74]], %[[VAL_75]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             %[[VAL_76:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_5]], %[[VAL_68]], %[[VAL_76]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             memref.dealloc %[[VAL_43]] : memref<3x2xf32>
// CHECK:           }
// CHECK:           %[[VAL_77:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_77]]) with (%[[VAL_77]] -> %[[VAL_78:.*]] = 0 to 4) {
// CHECK:             %[[VAL_79:.*]] = memref.alloc() : memref<3x2xf32>
// CHECK:             %[[VAL_80:.*]] = constant 1 : index
// CHECK:             %[[VAL_81:.*]] = constant 4 : index
// CHECK:             %[[VAL_82:.*]] = affine.apply #map(%[[VAL_78]]){{\[}}%[[VAL_81]]]
// CHECK:             %[[VAL_83:.*]] = constant 3 : index
// CHECK:             %[[VAL_84:.*]] = constant 2 : index
// CHECK:             %[[VAL_85:.*]] = constant 0 : index
// CHECK:             %[[VAL_86:.*]] = constant 0 : index
// CHECK:             %[[VAL_87:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_87]]#0, %[[VAL_87]]#1) with (%[[VAL_87]]#0 -> %[[VAL_88:.*]] = %[[VAL_85]] to %[[VAL_83]], %[[VAL_87]]#1 -> %[[VAL_89:.*]] = %[[VAL_86]] to %[[VAL_84]]) {
// CHECK:               %[[VAL_90:.*]]:2 = krnl.get_induction_var_value(%[[VAL_87]]#0, %[[VAL_87]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_91:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_82]], %[[VAL_90]]#0, %[[VAL_90]]#1] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_91]], %[[VAL_79]]{{\[}}%[[VAL_90]]#0, %[[VAL_90]]#1] : memref<3x2xf32>
// CHECK:             }
// CHECK:             %[[VAL_92:.*]] = "onnx.MatMul"(%[[VAL_79]], %[[VAL_32]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_93:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_37]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_94:.*]] = "onnx.Add"(%[[VAL_92]], %[[VAL_93]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_95:.*]] = "onnx.Sigmoid"(%[[VAL_94]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_96:.*]] = "onnx.MatMul"(%[[VAL_79]], %[[VAL_34]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_97:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_39]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_98:.*]] = "onnx.Add"(%[[VAL_96]], %[[VAL_97]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_99:.*]] = "onnx.Sigmoid"(%[[VAL_98]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_100:.*]] = "onnx.MatMul"(%[[VAL_79]], %[[VAL_35]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_101:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_40]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_102:.*]] = "onnx.Add"(%[[VAL_100]], %[[VAL_101]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_103:.*]] = "onnx.Tanh"(%[[VAL_102]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_104:.*]] = "onnx.Mul"(%[[VAL_99]], %[[VAL_3]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_105:.*]] = "onnx.Mul"(%[[VAL_95]], %[[VAL_103]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_106:.*]] = "onnx.Add"(%[[VAL_104]], %[[VAL_105]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_107:.*]] = "onnx.MatMul"(%[[VAL_79]], %[[VAL_33]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_108:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_38]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_109:.*]] = "onnx.Add"(%[[VAL_107]], %[[VAL_108]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_110:.*]] = "onnx.Sigmoid"(%[[VAL_109]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_111:.*]] = "onnx.Tanh"(%[[VAL_106]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_112:.*]] = "onnx.Mul"(%[[VAL_110]], %[[VAL_111]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_113:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_4]], %[[VAL_112]], %[[VAL_113]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             %[[VAL_114:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_3]], %[[VAL_106]], %[[VAL_114]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             memref.dealloc %[[VAL_79]] : memref<3x2xf32>
// CHECK:           }
// CHECK:           %[[VAL_115:.*]] = constant 3 : index
// CHECK:           %[[VAL_116:.*]] = constant 3 : index
// CHECK:           %[[VAL_117:.*]] = constant 0 : index
// CHECK:           %[[VAL_118:.*]] = constant 0 : index
// CHECK:           %[[VAL_119:.*]] = constant 0 : index
// CHECK:           %[[VAL_120:.*]] = constant 1 : index
// CHECK:           %[[VAL_121:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_121]]#0, %[[VAL_121]]#1) with (%[[VAL_121]]#0 -> %[[VAL_122:.*]] = %[[VAL_117]] to %[[VAL_115]], %[[VAL_121]]#1 -> %[[VAL_123:.*]] = %[[VAL_118]] to %[[VAL_116]]) {
// CHECK:             %[[VAL_124:.*]]:2 = krnl.get_induction_var_value(%[[VAL_121]]#0, %[[VAL_121]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             %[[VAL_125:.*]] = krnl.load %[[VAL_6]]{{\[}}%[[VAL_124]]#0, %[[VAL_124]]#1] : memref<3x3xf32>
// CHECK:             krnl.store %[[VAL_125]], %[[VAL_7]]{{\[}}%[[VAL_119]], %[[VAL_124]]#0, %[[VAL_124]]#1] : memref<2x3x3xf32>
// CHECK:             %[[VAL_126:.*]] = krnl.load %[[VAL_4]]{{\[}}%[[VAL_124]]#0, %[[VAL_124]]#1] : memref<3x3xf32>
// CHECK:             krnl.store %[[VAL_126]], %[[VAL_7]]{{\[}}%[[VAL_120]], %[[VAL_124]]#0, %[[VAL_124]]#1] : memref<2x3x3xf32>
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
// CHECK-SAME:                                                    %[[VAL_0:.*]]: memref<?x?x?xf32>,
// CHECK-SAME:                                                    %[[VAL_1:.*]]: memref<1x12x?xf32>,
// CHECK-SAME:                                                    %[[VAL_2:.*]]: memref<1x12x3xf32>) -> memref<1x?x3xf32> {
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
// CHECK:             %[[VAL_48:.*]] = constant 0 : index
// CHECK:             %[[VAL_49:.*]] = memref.dim %[[VAL_47]], %[[VAL_48]] : memref<?x?xf32>
// CHECK:             %[[VAL_50:.*]] = constant 1 : index
// CHECK:             %[[VAL_51:.*]] = memref.dim %[[VAL_47]], %[[VAL_50]] : memref<?x?xf32>
// CHECK:             %[[VAL_52:.*]] = constant 0 : index
// CHECK:             %[[VAL_53:.*]] = constant 0 : index
// CHECK:             %[[VAL_54:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_54]]#0, %[[VAL_54]]#1) with (%[[VAL_54]]#0 -> %[[VAL_55:.*]] = %[[VAL_52]] to %[[VAL_49]], %[[VAL_54]]#1 -> %[[VAL_56:.*]] = %[[VAL_53]] to %[[VAL_51]]) {
// CHECK:               %[[VAL_57:.*]]:2 = krnl.get_induction_var_value(%[[VAL_54]]#0, %[[VAL_54]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_58:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_41]], %[[VAL_57]]#0, %[[VAL_57]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store %[[VAL_58]], %[[VAL_47]]{{\[}}%[[VAL_57]]#0, %[[VAL_57]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK:             %[[VAL_59:.*]] = "onnx.MatMul"(%[[VAL_47]], %[[VAL_29]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_60:.*]] = "onnx.MatMul"(%[[VAL_14]], %[[VAL_34]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_61:.*]] = "onnx.Add"(%[[VAL_59]], %[[VAL_60]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_62:.*]] = "onnx.Sigmoid"(%[[VAL_61]]) : (memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_63:.*]] = "onnx.MatMul"(%[[VAL_47]], %[[VAL_31]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_64:.*]] = "onnx.MatMul"(%[[VAL_14]], %[[VAL_36]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_65:.*]] = "onnx.Add"(%[[VAL_63]], %[[VAL_64]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_66:.*]] = "onnx.Sigmoid"(%[[VAL_65]]) : (memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_67:.*]] = "onnx.MatMul"(%[[VAL_47]], %[[VAL_32]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_68:.*]] = "onnx.MatMul"(%[[VAL_14]], %[[VAL_37]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_69:.*]] = "onnx.Add"(%[[VAL_67]], %[[VAL_68]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_70:.*]] = "onnx.Tanh"(%[[VAL_69]]) : (memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_71:.*]] = "onnx.Mul"(%[[VAL_66]], %[[VAL_17]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_72:.*]] = "onnx.Mul"(%[[VAL_62]], %[[VAL_70]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_73:.*]] = "onnx.Add"(%[[VAL_71]], %[[VAL_72]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_74:.*]] = "onnx.MatMul"(%[[VAL_47]], %[[VAL_30]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_75:.*]] = "onnx.MatMul"(%[[VAL_14]], %[[VAL_35]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_76:.*]] = "onnx.Add"(%[[VAL_74]], %[[VAL_75]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_77:.*]] = "onnx.Sigmoid"(%[[VAL_76]]) : (memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_78:.*]] = "onnx.Tanh"(%[[VAL_73]]) : (memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_79:.*]] = "onnx.Mul"(%[[VAL_77]], %[[VAL_78]]) : (memref<?x3xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_80:.*]] = constant 12 : i64
// CHECK:             %[[VAL_81:.*]] = constant 0 : index
// CHECK:             %[[VAL_82:.*]] = memref.dim %[[VAL_79]], %[[VAL_81]] : memref<?x3xf32>
// CHECK:             %[[VAL_83:.*]] = index_cast %[[VAL_82]] : index to i64
// CHECK:             %[[VAL_84:.*]] = muli %[[VAL_80]], %[[VAL_83]] : i64
// CHECK:             "krnl.memcpy"(%[[VAL_14]], %[[VAL_79]], %[[VAL_84]]) : (memref<?x3xf32>, memref<?x3xf32>, i64) -> ()
// CHECK:             %[[VAL_85:.*]] = constant 12 : i64
// CHECK:             %[[VAL_86:.*]] = constant 0 : index
// CHECK:             %[[VAL_87:.*]] = memref.dim %[[VAL_73]], %[[VAL_86]] : memref<?x3xf32>
// CHECK:             %[[VAL_88:.*]] = index_cast %[[VAL_87]] : index to i64
// CHECK:             %[[VAL_89:.*]] = muli %[[VAL_85]], %[[VAL_88]] : i64
// CHECK:             "krnl.memcpy"(%[[VAL_17]], %[[VAL_73]], %[[VAL_89]]) : (memref<?x3xf32>, memref<?x3xf32>, i64) -> ()
// CHECK:             %[[VAL_90:.*]] = constant 0 : index
// CHECK:             %[[VAL_91:.*]] = memref.dim %[[VAL_79]], %[[VAL_90]] : memref<?x3xf32>
// CHECK:             %[[VAL_92:.*]] = constant 3 : index
// CHECK:             %[[VAL_93:.*]] = constant 0 : index
// CHECK:             %[[VAL_94:.*]] = constant 0 : index
// CHECK:             %[[VAL_95:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_95]]#0, %[[VAL_95]]#1) with (%[[VAL_95]]#0 -> %[[VAL_96:.*]] = %[[VAL_93]] to %[[VAL_91]], %[[VAL_95]]#1 -> %[[VAL_97:.*]] = %[[VAL_94]] to %[[VAL_92]]) {
// CHECK:               %[[VAL_98:.*]]:2 = krnl.get_induction_var_value(%[[VAL_95]]#0, %[[VAL_95]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_99:.*]] = krnl.load %[[VAL_79]]{{\[}}%[[VAL_98]]#0, %[[VAL_98]]#1] : memref<?x3xf32>
// CHECK:               krnl.store %[[VAL_99]], %[[VAL_8]]{{\[}}%[[VAL_41]], %[[VAL_42]], %[[VAL_98]]#0, %[[VAL_98]]#1] : memref<?x1x?x3xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_47]] : memref<?x?xf32>
// CHECK:           }
// CHECK:           %[[VAL_100:.*]] = constant 12 : i64
// CHECK:           %[[VAL_101:.*]] = constant 0 : index
// CHECK:           %[[VAL_102:.*]] = memref.dim %[[VAL_14]], %[[VAL_101]] : memref<?x3xf32>
// CHECK:           %[[VAL_103:.*]] = index_cast %[[VAL_102]] : index to i64
// CHECK:           %[[VAL_104:.*]] = muli %[[VAL_100]], %[[VAL_103]] : i64
// CHECK:           "krnl.memcpy"(%[[VAL_11]], %[[VAL_14]], %[[VAL_104]]) : (memref<1x?x3xf32>, memref<?x3xf32>, i64) -> ()
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
// CHECK:             %[[VAL_22:.*]] = constant 0 : index
// CHECK:             %[[VAL_23:.*]] = constant 0 : index
// CHECK:             %[[VAL_24:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_24]]#0, %[[VAL_24]]#1) with (%[[VAL_24]]#0 -> %[[VAL_25:.*]] = %[[VAL_22]] to %[[VAL_20]], %[[VAL_24]]#1 -> %[[VAL_26:.*]] = %[[VAL_23]] to %[[VAL_21]]) {
// CHECK:               %[[VAL_27:.*]]:2 = krnl.get_induction_var_value(%[[VAL_24]]#0, %[[VAL_24]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_28:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_17]], %[[VAL_27]]#0, %[[VAL_27]]#1] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_28]], %[[VAL_18]]{{\[}}%[[VAL_27]]#0, %[[VAL_27]]#1] : memref<3x2xf32>
// CHECK:             }
// CHECK:             %[[VAL_29:.*]] = "onnx.MatMul"(%[[VAL_18]], %[[VAL_14]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_30:.*]] = "onnx.MatMul"(%[[VAL_3]], %[[VAL_15]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_31:.*]] = "onnx.Add"(%[[VAL_29]], %[[VAL_30]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_32:.*]] = "onnx.Tanh"(%[[VAL_31]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_33:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[VAL_3]], %[[VAL_32]], %[[VAL_33]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             memref.dealloc %[[VAL_18]] : memref<3x2xf32>
// CHECK:           }
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

