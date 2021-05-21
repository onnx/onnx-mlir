// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='test-rnn-ops-lowering' %s -split-input-file

// FIXME: enable this.
// | FileCheck %s

func private @test_lstm_general_computation(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_lstm_general_computation(
// CHECK-SAME:                                                %[[X:.*]]: memref<4x3x2xf32>,
// CHECK-SAME:                                                %[[W:.*]]: memref<1x12x2xf32>,
// CHECK-SAME:                                                %[[B:.*]]: memref<1x12x3xf32>) -> memref<1x3x3xf32> {
// CHECK:           %[[Ct:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:           %[[Ht:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:           %[[Y_h:.*]] = memref.alloc() : memref<1x3x3xf32>
// CHECK:           %[[VAL_6:.*]] = constant unit
// CHECK:           %[[VAL_7:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_8:.*]] = constant 0 : index
// CHECK:           %[[VAL_9:.*]] = constant 1 : index
// CHECK:           %[[VAL_10:.*]]:2 = krnl.define_loops 2
// Initialize the intermediate states Ht and Ct.
// CHECK:           krnl.iterate(%[[VAL_10]]#0, %[[VAL_10]]#1) with (%[[VAL_10]]#0 -> %[[VAL_11:.*]] = 0 to 3, %[[VAL_10]]#1 -> %[[VAL_12:.*]] = 0 to 3) {
// CHECK:             krnl.store %[[VAL_7]], %[[Ht]]{{\[}}%[[VAL_11]], %[[VAL_12]]] : memref<3x3xf32>
// CHECK:             krnl.store %[[VAL_7]], %[[Ct]]{{\[}}%[[VAL_11]], %[[VAL_12]]] : memref<3x3xf32>
// CHECK:           }
// Prepare weights and biases.
// CHECK:           %[[VAL_13:.*]] = "onnx.Squeeze"(%[[W]]) {axes = [0]} : (memref<1x12x2xf32>) -> memref<12x2xf32>
// CHECK:           %[[VAL_14:.*]] = "onnx.Squeeze"(%[[B]]) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// Weights
// CHECK:           %[[VAL_15:.*]]:4 = "onnx.Split"(%[[VAL_13]]) {axis = 0 : si64} : (memref<12x2xf32>) -> (memref<3x2xf32>, memref<3x2xf32>, memref<3x2xf32>, memref<3x2xf32>)
// CHECK:           %[[Wf:.*]] = "onnx.Transpose"(%[[VAL_15]]#0) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[Wo:.*]] = "onnx.Transpose"(%[[VAL_15]]#1) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[Wi:.*]] = "onnx.Transpose"(%[[VAL_15]]#2) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// CHECK:           %[[Wc:.*]] = "onnx.Transpose"(%[[VAL_15]]#3) {perm = [1, 0]} : (memref<3x2xf32>) -> memref<2x3xf32>
// Biases.
// CHECK:           %[[VAL_20:.*]]:4 = "onnx.Split"(%[[VAL_14]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>, memref<3x3xf32>)
// CHECK:           %[[Rf:.*]] = "onnx.Transpose"(%[[VAL_20]]#0) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[Ro:.*]] = "onnx.Transpose"(%[[VAL_20]]#1) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[Ri:.*]] = "onnx.Transpose"(%[[VAL_20]]#2) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:           %[[Rc:.*]] = "onnx.Transpose"(%[[VAL_20]]#3) {perm = [1, 0]} : (memref<3x3xf32>) -> memref<3x3xf32>
// LSTM computation. Iterate over the sequence.
// CHECK:           %[[VAL_25:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_25]]) with (%[[VAL_25]] -> %[[VAL_26:.*]] = 0 to 4) {
// Get a slice of X for the current timestep.
// CHECK:             %[[Xt:.*]] = memref.alloc() : memref<3x2xf32>
// CHECK:             %[[VAL_28:.*]] = constant 0 : index
// CHECK:             %[[VAL_29:.*]] = constant 3 : index
// CHECK:             %[[VAL_30:.*]] = constant 2 : index
// CHECK:             %[[VAL_31:.*]] = constant 0 : index
// CHECK:             %[[VAL_32:.*]] = constant 0 : index
// CHECK:             %[[VAL_33:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_33]]#0, %[[VAL_33]]#1) with (%[[VAL_33]]#0 -> %[[VAL_34:.*]] = %[[VAL_31]] to %[[VAL_29]], %[[VAL_33]]#1 -> %[[VAL_35:.*]] = %[[VAL_32]] to %[[VAL_30]]) {
// CHECK:               %[[VAL_36:.*]]:2 = krnl.get_induction_var_value(%[[VAL_33]]#0, %[[VAL_33]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_37:.*]] = krnl.load %[[X]]{{\[}}%[[VAL_26]], %[[VAL_36]]#0, %[[VAL_36]]#1] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_37]], %[[Xt]]{{\[}}%[[VAL_36]]#0, %[[VAL_36]]#1] : memref<3x2xf32>
// CHECK:             }
// it = f(Xt*(Wi^T) + Ht-1*(Ri^T) + Pi (.) Ct-1 + Wbi + Rbi)
// CHECK:             %[[VAL_38:.*]] = "onnx.MatMul"(%[[Xt]], %[[Wf]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_39:.*]] = "onnx.MatMul"(%[[Ht]], %[[Rf]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_40:.*]] = "onnx.Add"(%[[VAL_38]], %[[VAL_39]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[it:.*]] = "onnx.Sigmoid"(%[[VAL_40]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// ft = f(Xt*(Wf^T) + Ht-1*(Rf^T) + Pf (.) Ct-1 + Wbf + Rbf)
// CHECK:             %[[VAL_42:.*]] = "onnx.MatMul"(%[[Xt]], %[[Wi]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_43:.*]] = "onnx.MatMul"(%[[Ht]], %[[Ri]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_44:.*]] = "onnx.Add"(%[[VAL_42]], %[[VAL_43]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[ft:.*]] = "onnx.Sigmoid"(%[[VAL_44]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// ct = g(Xt*(Wc^T) + Ht-1*(Rc^T) + Wbc + Rbc)
// CHECK:             %[[VAL_46:.*]] = "onnx.MatMul"(%[[Xt]], %[[Wc]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_47:.*]] = "onnx.MatMul"(%[[Ht]], %[[Rc]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_48:.*]] = "onnx.Add"(%[[VAL_46]], %[[VAL_47]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[ct:.*]] = "onnx.Tanh"(%[[VAL_48]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// Ct = ft (.) Ct-1 + it (.) ct
// CHECK:             %[[VAL_50:.*]] = "onnx.Mul"(%[[ft]], %[[Ct]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_51:.*]] = "onnx.Mul"(%[[it]], %[[ct]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[nextCt:.*]] = "onnx.Add"(%[[VAL_50]], %[[VAL_51]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// ot = f(Xt*(Wo^T) + Ht-1*(Ro^T) + Po (.) Ct + Wbo + Rbo)
// CHECK:             %[[VAL_53:.*]] = "onnx.MatMul"(%[[Xt]], %[[Wo]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_54:.*]] = "onnx.MatMul"(%[[Ht]], %[[Ro]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.Add"(%[[VAL_53]], %[[VAL_54]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[ot:.*]] = "onnx.Sigmoid"(%[[VAL_55]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// Ht = ot (.) h(Ct)
// CHECK:             %[[VAL_57:.*]] = "onnx.Tanh"(%[[nextCt]]) : (memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_58:.*]] = "onnx.Mul"(%[[ot]], %[[VAL_57]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// Store the current Ht.
// CHECK:             %[[VAL_59:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[Ht]], %[[VAL_58]], %[[VAL_59]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// Store the current Ct.
// CHECK:             %[[VAL_60:.*]] = constant 36 : i64
// CHECK:             "krnl.memcpy"(%[[Ct]], %[[nextCt]], %[[VAL_60]]) : (memref<3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:             memref.dealloc %[[Xt]] : memref<3x2xf32>
// CHECK:           }
// Store the intermediate states to the returned states.
// CHECK:           %[[VAL_61:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[Y_h]], %[[Ht]], %[[VAL_61]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[Ht]] : memref<3x3xf32>
// CHECK:           memref.dealloc %[[Ct]] : memref<3x3xf32>
// CHECK:           return %[[Y_h]] : memref<1x3x3xf32>
// CHECK:         }

}

// -----

func private @test_lstm_reverse_mode(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64, direction = "reverse"} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK: #map = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
// CHECK-LABEL:   func private @test_lstm_reverse_mode(
// CHECK-SAME:                                         %[[VAL_0:.*]]: memref<4x3x2xf32>,
// CHECK-SAME:                                         %[[VAL_1:.*]]: memref<1x12x2xf32>,
// CHECK-SAME:                                         %[[VAL_2:.*]]: memref<1x12x3xf32>) -> memref<1x3x3xf32> {
// Reverse direction. Load X from the last timestep.
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
// CHECK:          }

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

// Forward direction.
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
// CHECK:           }

// Reverse direction. Load X from the last timestep.
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
// CHECK:           }

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
