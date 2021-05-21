// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='test-rnn-ops-lowering' %s -split-input-file | FileCheck %s

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
// CHECK:             %[[VAL_38:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_16]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_39:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_21]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_40:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_18]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_41:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_23]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_42:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_19]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_43:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_24]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_44:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_17]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_45:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_22]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_46:.*]] = constant 3 : index
// CHECK:             %[[VAL_47:.*]] = constant 3 : index
// CHECK:             %[[VAL_48:.*]] = constant 0 : index
// CHECK:             %[[VAL_49:.*]] = constant 0 : index
// CHECK:             %[[VAL_50:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_50]]#0, %[[VAL_50]]#1) with (%[[VAL_50]]#0 -> %[[VAL_51:.*]] = %[[VAL_48]] to %[[VAL_46]], %[[VAL_50]]#1 -> %[[VAL_52:.*]] = %[[VAL_49]] to %[[VAL_47]]) {
// CHECK:               %[[VAL_53:.*]]:2 = krnl.get_induction_var_value(%[[VAL_50]]#0, %[[VAL_50]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_54:.*]] = krnl.load %[[VAL_3]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_55:.*]] = krnl.load %[[VAL_38]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_56:.*]] = krnl.load %[[VAL_39]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_57:.*]] = addf %[[VAL_55]], %[[VAL_56]] : f32
// CHECK:               %[[VAL_58:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_57]], %[[VAL_58]][] : memref<f32>
// CHECK:               %[[VAL_59:.*]] = "onnx.Sigmoid"(%[[VAL_58]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_60:.*]] = krnl.load %[[VAL_59]][] : memref<f32>
// CHECK:               %[[VAL_61:.*]] = krnl.load %[[VAL_40]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_62:.*]] = krnl.load %[[VAL_41]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_63:.*]] = addf %[[VAL_61]], %[[VAL_62]] : f32
// CHECK:               %[[VAL_64:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_63]], %[[VAL_64]][] : memref<f32>
// CHECK:               %[[VAL_65:.*]] = "onnx.Sigmoid"(%[[VAL_64]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_66:.*]] = krnl.load %[[VAL_65]][] : memref<f32>
// CHECK:               %[[VAL_67:.*]] = krnl.load %[[VAL_42]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_68:.*]] = krnl.load %[[VAL_43]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_69:.*]] = addf %[[VAL_67]], %[[VAL_68]] : f32
// CHECK:               %[[VAL_70:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_69]], %[[VAL_70]][] : memref<f32>
// CHECK:               %[[VAL_71:.*]] = "onnx.Tanh"(%[[VAL_70]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_72:.*]] = krnl.load %[[VAL_71]][] : memref<f32>
// CHECK:               %[[VAL_73:.*]] = mulf %[[VAL_66]], %[[VAL_54]] : f32
// CHECK:               %[[VAL_74:.*]] = mulf %[[VAL_60]], %[[VAL_72]] : f32
// CHECK:               %[[VAL_75:.*]] = addf %[[VAL_73]], %[[VAL_74]] : f32
// CHECK:               %[[VAL_76:.*]] = krnl.load %[[VAL_44]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_77:.*]] = krnl.load %[[VAL_45]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_78:.*]] = addf %[[VAL_76]], %[[VAL_77]] : f32
// CHECK:               %[[VAL_79:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_78]], %[[VAL_79]][] : memref<f32>
// CHECK:               %[[VAL_80:.*]] = "onnx.Sigmoid"(%[[VAL_79]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_81:.*]] = krnl.load %[[VAL_80]][] : memref<f32>
// CHECK:               %[[VAL_82:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_75]], %[[VAL_82]][] : memref<f32>
// CHECK:               %[[VAL_83:.*]] = "onnx.Tanh"(%[[VAL_82]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_84:.*]] = krnl.load %[[VAL_83]][] : memref<f32>
// CHECK:               %[[VAL_85:.*]] = mulf %[[VAL_81]], %[[VAL_84]] : f32
// CHECK:               krnl.store %[[VAL_75]], %[[VAL_3]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:               krnl.store %[[VAL_85]], %[[VAL_4]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_27]] : memref<3x2xf32>
// CHECK:           }
// CHECK:           %[[VAL_86:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_5]], %[[VAL_4]], %[[VAL_86]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
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
// CHECK:             %[[VAL_61:.*]] = "onnx.MatMul"(%[[VAL_47]], %[[VAL_31]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_62:.*]] = "onnx.MatMul"(%[[VAL_14]], %[[VAL_36]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_63:.*]] = "onnx.MatMul"(%[[VAL_47]], %[[VAL_32]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_64:.*]] = "onnx.MatMul"(%[[VAL_14]], %[[VAL_37]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_65:.*]] = "onnx.MatMul"(%[[VAL_47]], %[[VAL_30]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_66:.*]] = "onnx.MatMul"(%[[VAL_14]], %[[VAL_35]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_67:.*]] = constant 0 : index
// CHECK:             %[[VAL_68:.*]] = memref.dim %[[VAL_14]], %[[VAL_67]] : memref<?x3xf32>
// CHECK:             %[[VAL_69:.*]] = constant 3 : index
// CHECK:             %[[VAL_70:.*]] = constant 0 : index
// CHECK:             %[[VAL_71:.*]] = constant 0 : index
// CHECK:             %[[VAL_72:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_72]]#0, %[[VAL_72]]#1) with (%[[VAL_72]]#0 -> %[[VAL_73:.*]] = %[[VAL_70]] to %[[VAL_68]], %[[VAL_72]]#1 -> %[[VAL_74:.*]] = %[[VAL_71]] to %[[VAL_69]]) {
// CHECK:               %[[VAL_75:.*]]:2 = krnl.get_induction_var_value(%[[VAL_72]]#0, %[[VAL_72]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_76:.*]] = krnl.load %[[VAL_17]]{{\[}}%[[VAL_75]]#0, %[[VAL_75]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_77:.*]] = krnl.load %[[VAL_59]]{{\[}}%[[VAL_75]]#0, %[[VAL_75]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_78:.*]] = krnl.load %[[VAL_60]]{{\[}}%[[VAL_75]]#0, %[[VAL_75]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_79:.*]] = addf %[[VAL_77]], %[[VAL_78]] : f32
// CHECK:               %[[VAL_80:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_79]], %[[VAL_80]][] : memref<f32>
// CHECK:               %[[VAL_81:.*]] = "onnx.Sigmoid"(%[[VAL_80]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_82:.*]] = krnl.load %[[VAL_81]][] : memref<f32>
// CHECK:               %[[VAL_83:.*]] = krnl.load %[[VAL_61]]{{\[}}%[[VAL_75]]#0, %[[VAL_75]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_84:.*]] = krnl.load %[[VAL_62]]{{\[}}%[[VAL_75]]#0, %[[VAL_75]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_85:.*]] = addf %[[VAL_83]], %[[VAL_84]] : f32
// CHECK:               %[[VAL_86:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_85]], %[[VAL_86]][] : memref<f32>
// CHECK:               %[[VAL_87:.*]] = "onnx.Sigmoid"(%[[VAL_86]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_88:.*]] = krnl.load %[[VAL_87]][] : memref<f32>
// CHECK:               %[[VAL_89:.*]] = krnl.load %[[VAL_63]]{{\[}}%[[VAL_75]]#0, %[[VAL_75]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_90:.*]] = krnl.load %[[VAL_64]]{{\[}}%[[VAL_75]]#0, %[[VAL_75]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_91:.*]] = addf %[[VAL_89]], %[[VAL_90]] : f32
// CHECK:               %[[VAL_92:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_91]], %[[VAL_92]][] : memref<f32>
// CHECK:               %[[VAL_93:.*]] = "onnx.Tanh"(%[[VAL_92]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_94:.*]] = krnl.load %[[VAL_93]][] : memref<f32>
// CHECK:               %[[VAL_95:.*]] = mulf %[[VAL_88]], %[[VAL_76]] : f32
// CHECK:               %[[VAL_96:.*]] = mulf %[[VAL_82]], %[[VAL_94]] : f32
// CHECK:               %[[VAL_97:.*]] = addf %[[VAL_95]], %[[VAL_96]] : f32
// CHECK:               %[[VAL_98:.*]] = krnl.load %[[VAL_65]]{{\[}}%[[VAL_75]]#0, %[[VAL_75]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_99:.*]] = krnl.load %[[VAL_66]]{{\[}}%[[VAL_75]]#0, %[[VAL_75]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_100:.*]] = addf %[[VAL_98]], %[[VAL_99]] : f32
// CHECK:               %[[VAL_101:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_100]], %[[VAL_101]][] : memref<f32>
// CHECK:               %[[VAL_102:.*]] = "onnx.Sigmoid"(%[[VAL_101]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_103:.*]] = krnl.load %[[VAL_102]][] : memref<f32>
// CHECK:               %[[VAL_104:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_97]], %[[VAL_104]][] : memref<f32>
// CHECK:               %[[VAL_105:.*]] = "onnx.Tanh"(%[[VAL_104]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_106:.*]] = krnl.load %[[VAL_105]][] : memref<f32>
// CHECK:               %[[VAL_107:.*]] = mulf %[[VAL_103]], %[[VAL_106]] : f32
// CHECK:               krnl.store %[[VAL_97]], %[[VAL_17]]{{\[}}%[[VAL_75]]#0, %[[VAL_75]]#1] : memref<?x3xf32>
// CHECK:               krnl.store %[[VAL_107]], %[[VAL_14]]{{\[}}%[[VAL_75]]#0, %[[VAL_75]]#1] : memref<?x3xf32>
// CHECK:               krnl.store %[[VAL_107]], %[[VAL_8]]{{\[}}%[[VAL_41]], %[[VAL_42]], %[[VAL_75]]#0, %[[VAL_75]]#1] : memref<?x1x?x3xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_47]] : memref<?x?xf32>
// CHECK:           }
// CHECK:           %[[VAL_108:.*]] = constant 12 : i64
// CHECK:           %[[VAL_109:.*]] = constant 0 : index
// CHECK:           %[[VAL_110:.*]] = memref.dim %[[VAL_14]], %[[VAL_109]] : memref<?x3xf32>
// CHECK:           %[[VAL_111:.*]] = index_cast %[[VAL_110]] : index to i64
// CHECK:           %[[VAL_112:.*]] = muli %[[VAL_108]], %[[VAL_111]] : i64
// CHECK:           "krnl.memcpy"(%[[VAL_11]], %[[VAL_14]], %[[VAL_112]]) : (memref<1x?x3xf32>, memref<?x3xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_8]] : memref<?x1x?x3xf32>
// CHECK:           memref.dealloc %[[VAL_14]] : memref<?x3xf32>
// CHECK:           memref.dealloc %[[VAL_17]] : memref<?x3xf32>
// CHECK:           return %[[VAL_11]] : memref<1x?x3xf32>
// CHECK:         }

}
