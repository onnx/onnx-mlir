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
// CHECK:             %[[VAL_24:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:             %[[VAL_25:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:             %[[VAL_26:.*]] = memref.alloc() : memref<3x2xf32>
// CHECK:             %[[VAL_27:.*]] = constant 0 : index
// CHECK:             %[[VAL_28:.*]] = constant 3 : index
// CHECK:             %[[VAL_29:.*]] = constant 2 : index
// CHECK:             %[[VAL_30:.*]] = constant 0 : index
// CHECK:             %[[VAL_31:.*]] = constant 0 : index
// CHECK:             %[[VAL_32:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_32]]#0, %[[VAL_32]]#1) with (%[[VAL_32]]#0 -> %[[VAL_33:.*]] = %[[VAL_30]] to %[[VAL_28]], %[[VAL_32]]#1 -> %[[VAL_34:.*]] = %[[VAL_31]] to %[[VAL_29]]) {
// CHECK:               %[[VAL_35:.*]]:2 = krnl.get_induction_var_value(%[[VAL_32]]#0, %[[VAL_32]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_36:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_23]], %[[VAL_35]]#0, %[[VAL_35]]#1] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_36]], %[[VAL_26]]{{\[}}%[[VAL_35]]#0, %[[VAL_35]]#1] : memref<3x2xf32>
// CHECK:             }
// CHECK:             %[[VAL_37:.*]] = "onnx.MatMul"(%[[VAL_26]], %[[VAL_15]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_38:.*]] = "onnx.MatMul"(%[[VAL_3]], %[[VAL_19]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_39:.*]] = "onnx.MatMul"(%[[VAL_26]], %[[VAL_16]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_40:.*]] = "onnx.MatMul"(%[[VAL_3]], %[[VAL_20]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_41:.*]] = "onnx.MatMul"(%[[VAL_26]], %[[VAL_17]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_42:.*]] = constant 1.000000e+00 : f32
// CHECK:             %[[VAL_43:.*]] = constant 3 : index
// CHECK:             %[[VAL_44:.*]] = constant 3 : index
// CHECK:             %[[VAL_45:.*]] = constant 0 : index
// CHECK:             %[[VAL_46:.*]] = constant 0 : index
// CHECK:             %[[VAL_47:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_47]]#0, %[[VAL_47]]#1) with (%[[VAL_47]]#0 -> %[[VAL_48:.*]] = %[[VAL_45]] to %[[VAL_43]], %[[VAL_47]]#1 -> %[[VAL_49:.*]] = %[[VAL_46]] to %[[VAL_44]]) {
// CHECK:               %[[VAL_50:.*]]:2 = krnl.get_induction_var_value(%[[VAL_47]]#0, %[[VAL_47]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_51:.*]] = krnl.load %[[VAL_3]]{{\[}}%[[VAL_50]]#0, %[[VAL_50]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_52:.*]] = krnl.load %[[VAL_39]]{{\[}}%[[VAL_50]]#0, %[[VAL_50]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_53:.*]] = krnl.load %[[VAL_40]]{{\[}}%[[VAL_50]]#0, %[[VAL_50]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_54:.*]] = addf %[[VAL_52]], %[[VAL_53]] : f32
// CHECK:               %[[VAL_55:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_54]], %[[VAL_55]][] : memref<f32>
// CHECK:               %[[VAL_56:.*]] = "onnx.Sigmoid"(%[[VAL_55]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_57:.*]] = krnl.load %[[VAL_56]][] : memref<f32>
// CHECK:               krnl.store %[[VAL_57]], %[[VAL_25]]{{\[}}%[[VAL_50]]#0, %[[VAL_50]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_58:.*]] = mulf %[[VAL_57]], %[[VAL_51]] : f32
// CHECK:               krnl.store %[[VAL_58]], %[[VAL_24]]{{\[}}%[[VAL_50]]#0, %[[VAL_50]]#1] : memref<3x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_59:.*]] = "onnx.MatMul"(%[[VAL_24]], %[[VAL_21]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_60:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_60]]#0, %[[VAL_60]]#1) with (%[[VAL_60]]#0 -> %[[VAL_61:.*]] = %[[VAL_45]] to %[[VAL_43]], %[[VAL_60]]#1 -> %[[VAL_62:.*]] = %[[VAL_46]] to %[[VAL_44]]) {
// CHECK:               %[[VAL_63:.*]]:2 = krnl.get_induction_var_value(%[[VAL_60]]#0, %[[VAL_60]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_64:.*]] = krnl.load %[[VAL_3]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_65:.*]] = krnl.load %[[VAL_37]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_66:.*]] = krnl.load %[[VAL_38]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_67:.*]] = addf %[[VAL_65]], %[[VAL_66]] : f32
// CHECK:               %[[VAL_68:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_67]], %[[VAL_68]][] : memref<f32>
// CHECK:               %[[VAL_69:.*]] = "onnx.Sigmoid"(%[[VAL_68]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_70:.*]] = krnl.load %[[VAL_69]][] : memref<f32>
// CHECK:               %[[VAL_71:.*]] = krnl.load %[[VAL_41]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_72:.*]] = krnl.load %[[VAL_59]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_73:.*]] = addf %[[VAL_71]], %[[VAL_72]] : f32
// CHECK:               %[[VAL_74:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_73]], %[[VAL_74]][] : memref<f32>
// CHECK:               %[[VAL_75:.*]] = "onnx.Tanh"(%[[VAL_74]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_76:.*]] = krnl.load %[[VAL_75]][] : memref<f32>
// CHECK:               %[[VAL_77:.*]] = subf %[[VAL_42]], %[[VAL_70]] : f32
// CHECK:               %[[VAL_78:.*]] = mulf %[[VAL_77]], %[[VAL_76]] : f32
// CHECK:               %[[VAL_79:.*]] = mulf %[[VAL_70]], %[[VAL_64]] : f32
// CHECK:               %[[VAL_80:.*]] = addf %[[VAL_78]], %[[VAL_79]] : f32
// CHECK:               krnl.store %[[VAL_80]], %[[VAL_3]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<3x3xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_25]] : memref<3x3xf32>
// CHECK:             memref.dealloc %[[VAL_24]] : memref<3x3xf32>
// CHECK:             memref.dealloc %[[VAL_26]] : memref<3x2xf32>
// CHECK:           }
// CHECK:           %[[VAL_81:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_4]], %[[VAL_3]], %[[VAL_81]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
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
// CHECK:             %[[VAL_35:.*]] = "onnx.MatMul"(%[[VAL_24]], %[[VAL_15]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_36:.*]] = "onnx.MatMul"(%[[VAL_3]], %[[VAL_19]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_37:.*]] = "onnx.MatMul"(%[[VAL_24]], %[[VAL_16]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_38:.*]] = "onnx.MatMul"(%[[VAL_3]], %[[VAL_20]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_39:.*]] = "onnx.MatMul"(%[[VAL_24]], %[[VAL_17]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_40:.*]] = constant 1.000000e+00 : f32
// CHECK:             %[[VAL_41:.*]] = "onnx.MatMul"(%[[VAL_3]], %[[VAL_21]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_42:.*]] = constant 3 : index
// CHECK:             %[[VAL_43:.*]] = constant 3 : index
// CHECK:             %[[VAL_44:.*]] = constant 0 : index
// CHECK:             %[[VAL_45:.*]] = constant 0 : index
// CHECK:             %[[VAL_46:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_46]]#0, %[[VAL_46]]#1) with (%[[VAL_46]]#0 -> %[[VAL_47:.*]] = %[[VAL_44]] to %[[VAL_42]], %[[VAL_46]]#1 -> %[[VAL_48:.*]] = %[[VAL_45]] to %[[VAL_43]]) {
// CHECK:               %[[VAL_49:.*]]:2 = krnl.get_induction_var_value(%[[VAL_46]]#0, %[[VAL_46]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_50:.*]] = krnl.load %[[VAL_3]]{{\[}}%[[VAL_49]]#0, %[[VAL_49]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_51:.*]] = krnl.load %[[VAL_35]]{{\[}}%[[VAL_49]]#0, %[[VAL_49]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_52:.*]] = krnl.load %[[VAL_36]]{{\[}}%[[VAL_49]]#0, %[[VAL_49]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_53:.*]] = addf %[[VAL_51]], %[[VAL_52]] : f32
// CHECK:               %[[VAL_54:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_53]], %[[VAL_54]][] : memref<f32>
// CHECK:               %[[VAL_55:.*]] = "onnx.Sigmoid"(%[[VAL_54]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_56:.*]] = krnl.load %[[VAL_55]][] : memref<f32>
// CHECK:               %[[VAL_57:.*]] = krnl.load %[[VAL_37]]{{\[}}%[[VAL_49]]#0, %[[VAL_49]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_58:.*]] = krnl.load %[[VAL_38]]{{\[}}%[[VAL_49]]#0, %[[VAL_49]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_59:.*]] = addf %[[VAL_57]], %[[VAL_58]] : f32
// CHECK:               %[[VAL_60:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_59]], %[[VAL_60]][] : memref<f32>
// CHECK:               %[[VAL_61:.*]] = "onnx.Sigmoid"(%[[VAL_60]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_62:.*]] = krnl.load %[[VAL_61]][] : memref<f32>
// CHECK:               %[[VAL_63:.*]] = krnl.load %[[VAL_39]]{{\[}}%[[VAL_49]]#0, %[[VAL_49]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_64:.*]] = krnl.load %[[VAL_41]]{{\[}}%[[VAL_49]]#0, %[[VAL_49]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_65:.*]] = mulf %[[VAL_62]], %[[VAL_64]] : f32
// CHECK:               %[[VAL_66:.*]] = addf %[[VAL_63]], %[[VAL_65]] : f32
// CHECK:               %[[VAL_67:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_66]], %[[VAL_67]][] : memref<f32>
// CHECK:               %[[VAL_68:.*]] = "onnx.Tanh"(%[[VAL_67]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_69:.*]] = krnl.load %[[VAL_68]][] : memref<f32>
// CHECK:               %[[VAL_70:.*]] = subf %[[VAL_40]], %[[VAL_56]] : f32
// CHECK:               %[[VAL_71:.*]] = mulf %[[VAL_70]], %[[VAL_69]] : f32
// CHECK:               %[[VAL_72:.*]] = mulf %[[VAL_56]], %[[VAL_50]] : f32
// CHECK:               %[[VAL_73:.*]] = addf %[[VAL_71]], %[[VAL_72]] : f32
// CHECK:               krnl.store %[[VAL_73]], %[[VAL_3]]{{\[}}%[[VAL_49]]#0, %[[VAL_49]]#1] : memref<3x3xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_24]] : memref<3x2xf32>
// CHECK:           }
// CHECK:           %[[VAL_74:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_4]], %[[VAL_3]], %[[VAL_74]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
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
// CHECK:             %[[VAL_27:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:             %[[VAL_28:.*]] = memref.alloc() : memref<3x3xf32>
// CHECK:             %[[VAL_29:.*]] = memref.alloc() : memref<3x2xf32>
// CHECK:             %[[VAL_30:.*]] = constant 0 : index
// CHECK:             %[[VAL_31:.*]] = constant 3 : index
// CHECK:             %[[VAL_32:.*]] = constant 2 : index
// CHECK:             %[[VAL_33:.*]] = constant 0 : index
// CHECK:             %[[VAL_34:.*]] = constant 0 : index
// CHECK:             %[[VAL_35:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_35]]#0, %[[VAL_35]]#1) with (%[[VAL_35]]#0 -> %[[VAL_36:.*]] = %[[VAL_33]] to %[[VAL_31]], %[[VAL_35]]#1 -> %[[VAL_37:.*]] = %[[VAL_34]] to %[[VAL_32]]) {
// CHECK:               %[[VAL_38:.*]]:2 = krnl.get_induction_var_value(%[[VAL_35]]#0, %[[VAL_35]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_39:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_26]], %[[VAL_38]]#0, %[[VAL_38]]#1] : memref<4x3x2xf32>
// CHECK:               krnl.store %[[VAL_39]], %[[VAL_29]]{{\[}}%[[VAL_38]]#0, %[[VAL_38]]#1] : memref<3x2xf32>
// CHECK:             }
// CHECK:             %[[VAL_40:.*]] = "onnx.MatMul"(%[[VAL_29]], %[[VAL_16]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_41:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_20]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_42:.*]] = "onnx.MatMul"(%[[VAL_29]], %[[VAL_17]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_43:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_21]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_44:.*]] = "onnx.MatMul"(%[[VAL_29]], %[[VAL_18]]) : (memref<3x2xf32>, memref<2x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_45:.*]] = constant 1.000000e+00 : f32
// CHECK:             %[[VAL_46:.*]] = constant 3 : index
// CHECK:             %[[VAL_47:.*]] = constant 3 : index
// CHECK:             %[[VAL_48:.*]] = constant 0 : index
// CHECK:             %[[VAL_49:.*]] = constant 0 : index
// CHECK:             %[[VAL_50:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_50]]#0, %[[VAL_50]]#1) with (%[[VAL_50]]#0 -> %[[VAL_51:.*]] = %[[VAL_48]] to %[[VAL_46]], %[[VAL_50]]#1 -> %[[VAL_52:.*]] = %[[VAL_49]] to %[[VAL_47]]) {
// CHECK:               %[[VAL_53:.*]]:2 = krnl.get_induction_var_value(%[[VAL_50]]#0, %[[VAL_50]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_54:.*]] = krnl.load %[[VAL_4]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_55:.*]] = krnl.load %[[VAL_42]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_56:.*]] = krnl.load %[[VAL_43]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_57:.*]] = addf %[[VAL_55]], %[[VAL_56]] : f32
// CHECK:               %[[VAL_58:.*]] = krnl.load %[[VAL_24]]#1{{\[}}%[[VAL_53]]#1] : memref<3xf32>
// CHECK:               %[[VAL_59:.*]] = krnl.load %[[VAL_24]]#4{{\[}}%[[VAL_53]]#1] : memref<3xf32>
// CHECK:               %[[VAL_60:.*]] = addf %[[VAL_57]], %[[VAL_58]] : f32
// CHECK:               %[[VAL_61:.*]] = addf %[[VAL_60]], %[[VAL_59]] : f32
// CHECK:               %[[VAL_62:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_61]], %[[VAL_62]][] : memref<f32>
// CHECK:               %[[VAL_63:.*]] = "onnx.Sigmoid"(%[[VAL_62]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_64:.*]] = krnl.load %[[VAL_63]][] : memref<f32>
// CHECK:               krnl.store %[[VAL_64]], %[[VAL_28]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_65:.*]] = mulf %[[VAL_64]], %[[VAL_54]] : f32
// CHECK:               krnl.store %[[VAL_65]], %[[VAL_27]]{{\[}}%[[VAL_53]]#0, %[[VAL_53]]#1] : memref<3x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_66:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_22]]) : (memref<3x3xf32>, memref<3x3xf32>) -> memref<3x3xf32>
// CHECK:             %[[VAL_67:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_67]]#0, %[[VAL_67]]#1) with (%[[VAL_67]]#0 -> %[[VAL_68:.*]] = %[[VAL_48]] to %[[VAL_46]], %[[VAL_67]]#1 -> %[[VAL_69:.*]] = %[[VAL_49]] to %[[VAL_47]]) {
// CHECK:               %[[VAL_70:.*]]:2 = krnl.get_induction_var_value(%[[VAL_67]]#0, %[[VAL_67]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_71:.*]] = krnl.load %[[VAL_4]]{{\[}}%[[VAL_70]]#0, %[[VAL_70]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_72:.*]] = krnl.load %[[VAL_40]]{{\[}}%[[VAL_70]]#0, %[[VAL_70]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_73:.*]] = krnl.load %[[VAL_41]]{{\[}}%[[VAL_70]]#0, %[[VAL_70]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_74:.*]] = addf %[[VAL_72]], %[[VAL_73]] : f32
// CHECK:               %[[VAL_75:.*]] = krnl.load %[[VAL_24]]#0{{\[}}%[[VAL_70]]#1] : memref<3xf32>
// CHECK:               %[[VAL_76:.*]] = krnl.load %[[VAL_24]]#3{{\[}}%[[VAL_70]]#1] : memref<3xf32>
// CHECK:               %[[VAL_77:.*]] = addf %[[VAL_74]], %[[VAL_75]] : f32
// CHECK:               %[[VAL_78:.*]] = addf %[[VAL_77]], %[[VAL_76]] : f32
// CHECK:               %[[VAL_79:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_78]], %[[VAL_79]][] : memref<f32>
// CHECK:               %[[VAL_80:.*]] = "onnx.Sigmoid"(%[[VAL_79]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_81:.*]] = krnl.load %[[VAL_80]][] : memref<f32>
// CHECK:               %[[VAL_82:.*]] = krnl.load %[[VAL_44]]{{\[}}%[[VAL_70]]#0, %[[VAL_70]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_83:.*]] = krnl.load %[[VAL_66]]{{\[}}%[[VAL_70]]#0, %[[VAL_70]]#1] : memref<3x3xf32>
// CHECK:               %[[VAL_84:.*]] = addf %[[VAL_82]], %[[VAL_83]] : f32
// CHECK:               %[[VAL_85:.*]] = krnl.load %[[VAL_24]]#2{{\[}}%[[VAL_70]]#1] : memref<3xf32>
// CHECK:               %[[VAL_86:.*]] = krnl.load %[[VAL_24]]#5{{\[}}%[[VAL_70]]#1] : memref<3xf32>
// CHECK:               %[[VAL_87:.*]] = addf %[[VAL_84]], %[[VAL_85]] : f32
// CHECK:               %[[VAL_88:.*]] = addf %[[VAL_87]], %[[VAL_86]] : f32
// CHECK:               %[[VAL_89:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_88]], %[[VAL_89]][] : memref<f32>
// CHECK:               %[[VAL_90:.*]] = "onnx.Tanh"(%[[VAL_89]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_91:.*]] = krnl.load %[[VAL_90]][] : memref<f32>
// CHECK:               %[[VAL_92:.*]] = subf %[[VAL_45]], %[[VAL_81]] : f32
// CHECK:               %[[VAL_93:.*]] = mulf %[[VAL_92]], %[[VAL_91]] : f32
// CHECK:               %[[VAL_94:.*]] = mulf %[[VAL_81]], %[[VAL_71]] : f32
// CHECK:               %[[VAL_95:.*]] = addf %[[VAL_93]], %[[VAL_94]] : f32
// CHECK:               krnl.store %[[VAL_95]], %[[VAL_4]]{{\[}}%[[VAL_70]]#0, %[[VAL_70]]#1] : memref<3x3xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_28]] : memref<3x3xf32>
// CHECK:             memref.dealloc %[[VAL_27]] : memref<3x3xf32>
// CHECK:             memref.dealloc %[[VAL_29]] : memref<3x2xf32>
// CHECK:           }
// CHECK:           %[[VAL_96:.*]] = constant 36 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_5]], %[[VAL_4]], %[[VAL_96]]) : (memref<1x3x3xf32>, memref<3x3xf32>, i64) -> ()
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
// CHECK:             %[[VAL_49:.*]] = "onnx.MatMul"(%[[VAL_37]], %[[VAL_21]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_50:.*]] = "onnx.MatMul"(%[[VAL_9]], %[[VAL_25]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_51:.*]] = "onnx.MatMul"(%[[VAL_37]], %[[VAL_22]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_52:.*]] = "onnx.MatMul"(%[[VAL_9]], %[[VAL_26]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_53:.*]] = "onnx.MatMul"(%[[VAL_37]], %[[VAL_23]]) : (memref<?x?xf32>, memref<?x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_54:.*]] = constant 1.000000e+00 : f32
// CHECK:             %[[VAL_55:.*]] = constant 0 : index
// CHECK:             %[[VAL_56:.*]] = memref.dim %[[VAL_9]], %[[VAL_55]] : memref<?x3xf32>
// CHECK:             %[[VAL_57:.*]] = memref.alloc(%[[VAL_56]]) : memref<?x3xf32>
// CHECK:             %[[VAL_58:.*]] = memref.alloc(%[[VAL_56]]) : memref<?x3xf32>
// CHECK:             %[[VAL_59:.*]] = constant 0 : index
// CHECK:             %[[VAL_60:.*]] = memref.dim %[[VAL_9]], %[[VAL_59]] : memref<?x3xf32>
// CHECK:             %[[VAL_61:.*]] = constant 3 : index
// CHECK:             %[[VAL_62:.*]] = constant 0 : index
// CHECK:             %[[VAL_63:.*]] = constant 0 : index
// CHECK:             %[[VAL_64:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_64]]#0, %[[VAL_64]]#1) with (%[[VAL_64]]#0 -> %[[VAL_65:.*]] = %[[VAL_62]] to %[[VAL_60]], %[[VAL_64]]#1 -> %[[VAL_66:.*]] = %[[VAL_63]] to %[[VAL_61]]) {
// CHECK:               %[[VAL_67:.*]]:2 = krnl.get_induction_var_value(%[[VAL_64]]#0, %[[VAL_64]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_68:.*]] = krnl.load %[[VAL_9]]{{\[}}%[[VAL_67]]#0, %[[VAL_67]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_69:.*]] = krnl.load %[[VAL_51]]{{\[}}%[[VAL_67]]#0, %[[VAL_67]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_70:.*]] = krnl.load %[[VAL_52]]{{\[}}%[[VAL_67]]#0, %[[VAL_67]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_71:.*]] = addf %[[VAL_69]], %[[VAL_70]] : f32
// CHECK:               %[[VAL_72:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_71]], %[[VAL_72]][] : memref<f32>
// CHECK:               %[[VAL_73:.*]] = "onnx.Sigmoid"(%[[VAL_72]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_74:.*]] = krnl.load %[[VAL_73]][] : memref<f32>
// CHECK:               krnl.store %[[VAL_74]], %[[VAL_57]]{{\[}}%[[VAL_67]]#0, %[[VAL_67]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_75:.*]] = mulf %[[VAL_74]], %[[VAL_68]] : f32
// CHECK:               krnl.store %[[VAL_75]], %[[VAL_58]]{{\[}}%[[VAL_67]]#0, %[[VAL_67]]#1] : memref<?x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_76:.*]] = "onnx.MatMul"(%[[VAL_58]], %[[VAL_27]]) : (memref<?x3xf32>, memref<3x3xf32>) -> memref<?x3xf32>
// CHECK:             %[[VAL_77:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_77]]#0, %[[VAL_77]]#1) with (%[[VAL_77]]#0 -> %[[VAL_78:.*]] = %[[VAL_62]] to %[[VAL_60]], %[[VAL_77]]#1 -> %[[VAL_79:.*]] = %[[VAL_63]] to %[[VAL_61]]) {
// CHECK:               %[[VAL_80:.*]]:2 = krnl.get_induction_var_value(%[[VAL_77]]#0, %[[VAL_77]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_81:.*]] = krnl.load %[[VAL_9]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_82:.*]] = krnl.load %[[VAL_49]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_83:.*]] = krnl.load %[[VAL_50]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_84:.*]] = addf %[[VAL_82]], %[[VAL_83]] : f32
// CHECK:               %[[VAL_85:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_84]], %[[VAL_85]][] : memref<f32>
// CHECK:               %[[VAL_86:.*]] = "onnx.Sigmoid"(%[[VAL_85]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_87:.*]] = krnl.load %[[VAL_86]][] : memref<f32>
// CHECK:               %[[VAL_88:.*]] = krnl.load %[[VAL_53]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_89:.*]] = krnl.load %[[VAL_76]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x3xf32>
// CHECK:               %[[VAL_90:.*]] = addf %[[VAL_88]], %[[VAL_89]] : f32
// CHECK:               %[[VAL_91:.*]] = memref.alloc() : memref<f32>
// CHECK:               krnl.store %[[VAL_90]], %[[VAL_91]][] : memref<f32>
// CHECK:               %[[VAL_92:.*]] = "onnx.Tanh"(%[[VAL_91]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_93:.*]] = krnl.load %[[VAL_92]][] : memref<f32>
// CHECK:               %[[VAL_94:.*]] = subf %[[VAL_54]], %[[VAL_87]] : f32
// CHECK:               %[[VAL_95:.*]] = mulf %[[VAL_94]], %[[VAL_93]] : f32
// CHECK:               %[[VAL_96:.*]] = mulf %[[VAL_87]], %[[VAL_81]] : f32
// CHECK:               %[[VAL_97:.*]] = addf %[[VAL_95]], %[[VAL_96]] : f32
// CHECK:               krnl.store %[[VAL_97]], %[[VAL_9]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x3xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_57]] : memref<?x3xf32>
// CHECK:             memref.dealloc %[[VAL_58]] : memref<?x3xf32>
// CHECK:             memref.dealloc %[[VAL_37]] : memref<?x?xf32>
// CHECK:           }
// CHECK:           %[[VAL_98:.*]] = constant 12 : i64
// CHECK:           %[[VAL_99:.*]] = constant 0 : index
// CHECK:           %[[VAL_100:.*]] = memref.dim %[[VAL_9]], %[[VAL_99]] : memref<?x3xf32>
// CHECK:           %[[VAL_101:.*]] = index_cast %[[VAL_100]] : index to i64
// CHECK:           %[[VAL_102:.*]] = muli %[[VAL_98]], %[[VAL_101]] : i64
// CHECK:           "krnl.memcpy"(%[[VAL_6]], %[[VAL_9]], %[[VAL_102]]) : (memref<1x?x3xf32>, memref<?x3xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_9]] : memref<?x3xf32>
// CHECK:           return %[[VAL_6]] : memref<1x?x3xf32>
// CHECK:         }

}
