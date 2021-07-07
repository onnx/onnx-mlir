// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='check-rnn-ops-lowering' %s -split-input-file | FileCheck %s

func private @test_lstm_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x16x3xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x2x4xf32>, %arg5: tensor<1x2x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_lstm_forward_mode(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<7x2x3xf32>, %[[VAL_1:.*]]: memref<1x16x3xf32>, %[[VAL_2:.*]]: memref<1x16x4xf32>, %[[VAL_3:.*]]: memref<1x32xf32>, %[[VAL_4:.*]]: memref<1x2x4xf32>, %[[VAL_5:.*]]: memref<1x2x4xf32>, %[[VAL_6:.*]]: memref<1x12xf32>) -> memref<1x2x4xf32> {
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_8:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<1x2x4xf32>
// CHECK:           %[[VAL_10:.*]] = constant unit
// CHECK:           %[[VAL_11:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_12:.*]] = constant 0 : index
// CHECK:           %[[VAL_13:.*]] = constant 1 : index
// CHECK:           %[[VAL_14:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_14]]#0, %[[VAL_14]]#1) with (%[[VAL_14]]#0 -> %[[VAL_15:.*]] = 0 to 2, %[[VAL_14]]#1 -> %[[VAL_16:.*]] = 0 to 4) {
// CHECK:             %[[VAL_17:.*]] = krnl.load %[[VAL_4]]{{\[}}%[[VAL_12]], %[[VAL_15]], %[[VAL_16]]] : memref<1x2x4xf32>
// CHECK:             krnl.store %[[VAL_17]], %[[VAL_8]]{{\[}}%[[VAL_15]], %[[VAL_16]]] : memref<2x4xf32>
// CHECK:             %[[VAL_18:.*]] = krnl.load %[[VAL_5]]{{\[}}%[[VAL_12]], %[[VAL_15]], %[[VAL_16]]] : memref<1x2x4xf32>
// CHECK:             krnl.store %[[VAL_18]], %[[VAL_7]]{{\[}}%[[VAL_15]], %[[VAL_16]]] : memref<2x4xf32>
// CHECK:           }
// CHECK:           %[[VAL_19:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x16x3xf32>) -> memref<16x3xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Squeeze"(%[[VAL_3]]) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK:           %[[VAL_22:.*]]:8 = "onnx.Split"(%[[VAL_21]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_23:.*]] = "onnx.Squeeze"(%[[VAL_6]]) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK:           %[[VAL_24:.*]]:3 = "onnx.Split"(%[[VAL_23]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_25:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_25]]) with (%[[VAL_25]] -> %[[VAL_26:.*]] = 0 to 7) {
// CHECK:             %[[VAL_27:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_28:.*]] = constant 0 : index
// CHECK:             %[[VAL_29:.*]] = constant 2 : index
// CHECK:             %[[VAL_30:.*]] = constant 3 : index
// CHECK:             %[[VAL_31:.*]] = constant 0 : index
// CHECK:             %[[VAL_32:.*]] = constant 0 : index
// CHECK:             %[[VAL_33:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_33]]#0, %[[VAL_33]]#1) with (%[[VAL_33]]#0 -> %[[VAL_34:.*]] = %[[VAL_31]] to %[[VAL_29]], %[[VAL_33]]#1 -> %[[VAL_35:.*]] = %[[VAL_32]] to %[[VAL_30]]) {
// CHECK:               %[[VAL_36:.*]]:2 = krnl.get_induction_var_value(%[[VAL_33]]#0, %[[VAL_33]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_37:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_26]], %[[VAL_36]]#0, %[[VAL_36]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_37]], %[[VAL_27]]{{\[}}%[[VAL_36]]#0, %[[VAL_36]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_38:.*]] = "onnx.Transpose"(%[[VAL_27]]) {perm = [1, 0]} : (memref<2x3xf32>) -> memref<3x2xf32>
// CHECK:             %[[VAL_39:.*]] = "onnx.MatMul"(%[[VAL_19]], %[[VAL_38]]) : (memref<16x3xf32>, memref<3x2xf32>) -> memref<16x2xf32>
// CHECK:             %[[VAL_40:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [1, 0]} : (memref<2x4xf32>) -> memref<4x2xf32>
// CHECK:             %[[VAL_41:.*]] = "onnx.MatMul"(%[[VAL_20]], %[[VAL_40]]) : (memref<16x4xf32>, memref<4x2xf32>) -> memref<16x2xf32>
// CHECK:             %[[VAL_42:.*]] = constant 2 : index
// CHECK:             %[[VAL_43:.*]] = constant 4 : index
// CHECK:             %[[VAL_44:.*]] = constant 0 : index
// CHECK:             %[[VAL_45:.*]] = constant 0 : index
// CHECK:             %[[VAL_46:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_46]]#0, %[[VAL_46]]#1) with (%[[VAL_46]]#0 -> %[[VAL_47:.*]] = %[[VAL_44]] to %[[VAL_42]], %[[VAL_46]]#1 -> %[[VAL_48:.*]] = %[[VAL_45]] to %[[VAL_43]]) {
// CHECK:               %[[VAL_49:.*]]:2 = krnl.get_induction_var_value(%[[VAL_46]]#0, %[[VAL_46]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_50:.*]] = krnl.load %[[VAL_7]]{{\[}}%[[VAL_49]]#0, %[[VAL_49]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_51:.*]] = krnl.load %[[VAL_39]]{{\[}}%[[VAL_49]]#1, %[[VAL_49]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_52:.*]] = krnl.load %[[VAL_41]]{{\[}}%[[VAL_49]]#1, %[[VAL_49]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_53:.*]] = addf %[[VAL_51]], %[[VAL_52]] : f32
// CHECK:               %[[VAL_54:.*]] = krnl.load %[[VAL_22]]#0{{\[}}%[[VAL_49]]#1] : memref<4xf32>
// CHECK:               %[[VAL_55:.*]] = krnl.load %[[VAL_22]]#4{{\[}}%[[VAL_49]]#1] : memref<4xf32>
// CHECK:               %[[VAL_56:.*]] = addf %[[VAL_53]], %[[VAL_54]] : f32
// CHECK:               %[[VAL_57:.*]] = addf %[[VAL_56]], %[[VAL_55]] : f32
// CHECK:               %[[VAL_58:.*]] = krnl.load %[[VAL_24]]#0{{\[}}%[[VAL_49]]#1] : memref<4xf32>
// CHECK:               %[[VAL_59:.*]] = mulf %[[VAL_58]], %[[VAL_50]] : f32
// CHECK:               %[[VAL_60:.*]] = addf %[[VAL_57]], %[[VAL_59]] : f32
// CHECK:               %[[VAL_61:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_60]], %[[VAL_61]][] : memref<f32>
// CHECK:               %[[VAL_62:.*]] = "onnx.Sigmoid"(%[[VAL_61]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_63:.*]] = krnl.load %[[VAL_62]][] : memref<f32>
// CHECK:               %[[VAL_64:.*]] = constant 8 : index
// CHECK:               %[[VAL_65:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_49]]#1]
// CHECK:               %[[VAL_66:.*]] = krnl.load %[[VAL_39]]{{\[}}%[[VAL_65]], %[[VAL_49]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_67:.*]] = constant 8 : index
// CHECK:               %[[VAL_68:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_49]]#1]
// CHECK:               %[[VAL_69:.*]] = krnl.load %[[VAL_41]]{{\[}}%[[VAL_68]], %[[VAL_49]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_70:.*]] = addf %[[VAL_66]], %[[VAL_69]] : f32
// CHECK:               %[[VAL_71:.*]] = krnl.load %[[VAL_22]]#2{{\[}}%[[VAL_49]]#1] : memref<4xf32>
// CHECK:               %[[VAL_72:.*]] = krnl.load %[[VAL_22]]#6{{\[}}%[[VAL_49]]#1] : memref<4xf32>
// CHECK:               %[[VAL_73:.*]] = addf %[[VAL_70]], %[[VAL_71]] : f32
// CHECK:               %[[VAL_74:.*]] = addf %[[VAL_73]], %[[VAL_72]] : f32
// CHECK:               %[[VAL_75:.*]] = krnl.load %[[VAL_24]]#2{{\[}}%[[VAL_49]]#1] : memref<4xf32>
// CHECK:               %[[VAL_76:.*]] = mulf %[[VAL_75]], %[[VAL_50]] : f32
// CHECK:               %[[VAL_77:.*]] = addf %[[VAL_74]], %[[VAL_76]] : f32
// CHECK:               %[[VAL_78:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_77]], %[[VAL_78]][] : memref<f32>
// CHECK:               %[[VAL_79:.*]] = "onnx.Sigmoid"(%[[VAL_78]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_80:.*]] = krnl.load %[[VAL_79]][] : memref<f32>
// CHECK:               %[[VAL_81:.*]] = constant 12 : index
// CHECK:               %[[VAL_82:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_49]]#1]
// CHECK:               %[[VAL_83:.*]] = krnl.load %[[VAL_39]]{{\[}}%[[VAL_82]], %[[VAL_49]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_84:.*]] = constant 12 : index
// CHECK:               %[[VAL_85:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_49]]#1]
// CHECK:               %[[VAL_86:.*]] = krnl.load %[[VAL_41]]{{\[}}%[[VAL_85]], %[[VAL_49]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_87:.*]] = addf %[[VAL_83]], %[[VAL_86]] : f32
// CHECK:               %[[VAL_88:.*]] = krnl.load %[[VAL_22]]#3{{\[}}%[[VAL_49]]#1] : memref<4xf32>
// CHECK:               %[[VAL_89:.*]] = krnl.load %[[VAL_22]]#7{{\[}}%[[VAL_49]]#1] : memref<4xf32>
// CHECK:               %[[VAL_90:.*]] = addf %[[VAL_87]], %[[VAL_88]] : f32
// CHECK:               %[[VAL_91:.*]] = addf %[[VAL_90]], %[[VAL_89]] : f32
// CHECK:               %[[VAL_92:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_91]], %[[VAL_92]][] : memref<f32>
// CHECK:               %[[VAL_93:.*]] = "onnx.Tanh"(%[[VAL_92]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_94:.*]] = krnl.load %[[VAL_93]][] : memref<f32>
// CHECK:               %[[VAL_95:.*]] = mulf %[[VAL_80]], %[[VAL_50]] : f32
// CHECK:               %[[VAL_96:.*]] = mulf %[[VAL_63]], %[[VAL_94]] : f32
// CHECK:               %[[VAL_97:.*]] = addf %[[VAL_95]], %[[VAL_96]] : f32
// CHECK:               %[[VAL_98:.*]] = constant 4 : index
// CHECK:               %[[VAL_99:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_49]]#1]
// CHECK:               %[[VAL_100:.*]] = krnl.load %[[VAL_39]]{{\[}}%[[VAL_99]], %[[VAL_49]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_101:.*]] = constant 4 : index
// CHECK:               %[[VAL_102:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_49]]#1]
// CHECK:               %[[VAL_103:.*]] = krnl.load %[[VAL_41]]{{\[}}%[[VAL_102]], %[[VAL_49]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_104:.*]] = addf %[[VAL_100]], %[[VAL_103]] : f32
// CHECK:               %[[VAL_105:.*]] = krnl.load %[[VAL_22]]#1{{\[}}%[[VAL_49]]#1] : memref<4xf32>
// CHECK:               %[[VAL_106:.*]] = krnl.load %[[VAL_22]]#5{{\[}}%[[VAL_49]]#1] : memref<4xf32>
// CHECK:               %[[VAL_107:.*]] = addf %[[VAL_104]], %[[VAL_105]] : f32
// CHECK:               %[[VAL_108:.*]] = addf %[[VAL_107]], %[[VAL_106]] : f32
// CHECK:               %[[VAL_109:.*]] = krnl.load %[[VAL_24]]#1{{\[}}%[[VAL_49]]#1] : memref<4xf32>
// CHECK:               %[[VAL_110:.*]] = mulf %[[VAL_109]], %[[VAL_97]] : f32
// CHECK:               %[[VAL_111:.*]] = addf %[[VAL_108]], %[[VAL_110]] : f32
// CHECK:               %[[VAL_112:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_111]], %[[VAL_112]][] : memref<f32>
// CHECK:               %[[VAL_113:.*]] = "onnx.Sigmoid"(%[[VAL_112]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_114:.*]] = krnl.load %[[VAL_113]][] : memref<f32>
// CHECK:               %[[VAL_115:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_97]], %[[VAL_115]][] : memref<f32>
// CHECK:               %[[VAL_116:.*]] = "onnx.Tanh"(%[[VAL_115]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_117:.*]] = krnl.load %[[VAL_116]][] : memref<f32>
// CHECK:               %[[VAL_118:.*]] = mulf %[[VAL_114]], %[[VAL_117]] : f32
// CHECK:               krnl.store %[[VAL_97]], %[[VAL_7]]{{\[}}%[[VAL_49]]#0, %[[VAL_49]]#1] : memref<2x4xf32>
// CHECK:               krnl.store %[[VAL_118]], %[[VAL_8]]{{\[}}%[[VAL_49]]#0, %[[VAL_49]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_27]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_119:.*]] = constant 32 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_9]], %[[VAL_8]], %[[VAL_119]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_8]] : memref<2x4xf32>
// CHECK:           memref.dealloc %[[VAL_7]] : memref<2x4xf32>
// CHECK:           return %[[VAL_9]] : memref<1x2x4xf32>
// CHECK:         }

}

// -----

func private @test_lstm_forward_mode_constant_weight_and_bias(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x2x4xf32>, %arg2: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %w = "onnx.Constant"() {value = dense<[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]> : tensor<1x16x3xf32>} : () -> tensor<1x16x3xf32> 
  %r = "onnx.Constant"() {value = dense<[[[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]]]> : tensor<1x16x4xf32>} : () -> tensor<1x16x4xf32> 
  %b = "onnx.Constant"() {value = dense<[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24., 25., 26., 27., 28., 29., 30., 31., 32.]]> : tensor<1x32xf32>} : () -> tensor<1x32xf32> 
  %p = "onnx.Constant"() {value = dense<[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12.]]> : tensor<1x12xf32>} : () -> tensor<1x12xf32> 

  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %w, %r, %b, %cst, %arg1, %arg2, %p) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_lstm_forward_mode_constant_weight_and_bias(
// CHECK-SAME:                                                                  %[[VAL_0:.*]]: memref<7x2x3xf32>,
// CHECK-SAME:                                                                  %[[VAL_1:.*]]: memref<1x2x4xf32>,
// CHECK-SAME:                                                                  %[[VAL_2:.*]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_4:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_5:.*]] = memref.alloc() : memref<1x2x4xf32>
// CHECK:           %[[VAL_6:.*]] = constant unit
// CHECK:           %[[VAL_7:.*]] = "krnl.global"() {name = "constant_0", shape = [1, 16, 3], value = dense<1.000000e+00> : tensor<1x16x3xf32>} : () -> memref<1x16x3xf32>
// CHECK:           %[[VAL_8:.*]] = "krnl.global"() {name = "constant_1", shape = [1, 16, 4], value = dense<2.000000e+00> : tensor<1x16x4xf32>} : () -> memref<1x16x4xf32>
// CHECK:           %[[VAL_9:.*]] = "krnl.global"() {name = "constant_2", shape = [1, 32], value = dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01]]> : tensor<1x32xf32>} : () -> memref<1x32xf32>
// CHECK:           %[[VAL_10:.*]] = "krnl.global"() {name = "constant_3", shape = [1, 12], value = dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]]> : tensor<1x12xf32>} : () -> memref<1x12xf32>
// CHECK:           %[[VAL_11:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_12:.*]] = constant 0 : index
// CHECK:           %[[VAL_13:.*]] = constant 1 : index
// CHECK:           %[[VAL_14:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_14]]#0, %[[VAL_14]]#1) with (%[[VAL_14]]#0 -> %[[VAL_15:.*]] = 0 to 2, %[[VAL_14]]#1 -> %[[VAL_16:.*]] = 0 to 4) {
// CHECK:             %[[VAL_17:.*]] = krnl.load %[[VAL_1]]{{\[}}%[[VAL_12]], %[[VAL_15]], %[[VAL_16]]] : memref<1x2x4xf32>
// CHECK:             krnl.store %[[VAL_17]], %[[VAL_4]]{{\[}}%[[VAL_15]], %[[VAL_16]]] : memref<2x4xf32>
// CHECK:             %[[VAL_18:.*]] = krnl.load %[[VAL_2]]{{\[}}%[[VAL_12]], %[[VAL_15]], %[[VAL_16]]] : memref<1x2x4xf32>
// CHECK:             krnl.store %[[VAL_18]], %[[VAL_3]]{{\[}}%[[VAL_15]], %[[VAL_16]]] : memref<2x4xf32>
// CHECK:           }
// CHECK:           %[[VAL_19:.*]] = "krnl.global"() {name = "constant_4", shape = [16, 3], value = dense<1.000000e+00> : tensor<16x3xf32>} : () -> memref<16x3xf32>
// CHECK:           %[[VAL_20:.*]] = "krnl.global"() {name = "constant_5", shape = [16, 4], value = dense<2.000000e+00> : tensor<16x4xf32>} : () -> memref<16x4xf32>
// CHECK:           %[[VAL_21:.*]] = "krnl.global"() {name = "constant_6", shape = [32], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01]> : tensor<32xf32>} : () -> memref<32xf32>
// CHECK:           %[[VAL_22:.*]] = "krnl.global"() {name = "constant_7", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_23:.*]] = "krnl.global"() {name = "constant_8", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_24:.*]] = "krnl.global"() {name = "constant_9", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_25:.*]] = "krnl.global"() {name = "constant_10", shape = [4], value = dense<[1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_26:.*]] = "krnl.global"() {name = "constant_11", shape = [4], value = dense<[1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_27:.*]] = "krnl.global"() {name = "constant_12", shape = [4], value = dense<[2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_28:.*]] = "krnl.global"() {name = "constant_13", shape = [4], value = dense<[2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_29:.*]] = "krnl.global"() {name = "constant_14", shape = [4], value = dense<[2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_30:.*]] = "krnl.global"() {name = "constant_15", shape = [12], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<12xf32>} : () -> memref<12xf32>
// CHECK:           %[[VAL_31:.*]] = "krnl.global"() {name = "constant_16", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_32:.*]] = "krnl.global"() {name = "constant_17", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_33:.*]] = "krnl.global"() {name = "constant_18", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_34:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_34]]) with (%[[VAL_34]] -> %[[VAL_35:.*]] = 0 to 7) {
// CHECK:             %[[VAL_36:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_37:.*]] = constant 0 : index
// CHECK:             %[[VAL_38:.*]] = constant 2 : index
// CHECK:             %[[VAL_39:.*]] = constant 3 : index
// CHECK:             %[[VAL_40:.*]] = constant 0 : index
// CHECK:             %[[VAL_41:.*]] = constant 0 : index
// CHECK:             %[[VAL_42:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_42]]#0, %[[VAL_42]]#1) with (%[[VAL_42]]#0 -> %[[VAL_43:.*]] = %[[VAL_40]] to %[[VAL_38]], %[[VAL_42]]#1 -> %[[VAL_44:.*]] = %[[VAL_41]] to %[[VAL_39]]) {
// CHECK:               %[[VAL_45:.*]]:2 = krnl.get_induction_var_value(%[[VAL_42]]#0, %[[VAL_42]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_46:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_35]], %[[VAL_45]]#0, %[[VAL_45]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_46]], %[[VAL_36]]{{\[}}%[[VAL_45]]#0, %[[VAL_45]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_47:.*]] = "onnx.Transpose"(%[[VAL_36]]) {perm = [1, 0]} : (memref<2x3xf32>) -> memref<3x2xf32>
// CHECK:             %[[VAL_48:.*]] = "onnx.MatMul"(%[[VAL_19]], %[[VAL_47]]) : (memref<16x3xf32>, memref<3x2xf32>) -> memref<16x2xf32>
// CHECK:             %[[VAL_49:.*]] = "onnx.Transpose"(%[[VAL_4]]) {perm = [1, 0]} : (memref<2x4xf32>) -> memref<4x2xf32>
// CHECK:             %[[VAL_50:.*]] = "onnx.MatMul"(%[[VAL_20]], %[[VAL_49]]) : (memref<16x4xf32>, memref<4x2xf32>) -> memref<16x2xf32>
// CHECK:             %[[VAL_51:.*]] = constant 2 : index
// CHECK:             %[[VAL_52:.*]] = constant 4 : index
// CHECK:             %[[VAL_53:.*]] = constant 0 : index
// CHECK:             %[[VAL_54:.*]] = constant 0 : index
// CHECK:             %[[VAL_55:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_55]]#0, %[[VAL_55]]#1) with (%[[VAL_55]]#0 -> %[[VAL_45:.*]] = %[[VAL_53]] to %[[VAL_51]], %[[VAL_55]]#1 -> %[[VAL_46:.*]] = %[[VAL_54]] to %[[VAL_52]]) {
// CHECK:               %[[VAL_56:.*]]:2 = krnl.get_induction_var_value(%[[VAL_55]]#0, %[[VAL_55]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_57:.*]] = krnl.load %[[VAL_3]]{{\[}}%[[VAL_56]]#0, %[[VAL_56]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_58:.*]] = krnl.load %[[VAL_48]]{{\[}}%[[VAL_56]]#1, %[[VAL_56]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_59:.*]] = krnl.load %[[VAL_50]]{{\[}}%[[VAL_56]]#1, %[[VAL_56]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_60:.*]] = addf %[[VAL_58]], %[[VAL_59]] : f32
// CHECK:               %[[VAL_61:.*]] = krnl.load %[[VAL_22]]{{\[}}%[[VAL_56]]#1] : memref<4xf32>
// CHECK:               %[[VAL_62:.*]] = krnl.load %[[VAL_26]]{{\[}}%[[VAL_56]]#1] : memref<4xf32>
// CHECK:               %[[VAL_63:.*]] = addf %[[VAL_60]], %[[VAL_61]] : f32
// CHECK:               %[[VAL_64:.*]] = addf %[[VAL_63]], %[[VAL_62]] : f32
// CHECK:               %[[VAL_65:.*]] = krnl.load %[[VAL_31]]{{\[}}%[[VAL_56]]#1] : memref<4xf32>
// CHECK:               %[[VAL_66:.*]] = mulf %[[VAL_65]], %[[VAL_57]] : f32
// CHECK:               %[[VAL_67:.*]] = addf %[[VAL_64]], %[[VAL_66]] : f32
// CHECK:               %[[VAL_68:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_67]], %[[VAL_68]][] : memref<f32>
// CHECK:               %[[VAL_69:.*]] = "onnx.Sigmoid"(%[[VAL_68]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_70:.*]] = krnl.load %[[VAL_69]][] : memref<f32>
// CHECK:               %[[VAL_71:.*]] = constant 8 : index
// CHECK:               %[[VAL_72:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_56]]#1]
// CHECK:               %[[VAL_73:.*]] = krnl.load %[[VAL_48]]{{\[}}%[[VAL_72]], %[[VAL_56]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_74:.*]] = constant 8 : index
// CHECK:               %[[VAL_75:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_56]]#1]
// CHECK:               %[[VAL_76:.*]] = krnl.load %[[VAL_50]]{{\[}}%[[VAL_75]], %[[VAL_56]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_77:.*]] = addf %[[VAL_73]], %[[VAL_76]] : f32
// CHECK:               %[[VAL_78:.*]] = krnl.load %[[VAL_24]]{{\[}}%[[VAL_56]]#1] : memref<4xf32>
// CHECK:               %[[VAL_79:.*]] = krnl.load %[[VAL_28]]{{\[}}%[[VAL_56]]#1] : memref<4xf32>
// CHECK:               %[[VAL_80:.*]] = addf %[[VAL_77]], %[[VAL_78]] : f32
// CHECK:               %[[VAL_81:.*]] = addf %[[VAL_80]], %[[VAL_79]] : f32
// CHECK:               %[[VAL_82:.*]] = krnl.load %[[VAL_33]]{{\[}}%[[VAL_56]]#1] : memref<4xf32>
// CHECK:               %[[VAL_83:.*]] = mulf %[[VAL_82]], %[[VAL_57]] : f32
// CHECK:               %[[VAL_84:.*]] = addf %[[VAL_81]], %[[VAL_83]] : f32
// CHECK:               %[[VAL_85:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_84]], %[[VAL_85]][] : memref<f32>
// CHECK:               %[[VAL_86:.*]] = "onnx.Sigmoid"(%[[VAL_85]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_87:.*]] = krnl.load %[[VAL_86]][] : memref<f32>
// CHECK:               %[[VAL_88:.*]] = constant 12 : index
// CHECK:               %[[VAL_89:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_56]]#1]
// CHECK:               %[[VAL_90:.*]] = krnl.load %[[VAL_48]]{{\[}}%[[VAL_89]], %[[VAL_56]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_91:.*]] = constant 12 : index
// CHECK:               %[[VAL_92:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_56]]#1]
// CHECK:               %[[VAL_93:.*]] = krnl.load %[[VAL_50]]{{\[}}%[[VAL_92]], %[[VAL_56]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_94:.*]] = addf %[[VAL_90]], %[[VAL_93]] : f32
// CHECK:               %[[VAL_95:.*]] = krnl.load %[[VAL_25]]{{\[}}%[[VAL_56]]#1] : memref<4xf32>
// CHECK:               %[[VAL_96:.*]] = krnl.load %[[VAL_29]]{{\[}}%[[VAL_56]]#1] : memref<4xf32>
// CHECK:               %[[VAL_97:.*]] = addf %[[VAL_94]], %[[VAL_95]] : f32
// CHECK:               %[[VAL_98:.*]] = addf %[[VAL_97]], %[[VAL_96]] : f32
// CHECK:               %[[VAL_99:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_98]], %[[VAL_99]][] : memref<f32>
// CHECK:               %[[VAL_100:.*]] = "onnx.Tanh"(%[[VAL_99]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_101:.*]] = krnl.load %[[VAL_100]][] : memref<f32>
// CHECK:               %[[VAL_102:.*]] = mulf %[[VAL_87]], %[[VAL_57]] : f32
// CHECK:               %[[VAL_103:.*]] = mulf %[[VAL_70]], %[[VAL_101]] : f32
// CHECK:               %[[VAL_104:.*]] = addf %[[VAL_102]], %[[VAL_103]] : f32
// CHECK:               %[[VAL_105:.*]] = constant 4 : index
// CHECK:               %[[VAL_106:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_56]]#1]
// CHECK:               %[[VAL_107:.*]] = krnl.load %[[VAL_48]]{{\[}}%[[VAL_106]], %[[VAL_56]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_108:.*]] = constant 4 : index
// CHECK:               %[[VAL_109:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_56]]#1]
// CHECK:               %[[VAL_110:.*]] = krnl.load %[[VAL_50]]{{\[}}%[[VAL_109]], %[[VAL_56]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_111:.*]] = addf %[[VAL_107]], %[[VAL_110]] : f32
// CHECK:               %[[VAL_112:.*]] = krnl.load %[[VAL_23]]{{\[}}%[[VAL_56]]#1] : memref<4xf32>
// CHECK:               %[[VAL_113:.*]] = krnl.load %[[VAL_27]]{{\[}}%[[VAL_56]]#1] : memref<4xf32>
// CHECK:               %[[VAL_114:.*]] = addf %[[VAL_111]], %[[VAL_112]] : f32
// CHECK:               %[[VAL_115:.*]] = addf %[[VAL_114]], %[[VAL_113]] : f32
// CHECK:               %[[VAL_116:.*]] = krnl.load %[[VAL_32]]{{\[}}%[[VAL_56]]#1] : memref<4xf32>
// CHECK:               %[[VAL_117:.*]] = mulf %[[VAL_116]], %[[VAL_104]] : f32
// CHECK:               %[[VAL_118:.*]] = addf %[[VAL_115]], %[[VAL_117]] : f32
// CHECK:               %[[VAL_119:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_118]], %[[VAL_119]][] : memref<f32>
// CHECK:               %[[VAL_120:.*]] = "onnx.Sigmoid"(%[[VAL_119]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_121:.*]] = krnl.load %[[VAL_120]][] : memref<f32>
// CHECK:               %[[VAL_122:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_104]], %[[VAL_122]][] : memref<f32>
// CHECK:               %[[VAL_123:.*]] = "onnx.Tanh"(%[[VAL_122]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_124:.*]] = krnl.load %[[VAL_123]][] : memref<f32>
// CHECK:               %[[VAL_125:.*]] = mulf %[[VAL_121]], %[[VAL_124]] : f32
// CHECK:               krnl.store %[[VAL_104]], %[[VAL_3]]{{\[}}%[[VAL_56]]#0, %[[VAL_56]]#1] : memref<2x4xf32>
// CHECK:               krnl.store %[[VAL_125]], %[[VAL_4]]{{\[}}%[[VAL_56]]#0, %[[VAL_56]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_36]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_126:.*]] = constant 32 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_5]], %[[VAL_4]], %[[VAL_126]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_4]] : memref<2x4xf32>
// CHECK:           memref.dealloc %[[VAL_3]] : memref<2x4xf32>
// CHECK:           return %[[VAL_5]] : memref<1x2x4xf32>
// CHECK:         }

}

// -----

func private @test_lstm_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x16x3xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x2x4xf32>, %arg5: tensor<1x2x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x16x3xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x2x4xf32>, tensor<1x2x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_lstm_reverse_mode(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<7x2x3xf32>, %[[VAL_1:.*]]: memref<1x16x3xf32>, %[[VAL_2:.*]]: memref<1x16x4xf32>, %[[VAL_3:.*]]: memref<1x32xf32>, %[[VAL_4:.*]]: memref<1x2x4xf32>, %[[VAL_5:.*]]: memref<1x2x4xf32>, %[[VAL_6:.*]]: memref<1x12xf32>) -> memref<1x2x4xf32> {
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_8:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<1x2x4xf32>
// CHECK:           %[[VAL_10:.*]] = constant unit
// CHECK:           %[[VAL_11:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_12:.*]] = constant 0 : index
// CHECK:           %[[VAL_13:.*]] = constant 1 : index
// CHECK:           %[[VAL_14:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_14]]#0, %[[VAL_14]]#1) with (%[[VAL_14]]#0 -> %[[VAL_15:.*]] = 0 to 2, %[[VAL_14]]#1 -> %[[VAL_16:.*]] = 0 to 4) {
// CHECK:             %[[VAL_17:.*]] = krnl.load %[[VAL_4]]{{\[}}%[[VAL_12]], %[[VAL_15]], %[[VAL_16]]] : memref<1x2x4xf32>
// CHECK:             krnl.store %[[VAL_17]], %[[VAL_8]]{{\[}}%[[VAL_15]], %[[VAL_16]]] : memref<2x4xf32>
// CHECK:             %[[VAL_18:.*]] = krnl.load %[[VAL_5]]{{\[}}%[[VAL_12]], %[[VAL_15]], %[[VAL_16]]] : memref<1x2x4xf32>
// CHECK:             krnl.store %[[VAL_18]], %[[VAL_7]]{{\[}}%[[VAL_15]], %[[VAL_16]]] : memref<2x4xf32>
// CHECK:           }
// CHECK:           %[[VAL_19:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x16x3xf32>) -> memref<16x3xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK:           %[[VAL_21:.*]] = "onnx.Squeeze"(%[[VAL_3]]) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK:           %[[VAL_22:.*]]:8 = "onnx.Split"(%[[VAL_21]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_23:.*]] = "onnx.Squeeze"(%[[VAL_6]]) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK:           %[[VAL_24:.*]]:3 = "onnx.Split"(%[[VAL_23]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_25:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_25]]) with (%[[VAL_25]] -> %[[VAL_26:.*]] = 0 to 7) {
// CHECK:             %[[VAL_27:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_28:.*]] = constant 0 : index
// CHECK:             %[[VAL_29:.*]] = constant 7 : index
// CHECK:             %[[VAL_30:.*]] = affine.apply #map{{.+}}(%[[VAL_26]]){{\[}}%[[VAL_29]]]
// CHECK:             %[[VAL_31:.*]] = constant 2 : index
// CHECK:             %[[VAL_32:.*]] = constant 3 : index
// CHECK:             %[[VAL_33:.*]] = constant 0 : index
// CHECK:             %[[VAL_34:.*]] = constant 0 : index
// CHECK:             %[[VAL_35:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_35]]#0, %[[VAL_35]]#1) with (%[[VAL_35]]#0 -> %[[VAL_36:.*]] = %[[VAL_33]] to %[[VAL_31]], %[[VAL_35]]#1 -> %[[VAL_37:.*]] = %[[VAL_34]] to %[[VAL_32]]) {
// CHECK:               %[[VAL_38:.*]]:2 = krnl.get_induction_var_value(%[[VAL_35]]#0, %[[VAL_35]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_39:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_30]], %[[VAL_38]]#0, %[[VAL_38]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_39]], %[[VAL_27]]{{\[}}%[[VAL_38]]#0, %[[VAL_38]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_40:.*]] = "onnx.Transpose"(%[[VAL_27]]) {perm = [1, 0]} : (memref<2x3xf32>) -> memref<3x2xf32>
// CHECK:             %[[VAL_41:.*]] = "onnx.MatMul"(%[[VAL_19]], %[[VAL_40]]) : (memref<16x3xf32>, memref<3x2xf32>) -> memref<16x2xf32>
// CHECK:             %[[VAL_42:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [1, 0]} : (memref<2x4xf32>) -> memref<4x2xf32>
// CHECK:             %[[VAL_43:.*]] = "onnx.MatMul"(%[[VAL_20]], %[[VAL_42]]) : (memref<16x4xf32>, memref<4x2xf32>) -> memref<16x2xf32>
// CHECK:             %[[VAL_44:.*]] = constant 2 : index
// CHECK:             %[[VAL_45:.*]] = constant 4 : index
// CHECK:             %[[VAL_46:.*]] = constant 0 : index
// CHECK:             %[[VAL_47:.*]] = constant 0 : index
// CHECK:             %[[VAL_48:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_48]]#0, %[[VAL_48]]#1) with (%[[VAL_48]]#0 -> %[[VAL_49:.*]] = %[[VAL_46]] to %[[VAL_44]], %[[VAL_48]]#1 -> %[[VAL_50:.*]] = %[[VAL_47]] to %[[VAL_45]]) {
// CHECK:               %[[VAL_51:.*]]:2 = krnl.get_induction_var_value(%[[VAL_48]]#0, %[[VAL_48]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_52:.*]] = krnl.load %[[VAL_7]]{{\[}}%[[VAL_51]]#0, %[[VAL_51]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_53:.*]] = krnl.load %[[VAL_41]]{{\[}}%[[VAL_51]]#1, %[[VAL_51]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_54:.*]] = krnl.load %[[VAL_43]]{{\[}}%[[VAL_51]]#1, %[[VAL_51]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_55:.*]] = addf %[[VAL_53]], %[[VAL_54]] : f32
// CHECK:               %[[VAL_56:.*]] = krnl.load %[[VAL_22]]#0{{\[}}%[[VAL_51]]#1] : memref<4xf32>
// CHECK:               %[[VAL_57:.*]] = krnl.load %[[VAL_22]]#4{{\[}}%[[VAL_51]]#1] : memref<4xf32>
// CHECK:               %[[VAL_58:.*]] = addf %[[VAL_55]], %[[VAL_56]] : f32
// CHECK:               %[[VAL_59:.*]] = addf %[[VAL_58]], %[[VAL_57]] : f32
// CHECK:               %[[VAL_60:.*]] = krnl.load %[[VAL_24]]#0{{\[}}%[[VAL_51]]#1] : memref<4xf32>
// CHECK:               %[[VAL_61:.*]] = mulf %[[VAL_60]], %[[VAL_52]] : f32
// CHECK:               %[[VAL_62:.*]] = addf %[[VAL_59]], %[[VAL_61]] : f32
// CHECK:               %[[VAL_63:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_62]], %[[VAL_63]][] : memref<f32>
// CHECK:               %[[VAL_64:.*]] = "onnx.Sigmoid"(%[[VAL_63]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_65:.*]] = krnl.load %[[VAL_64]][] : memref<f32>
// CHECK:               %[[VAL_66:.*]] = constant 8 : index
// CHECK:               %[[VAL_67:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_51]]#1]
// CHECK:               %[[VAL_68:.*]] = krnl.load %[[VAL_41]]{{\[}}%[[VAL_67]], %[[VAL_51]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_69:.*]] = constant 8 : index
// CHECK:               %[[VAL_70:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_51]]#1]
// CHECK:               %[[VAL_71:.*]] = krnl.load %[[VAL_43]]{{\[}}%[[VAL_70]], %[[VAL_51]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_72:.*]] = addf %[[VAL_68]], %[[VAL_71]] : f32
// CHECK:               %[[VAL_73:.*]] = krnl.load %[[VAL_22]]#2{{\[}}%[[VAL_51]]#1] : memref<4xf32>
// CHECK:               %[[VAL_74:.*]] = krnl.load %[[VAL_22]]#6{{\[}}%[[VAL_51]]#1] : memref<4xf32>
// CHECK:               %[[VAL_75:.*]] = addf %[[VAL_72]], %[[VAL_73]] : f32
// CHECK:               %[[VAL_76:.*]] = addf %[[VAL_75]], %[[VAL_74]] : f32
// CHECK:               %[[VAL_77:.*]] = krnl.load %[[VAL_24]]#2{{\[}}%[[VAL_51]]#1] : memref<4xf32>
// CHECK:               %[[VAL_78:.*]] = mulf %[[VAL_77]], %[[VAL_52]] : f32
// CHECK:               %[[VAL_79:.*]] = addf %[[VAL_76]], %[[VAL_78]] : f32
// CHECK:               %[[VAL_80:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_79]], %[[VAL_80]][] : memref<f32>
// CHECK:               %[[VAL_81:.*]] = "onnx.Sigmoid"(%[[VAL_80]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_82:.*]] = krnl.load %[[VAL_81]][] : memref<f32>
// CHECK:               %[[VAL_83:.*]] = constant 12 : index
// CHECK:               %[[VAL_84:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_51]]#1]
// CHECK:               %[[VAL_85:.*]] = krnl.load %[[VAL_41]]{{\[}}%[[VAL_84]], %[[VAL_51]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_86:.*]] = constant 12 : index
// CHECK:               %[[VAL_87:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_51]]#1]
// CHECK:               %[[VAL_88:.*]] = krnl.load %[[VAL_43]]{{\[}}%[[VAL_87]], %[[VAL_51]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_89:.*]] = addf %[[VAL_85]], %[[VAL_88]] : f32
// CHECK:               %[[VAL_90:.*]] = krnl.load %[[VAL_22]]#3{{\[}}%[[VAL_51]]#1] : memref<4xf32>
// CHECK:               %[[VAL_91:.*]] = krnl.load %[[VAL_22]]#7{{\[}}%[[VAL_51]]#1] : memref<4xf32>
// CHECK:               %[[VAL_92:.*]] = addf %[[VAL_89]], %[[VAL_90]] : f32
// CHECK:               %[[VAL_93:.*]] = addf %[[VAL_92]], %[[VAL_91]] : f32
// CHECK:               %[[VAL_94:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_93]], %[[VAL_94]][] : memref<f32>
// CHECK:               %[[VAL_95:.*]] = "onnx.Tanh"(%[[VAL_94]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_96:.*]] = krnl.load %[[VAL_95]][] : memref<f32>
// CHECK:               %[[VAL_97:.*]] = mulf %[[VAL_82]], %[[VAL_52]] : f32
// CHECK:               %[[VAL_98:.*]] = mulf %[[VAL_65]], %[[VAL_96]] : f32
// CHECK:               %[[VAL_99:.*]] = addf %[[VAL_97]], %[[VAL_98]] : f32
// CHECK:               %[[VAL_100:.*]] = constant 4 : index
// CHECK:               %[[VAL_101:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_51]]#1]
// CHECK:               %[[VAL_102:.*]] = krnl.load %[[VAL_41]]{{\[}}%[[VAL_101]], %[[VAL_51]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_103:.*]] = constant 4 : index
// CHECK:               %[[VAL_104:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_51]]#1]
// CHECK:               %[[VAL_105:.*]] = krnl.load %[[VAL_43]]{{\[}}%[[VAL_104]], %[[VAL_51]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_106:.*]] = addf %[[VAL_102]], %[[VAL_105]] : f32
// CHECK:               %[[VAL_107:.*]] = krnl.load %[[VAL_22]]#1{{\[}}%[[VAL_51]]#1] : memref<4xf32>
// CHECK:               %[[VAL_108:.*]] = krnl.load %[[VAL_22]]#5{{\[}}%[[VAL_51]]#1] : memref<4xf32>
// CHECK:               %[[VAL_109:.*]] = addf %[[VAL_106]], %[[VAL_107]] : f32
// CHECK:               %[[VAL_110:.*]] = addf %[[VAL_109]], %[[VAL_108]] : f32
// CHECK:               %[[VAL_111:.*]] = krnl.load %[[VAL_24]]#1{{\[}}%[[VAL_51]]#1] : memref<4xf32>
// CHECK:               %[[VAL_112:.*]] = mulf %[[VAL_111]], %[[VAL_99]] : f32
// CHECK:               %[[VAL_113:.*]] = addf %[[VAL_110]], %[[VAL_112]] : f32
// CHECK:               %[[VAL_114:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_113]], %[[VAL_114]][] : memref<f32>
// CHECK:               %[[VAL_115:.*]] = "onnx.Sigmoid"(%[[VAL_114]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_116:.*]] = krnl.load %[[VAL_115]][] : memref<f32>
// CHECK:               %[[VAL_117:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_99]], %[[VAL_117]][] : memref<f32>
// CHECK:               %[[VAL_118:.*]] = "onnx.Tanh"(%[[VAL_117]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_119:.*]] = krnl.load %[[VAL_118]][] : memref<f32>
// CHECK:               %[[VAL_120:.*]] = mulf %[[VAL_116]], %[[VAL_119]] : f32
// CHECK:               krnl.store %[[VAL_99]], %[[VAL_7]]{{\[}}%[[VAL_51]]#0, %[[VAL_51]]#1] : memref<2x4xf32>
// CHECK:               krnl.store %[[VAL_120]], %[[VAL_8]]{{\[}}%[[VAL_51]]#0, %[[VAL_51]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_27]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_121:.*]] = constant 32 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_9]], %[[VAL_8]], %[[VAL_121]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_8]] : memref<2x4xf32>
// CHECK:           memref.dealloc %[[VAL_7]] : memref<2x4xf32>
// CHECK:           return %[[VAL_9]] : memref<1x2x4xf32>
// CHECK:         }

}

// -----

func private @test_lstm_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x16x3xf32>, %arg2: tensor<2x16x4xf32>, %arg3: tensor<2x32xf32>, %arg4: tensor<2x2x4xf32>, %arg5: tensor<2x2x4xf32>, %arg6: tensor<2x12xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x16x3xf32>, tensor<2x16x4xf32>, tensor<2x32xf32>, none, tensor<2x2x4xf32>, tensor<2x2x4xf32>, tensor<2x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_lstm_bidirectional_mode(
// CHECK-SAME:        %[[VAL_0:.*]]: memref<7x2x3xf32>, %[[VAL_1:.*]]: memref<2x16x3xf32>, %[[VAL_2:.*]]: memref<2x16x4xf32>, %[[VAL_3:.*]]: memref<2x32xf32>, %[[VAL_4:.*]]: memref<2x2x4xf32>, %[[VAL_5:.*]]: memref<2x2x4xf32>, %[[VAL_6:.*]]: memref<2x12xf32>) -> memref<2x2x4xf32> {
// CHECK:           %[[VAL_7:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_8:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_9:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_10:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_11:.*]] = memref.alloc() : memref<2x2x4xf32>
// CHECK:           %[[VAL_12:.*]] = constant unit
// CHECK:           %[[VAL_13:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_14:.*]] = constant 0 : index
// CHECK:           %[[VAL_15:.*]] = constant 1 : index
// CHECK:           %[[VAL_16:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_16]]#0, %[[VAL_16]]#1) with (%[[VAL_16]]#0 -> %[[VAL_17:.*]] = 0 to 2, %[[VAL_16]]#1 -> %[[VAL_18:.*]] = 0 to 4) {
// CHECK:             %[[VAL_19:.*]] = krnl.load %[[VAL_4]]{{\[}}%[[VAL_14]], %[[VAL_17]], %[[VAL_18]]] : memref<2x2x4xf32>
// CHECK:             krnl.store %[[VAL_19]], %[[VAL_10]]{{\[}}%[[VAL_17]], %[[VAL_18]]] : memref<2x4xf32>
// CHECK:             %[[VAL_20:.*]] = krnl.load %[[VAL_5]]{{\[}}%[[VAL_14]], %[[VAL_17]], %[[VAL_18]]] : memref<2x2x4xf32>
// CHECK:             krnl.store %[[VAL_20]], %[[VAL_9]]{{\[}}%[[VAL_17]], %[[VAL_18]]] : memref<2x4xf32>
// CHECK:             %[[VAL_21:.*]] = krnl.load %[[VAL_4]]{{\[}}%[[VAL_15]], %[[VAL_17]], %[[VAL_18]]] : memref<2x2x4xf32>
// CHECK:             krnl.store %[[VAL_21]], %[[VAL_8]]{{\[}}%[[VAL_17]], %[[VAL_18]]] : memref<2x4xf32>
// CHECK:             %[[VAL_22:.*]] = krnl.load %[[VAL_5]]{{\[}}%[[VAL_15]], %[[VAL_17]], %[[VAL_18]]] : memref<2x2x4xf32>
// CHECK:             krnl.store %[[VAL_22]], %[[VAL_7]]{{\[}}%[[VAL_17]], %[[VAL_18]]] : memref<2x4xf32>
// CHECK:           }
// CHECK:           %[[VAL_23:.*]]:2 = "onnx.Split"(%[[VAL_1]]) {axis = 0 : si64} : (memref<2x16x3xf32>) -> (memref<1x16x3xf32>, memref<1x16x3xf32>)
// CHECK:           %[[VAL_24:.*]] = "onnx.Squeeze"(%[[VAL_23]]#0) {axes = [0]} : (memref<1x16x3xf32>) -> memref<16x3xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Squeeze"(%[[VAL_23]]#1) {axes = [0]} : (memref<1x16x3xf32>) -> memref<16x3xf32>
// CHECK:           %[[VAL_26:.*]]:2 = "onnx.Split"(%[[VAL_2]]) {axis = 0 : si64} : (memref<2x16x4xf32>) -> (memref<1x16x4xf32>, memref<1x16x4xf32>)
// CHECK:           %[[VAL_27:.*]] = "onnx.Squeeze"(%[[VAL_26]]#0) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.Squeeze"(%[[VAL_26]]#1) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK:           %[[VAL_29:.*]]:2 = "onnx.Split"(%[[VAL_3]]) {axis = 0 : si64} : (memref<2x32xf32>) -> (memref<1x32xf32>, memref<1x32xf32>)
// CHECK:           %[[VAL_30:.*]] = "onnx.Squeeze"(%[[VAL_29]]#0) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Squeeze"(%[[VAL_29]]#1) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK:           %[[VAL_32:.*]]:8 = "onnx.Split"(%[[VAL_30]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_33:.*]]:8 = "onnx.Split"(%[[VAL_31]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_34:.*]]:2 = "onnx.Split"(%[[VAL_6]]) {axis = 0 : si64} : (memref<2x12xf32>) -> (memref<1x12xf32>, memref<1x12xf32>)
// CHECK:           %[[VAL_35:.*]] = "onnx.Squeeze"(%[[VAL_34]]#0) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Squeeze"(%[[VAL_34]]#1) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK:           %[[VAL_37:.*]]:3 = "onnx.Split"(%[[VAL_35]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_38:.*]]:3 = "onnx.Split"(%[[VAL_36]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_39:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_39]]) with (%[[VAL_39]] -> %[[VAL_40:.*]] = 0 to 7) {
// CHECK:             %[[VAL_41:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_42:.*]] = constant 0 : index
// CHECK:             %[[VAL_43:.*]] = constant 2 : index
// CHECK:             %[[VAL_44:.*]] = constant 3 : index
// CHECK:             %[[VAL_45:.*]] = constant 0 : index
// CHECK:             %[[VAL_46:.*]] = constant 0 : index
// CHECK:             %[[VAL_47:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_47]]#0, %[[VAL_47]]#1) with (%[[VAL_47]]#0 -> %[[VAL_48:.*]] = %[[VAL_45]] to %[[VAL_43]], %[[VAL_47]]#1 -> %[[VAL_49:.*]] = %[[VAL_46]] to %[[VAL_44]]) {
// CHECK:               %[[VAL_50:.*]]:2 = krnl.get_induction_var_value(%[[VAL_47]]#0, %[[VAL_47]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_51:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_40]], %[[VAL_50]]#0, %[[VAL_50]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_51]], %[[VAL_41]]{{\[}}%[[VAL_50]]#0, %[[VAL_50]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_52:.*]] = "onnx.Transpose"(%[[VAL_41]]) {perm = [1, 0]} : (memref<2x3xf32>) -> memref<3x2xf32>
// CHECK:             %[[VAL_53:.*]] = "onnx.MatMul"(%[[VAL_24]], %[[VAL_52]]) : (memref<16x3xf32>, memref<3x2xf32>) -> memref<16x2xf32>
// CHECK:             %[[VAL_54:.*]] = "onnx.Transpose"(%[[VAL_10]]) {perm = [1, 0]} : (memref<2x4xf32>) -> memref<4x2xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_54]]) : (memref<16x4xf32>, memref<4x2xf32>) -> memref<16x2xf32>
// CHECK:             %[[VAL_56:.*]] = constant 2 : index
// CHECK:             %[[VAL_57:.*]] = constant 4 : index
// CHECK:             %[[VAL_58:.*]] = constant 0 : index
// CHECK:             %[[VAL_59:.*]] = constant 0 : index
// CHECK:             %[[VAL_60:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_60]]#0, %[[VAL_60]]#1) with (%[[VAL_60]]#0 -> %[[VAL_61:.*]] = %[[VAL_58]] to %[[VAL_56]], %[[VAL_60]]#1 -> %[[VAL_62:.*]] = %[[VAL_59]] to %[[VAL_57]]) {
// CHECK:               %[[VAL_63:.*]]:2 = krnl.get_induction_var_value(%[[VAL_60]]#0, %[[VAL_60]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_64:.*]] = krnl.load %[[VAL_9]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_65:.*]] = krnl.load %[[VAL_53]]{{\[}}%[[VAL_63]]#1, %[[VAL_63]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_66:.*]] = krnl.load %[[VAL_55]]{{\[}}%[[VAL_63]]#1, %[[VAL_63]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_67:.*]] = addf %[[VAL_65]], %[[VAL_66]] : f32
// CHECK:               %[[VAL_68:.*]] = krnl.load %[[VAL_32]]#0{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_69:.*]] = krnl.load %[[VAL_32]]#4{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_70:.*]] = addf %[[VAL_67]], %[[VAL_68]] : f32
// CHECK:               %[[VAL_71:.*]] = addf %[[VAL_70]], %[[VAL_69]] : f32
// CHECK:               %[[VAL_72:.*]] = krnl.load %[[VAL_37]]#0{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_73:.*]] = mulf %[[VAL_72]], %[[VAL_64]] : f32
// CHECK:               %[[VAL_74:.*]] = addf %[[VAL_71]], %[[VAL_73]] : f32
// CHECK:               %[[VAL_75:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_74]], %[[VAL_75]][] : memref<f32>
// CHECK:               %[[VAL_76:.*]] = "onnx.Sigmoid"(%[[VAL_75]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_77:.*]] = krnl.load %[[VAL_76]][] : memref<f32>
// CHECK:               %[[VAL_78:.*]] = constant 8 : index
// CHECK:               %[[VAL_79:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_63]]#1]
// CHECK:               %[[VAL_80:.*]] = krnl.load %[[VAL_53]]{{\[}}%[[VAL_79]], %[[VAL_63]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_81:.*]] = constant 8 : index
// CHECK:               %[[VAL_82:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_63]]#1]
// CHECK:               %[[VAL_83:.*]] = krnl.load %[[VAL_55]]{{\[}}%[[VAL_82]], %[[VAL_63]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_84:.*]] = addf %[[VAL_80]], %[[VAL_83]] : f32
// CHECK:               %[[VAL_85:.*]] = krnl.load %[[VAL_32]]#2{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_86:.*]] = krnl.load %[[VAL_32]]#6{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_87:.*]] = addf %[[VAL_84]], %[[VAL_85]] : f32
// CHECK:               %[[VAL_88:.*]] = addf %[[VAL_87]], %[[VAL_86]] : f32
// CHECK:               %[[VAL_89:.*]] = krnl.load %[[VAL_37]]#2{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_90:.*]] = mulf %[[VAL_89]], %[[VAL_64]] : f32
// CHECK:               %[[VAL_91:.*]] = addf %[[VAL_88]], %[[VAL_90]] : f32
// CHECK:               %[[VAL_92:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_91]], %[[VAL_92]][] : memref<f32>
// CHECK:               %[[VAL_93:.*]] = "onnx.Sigmoid"(%[[VAL_92]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_94:.*]] = krnl.load %[[VAL_93]][] : memref<f32>
// CHECK:               %[[VAL_95:.*]] = constant 12 : index
// CHECK:               %[[VAL_96:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_63]]#1]
// CHECK:               %[[VAL_97:.*]] = krnl.load %[[VAL_53]]{{\[}}%[[VAL_96]], %[[VAL_63]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_98:.*]] = constant 12 : index
// CHECK:               %[[VAL_99:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_63]]#1]
// CHECK:               %[[VAL_100:.*]] = krnl.load %[[VAL_55]]{{\[}}%[[VAL_99]], %[[VAL_63]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_101:.*]] = addf %[[VAL_97]], %[[VAL_100]] : f32
// CHECK:               %[[VAL_102:.*]] = krnl.load %[[VAL_32]]#3{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_103:.*]] = krnl.load %[[VAL_32]]#7{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_104:.*]] = addf %[[VAL_101]], %[[VAL_102]] : f32
// CHECK:               %[[VAL_105:.*]] = addf %[[VAL_104]], %[[VAL_103]] : f32
// CHECK:               %[[VAL_106:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_105]], %[[VAL_106]][] : memref<f32>
// CHECK:               %[[VAL_107:.*]] = "onnx.Tanh"(%[[VAL_106]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_108:.*]] = krnl.load %[[VAL_107]][] : memref<f32>
// CHECK:               %[[VAL_109:.*]] = mulf %[[VAL_94]], %[[VAL_64]] : f32
// CHECK:               %[[VAL_110:.*]] = mulf %[[VAL_77]], %[[VAL_108]] : f32
// CHECK:               %[[VAL_111:.*]] = addf %[[VAL_109]], %[[VAL_110]] : f32
// CHECK:               %[[VAL_112:.*]] = constant 4 : index
// CHECK:               %[[VAL_113:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_63]]#1]
// CHECK:               %[[VAL_114:.*]] = krnl.load %[[VAL_53]]{{\[}}%[[VAL_113]], %[[VAL_63]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_115:.*]] = constant 4 : index
// CHECK:               %[[VAL_116:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_63]]#1]
// CHECK:               %[[VAL_117:.*]] = krnl.load %[[VAL_55]]{{\[}}%[[VAL_116]], %[[VAL_63]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_118:.*]] = addf %[[VAL_114]], %[[VAL_117]] : f32
// CHECK:               %[[VAL_119:.*]] = krnl.load %[[VAL_32]]#1{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_120:.*]] = krnl.load %[[VAL_32]]#5{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_121:.*]] = addf %[[VAL_118]], %[[VAL_119]] : f32
// CHECK:               %[[VAL_122:.*]] = addf %[[VAL_121]], %[[VAL_120]] : f32
// CHECK:               %[[VAL_123:.*]] = krnl.load %[[VAL_37]]#1{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_124:.*]] = mulf %[[VAL_123]], %[[VAL_111]] : f32
// CHECK:               %[[VAL_125:.*]] = addf %[[VAL_122]], %[[VAL_124]] : f32
// CHECK:               %[[VAL_126:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_125]], %[[VAL_126]][] : memref<f32>
// CHECK:               %[[VAL_127:.*]] = "onnx.Sigmoid"(%[[VAL_126]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_128:.*]] = krnl.load %[[VAL_127]][] : memref<f32>
// CHECK:               %[[VAL_129:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_111]], %[[VAL_129]][] : memref<f32>
// CHECK:               %[[VAL_130:.*]] = "onnx.Tanh"(%[[VAL_129]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_131:.*]] = krnl.load %[[VAL_130]][] : memref<f32>
// CHECK:               %[[VAL_132:.*]] = mulf %[[VAL_128]], %[[VAL_131]] : f32
// CHECK:               krnl.store %[[VAL_111]], %[[VAL_9]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<2x4xf32>
// CHECK:               krnl.store %[[VAL_132]], %[[VAL_10]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_41]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_133:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_133]]) with (%[[VAL_133]] -> %[[VAL_134:.*]] = 0 to 7) {
// CHECK:             %[[VAL_135:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_136:.*]] = constant 1 : index
// CHECK:             %[[VAL_137:.*]] = constant 7 : index
// CHECK:             %[[VAL_138:.*]] = affine.apply #map{{.+}}(%[[VAL_134]]){{\[}}%[[VAL_137]]]
// CHECK:             %[[VAL_139:.*]] = constant 2 : index
// CHECK:             %[[VAL_140:.*]] = constant 3 : index
// CHECK:             %[[VAL_141:.*]] = constant 0 : index
// CHECK:             %[[VAL_142:.*]] = constant 0 : index
// CHECK:             %[[VAL_143:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_143]]#0, %[[VAL_143]]#1) with (%[[VAL_143]]#0 -> %[[VAL_144:.*]] = %[[VAL_141]] to %[[VAL_139]], %[[VAL_143]]#1 -> %[[VAL_145:.*]] = %[[VAL_142]] to %[[VAL_140]]) {
// CHECK:               %[[VAL_146:.*]]:2 = krnl.get_induction_var_value(%[[VAL_143]]#0, %[[VAL_143]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_147:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_138]], %[[VAL_146]]#0, %[[VAL_146]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_147]], %[[VAL_135]]{{\[}}%[[VAL_146]]#0, %[[VAL_146]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_148:.*]] = "onnx.Transpose"(%[[VAL_135]]) {perm = [1, 0]} : (memref<2x3xf32>) -> memref<3x2xf32>
// CHECK:             %[[VAL_149:.*]] = "onnx.MatMul"(%[[VAL_25]], %[[VAL_148]]) : (memref<16x3xf32>, memref<3x2xf32>) -> memref<16x2xf32>
// CHECK:             %[[VAL_150:.*]] = "onnx.Transpose"(%[[VAL_8]]) {perm = [1, 0]} : (memref<2x4xf32>) -> memref<4x2xf32>
// CHECK:             %[[VAL_151:.*]] = "onnx.MatMul"(%[[VAL_28]], %[[VAL_150]]) : (memref<16x4xf32>, memref<4x2xf32>) -> memref<16x2xf32>
// CHECK:             %[[VAL_152:.*]] = constant 2 : index
// CHECK:             %[[VAL_153:.*]] = constant 4 : index
// CHECK:             %[[VAL_154:.*]] = constant 0 : index
// CHECK:             %[[VAL_155:.*]] = constant 0 : index
// CHECK:             %[[VAL_156:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_156]]#0, %[[VAL_156]]#1) with (%[[VAL_156]]#0 -> %[[VAL_157:.*]] = %[[VAL_154]] to %[[VAL_152]], %[[VAL_156]]#1 -> %[[VAL_158:.*]] = %[[VAL_155]] to %[[VAL_153]]) {
// CHECK:               %[[VAL_159:.*]]:2 = krnl.get_induction_var_value(%[[VAL_156]]#0, %[[VAL_156]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_160:.*]] = krnl.load %[[VAL_7]]{{\[}}%[[VAL_159]]#0, %[[VAL_159]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_161:.*]] = krnl.load %[[VAL_149]]{{\[}}%[[VAL_159]]#1, %[[VAL_159]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_162:.*]] = krnl.load %[[VAL_151]]{{\[}}%[[VAL_159]]#1, %[[VAL_159]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_163:.*]] = addf %[[VAL_161]], %[[VAL_162]] : f32
// CHECK:               %[[VAL_164:.*]] = krnl.load %[[VAL_33]]#0{{\[}}%[[VAL_159]]#1] : memref<4xf32>
// CHECK:               %[[VAL_165:.*]] = krnl.load %[[VAL_33]]#4{{\[}}%[[VAL_159]]#1] : memref<4xf32>
// CHECK:               %[[VAL_166:.*]] = addf %[[VAL_163]], %[[VAL_164]] : f32
// CHECK:               %[[VAL_167:.*]] = addf %[[VAL_166]], %[[VAL_165]] : f32
// CHECK:               %[[VAL_168:.*]] = krnl.load %[[VAL_38]]#0{{\[}}%[[VAL_159]]#1] : memref<4xf32>
// CHECK:               %[[VAL_169:.*]] = mulf %[[VAL_168]], %[[VAL_160]] : f32
// CHECK:               %[[VAL_170:.*]] = addf %[[VAL_167]], %[[VAL_169]] : f32
// CHECK:               %[[VAL_171:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_170]], %[[VAL_171]][] : memref<f32>
// CHECK:               %[[VAL_172:.*]] = "onnx.Sigmoid"(%[[VAL_171]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_173:.*]] = krnl.load %[[VAL_172]][] : memref<f32>
// CHECK:               %[[VAL_174:.*]] = constant 8 : index
// CHECK:               %[[VAL_175:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_159]]#1]
// CHECK:               %[[VAL_176:.*]] = krnl.load %[[VAL_149]]{{\[}}%[[VAL_175]], %[[VAL_159]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_177:.*]] = constant 8 : index
// CHECK:               %[[VAL_178:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_159]]#1]
// CHECK:               %[[VAL_179:.*]] = krnl.load %[[VAL_151]]{{\[}}%[[VAL_178]], %[[VAL_159]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_180:.*]] = addf %[[VAL_176]], %[[VAL_179]] : f32
// CHECK:               %[[VAL_181:.*]] = krnl.load %[[VAL_33]]#2{{\[}}%[[VAL_159]]#1] : memref<4xf32>
// CHECK:               %[[VAL_182:.*]] = krnl.load %[[VAL_33]]#6{{\[}}%[[VAL_159]]#1] : memref<4xf32>
// CHECK:               %[[VAL_183:.*]] = addf %[[VAL_180]], %[[VAL_181]] : f32
// CHECK:               %[[VAL_184:.*]] = addf %[[VAL_183]], %[[VAL_182]] : f32
// CHECK:               %[[VAL_185:.*]] = krnl.load %[[VAL_38]]#2{{\[}}%[[VAL_159]]#1] : memref<4xf32>
// CHECK:               %[[VAL_186:.*]] = mulf %[[VAL_185]], %[[VAL_160]] : f32
// CHECK:               %[[VAL_187:.*]] = addf %[[VAL_184]], %[[VAL_186]] : f32
// CHECK:               %[[VAL_188:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_187]], %[[VAL_188]][] : memref<f32>
// CHECK:               %[[VAL_189:.*]] = "onnx.Sigmoid"(%[[VAL_188]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_190:.*]] = krnl.load %[[VAL_189]][] : memref<f32>
// CHECK:               %[[VAL_191:.*]] = constant 12 : index
// CHECK:               %[[VAL_192:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_159]]#1]
// CHECK:               %[[VAL_193:.*]] = krnl.load %[[VAL_149]]{{\[}}%[[VAL_192]], %[[VAL_159]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_194:.*]] = constant 12 : index
// CHECK:               %[[VAL_195:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_159]]#1]
// CHECK:               %[[VAL_196:.*]] = krnl.load %[[VAL_151]]{{\[}}%[[VAL_195]], %[[VAL_159]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_197:.*]] = addf %[[VAL_193]], %[[VAL_196]] : f32
// CHECK:               %[[VAL_198:.*]] = krnl.load %[[VAL_33]]#3{{\[}}%[[VAL_159]]#1] : memref<4xf32>
// CHECK:               %[[VAL_199:.*]] = krnl.load %[[VAL_33]]#7{{\[}}%[[VAL_159]]#1] : memref<4xf32>
// CHECK:               %[[VAL_200:.*]] = addf %[[VAL_197]], %[[VAL_198]] : f32
// CHECK:               %[[VAL_201:.*]] = addf %[[VAL_200]], %[[VAL_199]] : f32
// CHECK:               %[[VAL_202:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_201]], %[[VAL_202]][] : memref<f32>
// CHECK:               %[[VAL_203:.*]] = "onnx.Tanh"(%[[VAL_202]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_204:.*]] = krnl.load %[[VAL_203]][] : memref<f32>
// CHECK:               %[[VAL_205:.*]] = mulf %[[VAL_190]], %[[VAL_160]] : f32
// CHECK:               %[[VAL_206:.*]] = mulf %[[VAL_173]], %[[VAL_204]] : f32
// CHECK:               %[[VAL_207:.*]] = addf %[[VAL_205]], %[[VAL_206]] : f32
// CHECK:               %[[VAL_208:.*]] = constant 4 : index
// CHECK:               %[[VAL_209:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_159]]#1]
// CHECK:               %[[VAL_210:.*]] = krnl.load %[[VAL_149]]{{\[}}%[[VAL_209]], %[[VAL_159]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_211:.*]] = constant 4 : index
// CHECK:               %[[VAL_212:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_159]]#1]
// CHECK:               %[[VAL_213:.*]] = krnl.load %[[VAL_151]]{{\[}}%[[VAL_212]], %[[VAL_159]]#0] : memref<16x2xf32>
// CHECK:               %[[VAL_214:.*]] = addf %[[VAL_210]], %[[VAL_213]] : f32
// CHECK:               %[[VAL_215:.*]] = krnl.load %[[VAL_33]]#1{{\[}}%[[VAL_159]]#1] : memref<4xf32>
// CHECK:               %[[VAL_216:.*]] = krnl.load %[[VAL_33]]#5{{\[}}%[[VAL_159]]#1] : memref<4xf32>
// CHECK:               %[[VAL_217:.*]] = addf %[[VAL_214]], %[[VAL_215]] : f32
// CHECK:               %[[VAL_218:.*]] = addf %[[VAL_217]], %[[VAL_216]] : f32
// CHECK:               %[[VAL_219:.*]] = krnl.load %[[VAL_38]]#1{{\[}}%[[VAL_159]]#1] : memref<4xf32>
// CHECK:               %[[VAL_220:.*]] = mulf %[[VAL_219]], %[[VAL_207]] : f32
// CHECK:               %[[VAL_221:.*]] = addf %[[VAL_218]], %[[VAL_220]] : f32
// CHECK:               %[[VAL_222:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_221]], %[[VAL_222]][] : memref<f32>
// CHECK:               %[[VAL_223:.*]] = "onnx.Sigmoid"(%[[VAL_222]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_224:.*]] = krnl.load %[[VAL_223]][] : memref<f32>
// CHECK:               %[[VAL_225:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_207]], %[[VAL_225]][] : memref<f32>
// CHECK:               %[[VAL_226:.*]] = "onnx.Tanh"(%[[VAL_225]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_227:.*]] = krnl.load %[[VAL_226]][] : memref<f32>
// CHECK:               %[[VAL_228:.*]] = mulf %[[VAL_224]], %[[VAL_227]] : f32
// CHECK:               krnl.store %[[VAL_207]], %[[VAL_7]]{{\[}}%[[VAL_159]]#0, %[[VAL_159]]#1] : memref<2x4xf32>
// CHECK:               krnl.store %[[VAL_228]], %[[VAL_8]]{{\[}}%[[VAL_159]]#0, %[[VAL_159]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_135]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_229:.*]] = constant 2 : index
// CHECK:           %[[VAL_230:.*]] = constant 4 : index
// CHECK:           %[[VAL_231:.*]] = constant 0 : index
// CHECK:           %[[VAL_232:.*]] = constant 0 : index
// CHECK:           %[[VAL_233:.*]] = constant 0 : index
// CHECK:           %[[VAL_234:.*]] = constant 1 : index
// CHECK:           %[[VAL_235:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_235]]#0, %[[VAL_235]]#1) with (%[[VAL_235]]#0 -> %[[VAL_236:.*]] = %[[VAL_231]] to %[[VAL_229]], %[[VAL_235]]#1 -> %[[VAL_237:.*]] = %[[VAL_232]] to %[[VAL_230]]) {
// CHECK:             %[[VAL_238:.*]]:2 = krnl.get_induction_var_value(%[[VAL_235]]#0, %[[VAL_235]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             %[[VAL_239:.*]] = krnl.load %[[VAL_10]]{{\[}}%[[VAL_238]]#0, %[[VAL_238]]#1] : memref<2x4xf32>
// CHECK:             krnl.store %[[VAL_239]], %[[VAL_11]]{{\[}}%[[VAL_233]], %[[VAL_238]]#0, %[[VAL_238]]#1] : memref<2x2x4xf32>
// CHECK:             %[[VAL_240:.*]] = krnl.load %[[VAL_8]]{{\[}}%[[VAL_238]]#0, %[[VAL_238]]#1] : memref<2x4xf32>
// CHECK:             krnl.store %[[VAL_240]], %[[VAL_11]]{{\[}}%[[VAL_234]], %[[VAL_238]]#0, %[[VAL_238]]#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           memref.dealloc %[[VAL_10]] : memref<2x4xf32>
// CHECK:           memref.dealloc %[[VAL_9]] : memref<2x4xf32>
// CHECK:           memref.dealloc %[[VAL_8]] : memref<2x4xf32>
// CHECK:           memref.dealloc %[[VAL_7]] : memref<2x4xf32>
// CHECK:           return %[[VAL_11]] : memref<2x2x4xf32>
// CHECK:         }

}

// -----

func private @test_lstm_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x16x?xf32>, %arg2: tensor<1x16x4xf32>, %arg3: tensor<1x32xf32>, %arg4: tensor<1x?x4xf32>, %arg5: tensor<1x?x4xf32>, %arg6: tensor<1x12xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4, %arg5, %arg6) {hidden_size = 4 : si64} : (tensor<?x?x?xf32>, tensor<1x16x?xf32>, tensor<1x16x4xf32>, tensor<1x32xf32>, none, tensor<1x?x4xf32>, tensor<1x?x4xf32>, tensor<1x12xf32>) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_lstm_unknown_dims(
// CHECK-SAME:        %[[VAL_0:.*]]: memref<?x?x?xf32>, %[[VAL_1:.*]]: memref<1x16x?xf32>, %[[VAL_2:.*]]: memref<1x16x4xf32>, %[[VAL_3:.*]]: memref<1x32xf32>, %[[VAL_4:.*]]: memref<1x?x4xf32>, %[[VAL_5:.*]]: memref<1x?x4xf32>, %[[VAL_6:.*]]: memref<1x12xf32>) -> memref<1x?x4xf32> {
// CHECK:           %[[VAL_7:.*]] = constant unit
// CHECK:           %[[VAL_8:.*]] = constant 1 : index
// CHECK:           %[[VAL_9:.*]] = memref.dim %[[VAL_0]], %[[VAL_8]] : memref<?x?x?xf32>
// CHECK:           %[[VAL_10:.*]] = memref.alloc(%[[VAL_9]]) : memref<1x?x4xf32>
// CHECK:           %[[VAL_11:.*]] = constant 1 : index
// CHECK:           %[[VAL_12:.*]] = memref.dim %[[VAL_0]], %[[VAL_11]] : memref<?x?x?xf32>
// CHECK:           %[[VAL_13:.*]] = memref.alloc(%[[VAL_12]]) : memref<?x4xf32>
// CHECK:           %[[VAL_14:.*]] = constant 1 : index
// CHECK:           %[[VAL_15:.*]] = memref.dim %[[VAL_0]], %[[VAL_14]] : memref<?x?x?xf32>
// CHECK:           %[[VAL_16:.*]] = memref.alloc(%[[VAL_15]]) : memref<?x4xf32>
// CHECK:           %[[VAL_17:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_18:.*]] = constant 0 : index
// CHECK:           %[[VAL_19:.*]] = constant 1 : index
// CHECK:           %[[VAL_20:.*]]:2 = krnl.define_loops 2
// CHECK:           %[[VAL_21:.*]] = constant 0 : index
// CHECK:           %[[VAL_22:.*]] = memref.dim %[[VAL_13]], %[[VAL_21]] : memref<?x4xf32>
// CHECK:           krnl.iterate(%[[VAL_20]]#0, %[[VAL_20]]#1) with (%[[VAL_20]]#0 -> %[[VAL_23:.*]] = 0 to %[[VAL_22]], %[[VAL_20]]#1 -> %[[VAL_24:.*]] = 0 to 4) {
// CHECK:             %[[VAL_25:.*]] = krnl.load %[[VAL_4]]{{\[}}%[[VAL_18]], %[[VAL_23]], %[[VAL_24]]] : memref<1x?x4xf32>
// CHECK:             krnl.store %[[VAL_25]], %[[VAL_13]]{{\[}}%[[VAL_23]], %[[VAL_24]]] : memref<?x4xf32>
// CHECK:             %[[VAL_26:.*]] = krnl.load %[[VAL_5]]{{\[}}%[[VAL_18]], %[[VAL_23]], %[[VAL_24]]] : memref<1x?x4xf32>
// CHECK:             krnl.store %[[VAL_26]], %[[VAL_16]]{{\[}}%[[VAL_23]], %[[VAL_24]]] : memref<?x4xf32>
// CHECK:           }
// CHECK:           %[[VAL_27:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x16x?xf32>) -> memref<16x?xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x16x4xf32>) -> memref<16x4xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Squeeze"(%[[VAL_3]]) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK:           %[[VAL_30:.*]]:8 = "onnx.Split"(%[[VAL_29]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_31:.*]] = "onnx.Squeeze"(%[[VAL_6]]) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK:           %[[VAL_32:.*]]:3 = "onnx.Split"(%[[VAL_31]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_33:.*]] = krnl.define_loops 1
// CHECK:           %[[VAL_34:.*]] = constant 0 : index
// CHECK:           %[[VAL_35:.*]] = memref.dim %[[VAL_0]], %[[VAL_34]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate(%[[VAL_33]]) with (%[[VAL_33]] -> %[[VAL_36:.*]] = 0 to %[[VAL_35]]) {
// CHECK:             %[[VAL_37:.*]] = constant 0 : index
// CHECK:             %[[VAL_38:.*]] = constant 1 : index
// CHECK:             %[[VAL_39:.*]] = memref.dim %[[VAL_0]], %[[VAL_38]] : memref<?x?x?xf32>
// CHECK:             %[[VAL_40:.*]] = constant 2 : index
// CHECK:             %[[VAL_41:.*]] = memref.dim %[[VAL_0]], %[[VAL_40]] : memref<?x?x?xf32>
// CHECK:             %[[VAL_42:.*]] = memref.alloc(%[[VAL_39]], %[[VAL_41]]) : memref<?x?xf32>
// CHECK:             %[[VAL_43:.*]] = constant 0 : index
// CHECK:             %[[VAL_44:.*]] = memref.dim %[[VAL_42]], %[[VAL_43]] : memref<?x?xf32>
// CHECK:             %[[VAL_45:.*]] = constant 1 : index
// CHECK:             %[[VAL_46:.*]] = memref.dim %[[VAL_42]], %[[VAL_45]] : memref<?x?xf32>
// CHECK:             %[[VAL_66:.*]] = constant 0 : index
// CHECK:             %[[VAL_48:.*]] = constant 0 : index
// CHECK:             %[[VAL_49:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_49]]#0, %[[VAL_49]]#1) with (%[[VAL_49]]#0 -> %[[VAL_50:.*]] = %[[VAL_66]] to %[[VAL_44]], %[[VAL_49]]#1 -> %[[VAL_51:.*]] = %[[VAL_48]] to %[[VAL_46]]) {
// CHECK:               %[[VAL_52:.*]]:2 = krnl.get_induction_var_value(%[[VAL_49]]#0, %[[VAL_49]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_53:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_36]], %[[VAL_52]]#0, %[[VAL_52]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store %[[VAL_53]], %[[VAL_42]]{{\[}}%[[VAL_52]]#0, %[[VAL_52]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK:             %[[VAL_54:.*]] = "onnx.Transpose"(%[[VAL_42]]) {perm = [1, 0]} : (memref<?x?xf32>) -> memref<?x?xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.MatMul"(%[[VAL_27]], %[[VAL_54]]) : (memref<16x?xf32>, memref<?x?xf32>) -> memref<16x?xf32>
// CHECK:             %[[VAL_56:.*]] = "onnx.Transpose"(%[[VAL_13]]) {perm = [1, 0]} : (memref<?x4xf32>) -> memref<4x?xf32>
// CHECK:             %[[VAL_57:.*]] = "onnx.MatMul"(%[[VAL_28]], %[[VAL_56]]) : (memref<16x4xf32>, memref<4x?xf32>) -> memref<16x?xf32>
// CHECK:             %[[VAL_58:.*]] = constant 0 : index
// CHECK:             %[[VAL_59:.*]] = memref.dim %[[VAL_13]], %[[VAL_58]] : memref<?x4xf32>
// CHECK:             %[[VAL_60:.*]] = constant 4 : index
// CHECK:             %[[VAL_61:.*]] = constant 0 : index
// CHECK:             %[[VAL_62:.*]] = constant 0 : index
// CHECK:             %[[VAL_63:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_63]]#0, %[[VAL_63]]#1) with (%[[VAL_63]]#0 -> %[[VAL_64:.*]] = %[[VAL_61]] to %[[VAL_59]], %[[VAL_63]]#1 -> %[[VAL_65:.*]] = %[[VAL_62]] to %[[VAL_60]]) {
// CHECK:               %[[VAL_66:.*]]:2 = krnl.get_induction_var_value(%[[VAL_63]]#0, %[[VAL_63]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_67:.*]] = krnl.load %[[VAL_16]]{{\[}}%[[VAL_66]]#0, %[[VAL_66]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_68:.*]] = krnl.load %[[VAL_55]]{{\[}}%[[VAL_66]]#1, %[[VAL_66]]#0] : memref<16x?xf32>
// CHECK:               %[[VAL_69:.*]] = krnl.load %[[VAL_57]]{{\[}}%[[VAL_66]]#1, %[[VAL_66]]#0] : memref<16x?xf32>
// CHECK:               %[[VAL_70:.*]] = addf %[[VAL_68]], %[[VAL_69]] : f32
// CHECK:               %[[VAL_71:.*]] = krnl.load %[[VAL_30]]#0{{\[}}%[[VAL_66]]#1] : memref<4xf32>
// CHECK:               %[[VAL_72:.*]] = krnl.load %[[VAL_30]]#4{{\[}}%[[VAL_66]]#1] : memref<4xf32>
// CHECK:               %[[VAL_73:.*]] = addf %[[VAL_70]], %[[VAL_71]] : f32
// CHECK:               %[[VAL_74:.*]] = addf %[[VAL_73]], %[[VAL_72]] : f32
// CHECK:               %[[VAL_75:.*]] = krnl.load %[[VAL_32]]#0{{\[}}%[[VAL_66]]#1] : memref<4xf32>
// CHECK:               %[[VAL_76:.*]] = mulf %[[VAL_75]], %[[VAL_67]] : f32
// CHECK:               %[[VAL_77:.*]] = addf %[[VAL_74]], %[[VAL_76]] : f32
// CHECK:               %[[VAL_78:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_77]], %[[VAL_78]][] : memref<f32>
// CHECK:               %[[VAL_79:.*]] = "onnx.Sigmoid"(%[[VAL_78]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_80:.*]] = krnl.load %[[VAL_79]][] : memref<f32>

// CHECK:               %[[VAL_81:.*]] = constant 8 : index
// CHECK:               %[[VAL_82:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_66]]#1]
// CHECK:               %[[VAL_83:.*]] = krnl.load %[[VAL_55]]{{\[}}%[[VAL_82]], %[[VAL_66]]#0] : memref<16x?xf32>
// CHECK:               %[[VAL_84:.*]] = constant 8 : index
// CHECK:               %[[VAL_85:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_66]]#1]
// CHECK:               %[[VAL_86:.*]] = krnl.load %[[VAL_57]]{{\[}}%[[VAL_85]], %[[VAL_66]]#0] : memref<16x?xf32>
// CHECK:               %[[VAL_87:.*]] = addf %[[VAL_83]], %[[VAL_86]] : f32
// CHECK:               %[[VAL_88:.*]] = krnl.load %[[VAL_30]]#2{{\[}}%[[VAL_66]]#1] : memref<4xf32>
// CHECK:               %[[VAL_89:.*]] = krnl.load %[[VAL_30]]#6{{\[}}%[[VAL_66]]#1] : memref<4xf32>
// CHECK:               %[[VAL_90:.*]] = addf %[[VAL_87]], %[[VAL_88]] : f32
// CHECK:               %[[VAL_91:.*]] = addf %[[VAL_90]], %[[VAL_89]] : f32
// CHECK:               %[[VAL_92:.*]] = krnl.load %[[VAL_32]]#2{{\[}}%[[VAL_66]]#1] : memref<4xf32>
// CHECK:               %[[VAL_93:.*]] = mulf %[[VAL_92]], %[[VAL_67]] : f32
// CHECK:               %[[VAL_94:.*]] = addf %[[VAL_91]], %[[VAL_93]] : f32
// CHECK:               %[[VAL_95:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_94]], %[[VAL_95]][] : memref<f32>
// CHECK:               %[[VAL_96:.*]] = "onnx.Sigmoid"(%[[VAL_95]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_97:.*]] = krnl.load %[[VAL_96]][] : memref<f32>

// CHECK:               %[[VAL_98:.*]] = constant 12 : index
// CHECK:               %[[VAL_99:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_66]]#1]
// CHECK:               %[[VAL_100:.*]] = krnl.load %[[VAL_55]]{{\[}}%[[VAL_99]], %[[VAL_66]]#0] : memref<16x?xf32>
// CHECK:               %[[VAL_101:.*]] = constant 12 : index
// CHECK:               %[[VAL_102:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_66]]#1]
// CHECK:               %[[VAL_103:.*]] = krnl.load %[[VAL_57]]{{\[}}%[[VAL_102]], %[[VAL_66]]#0] : memref<16x?xf32>
// CHECK:               %[[VAL_104:.*]] = addf %[[VAL_100]], %[[VAL_103]] : f32
// CHECK:               %[[VAL_105:.*]] = krnl.load %[[VAL_30]]#3{{\[}}%[[VAL_66]]#1] : memref<4xf32>
// CHECK:               %[[VAL_106:.*]] = krnl.load %[[VAL_30]]#7{{\[}}%[[VAL_66]]#1] : memref<4xf32>
// CHECK:               %[[VAL_107:.*]] = addf %[[VAL_104]], %[[VAL_105]] : f32
// CHECK:               %[[VAL_108:.*]] = addf %[[VAL_107]], %[[VAL_106]] : f32
// CHECK:               %[[VAL_109:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_108]], %[[VAL_109]][] : memref<f32>
// CHECK:               %[[VAL_110:.*]] = "onnx.Tanh"(%[[VAL_109]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_111:.*]] = krnl.load %[[VAL_110]][] : memref<f32>
// CHECK:               %[[VAL_112:.*]] = mulf %[[VAL_97]], %[[VAL_67]] : f32
// CHECK:               %[[VAL_113:.*]] = mulf %[[VAL_80]], %[[VAL_111]] : f32
// CHECK:               %[[VAL_114:.*]] = addf %[[VAL_112]], %[[VAL_113]] : f32

// CHECK:               %[[VAL_115:.*]] = constant 4 : index
// CHECK:               %[[VAL_116:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_66]]#1]
// CHECK:               %[[VAL_117:.*]] = krnl.load %[[VAL_55]]{{\[}}%[[VAL_116]], %[[VAL_66]]#0] : memref<16x?xf32>
// CHECK:               %[[VAL_118:.*]] = constant 4 : index
// CHECK:               %[[VAL_119:.*]] = affine.apply #map{{.+}}(){{\[}}%[[VAL_66]]#1]
// CHECK:               %[[VAL_120:.*]] = krnl.load %[[VAL_57]]{{\[}}%[[VAL_119]], %[[VAL_66]]#0] : memref<16x?xf32>
// CHECK:               %[[VAL_121:.*]] = addf %[[VAL_117]], %[[VAL_120]] : f32
// CHECK:               %[[VAL_122:.*]] = krnl.load %[[VAL_30]]#1{{\[}}%[[VAL_66]]#1] : memref<4xf32>
// CHECK:               %[[VAL_123:.*]] = krnl.load %[[VAL_30]]#5{{\[}}%[[VAL_66]]#1] : memref<4xf32>
// CHECK:               %[[VAL_124:.*]] = addf %[[VAL_121]], %[[VAL_122]] : f32
// CHECK:               %[[VAL_125:.*]] = addf %[[VAL_124]], %[[VAL_123]] : f32
// CHECK:               %[[VAL_126:.*]] = krnl.load %[[VAL_32]]#1{{\[}}%[[VAL_66]]#1] : memref<4xf32>
// CHECK:               %[[VAL_127:.*]] = mulf %[[VAL_126]], %[[VAL_114]] : f32
// CHECK:               %[[VAL_128:.*]] = addf %[[VAL_125]], %[[VAL_127]] : f32
// CHECK:               %[[VAL_129:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_128]], %[[VAL_129]][] : memref<f32>
// CHECK:               %[[VAL_130:.*]] = "onnx.Sigmoid"(%[[VAL_129]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_131:.*]] = krnl.load %[[VAL_130]][] : memref<f32>
// CHECK:               %[[VAL_132:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_114]], %[[VAL_132]][] : memref<f32>
// CHECK:               %[[VAL_133:.*]] = "onnx.Tanh"(%[[VAL_132]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_134:.*]] = krnl.load %[[VAL_133]][] : memref<f32>
// CHECK:               %[[VAL_135:.*]] = mulf %[[VAL_131]], %[[VAL_134]] : f32
// CHECK:               krnl.store %[[VAL_114]], %[[VAL_16]]{{\[}}%[[VAL_66]]#0, %[[VAL_66]]#1] : memref<?x4xf32>
// CHECK:               krnl.store %[[VAL_135]], %[[VAL_13]]{{\[}}%[[VAL_66]]#0, %[[VAL_66]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_42]] : memref<?x?xf32>
// CHECK:           }
// CHECK:           %[[VAL_136:.*]] = constant 16 : i64
// CHECK:           %[[VAL_137:.*]] = constant 0 : index
// CHECK:           %[[VAL_138:.*]] = memref.dim %[[VAL_13]], %[[VAL_137]] : memref<?x4xf32>
// CHECK:           %[[VAL_139:.*]] = index_cast %[[VAL_138]] : index to i64
// CHECK:           %[[VAL_140:.*]] = muli %[[VAL_136]], %[[VAL_139]] : i64
// CHECK:           "krnl.memcpy"(%[[VAL_10]], %[[VAL_13]], %[[VAL_140]]) : (memref<1x?x4xf32>, memref<?x4xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_13]] : memref<?x4xf32>
// CHECK:           memref.dealloc %[[VAL_16]] : memref<?x4xf32>
// CHECK:           return %[[VAL_10]] : memref<1x?x4xf32>
// CHECK:         }

}
