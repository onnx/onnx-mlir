// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='test-rnn-ops-lowering' %s -split-input-file | FileCheck %s

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
// CHECK:           %[[VAL_21:.*]]:4 = "onnx.Split"(%[[VAL_19]]) {axis = 0 : si64} : (memref<16x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK:           %[[VAL_22:.*]] = "onnx.Transpose"(%[[VAL_21]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Transpose"(%[[VAL_21]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Transpose"(%[[VAL_21]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Transpose"(%[[VAL_21]]#3) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_26:.*]]:4 = "onnx.Split"(%[[VAL_20]]) {axis = 0 : si64} : (memref<16x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK:           %[[VAL_27:.*]] = "onnx.Transpose"(%[[VAL_26]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.Transpose"(%[[VAL_26]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Transpose"(%[[VAL_26]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.Transpose"(%[[VAL_26]]#3) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Squeeze"(%[[VAL_3]]) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK:           %[[VAL_32:.*]]:8 = "onnx.Split"(%[[VAL_31]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_33:.*]] = "onnx.Squeeze"(%[[VAL_6]]) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK:           %[[VAL_34:.*]]:3 = "onnx.Split"(%[[VAL_33]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_35:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_35]]) with (%[[VAL_35]] -> %[[VAL_36:.*]] = 0 to 7) {
// CHECK:             %[[VAL_37:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_38:.*]] = constant 0 : index
// CHECK:             %[[VAL_39:.*]] = constant 2 : index
// CHECK:             %[[VAL_40:.*]] = constant 3 : index
// CHECK:             %[[VAL_41:.*]] = constant 0 : index
// CHECK:             %[[VAL_42:.*]] = constant 0 : index
// CHECK:             %[[VAL_43:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_43]]#0, %[[VAL_43]]#1) with (%[[VAL_43]]#0 -> %[[VAL_44:.*]] = %[[VAL_41]] to %[[VAL_39]], %[[VAL_43]]#1 -> %[[VAL_45:.*]] = %[[VAL_42]] to %[[VAL_40]]) {
// CHECK:               %[[VAL_46:.*]]:2 = krnl.get_induction_var_value(%[[VAL_43]]#0, %[[VAL_43]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_47:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_36]], %[[VAL_46]]#0, %[[VAL_46]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_47]], %[[VAL_37]]{{\[}}%[[VAL_46]]#0, %[[VAL_46]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_48:.*]] = "onnx.MatMul"(%[[VAL_37]], %[[VAL_22]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_49:.*]] = "onnx.MatMul"(%[[VAL_8]], %[[VAL_27]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_50:.*]] = "onnx.MatMul"(%[[VAL_37]], %[[VAL_24]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_51:.*]] = "onnx.MatMul"(%[[VAL_8]], %[[VAL_29]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_52:.*]] = "onnx.MatMul"(%[[VAL_37]], %[[VAL_25]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_53:.*]] = "onnx.MatMul"(%[[VAL_8]], %[[VAL_30]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_54:.*]] = "onnx.MatMul"(%[[VAL_37]], %[[VAL_23]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.MatMul"(%[[VAL_8]], %[[VAL_28]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_56:.*]] = constant 2 : index
// CHECK:             %[[VAL_57:.*]] = constant 4 : index
// CHECK:             %[[VAL_58:.*]] = constant 0 : index
// CHECK:             %[[VAL_59:.*]] = constant 0 : index
// CHECK:             %[[VAL_60:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_60]]#0, %[[VAL_60]]#1) with (%[[VAL_60]]#0 -> %[[VAL_61:.*]] = %[[VAL_58]] to %[[VAL_56]], %[[VAL_60]]#1 -> %[[VAL_62:.*]] = %[[VAL_59]] to %[[VAL_57]]) {
// CHECK:               %[[VAL_63:.*]]:2 = krnl.get_induction_var_value(%[[VAL_60]]#0, %[[VAL_60]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_64:.*]] = krnl.load %[[VAL_7]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_65:.*]] = krnl.load %[[VAL_48]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_66:.*]] = krnl.load %[[VAL_49]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_67:.*]] = addf %[[VAL_65]], %[[VAL_66]] : f32
// CHECK:               %[[VAL_68:.*]] = krnl.load %[[VAL_32]]#0{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_69:.*]] = krnl.load %[[VAL_32]]#4{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_70:.*]] = addf %[[VAL_67]], %[[VAL_68]] : f32
// CHECK:               %[[VAL_71:.*]] = addf %[[VAL_70]], %[[VAL_69]] : f32
// CHECK:               %[[VAL_72:.*]] = krnl.load %[[VAL_34]]#0{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_73:.*]] = mulf %[[VAL_72]], %[[VAL_64]] : f32
// CHECK:               %[[VAL_74:.*]] = addf %[[VAL_71]], %[[VAL_73]] : f32
// CHECK:               %[[VAL_75:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_74]], %[[VAL_75]][] : memref<f32>
// CHECK:               %[[VAL_76:.*]] = "onnx.Sigmoid"(%[[VAL_75]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_77:.*]] = krnl.load %[[VAL_76]][] : memref<f32>
// CHECK:               %[[VAL_78:.*]] = krnl.load %[[VAL_50]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_79:.*]] = krnl.load %[[VAL_51]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_80:.*]] = addf %[[VAL_78]], %[[VAL_79]] : f32
// CHECK:               %[[VAL_81:.*]] = krnl.load %[[VAL_32]]#2{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_82:.*]] = krnl.load %[[VAL_32]]#6{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_83:.*]] = addf %[[VAL_80]], %[[VAL_81]] : f32
// CHECK:               %[[VAL_84:.*]] = addf %[[VAL_83]], %[[VAL_82]] : f32
// CHECK:               %[[VAL_85:.*]] = krnl.load %[[VAL_34]]#2{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_86:.*]] = mulf %[[VAL_85]], %[[VAL_64]] : f32
// CHECK:               %[[VAL_87:.*]] = addf %[[VAL_84]], %[[VAL_86]] : f32
// CHECK:               %[[VAL_88:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_87]], %[[VAL_88]][] : memref<f32>
// CHECK:               %[[VAL_89:.*]] = "onnx.Sigmoid"(%[[VAL_88]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_90:.*]] = krnl.load %[[VAL_89]][] : memref<f32>
// CHECK:               %[[VAL_91:.*]] = krnl.load %[[VAL_52]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_92:.*]] = krnl.load %[[VAL_53]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_93:.*]] = addf %[[VAL_91]], %[[VAL_92]] : f32
// CHECK:               %[[VAL_94:.*]] = krnl.load %[[VAL_32]]#3{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_95:.*]] = krnl.load %[[VAL_32]]#7{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_96:.*]] = addf %[[VAL_93]], %[[VAL_94]] : f32
// CHECK:               %[[VAL_97:.*]] = addf %[[VAL_96]], %[[VAL_95]] : f32
// CHECK:               %[[VAL_98:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_97]], %[[VAL_98]][] : memref<f32>
// CHECK:               %[[VAL_99:.*]] = "onnx.Tanh"(%[[VAL_98]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_100:.*]] = krnl.load %[[VAL_99]][] : memref<f32>
// CHECK:               %[[VAL_101:.*]] = mulf %[[VAL_90]], %[[VAL_64]] : f32
// CHECK:               %[[VAL_102:.*]] = mulf %[[VAL_77]], %[[VAL_100]] : f32
// CHECK:               %[[VAL_103:.*]] = addf %[[VAL_101]], %[[VAL_102]] : f32
// CHECK:               %[[VAL_104:.*]] = krnl.load %[[VAL_54]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_105:.*]] = krnl.load %[[VAL_55]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_106:.*]] = addf %[[VAL_104]], %[[VAL_105]] : f32
// CHECK:               %[[VAL_107:.*]] = krnl.load %[[VAL_32]]#1{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_108:.*]] = krnl.load %[[VAL_32]]#5{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_109:.*]] = addf %[[VAL_106]], %[[VAL_107]] : f32
// CHECK:               %[[VAL_110:.*]] = addf %[[VAL_109]], %[[VAL_108]] : f32
// CHECK:               %[[VAL_111:.*]] = krnl.load %[[VAL_34]]#1{{\[}}%[[VAL_63]]#1] : memref<4xf32>
// CHECK:               %[[VAL_112:.*]] = mulf %[[VAL_111]], %[[VAL_103]] : f32
// CHECK:               %[[VAL_113:.*]] = addf %[[VAL_110]], %[[VAL_112]] : f32
// CHECK:               %[[VAL_114:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_113]], %[[VAL_114]][] : memref<f32>
// CHECK:               %[[VAL_115:.*]] = "onnx.Sigmoid"(%[[VAL_114]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_116:.*]] = krnl.load %[[VAL_115]][] : memref<f32>
// CHECK:               %[[VAL_117:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_103]], %[[VAL_117]][] : memref<f32>
// CHECK:               %[[VAL_118:.*]] = "onnx.Tanh"(%[[VAL_117]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_119:.*]] = krnl.load %[[VAL_118]][] : memref<f32>
// CHECK:               %[[VAL_120:.*]] = mulf %[[VAL_116]], %[[VAL_119]] : f32
// CHECK:               krnl.store %[[VAL_103]], %[[VAL_7]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<2x4xf32>
// CHECK:               krnl.store %[[VAL_120]], %[[VAL_8]]{{\[}}%[[VAL_63]]#0, %[[VAL_63]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_37]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_121:.*]] = constant 32 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_9]], %[[VAL_8]], %[[VAL_121]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
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
// CHECK:           %[[VAL_21:.*]] = "krnl.global"() {name = "constant_6", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK:           %[[VAL_22:.*]] = "krnl.global"() {name = "constant_7", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK:           %[[VAL_23:.*]] = "krnl.global"() {name = "constant_8", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK:           %[[VAL_24:.*]] = "krnl.global"() {name = "constant_9", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK:           %[[VAL_25:.*]] = "krnl.global"() {name = "constant_10", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK:           %[[VAL_26:.*]] = "krnl.global"() {name = "constant_11", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK:           %[[VAL_27:.*]] = "krnl.global"() {name = "constant_12", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK:           %[[VAL_28:.*]] = "krnl.global"() {name = "constant_13", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK:           %[[VAL_29:.*]] = "krnl.global"() {name = "constant_14", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_30:.*]] = "krnl.global"() {name = "constant_15", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_31:.*]] = "krnl.global"() {name = "constant_16", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_32:.*]] = "krnl.global"() {name = "constant_17", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_33:.*]] = "krnl.global"() {name = "constant_18", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_34:.*]] = "krnl.global"() {name = "constant_19", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_35:.*]] = "krnl.global"() {name = "constant_20", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_36:.*]] = "krnl.global"() {name = "constant_21", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_37:.*]] = "krnl.global"() {name = "constant_22", shape = [32], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01, 2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01, 2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01]> : tensor<32xf32>} : () -> memref<32xf32>
// CHECK:           %[[VAL_38:.*]] = "krnl.global"() {name = "constant_23", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_39:.*]] = "krnl.global"() {name = "constant_24", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_40:.*]] = "krnl.global"() {name = "constant_25", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_41:.*]] = "krnl.global"() {name = "constant_26", shape = [4], value = dense<[1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_42:.*]] = "krnl.global"() {name = "constant_27", shape = [4], value = dense<[1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_43:.*]] = "krnl.global"() {name = "constant_28", shape = [4], value = dense<[2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_44:.*]] = "krnl.global"() {name = "constant_29", shape = [4], value = dense<[2.500000e+01, 2.600000e+01, 2.700000e+01, 2.800000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_45:.*]] = "krnl.global"() {name = "constant_30", shape = [4], value = dense<[2.900000e+01, 3.000000e+01, 3.100000e+01, 3.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_46:.*]] = "krnl.global"() {name = "constant_31", shape = [12], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<12xf32>} : () -> memref<12xf32>
// CHECK:           %[[VAL_47:.*]] = "krnl.global"() {name = "constant_32", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_48:.*]] = "krnl.global"() {name = "constant_33", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_49:.*]] = "krnl.global"() {name = "constant_34", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_50:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_50]]) with (%[[VAL_50]] -> %[[VAL_51:.*]] = 0 to 7) {
// CHECK:             %[[VAL_52:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_53:.*]] = constant 0 : index
// CHECK:             %[[VAL_54:.*]] = constant 2 : index
// CHECK:             %[[VAL_55:.*]] = constant 3 : index
// CHECK:             %[[VAL_56:.*]] = constant 0 : index
// CHECK:             %[[VAL_57:.*]] = constant 0 : index
// CHECK:             %[[VAL_58:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_58]]#0, %[[VAL_58]]#1) with (%[[VAL_58]]#0 -> %[[VAL_59:.*]] = %[[VAL_56]] to %[[VAL_54]], %[[VAL_58]]#1 -> %[[VAL_60:.*]] = %[[VAL_57]] to %[[VAL_55]]) {
// CHECK:               %[[VAL_61:.*]]:2 = krnl.get_induction_var_value(%[[VAL_58]]#0, %[[VAL_58]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_62:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_51]], %[[VAL_61]]#0, %[[VAL_61]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_62]], %[[VAL_52]]{{\[}}%[[VAL_61]]#0, %[[VAL_61]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_63:.*]] = "onnx.MatMul"(%[[VAL_52]], %[[VAL_25]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_64:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_33]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_65:.*]] = "onnx.MatMul"(%[[VAL_52]], %[[VAL_27]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_66:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_35]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_67:.*]] = "onnx.MatMul"(%[[VAL_52]], %[[VAL_28]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_68:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_36]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_69:.*]] = "onnx.MatMul"(%[[VAL_52]], %[[VAL_26]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_70:.*]] = "onnx.MatMul"(%[[VAL_4]], %[[VAL_34]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_71:.*]] = constant 2 : index
// CHECK:             %[[VAL_72:.*]] = constant 4 : index
// CHECK:             %[[VAL_73:.*]] = constant 0 : index
// CHECK:             %[[VAL_74:.*]] = constant 0 : index
// CHECK:             %[[VAL_75:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_75]]#0, %[[VAL_75]]#1) with (%[[VAL_75]]#0 -> %[[VAL_76:.*]] = %[[VAL_73]] to %[[VAL_71]], %[[VAL_75]]#1 -> %[[VAL_77:.*]] = %[[VAL_74]] to %[[VAL_72]]) {
// CHECK:               %[[VAL_78:.*]]:2 = krnl.get_induction_var_value(%[[VAL_75]]#0, %[[VAL_75]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_79:.*]] = krnl.load %[[VAL_3]]{{\[}}%[[VAL_78]]#0, %[[VAL_78]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_80:.*]] = krnl.load %[[VAL_63]]{{\[}}%[[VAL_78]]#0, %[[VAL_78]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_81:.*]] = krnl.load %[[VAL_64]]{{\[}}%[[VAL_78]]#0, %[[VAL_78]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_82:.*]] = addf %[[VAL_80]], %[[VAL_81]] : f32
// CHECK:               %[[VAL_83:.*]] = krnl.load %[[VAL_38]]{{\[}}%[[VAL_78]]#1] : memref<4xf32>
// CHECK:               %[[VAL_84:.*]] = krnl.load %[[VAL_42]]{{\[}}%[[VAL_78]]#1] : memref<4xf32>
// CHECK:               %[[VAL_85:.*]] = addf %[[VAL_82]], %[[VAL_83]] : f32
// CHECK:               %[[VAL_86:.*]] = addf %[[VAL_85]], %[[VAL_84]] : f32
// CHECK:               %[[VAL_87:.*]] = krnl.load %[[VAL_47]]{{\[}}%[[VAL_78]]#1] : memref<4xf32>
// CHECK:               %[[VAL_88:.*]] = mulf %[[VAL_87]], %[[VAL_79]] : f32
// CHECK:               %[[VAL_89:.*]] = addf %[[VAL_86]], %[[VAL_88]] : f32
// CHECK:               %[[VAL_90:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_89]], %[[VAL_90]][] : memref<f32>
// CHECK:               %[[VAL_91:.*]] = "onnx.Sigmoid"(%[[VAL_90]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_92:.*]] = krnl.load %[[VAL_91]][] : memref<f32>
// CHECK:               %[[VAL_93:.*]] = krnl.load %[[VAL_65]]{{\[}}%[[VAL_78]]#0, %[[VAL_78]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_94:.*]] = krnl.load %[[VAL_66]]{{\[}}%[[VAL_78]]#0, %[[VAL_78]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_95:.*]] = addf %[[VAL_93]], %[[VAL_94]] : f32
// CHECK:               %[[VAL_96:.*]] = krnl.load %[[VAL_40]]{{\[}}%[[VAL_78]]#1] : memref<4xf32>
// CHECK:               %[[VAL_97:.*]] = krnl.load %[[VAL_44]]{{\[}}%[[VAL_78]]#1] : memref<4xf32>
// CHECK:               %[[VAL_98:.*]] = addf %[[VAL_95]], %[[VAL_96]] : f32
// CHECK:               %[[VAL_99:.*]] = addf %[[VAL_98]], %[[VAL_97]] : f32
// CHECK:               %[[VAL_100:.*]] = krnl.load %[[VAL_49]]{{\[}}%[[VAL_78]]#1] : memref<4xf32>
// CHECK:               %[[VAL_101:.*]] = mulf %[[VAL_100]], %[[VAL_79]] : f32
// CHECK:               %[[VAL_102:.*]] = addf %[[VAL_99]], %[[VAL_101]] : f32
// CHECK:               %[[VAL_103:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_102]], %[[VAL_103]][] : memref<f32>
// CHECK:               %[[VAL_104:.*]] = "onnx.Sigmoid"(%[[VAL_103]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_105:.*]] = krnl.load %[[VAL_104]][] : memref<f32>
// CHECK:               %[[VAL_106:.*]] = krnl.load %[[VAL_67]]{{\[}}%[[VAL_78]]#0, %[[VAL_78]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_107:.*]] = krnl.load %[[VAL_68]]{{\[}}%[[VAL_78]]#0, %[[VAL_78]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_108:.*]] = addf %[[VAL_106]], %[[VAL_107]] : f32
// CHECK:               %[[VAL_109:.*]] = krnl.load %[[VAL_41]]{{\[}}%[[VAL_78]]#1] : memref<4xf32>
// CHECK:               %[[VAL_110:.*]] = krnl.load %[[VAL_45]]{{\[}}%[[VAL_78]]#1] : memref<4xf32>
// CHECK:               %[[VAL_111:.*]] = addf %[[VAL_108]], %[[VAL_109]] : f32
// CHECK:               %[[VAL_112:.*]] = addf %[[VAL_111]], %[[VAL_110]] : f32
// CHECK:               %[[VAL_113:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_112]], %[[VAL_113]][] : memref<f32>
// CHECK:               %[[VAL_114:.*]] = "onnx.Tanh"(%[[VAL_113]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_115:.*]] = krnl.load %[[VAL_114]][] : memref<f32>
// CHECK:               %[[VAL_116:.*]] = mulf %[[VAL_105]], %[[VAL_79]] : f32
// CHECK:               %[[VAL_117:.*]] = mulf %[[VAL_92]], %[[VAL_115]] : f32
// CHECK:               %[[VAL_118:.*]] = addf %[[VAL_116]], %[[VAL_117]] : f32
// CHECK:               %[[VAL_119:.*]] = krnl.load %[[VAL_69]]{{\[}}%[[VAL_78]]#0, %[[VAL_78]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_120:.*]] = krnl.load %[[VAL_70]]{{\[}}%[[VAL_78]]#0, %[[VAL_78]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_121:.*]] = addf %[[VAL_119]], %[[VAL_120]] : f32
// CHECK:               %[[VAL_122:.*]] = krnl.load %[[VAL_39]]{{\[}}%[[VAL_78]]#1] : memref<4xf32>
// CHECK:               %[[VAL_123:.*]] = krnl.load %[[VAL_43]]{{\[}}%[[VAL_78]]#1] : memref<4xf32>
// CHECK:               %[[VAL_124:.*]] = addf %[[VAL_121]], %[[VAL_122]] : f32
// CHECK:               %[[VAL_125:.*]] = addf %[[VAL_124]], %[[VAL_123]] : f32
// CHECK:               %[[VAL_126:.*]] = krnl.load %[[VAL_48]]{{\[}}%[[VAL_78]]#1] : memref<4xf32>
// CHECK:               %[[VAL_127:.*]] = mulf %[[VAL_126]], %[[VAL_118]] : f32
// CHECK:               %[[VAL_128:.*]] = addf %[[VAL_125]], %[[VAL_127]] : f32
// CHECK:               %[[VAL_129:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_128]], %[[VAL_129]][] : memref<f32>
// CHECK:               %[[VAL_130:.*]] = "onnx.Sigmoid"(%[[VAL_129]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_131:.*]] = krnl.load %[[VAL_130]][] : memref<f32>
// CHECK:               %[[VAL_132:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_118]], %[[VAL_132]][] : memref<f32>
// CHECK:               %[[VAL_133:.*]] = "onnx.Tanh"(%[[VAL_132]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_134:.*]] = krnl.load %[[VAL_133]][] : memref<f32>
// CHECK:               %[[VAL_135:.*]] = mulf %[[VAL_131]], %[[VAL_134]] : f32
// CHECK:               krnl.store %[[VAL_118]], %[[VAL_3]]{{\[}}%[[VAL_78]]#0, %[[VAL_78]]#1] : memref<2x4xf32>
// CHECK:               krnl.store %[[VAL_135]], %[[VAL_4]]{{\[}}%[[VAL_78]]#0, %[[VAL_78]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_52]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_136:.*]] = constant 32 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_5]], %[[VAL_4]], %[[VAL_136]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
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
// CHECK-SAME:        %[[VAL_0:.*]]: memref<7x2x3xf32>, %[[VAL_1:.*]]: memref<1x16x3xf32>, %[[VAL_2:.*]]: memref<1x16x4xf32>, %[[VAL_3:.*]]: memref<1x32xf32>, %[[VAL_4:.*]]: memref<1x2x4xf32>, %[[VAL_5:.*]]: memref<1x2x4xf32>, %[[VAL_6:.*]]: memref<1x12xf32>) -> memref<1x2x4xf32> {
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
// CHECK:           %[[VAL_21:.*]]:4 = "onnx.Split"(%[[VAL_19]]) {axis = 0 : si64} : (memref<16x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK:           %[[VAL_22:.*]] = "onnx.Transpose"(%[[VAL_21]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Transpose"(%[[VAL_21]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Transpose"(%[[VAL_21]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Transpose"(%[[VAL_21]]#3) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_26:.*]]:4 = "onnx.Split"(%[[VAL_20]]) {axis = 0 : si64} : (memref<16x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK:           %[[VAL_27:.*]] = "onnx.Transpose"(%[[VAL_26]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_28:.*]] = "onnx.Transpose"(%[[VAL_26]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Transpose"(%[[VAL_26]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.Transpose"(%[[VAL_26]]#3) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Squeeze"(%[[VAL_3]]) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK:           %[[VAL_32:.*]]:8 = "onnx.Split"(%[[VAL_31]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_33:.*]] = "onnx.Squeeze"(%[[VAL_6]]) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK:           %[[VAL_34:.*]]:3 = "onnx.Split"(%[[VAL_33]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_35:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_35]]) with (%[[VAL_35]] -> %[[VAL_36:.*]] = 0 to 7) {
// CHECK:             %[[VAL_37:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_38:.*]] = constant 0 : index
// CHECK:             %[[VAL_39:.*]] = constant 7 : index
// CHECK:             %[[VAL_40:.*]] = affine.apply #map(%[[VAL_36]]){{\[}}%[[VAL_39]]]
// CHECK:             %[[VAL_41:.*]] = constant 2 : index
// CHECK:             %[[VAL_42:.*]] = constant 3 : index
// CHECK:             %[[VAL_43:.*]] = constant 0 : index
// CHECK:             %[[VAL_44:.*]] = constant 0 : index
// CHECK:             %[[VAL_45:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_45]]#0, %[[VAL_45]]#1) with (%[[VAL_45]]#0 -> %[[VAL_46:.*]] = %[[VAL_43]] to %[[VAL_41]], %[[VAL_45]]#1 -> %[[VAL_47:.*]] = %[[VAL_44]] to %[[VAL_42]]) {
// CHECK:               %[[VAL_48:.*]]:2 = krnl.get_induction_var_value(%[[VAL_45]]#0, %[[VAL_45]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_49:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_40]], %[[VAL_48]]#0, %[[VAL_48]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_49]], %[[VAL_37]]{{\[}}%[[VAL_48]]#0, %[[VAL_48]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_50:.*]] = "onnx.MatMul"(%[[VAL_37]], %[[VAL_22]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_51:.*]] = "onnx.MatMul"(%[[VAL_8]], %[[VAL_27]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_52:.*]] = "onnx.MatMul"(%[[VAL_37]], %[[VAL_24]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_53:.*]] = "onnx.MatMul"(%[[VAL_8]], %[[VAL_29]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_54:.*]] = "onnx.MatMul"(%[[VAL_37]], %[[VAL_25]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.MatMul"(%[[VAL_8]], %[[VAL_30]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_56:.*]] = "onnx.MatMul"(%[[VAL_37]], %[[VAL_23]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_57:.*]] = "onnx.MatMul"(%[[VAL_8]], %[[VAL_28]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_58:.*]] = constant 2 : index
// CHECK:             %[[VAL_59:.*]] = constant 4 : index
// CHECK:             %[[VAL_60:.*]] = constant 0 : index
// CHECK:             %[[VAL_61:.*]] = constant 0 : index
// CHECK:             %[[VAL_62:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_62]]#0, %[[VAL_62]]#1) with (%[[VAL_62]]#0 -> %[[VAL_63:.*]] = %[[VAL_60]] to %[[VAL_58]], %[[VAL_62]]#1 -> %[[VAL_64:.*]] = %[[VAL_61]] to %[[VAL_59]]) {
// CHECK:               %[[VAL_65:.*]]:2 = krnl.get_induction_var_value(%[[VAL_62]]#0, %[[VAL_62]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_66:.*]] = krnl.load %[[VAL_7]]{{\[}}%[[VAL_65]]#0, %[[VAL_65]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_67:.*]] = krnl.load %[[VAL_50]]{{\[}}%[[VAL_65]]#0, %[[VAL_65]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_68:.*]] = krnl.load %[[VAL_51]]{{\[}}%[[VAL_65]]#0, %[[VAL_65]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_69:.*]] = addf %[[VAL_67]], %[[VAL_68]] : f32
// CHECK:               %[[VAL_70:.*]] = krnl.load %[[VAL_32]]#0{{\[}}%[[VAL_65]]#1] : memref<4xf32>
// CHECK:               %[[VAL_71:.*]] = krnl.load %[[VAL_32]]#4{{\[}}%[[VAL_65]]#1] : memref<4xf32>
// CHECK:               %[[VAL_72:.*]] = addf %[[VAL_69]], %[[VAL_70]] : f32
// CHECK:               %[[VAL_73:.*]] = addf %[[VAL_72]], %[[VAL_71]] : f32
// CHECK:               %[[VAL_74:.*]] = krnl.load %[[VAL_34]]#0{{\[}}%[[VAL_65]]#1] : memref<4xf32>
// CHECK:               %[[VAL_75:.*]] = mulf %[[VAL_74]], %[[VAL_66]] : f32
// CHECK:               %[[VAL_76:.*]] = addf %[[VAL_73]], %[[VAL_75]] : f32
// CHECK:               %[[VAL_77:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_76]], %[[VAL_77]][] : memref<f32>
// CHECK:               %[[VAL_78:.*]] = "onnx.Sigmoid"(%[[VAL_77]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_79:.*]] = krnl.load %[[VAL_78]][] : memref<f32>
// CHECK:               %[[VAL_80:.*]] = krnl.load %[[VAL_52]]{{\[}}%[[VAL_65]]#0, %[[VAL_65]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_81:.*]] = krnl.load %[[VAL_53]]{{\[}}%[[VAL_65]]#0, %[[VAL_65]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_82:.*]] = addf %[[VAL_80]], %[[VAL_81]] : f32
// CHECK:               %[[VAL_83:.*]] = krnl.load %[[VAL_32]]#2{{\[}}%[[VAL_65]]#1] : memref<4xf32>
// CHECK:               %[[VAL_84:.*]] = krnl.load %[[VAL_32]]#6{{\[}}%[[VAL_65]]#1] : memref<4xf32>
// CHECK:               %[[VAL_85:.*]] = addf %[[VAL_82]], %[[VAL_83]] : f32
// CHECK:               %[[VAL_86:.*]] = addf %[[VAL_85]], %[[VAL_84]] : f32
// CHECK:               %[[VAL_87:.*]] = krnl.load %[[VAL_34]]#2{{\[}}%[[VAL_65]]#1] : memref<4xf32>
// CHECK:               %[[VAL_88:.*]] = mulf %[[VAL_87]], %[[VAL_66]] : f32
// CHECK:               %[[VAL_89:.*]] = addf %[[VAL_86]], %[[VAL_88]] : f32
// CHECK:               %[[VAL_90:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_89]], %[[VAL_90]][] : memref<f32>
// CHECK:               %[[VAL_91:.*]] = "onnx.Sigmoid"(%[[VAL_90]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_92:.*]] = krnl.load %[[VAL_91]][] : memref<f32>
// CHECK:               %[[VAL_93:.*]] = krnl.load %[[VAL_54]]{{\[}}%[[VAL_65]]#0, %[[VAL_65]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_94:.*]] = krnl.load %[[VAL_55]]{{\[}}%[[VAL_65]]#0, %[[VAL_65]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_95:.*]] = addf %[[VAL_93]], %[[VAL_94]] : f32
// CHECK:               %[[VAL_96:.*]] = krnl.load %[[VAL_32]]#3{{\[}}%[[VAL_65]]#1] : memref<4xf32>
// CHECK:               %[[VAL_97:.*]] = krnl.load %[[VAL_32]]#7{{\[}}%[[VAL_65]]#1] : memref<4xf32>
// CHECK:               %[[VAL_98:.*]] = addf %[[VAL_95]], %[[VAL_96]] : f32
// CHECK:               %[[VAL_99:.*]] = addf %[[VAL_98]], %[[VAL_97]] : f32
// CHECK:               %[[VAL_100:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_99]], %[[VAL_100]][] : memref<f32>
// CHECK:               %[[VAL_101:.*]] = "onnx.Tanh"(%[[VAL_100]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_102:.*]] = krnl.load %[[VAL_101]][] : memref<f32>
// CHECK:               %[[VAL_103:.*]] = mulf %[[VAL_92]], %[[VAL_66]] : f32
// CHECK:               %[[VAL_104:.*]] = mulf %[[VAL_79]], %[[VAL_102]] : f32
// CHECK:               %[[VAL_105:.*]] = addf %[[VAL_103]], %[[VAL_104]] : f32
// CHECK:               %[[VAL_106:.*]] = krnl.load %[[VAL_56]]{{\[}}%[[VAL_65]]#0, %[[VAL_65]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_107:.*]] = krnl.load %[[VAL_57]]{{\[}}%[[VAL_65]]#0, %[[VAL_65]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_108:.*]] = addf %[[VAL_106]], %[[VAL_107]] : f32
// CHECK:               %[[VAL_109:.*]] = krnl.load %[[VAL_32]]#1{{\[}}%[[VAL_65]]#1] : memref<4xf32>
// CHECK:               %[[VAL_110:.*]] = krnl.load %[[VAL_32]]#5{{\[}}%[[VAL_65]]#1] : memref<4xf32>
// CHECK:               %[[VAL_111:.*]] = addf %[[VAL_108]], %[[VAL_109]] : f32
// CHECK:               %[[VAL_112:.*]] = addf %[[VAL_111]], %[[VAL_110]] : f32
// CHECK:               %[[VAL_113:.*]] = krnl.load %[[VAL_34]]#1{{\[}}%[[VAL_65]]#1] : memref<4xf32>
// CHECK:               %[[VAL_114:.*]] = mulf %[[VAL_113]], %[[VAL_105]] : f32
// CHECK:               %[[VAL_115:.*]] = addf %[[VAL_112]], %[[VAL_114]] : f32
// CHECK:               %[[VAL_116:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_115]], %[[VAL_116]][] : memref<f32>
// CHECK:               %[[VAL_117:.*]] = "onnx.Sigmoid"(%[[VAL_116]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_118:.*]] = krnl.load %[[VAL_117]][] : memref<f32>
// CHECK:               %[[VAL_119:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_105]], %[[VAL_119]][] : memref<f32>
// CHECK:               %[[VAL_120:.*]] = "onnx.Tanh"(%[[VAL_119]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_121:.*]] = krnl.load %[[VAL_120]][] : memref<f32>
// CHECK:               %[[VAL_122:.*]] = mulf %[[VAL_118]], %[[VAL_121]] : f32
// CHECK:               krnl.store %[[VAL_105]], %[[VAL_7]]{{\[}}%[[VAL_65]]#0, %[[VAL_65]]#1] : memref<2x4xf32>
// CHECK:               krnl.store %[[VAL_122]], %[[VAL_8]]{{\[}}%[[VAL_65]]#0, %[[VAL_65]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_37]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_123:.*]] = constant 32 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_9]], %[[VAL_8]], %[[VAL_123]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
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
// CHECK:           %[[VAL_29:.*]]:4 = "onnx.Split"(%[[VAL_24]]) {axis = 0 : si64} : (memref<16x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK:           %[[VAL_30:.*]] = "onnx.Transpose"(%[[VAL_29]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Transpose"(%[[VAL_29]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.Transpose"(%[[VAL_29]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_33:.*]] = "onnx.Transpose"(%[[VAL_29]]#3) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_34:.*]]:4 = "onnx.Split"(%[[VAL_27]]) {axis = 0 : si64} : (memref<16x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK:           %[[VAL_35:.*]] = "onnx.Transpose"(%[[VAL_34]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Transpose"(%[[VAL_34]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.Transpose"(%[[VAL_34]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Transpose"(%[[VAL_34]]#3) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_39:.*]]:4 = "onnx.Split"(%[[VAL_25]]) {axis = 0 : si64} : (memref<16x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK:           %[[VAL_40:.*]] = "onnx.Transpose"(%[[VAL_39]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_41:.*]] = "onnx.Transpose"(%[[VAL_39]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_42:.*]] = "onnx.Transpose"(%[[VAL_39]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_43:.*]] = "onnx.Transpose"(%[[VAL_39]]#3) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_44:.*]]:4 = "onnx.Split"(%[[VAL_28]]) {axis = 0 : si64} : (memref<16x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK:           %[[VAL_45:.*]] = "onnx.Transpose"(%[[VAL_44]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_46:.*]] = "onnx.Transpose"(%[[VAL_44]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_47:.*]] = "onnx.Transpose"(%[[VAL_44]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_48:.*]] = "onnx.Transpose"(%[[VAL_44]]#3) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_49:.*]]:2 = "onnx.Split"(%[[VAL_3]]) {axis = 0 : si64} : (memref<2x32xf32>) -> (memref<1x32xf32>, memref<1x32xf32>)
// CHECK:           %[[VAL_50:.*]] = "onnx.Squeeze"(%[[VAL_49]]#0) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK:           %[[VAL_51:.*]] = "onnx.Squeeze"(%[[VAL_49]]#1) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK:           %[[VAL_52:.*]]:8 = "onnx.Split"(%[[VAL_50]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_53:.*]]:8 = "onnx.Split"(%[[VAL_51]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_54:.*]]:2 = "onnx.Split"(%[[VAL_6]]) {axis = 0 : si64} : (memref<2x12xf32>) -> (memref<1x12xf32>, memref<1x12xf32>)
// CHECK:           %[[VAL_55:.*]] = "onnx.Squeeze"(%[[VAL_54]]#0) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK:           %[[VAL_56:.*]] = "onnx.Squeeze"(%[[VAL_54]]#1) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK:           %[[VAL_57:.*]]:3 = "onnx.Split"(%[[VAL_55]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_58:.*]]:3 = "onnx.Split"(%[[VAL_56]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_59:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_59]]) with (%[[VAL_59]] -> %[[VAL_60:.*]] = 0 to 7) {
// CHECK:             %[[VAL_61:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_62:.*]] = constant 0 : index
// CHECK:             %[[VAL_63:.*]] = constant 2 : index
// CHECK:             %[[VAL_64:.*]] = constant 3 : index
// CHECK:             %[[VAL_65:.*]] = constant 0 : index
// CHECK:             %[[VAL_66:.*]] = constant 0 : index
// CHECK:             %[[VAL_67:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_67]]#0, %[[VAL_67]]#1) with (%[[VAL_67]]#0 -> %[[VAL_68:.*]] = %[[VAL_65]] to %[[VAL_63]], %[[VAL_67]]#1 -> %[[VAL_69:.*]] = %[[VAL_66]] to %[[VAL_64]]) {
// CHECK:               %[[VAL_70:.*]]:2 = krnl.get_induction_var_value(%[[VAL_67]]#0, %[[VAL_67]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_71:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_60]], %[[VAL_70]]#0, %[[VAL_70]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_71]], %[[VAL_61]]{{\[}}%[[VAL_70]]#0, %[[VAL_70]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_72:.*]] = "onnx.MatMul"(%[[VAL_61]], %[[VAL_30]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_73:.*]] = "onnx.MatMul"(%[[VAL_10]], %[[VAL_35]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_74:.*]] = "onnx.MatMul"(%[[VAL_61]], %[[VAL_32]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_75:.*]] = "onnx.MatMul"(%[[VAL_10]], %[[VAL_37]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_76:.*]] = "onnx.MatMul"(%[[VAL_61]], %[[VAL_33]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_77:.*]] = "onnx.MatMul"(%[[VAL_10]], %[[VAL_38]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_78:.*]] = "onnx.MatMul"(%[[VAL_61]], %[[VAL_31]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_79:.*]] = "onnx.MatMul"(%[[VAL_10]], %[[VAL_36]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_80:.*]] = constant 2 : index
// CHECK:             %[[VAL_81:.*]] = constant 4 : index
// CHECK:             %[[VAL_82:.*]] = constant 0 : index
// CHECK:             %[[VAL_83:.*]] = constant 0 : index
// CHECK:             %[[VAL_84:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_84]]#0, %[[VAL_84]]#1) with (%[[VAL_84]]#0 -> %[[VAL_85:.*]] = %[[VAL_82]] to %[[VAL_80]], %[[VAL_84]]#1 -> %[[VAL_86:.*]] = %[[VAL_83]] to %[[VAL_81]]) {
// CHECK:               %[[VAL_87:.*]]:2 = krnl.get_induction_var_value(%[[VAL_84]]#0, %[[VAL_84]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_88:.*]] = krnl.load %[[VAL_9]]{{\[}}%[[VAL_87]]#0, %[[VAL_87]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_89:.*]] = krnl.load %[[VAL_72]]{{\[}}%[[VAL_87]]#0, %[[VAL_87]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_90:.*]] = krnl.load %[[VAL_73]]{{\[}}%[[VAL_87]]#0, %[[VAL_87]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_91:.*]] = addf %[[VAL_89]], %[[VAL_90]] : f32
// CHECK:               %[[VAL_92:.*]] = krnl.load %[[VAL_52]]#0{{\[}}%[[VAL_87]]#1] : memref<4xf32>
// CHECK:               %[[VAL_93:.*]] = krnl.load %[[VAL_52]]#4{{\[}}%[[VAL_87]]#1] : memref<4xf32>
// CHECK:               %[[VAL_94:.*]] = addf %[[VAL_91]], %[[VAL_92]] : f32
// CHECK:               %[[VAL_95:.*]] = addf %[[VAL_94]], %[[VAL_93]] : f32
// CHECK:               %[[VAL_96:.*]] = krnl.load %[[VAL_57]]#0{{\[}}%[[VAL_87]]#1] : memref<4xf32>
// CHECK:               %[[VAL_97:.*]] = mulf %[[VAL_96]], %[[VAL_88]] : f32
// CHECK:               %[[VAL_98:.*]] = addf %[[VAL_95]], %[[VAL_97]] : f32
// CHECK:               %[[VAL_99:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_98]], %[[VAL_99]][] : memref<f32>
// CHECK:               %[[VAL_100:.*]] = "onnx.Sigmoid"(%[[VAL_99]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_101:.*]] = krnl.load %[[VAL_100]][] : memref<f32>
// CHECK:               %[[VAL_102:.*]] = krnl.load %[[VAL_74]]{{\[}}%[[VAL_87]]#0, %[[VAL_87]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_103:.*]] = krnl.load %[[VAL_75]]{{\[}}%[[VAL_87]]#0, %[[VAL_87]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_104:.*]] = addf %[[VAL_102]], %[[VAL_103]] : f32
// CHECK:               %[[VAL_105:.*]] = krnl.load %[[VAL_52]]#2{{\[}}%[[VAL_87]]#1] : memref<4xf32>
// CHECK:               %[[VAL_106:.*]] = krnl.load %[[VAL_52]]#6{{\[}}%[[VAL_87]]#1] : memref<4xf32>
// CHECK:               %[[VAL_107:.*]] = addf %[[VAL_104]], %[[VAL_105]] : f32
// CHECK:               %[[VAL_108:.*]] = addf %[[VAL_107]], %[[VAL_106]] : f32
// CHECK:               %[[VAL_109:.*]] = krnl.load %[[VAL_57]]#2{{\[}}%[[VAL_87]]#1] : memref<4xf32>
// CHECK:               %[[VAL_110:.*]] = mulf %[[VAL_109]], %[[VAL_88]] : f32
// CHECK:               %[[VAL_111:.*]] = addf %[[VAL_108]], %[[VAL_110]] : f32
// CHECK:               %[[VAL_112:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_111]], %[[VAL_112]][] : memref<f32>
// CHECK:               %[[VAL_113:.*]] = "onnx.Sigmoid"(%[[VAL_112]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_114:.*]] = krnl.load %[[VAL_113]][] : memref<f32>
// CHECK:               %[[VAL_115:.*]] = krnl.load %[[VAL_76]]{{\[}}%[[VAL_87]]#0, %[[VAL_87]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_116:.*]] = krnl.load %[[VAL_77]]{{\[}}%[[VAL_87]]#0, %[[VAL_87]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_117:.*]] = addf %[[VAL_115]], %[[VAL_116]] : f32
// CHECK:               %[[VAL_118:.*]] = krnl.load %[[VAL_52]]#3{{\[}}%[[VAL_87]]#1] : memref<4xf32>
// CHECK:               %[[VAL_119:.*]] = krnl.load %[[VAL_52]]#7{{\[}}%[[VAL_87]]#1] : memref<4xf32>
// CHECK:               %[[VAL_120:.*]] = addf %[[VAL_117]], %[[VAL_118]] : f32
// CHECK:               %[[VAL_121:.*]] = addf %[[VAL_120]], %[[VAL_119]] : f32
// CHECK:               %[[VAL_122:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_121]], %[[VAL_122]][] : memref<f32>
// CHECK:               %[[VAL_123:.*]] = "onnx.Tanh"(%[[VAL_122]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_124:.*]] = krnl.load %[[VAL_123]][] : memref<f32>
// CHECK:               %[[VAL_125:.*]] = mulf %[[VAL_114]], %[[VAL_88]] : f32
// CHECK:               %[[VAL_126:.*]] = mulf %[[VAL_101]], %[[VAL_124]] : f32
// CHECK:               %[[VAL_127:.*]] = addf %[[VAL_125]], %[[VAL_126]] : f32
// CHECK:               %[[VAL_128:.*]] = krnl.load %[[VAL_78]]{{\[}}%[[VAL_87]]#0, %[[VAL_87]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_129:.*]] = krnl.load %[[VAL_79]]{{\[}}%[[VAL_87]]#0, %[[VAL_87]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_130:.*]] = addf %[[VAL_128]], %[[VAL_129]] : f32
// CHECK:               %[[VAL_131:.*]] = krnl.load %[[VAL_52]]#1{{\[}}%[[VAL_87]]#1] : memref<4xf32>
// CHECK:               %[[VAL_132:.*]] = krnl.load %[[VAL_52]]#5{{\[}}%[[VAL_87]]#1] : memref<4xf32>
// CHECK:               %[[VAL_133:.*]] = addf %[[VAL_130]], %[[VAL_131]] : f32
// CHECK:               %[[VAL_134:.*]] = addf %[[VAL_133]], %[[VAL_132]] : f32
// CHECK:               %[[VAL_135:.*]] = krnl.load %[[VAL_57]]#1{{\[}}%[[VAL_87]]#1] : memref<4xf32>
// CHECK:               %[[VAL_136:.*]] = mulf %[[VAL_135]], %[[VAL_127]] : f32
// CHECK:               %[[VAL_137:.*]] = addf %[[VAL_134]], %[[VAL_136]] : f32
// CHECK:               %[[VAL_138:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_137]], %[[VAL_138]][] : memref<f32>
// CHECK:               %[[VAL_139:.*]] = "onnx.Sigmoid"(%[[VAL_138]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_140:.*]] = krnl.load %[[VAL_139]][] : memref<f32>
// CHECK:               %[[VAL_141:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_127]], %[[VAL_141]][] : memref<f32>
// CHECK:               %[[VAL_142:.*]] = "onnx.Tanh"(%[[VAL_141]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_143:.*]] = krnl.load %[[VAL_142]][] : memref<f32>
// CHECK:               %[[VAL_144:.*]] = mulf %[[VAL_140]], %[[VAL_143]] : f32
// CHECK:               krnl.store %[[VAL_127]], %[[VAL_9]]{{\[}}%[[VAL_87]]#0, %[[VAL_87]]#1] : memref<2x4xf32>
// CHECK:               krnl.store %[[VAL_144]], %[[VAL_10]]{{\[}}%[[VAL_87]]#0, %[[VAL_87]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_61]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_145:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_145]]) with (%[[VAL_145]] -> %[[VAL_146:.*]] = 0 to 7) {
// CHECK:             %[[VAL_147:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_148:.*]] = constant 1 : index
// CHECK:             %[[VAL_149:.*]] = constant 7 : index
// CHECK:             %[[VAL_150:.*]] = affine.apply #map(%[[VAL_146]]){{\[}}%[[VAL_149]]]
// CHECK:             %[[VAL_151:.*]] = constant 2 : index
// CHECK:             %[[VAL_152:.*]] = constant 3 : index
// CHECK:             %[[VAL_153:.*]] = constant 0 : index
// CHECK:             %[[VAL_154:.*]] = constant 0 : index
// CHECK:             %[[VAL_155:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_155]]#0, %[[VAL_155]]#1) with (%[[VAL_155]]#0 -> %[[VAL_156:.*]] = %[[VAL_153]] to %[[VAL_151]], %[[VAL_155]]#1 -> %[[VAL_157:.*]] = %[[VAL_154]] to %[[VAL_152]]) {
// CHECK:               %[[VAL_158:.*]]:2 = krnl.get_induction_var_value(%[[VAL_155]]#0, %[[VAL_155]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_159:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_150]], %[[VAL_158]]#0, %[[VAL_158]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_159]], %[[VAL_147]]{{\[}}%[[VAL_158]]#0, %[[VAL_158]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_160:.*]] = "onnx.MatMul"(%[[VAL_147]], %[[VAL_40]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_161:.*]] = "onnx.MatMul"(%[[VAL_8]], %[[VAL_45]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_162:.*]] = "onnx.MatMul"(%[[VAL_147]], %[[VAL_42]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_163:.*]] = "onnx.MatMul"(%[[VAL_8]], %[[VAL_47]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_164:.*]] = "onnx.MatMul"(%[[VAL_147]], %[[VAL_43]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_165:.*]] = "onnx.MatMul"(%[[VAL_8]], %[[VAL_48]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_166:.*]] = "onnx.MatMul"(%[[VAL_147]], %[[VAL_41]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_167:.*]] = "onnx.MatMul"(%[[VAL_8]], %[[VAL_46]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_168:.*]] = constant 2 : index
// CHECK:             %[[VAL_169:.*]] = constant 4 : index
// CHECK:             %[[VAL_170:.*]] = constant 0 : index
// CHECK:             %[[VAL_171:.*]] = constant 0 : index
// CHECK:             %[[VAL_172:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_172]]#0, %[[VAL_172]]#1) with (%[[VAL_172]]#0 -> %[[VAL_173:.*]] = %[[VAL_170]] to %[[VAL_168]], %[[VAL_172]]#1 -> %[[VAL_174:.*]] = %[[VAL_171]] to %[[VAL_169]]) {
// CHECK:               %[[VAL_175:.*]]:2 = krnl.get_induction_var_value(%[[VAL_172]]#0, %[[VAL_172]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_176:.*]] = krnl.load %[[VAL_7]]{{\[}}%[[VAL_175]]#0, %[[VAL_175]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_177:.*]] = krnl.load %[[VAL_160]]{{\[}}%[[VAL_175]]#0, %[[VAL_175]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_178:.*]] = krnl.load %[[VAL_161]]{{\[}}%[[VAL_175]]#0, %[[VAL_175]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_179:.*]] = addf %[[VAL_177]], %[[VAL_178]] : f32
// CHECK:               %[[VAL_180:.*]] = krnl.load %[[VAL_53]]#0{{\[}}%[[VAL_175]]#1] : memref<4xf32>
// CHECK:               %[[VAL_181:.*]] = krnl.load %[[VAL_53]]#4{{\[}}%[[VAL_175]]#1] : memref<4xf32>
// CHECK:               %[[VAL_182:.*]] = addf %[[VAL_179]], %[[VAL_180]] : f32
// CHECK:               %[[VAL_183:.*]] = addf %[[VAL_182]], %[[VAL_181]] : f32
// CHECK:               %[[VAL_184:.*]] = krnl.load %[[VAL_58]]#0{{\[}}%[[VAL_175]]#1] : memref<4xf32>
// CHECK:               %[[VAL_185:.*]] = mulf %[[VAL_184]], %[[VAL_176]] : f32
// CHECK:               %[[VAL_186:.*]] = addf %[[VAL_183]], %[[VAL_185]] : f32
// CHECK:               %[[VAL_187:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_186]], %[[VAL_187]][] : memref<f32>
// CHECK:               %[[VAL_188:.*]] = "onnx.Sigmoid"(%[[VAL_187]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_189:.*]] = krnl.load %[[VAL_188]][] : memref<f32>
// CHECK:               %[[VAL_190:.*]] = krnl.load %[[VAL_162]]{{\[}}%[[VAL_175]]#0, %[[VAL_175]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_191:.*]] = krnl.load %[[VAL_163]]{{\[}}%[[VAL_175]]#0, %[[VAL_175]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_192:.*]] = addf %[[VAL_190]], %[[VAL_191]] : f32
// CHECK:               %[[VAL_193:.*]] = krnl.load %[[VAL_53]]#2{{\[}}%[[VAL_175]]#1] : memref<4xf32>
// CHECK:               %[[VAL_194:.*]] = krnl.load %[[VAL_53]]#6{{\[}}%[[VAL_175]]#1] : memref<4xf32>
// CHECK:               %[[VAL_195:.*]] = addf %[[VAL_192]], %[[VAL_193]] : f32
// CHECK:               %[[VAL_196:.*]] = addf %[[VAL_195]], %[[VAL_194]] : f32
// CHECK:               %[[VAL_197:.*]] = krnl.load %[[VAL_58]]#2{{\[}}%[[VAL_175]]#1] : memref<4xf32>
// CHECK:               %[[VAL_198:.*]] = mulf %[[VAL_197]], %[[VAL_176]] : f32
// CHECK:               %[[VAL_199:.*]] = addf %[[VAL_196]], %[[VAL_198]] : f32
// CHECK:               %[[VAL_200:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_199]], %[[VAL_200]][] : memref<f32>
// CHECK:               %[[VAL_201:.*]] = "onnx.Sigmoid"(%[[VAL_200]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_202:.*]] = krnl.load %[[VAL_201]][] : memref<f32>
// CHECK:               %[[VAL_203:.*]] = krnl.load %[[VAL_164]]{{\[}}%[[VAL_175]]#0, %[[VAL_175]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_204:.*]] = krnl.load %[[VAL_165]]{{\[}}%[[VAL_175]]#0, %[[VAL_175]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_205:.*]] = addf %[[VAL_203]], %[[VAL_204]] : f32
// CHECK:               %[[VAL_206:.*]] = krnl.load %[[VAL_53]]#3{{\[}}%[[VAL_175]]#1] : memref<4xf32>
// CHECK:               %[[VAL_207:.*]] = krnl.load %[[VAL_53]]#7{{\[}}%[[VAL_175]]#1] : memref<4xf32>
// CHECK:               %[[VAL_208:.*]] = addf %[[VAL_205]], %[[VAL_206]] : f32
// CHECK:               %[[VAL_209:.*]] = addf %[[VAL_208]], %[[VAL_207]] : f32
// CHECK:               %[[VAL_210:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_209]], %[[VAL_210]][] : memref<f32>
// CHECK:               %[[VAL_211:.*]] = "onnx.Tanh"(%[[VAL_210]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_212:.*]] = krnl.load %[[VAL_211]][] : memref<f32>
// CHECK:               %[[VAL_213:.*]] = mulf %[[VAL_202]], %[[VAL_176]] : f32
// CHECK:               %[[VAL_214:.*]] = mulf %[[VAL_189]], %[[VAL_212]] : f32
// CHECK:               %[[VAL_215:.*]] = addf %[[VAL_213]], %[[VAL_214]] : f32
// CHECK:               %[[VAL_216:.*]] = krnl.load %[[VAL_166]]{{\[}}%[[VAL_175]]#0, %[[VAL_175]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_217:.*]] = krnl.load %[[VAL_167]]{{\[}}%[[VAL_175]]#0, %[[VAL_175]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_218:.*]] = addf %[[VAL_216]], %[[VAL_217]] : f32
// CHECK:               %[[VAL_219:.*]] = krnl.load %[[VAL_53]]#1{{\[}}%[[VAL_175]]#1] : memref<4xf32>
// CHECK:               %[[VAL_220:.*]] = krnl.load %[[VAL_53]]#5{{\[}}%[[VAL_175]]#1] : memref<4xf32>
// CHECK:               %[[VAL_221:.*]] = addf %[[VAL_218]], %[[VAL_219]] : f32
// CHECK:               %[[VAL_222:.*]] = addf %[[VAL_221]], %[[VAL_220]] : f32
// CHECK:               %[[VAL_223:.*]] = krnl.load %[[VAL_58]]#1{{\[}}%[[VAL_175]]#1] : memref<4xf32>
// CHECK:               %[[VAL_224:.*]] = mulf %[[VAL_223]], %[[VAL_215]] : f32
// CHECK:               %[[VAL_225:.*]] = addf %[[VAL_222]], %[[VAL_224]] : f32
// CHECK:               %[[VAL_226:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_225]], %[[VAL_226]][] : memref<f32>
// CHECK:               %[[VAL_227:.*]] = "onnx.Sigmoid"(%[[VAL_226]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_228:.*]] = krnl.load %[[VAL_227]][] : memref<f32>
// CHECK:               %[[VAL_229:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_215]], %[[VAL_229]][] : memref<f32>
// CHECK:               %[[VAL_230:.*]] = "onnx.Tanh"(%[[VAL_229]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_231:.*]] = krnl.load %[[VAL_230]][] : memref<f32>
// CHECK:               %[[VAL_232:.*]] = mulf %[[VAL_228]], %[[VAL_231]] : f32
// CHECK:               krnl.store %[[VAL_215]], %[[VAL_7]]{{\[}}%[[VAL_175]]#0, %[[VAL_175]]#1] : memref<2x4xf32>
// CHECK:               krnl.store %[[VAL_232]], %[[VAL_8]]{{\[}}%[[VAL_175]]#0, %[[VAL_175]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_147]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_233:.*]] = constant 2 : index
// CHECK:           %[[VAL_234:.*]] = constant 4 : index
// CHECK:           %[[VAL_235:.*]] = constant 0 : index
// CHECK:           %[[VAL_236:.*]] = constant 0 : index
// CHECK:           %[[VAL_237:.*]] = constant 0 : index
// CHECK:           %[[VAL_238:.*]] = constant 1 : index
// CHECK:           %[[VAL_239:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_239]]#0, %[[VAL_239]]#1) with (%[[VAL_239]]#0 -> %[[VAL_240:.*]] = %[[VAL_235]] to %[[VAL_233]], %[[VAL_239]]#1 -> %[[VAL_241:.*]] = %[[VAL_236]] to %[[VAL_234]]) {
// CHECK:             %[[VAL_242:.*]]:2 = krnl.get_induction_var_value(%[[VAL_239]]#0, %[[VAL_239]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             %[[VAL_243:.*]] = krnl.load %[[VAL_10]]{{\[}}%[[VAL_242]]#0, %[[VAL_242]]#1] : memref<2x4xf32>
// CHECK:             krnl.store %[[VAL_243]], %[[VAL_11]]{{\[}}%[[VAL_237]], %[[VAL_242]]#0, %[[VAL_242]]#1] : memref<2x2x4xf32>
// CHECK:             %[[VAL_244:.*]] = krnl.load %[[VAL_8]]{{\[}}%[[VAL_242]]#0, %[[VAL_242]]#1] : memref<2x4xf32>
// CHECK:             krnl.store %[[VAL_244]], %[[VAL_11]]{{\[}}%[[VAL_238]], %[[VAL_242]]#0, %[[VAL_242]]#1] : memref<2x2x4xf32>
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
// CHECK:           %[[VAL_29:.*]]:4 = "onnx.Split"(%[[VAL_27]]) {axis = 0 : si64} : (memref<16x?xf32>) -> (memref<4x?xf32>, memref<4x?xf32>, memref<4x?xf32>, memref<4x?xf32>)
// CHECK:           %[[VAL_30:.*]] = "onnx.Transpose"(%[[VAL_29]]#0) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Transpose"(%[[VAL_29]]#1) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK:           %[[VAL_32:.*]] = "onnx.Transpose"(%[[VAL_29]]#2) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK:           %[[VAL_33:.*]] = "onnx.Transpose"(%[[VAL_29]]#3) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK:           %[[VAL_34:.*]]:4 = "onnx.Split"(%[[VAL_28]]) {axis = 0 : si64} : (memref<16x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK:           %[[VAL_35:.*]] = "onnx.Transpose"(%[[VAL_34]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_36:.*]] = "onnx.Transpose"(%[[VAL_34]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.Transpose"(%[[VAL_34]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Transpose"(%[[VAL_34]]#3) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_39:.*]] = "onnx.Squeeze"(%[[VAL_3]]) {axes = [0]} : (memref<1x32xf32>) -> memref<32xf32>
// CHECK:           %[[VAL_40:.*]]:8 = "onnx.Split"(%[[VAL_39]]) {axis = 0 : si64} : (memref<32xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_41:.*]] = "onnx.Squeeze"(%[[VAL_6]]) {axes = [0]} : (memref<1x12xf32>) -> memref<12xf32>
// CHECK:           %[[VAL_42:.*]]:3 = "onnx.Split"(%[[VAL_41]]) {axis = 0 : si64} : (memref<12xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_43:.*]] = krnl.define_loops 1
// CHECK:           %[[VAL_44:.*]] = constant 0 : index
// CHECK:           %[[VAL_45:.*]] = memref.dim %[[VAL_0]], %[[VAL_44]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate(%[[VAL_43]]) with (%[[VAL_43]] -> %[[VAL_46:.*]] = 0 to %[[VAL_45]]) {
// CHECK:             %[[VAL_47:.*]] = constant 0 : index
// CHECK:             %[[VAL_48:.*]] = constant 1 : index
// CHECK:             %[[VAL_49:.*]] = memref.dim %[[VAL_0]], %[[VAL_48]] : memref<?x?x?xf32>
// CHECK:             %[[VAL_50:.*]] = constant 2 : index
// CHECK:             %[[VAL_51:.*]] = memref.dim %[[VAL_0]], %[[VAL_50]] : memref<?x?x?xf32>
// CHECK:             %[[VAL_52:.*]] = memref.alloc(%[[VAL_49]], %[[VAL_51]]) : memref<?x?xf32>
// CHECK:             %[[VAL_53:.*]] = constant 0 : index
// CHECK:             %[[VAL_54:.*]] = memref.dim %[[VAL_52]], %[[VAL_53]] : memref<?x?xf32>
// CHECK:             %[[VAL_55:.*]] = constant 1 : index
// CHECK:             %[[VAL_56:.*]] = memref.dim %[[VAL_52]], %[[VAL_55]] : memref<?x?xf32>
// CHECK:             %[[VAL_57:.*]] = constant 0 : index
// CHECK:             %[[VAL_58:.*]] = constant 0 : index
// CHECK:             %[[VAL_59:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_59]]#0, %[[VAL_59]]#1) with (%[[VAL_59]]#0 -> %[[VAL_60:.*]] = %[[VAL_57]] to %[[VAL_54]], %[[VAL_59]]#1 -> %[[VAL_61:.*]] = %[[VAL_58]] to %[[VAL_56]]) {
// CHECK:               %[[VAL_62:.*]]:2 = krnl.get_induction_var_value(%[[VAL_59]]#0, %[[VAL_59]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_63:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_46]], %[[VAL_62]]#0, %[[VAL_62]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store %[[VAL_63]], %[[VAL_52]]{{\[}}%[[VAL_62]]#0, %[[VAL_62]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK:             %[[VAL_64:.*]] = "onnx.MatMul"(%[[VAL_52]], %[[VAL_30]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_65:.*]] = "onnx.MatMul"(%[[VAL_13]], %[[VAL_35]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_66:.*]] = "onnx.MatMul"(%[[VAL_52]], %[[VAL_32]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_67:.*]] = "onnx.MatMul"(%[[VAL_13]], %[[VAL_37]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_68:.*]] = "onnx.MatMul"(%[[VAL_52]], %[[VAL_33]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_69:.*]] = "onnx.MatMul"(%[[VAL_13]], %[[VAL_38]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_70:.*]] = "onnx.MatMul"(%[[VAL_52]], %[[VAL_31]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_71:.*]] = "onnx.MatMul"(%[[VAL_13]], %[[VAL_36]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_72:.*]] = constant 0 : index
// CHECK:             %[[VAL_73:.*]] = memref.dim %[[VAL_13]], %[[VAL_72]] : memref<?x4xf32>
// CHECK:             %[[VAL_74:.*]] = constant 4 : index
// CHECK:             %[[VAL_75:.*]] = constant 0 : index
// CHECK:             %[[VAL_76:.*]] = constant 0 : index
// CHECK:             %[[VAL_77:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_77]]#0, %[[VAL_77]]#1) with (%[[VAL_77]]#0 -> %[[VAL_78:.*]] = %[[VAL_75]] to %[[VAL_73]], %[[VAL_77]]#1 -> %[[VAL_79:.*]] = %[[VAL_76]] to %[[VAL_74]]) {
// CHECK:               %[[VAL_80:.*]]:2 = krnl.get_induction_var_value(%[[VAL_77]]#0, %[[VAL_77]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_81:.*]] = krnl.load %[[VAL_16]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_82:.*]] = krnl.load %[[VAL_64]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_83:.*]] = krnl.load %[[VAL_65]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_84:.*]] = addf %[[VAL_82]], %[[VAL_83]] : f32
// CHECK:               %[[VAL_85:.*]] = krnl.load %[[VAL_40]]#0{{\[}}%[[VAL_80]]#1] : memref<4xf32>
// CHECK:               %[[VAL_86:.*]] = krnl.load %[[VAL_40]]#4{{\[}}%[[VAL_80]]#1] : memref<4xf32>
// CHECK:               %[[VAL_87:.*]] = addf %[[VAL_84]], %[[VAL_85]] : f32
// CHECK:               %[[VAL_88:.*]] = addf %[[VAL_87]], %[[VAL_86]] : f32
// CHECK:               %[[VAL_89:.*]] = krnl.load %[[VAL_42]]#0{{\[}}%[[VAL_80]]#1] : memref<4xf32>
// CHECK:               %[[VAL_90:.*]] = mulf %[[VAL_89]], %[[VAL_81]] : f32
// CHECK:               %[[VAL_91:.*]] = addf %[[VAL_88]], %[[VAL_90]] : f32
// CHECK:               %[[VAL_92:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_91]], %[[VAL_92]][] : memref<f32>
// CHECK:               %[[VAL_93:.*]] = "onnx.Sigmoid"(%[[VAL_92]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_94:.*]] = krnl.load %[[VAL_93]][] : memref<f32>
// CHECK:               %[[VAL_95:.*]] = krnl.load %[[VAL_66]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_96:.*]] = krnl.load %[[VAL_67]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_97:.*]] = addf %[[VAL_95]], %[[VAL_96]] : f32
// CHECK:               %[[VAL_98:.*]] = krnl.load %[[VAL_40]]#2{{\[}}%[[VAL_80]]#1] : memref<4xf32>
// CHECK:               %[[VAL_99:.*]] = krnl.load %[[VAL_40]]#6{{\[}}%[[VAL_80]]#1] : memref<4xf32>
// CHECK:               %[[VAL_100:.*]] = addf %[[VAL_97]], %[[VAL_98]] : f32
// CHECK:               %[[VAL_101:.*]] = addf %[[VAL_100]], %[[VAL_99]] : f32
// CHECK:               %[[VAL_102:.*]] = krnl.load %[[VAL_42]]#2{{\[}}%[[VAL_80]]#1] : memref<4xf32>
// CHECK:               %[[VAL_103:.*]] = mulf %[[VAL_102]], %[[VAL_81]] : f32
// CHECK:               %[[VAL_104:.*]] = addf %[[VAL_101]], %[[VAL_103]] : f32
// CHECK:               %[[VAL_105:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_104]], %[[VAL_105]][] : memref<f32>
// CHECK:               %[[VAL_106:.*]] = "onnx.Sigmoid"(%[[VAL_105]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_107:.*]] = krnl.load %[[VAL_106]][] : memref<f32>
// CHECK:               %[[VAL_108:.*]] = krnl.load %[[VAL_68]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_109:.*]] = krnl.load %[[VAL_69]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_110:.*]] = addf %[[VAL_108]], %[[VAL_109]] : f32
// CHECK:               %[[VAL_111:.*]] = krnl.load %[[VAL_40]]#3{{\[}}%[[VAL_80]]#1] : memref<4xf32>
// CHECK:               %[[VAL_112:.*]] = krnl.load %[[VAL_40]]#7{{\[}}%[[VAL_80]]#1] : memref<4xf32>
// CHECK:               %[[VAL_113:.*]] = addf %[[VAL_110]], %[[VAL_111]] : f32
// CHECK:               %[[VAL_114:.*]] = addf %[[VAL_113]], %[[VAL_112]] : f32
// CHECK:               %[[VAL_115:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_114]], %[[VAL_115]][] : memref<f32>
// CHECK:               %[[VAL_116:.*]] = "onnx.Tanh"(%[[VAL_115]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_117:.*]] = krnl.load %[[VAL_116]][] : memref<f32>
// CHECK:               %[[VAL_118:.*]] = mulf %[[VAL_107]], %[[VAL_81]] : f32
// CHECK:               %[[VAL_119:.*]] = mulf %[[VAL_94]], %[[VAL_117]] : f32
// CHECK:               %[[VAL_120:.*]] = addf %[[VAL_118]], %[[VAL_119]] : f32
// CHECK:               %[[VAL_121:.*]] = krnl.load %[[VAL_70]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_122:.*]] = krnl.load %[[VAL_71]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_123:.*]] = addf %[[VAL_121]], %[[VAL_122]] : f32
// CHECK:               %[[VAL_124:.*]] = krnl.load %[[VAL_40]]#1{{\[}}%[[VAL_80]]#1] : memref<4xf32>
// CHECK:               %[[VAL_125:.*]] = krnl.load %[[VAL_40]]#5{{\[}}%[[VAL_80]]#1] : memref<4xf32>
// CHECK:               %[[VAL_126:.*]] = addf %[[VAL_123]], %[[VAL_124]] : f32
// CHECK:               %[[VAL_127:.*]] = addf %[[VAL_126]], %[[VAL_125]] : f32
// CHECK:               %[[VAL_128:.*]] = krnl.load %[[VAL_42]]#1{{\[}}%[[VAL_80]]#1] : memref<4xf32>
// CHECK:               %[[VAL_129:.*]] = mulf %[[VAL_128]], %[[VAL_120]] : f32
// CHECK:               %[[VAL_130:.*]] = addf %[[VAL_127]], %[[VAL_129]] : f32
// CHECK:               %[[VAL_131:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_130]], %[[VAL_131]][] : memref<f32>
// CHECK:               %[[VAL_132:.*]] = "onnx.Sigmoid"(%[[VAL_131]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_133:.*]] = krnl.load %[[VAL_132]][] : memref<f32>
// CHECK:               %[[VAL_134:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_120]], %[[VAL_134]][] : memref<f32>
// CHECK:               %[[VAL_135:.*]] = "onnx.Tanh"(%[[VAL_134]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_136:.*]] = krnl.load %[[VAL_135]][] : memref<f32>
// CHECK:               %[[VAL_137:.*]] = mulf %[[VAL_133]], %[[VAL_136]] : f32
// CHECK:               krnl.store %[[VAL_120]], %[[VAL_16]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x4xf32>
// CHECK:               krnl.store %[[VAL_137]], %[[VAL_13]]{{\[}}%[[VAL_80]]#0, %[[VAL_80]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_52]] : memref<?x?xf32>
// CHECK:           }
// CHECK:           %[[VAL_138:.*]] = constant 16 : i64
// CHECK:           %[[VAL_139:.*]] = constant 0 : index
// CHECK:           %[[VAL_140:.*]] = memref.dim %[[VAL_13]], %[[VAL_139]] : memref<?x4xf32>
// CHECK:           %[[VAL_141:.*]] = index_cast %[[VAL_140]] : index to i64
// CHECK:           %[[VAL_142:.*]] = muli %[[VAL_138]], %[[VAL_141]] : i64
// CHECK:           "krnl.memcpy"(%[[VAL_10]], %[[VAL_13]], %[[VAL_142]]) : (memref<1x?x4xf32>, memref<?x4xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_13]] : memref<?x4xf32>
// CHECK:           memref.dealloc %[[VAL_16]] : memref<?x4xf32>
// CHECK:           return %[[VAL_10]] : memref<1x?x4xf32>
// CHECK:         }

}
