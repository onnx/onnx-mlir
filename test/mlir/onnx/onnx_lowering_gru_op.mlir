// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl='test-rnn-ops-lowering' %s -split-input-file | FileCheck %s

func private @test_gru_forward_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_gru_forward_mode(
// CHECK-SAME:                                        %[[VAL_0:.*]]: memref<7x2x3xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: memref<1x12x3xf32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: memref<1x12x4xf32>,
// CHECK-SAME:                                        %[[VAL_3:.*]]: memref<1x24xf32>,
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
// CHECK:           %[[VAL_15:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK:           %[[VAL_16:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK:           %[[VAL_17:.*]]:3 = "onnx.Split"(%[[VAL_15]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Transpose"(%[[VAL_17]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Transpose"(%[[VAL_17]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_21:.*]]:3 = "onnx.Split"(%[[VAL_16]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK:           %[[VAL_22:.*]] = "onnx.Transpose"(%[[VAL_21]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Transpose"(%[[VAL_21]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Transpose"(%[[VAL_21]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Squeeze"(%[[VAL_3]]) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK:           %[[VAL_26:.*]]:6 = "onnx.Split"(%[[VAL_25]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_27:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_27]]) with (%[[VAL_27]] -> %[[VAL_28:.*]] = 0 to 7) {
// CHECK:             %[[VAL_29:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:             %[[VAL_30:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:             %[[VAL_31:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_32:.*]] = constant 0 : index
// CHECK:             %[[VAL_33:.*]] = constant 2 : index
// CHECK:             %[[VAL_34:.*]] = constant 3 : index
// CHECK:             %[[VAL_35:.*]] = constant 0 : index
// CHECK:             %[[VAL_36:.*]] = constant 0 : index
// CHECK:             %[[VAL_37:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_37]]#0, %[[VAL_37]]#1) with (%[[VAL_37]]#0 -> %[[VAL_38:.*]] = %[[VAL_35]] to %[[VAL_33]], %[[VAL_37]]#1 -> %[[VAL_39:.*]] = %[[VAL_36]] to %[[VAL_34]]) {
// CHECK:               %[[VAL_40:.*]]:2 = krnl.get_induction_var_value(%[[VAL_37]]#0, %[[VAL_37]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_41:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_28]], %[[VAL_40]]#0, %[[VAL_40]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_41]], %[[VAL_31]]{{\[}}%[[VAL_40]]#0, %[[VAL_40]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_42:.*]] = "onnx.MatMul"(%[[VAL_31]], %[[VAL_18]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_43:.*]] = "onnx.MatMul"(%[[VAL_5]], %[[VAL_22]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_44:.*]] = "onnx.MatMul"(%[[VAL_31]], %[[VAL_19]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_45:.*]] = "onnx.MatMul"(%[[VAL_5]], %[[VAL_23]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_46:.*]] = "onnx.MatMul"(%[[VAL_31]], %[[VAL_20]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_47:.*]] = constant 1.000000e+00 : f32
// CHECK:             %[[VAL_48:.*]] = constant 2 : index
// CHECK:             %[[VAL_49:.*]] = constant 4 : index
// CHECK:             %[[VAL_50:.*]] = constant 0 : index
// CHECK:             %[[VAL_51:.*]] = constant 0 : index
// CHECK:             %[[VAL_52:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_52]]#0, %[[VAL_52]]#1) with (%[[VAL_52]]#0 -> %[[VAL_53:.*]] = %[[VAL_50]] to %[[VAL_48]], %[[VAL_52]]#1 -> %[[VAL_54:.*]] = %[[VAL_51]] to %[[VAL_49]]) {
// CHECK:               %[[VAL_55:.*]]:2 = krnl.get_induction_var_value(%[[VAL_52]]#0, %[[VAL_52]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_56:.*]] = krnl.load %[[VAL_5]]{{\[}}%[[VAL_55]]#0, %[[VAL_55]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_57:.*]] = krnl.load %[[VAL_44]]{{\[}}%[[VAL_55]]#0, %[[VAL_55]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_58:.*]] = krnl.load %[[VAL_45]]{{\[}}%[[VAL_55]]#0, %[[VAL_55]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_59:.*]] = addf %[[VAL_57]], %[[VAL_58]] : f32
// CHECK:               %[[VAL_60:.*]] = krnl.load %[[VAL_26]]#1{{\[}}%[[VAL_55]]#1] : memref<4xf32>
// CHECK:               %[[VAL_61:.*]] = krnl.load %[[VAL_26]]#4{{\[}}%[[VAL_55]]#1] : memref<4xf32>
// CHECK:               %[[VAL_62:.*]] = addf %[[VAL_59]], %[[VAL_60]] : f32
// CHECK:               %[[VAL_63:.*]] = addf %[[VAL_62]], %[[VAL_61]] : f32
// CHECK:               %[[VAL_64:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_63]], %[[VAL_64]][] : memref<f32>
// CHECK:               %[[VAL_65:.*]] = "onnx.Sigmoid"(%[[VAL_64]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_66:.*]] = krnl.load %[[VAL_65]][] : memref<f32>
// CHECK:               krnl.store %[[VAL_66]], %[[VAL_30]]{{\[}}%[[VAL_55]]#0, %[[VAL_55]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_67:.*]] = mulf %[[VAL_66]], %[[VAL_56]] : f32
// CHECK:               krnl.store %[[VAL_67]], %[[VAL_29]]{{\[}}%[[VAL_55]]#0, %[[VAL_55]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             %[[VAL_68:.*]] = "onnx.MatMul"(%[[VAL_29]], %[[VAL_24]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_69:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_69]]#0, %[[VAL_69]]#1) with (%[[VAL_69]]#0 -> %[[VAL_70:.*]] = %[[VAL_50]] to %[[VAL_48]], %[[VAL_69]]#1 -> %[[VAL_71:.*]] = %[[VAL_51]] to %[[VAL_49]]) {
// CHECK:               %[[VAL_72:.*]]:2 = krnl.get_induction_var_value(%[[VAL_69]]#0, %[[VAL_69]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_73:.*]] = krnl.load %[[VAL_5]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_74:.*]] = krnl.load %[[VAL_42]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_75:.*]] = krnl.load %[[VAL_43]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_76:.*]] = addf %[[VAL_74]], %[[VAL_75]] : f32
// CHECK:               %[[VAL_77:.*]] = krnl.load %[[VAL_26]]#0{{\[}}%[[VAL_72]]#1] : memref<4xf32>
// CHECK:               %[[VAL_78:.*]] = krnl.load %[[VAL_26]]#3{{\[}}%[[VAL_72]]#1] : memref<4xf32>
// CHECK:               %[[VAL_79:.*]] = addf %[[VAL_76]], %[[VAL_77]] : f32
// CHECK:               %[[VAL_80:.*]] = addf %[[VAL_79]], %[[VAL_78]] : f32
// CHECK:               %[[VAL_81:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_80]], %[[VAL_81]][] : memref<f32>
// CHECK:               %[[VAL_82:.*]] = "onnx.Sigmoid"(%[[VAL_81]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_83:.*]] = krnl.load %[[VAL_82]][] : memref<f32>
// CHECK:               %[[VAL_84:.*]] = krnl.load %[[VAL_46]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_85:.*]] = krnl.load %[[VAL_68]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_86:.*]] = addf %[[VAL_84]], %[[VAL_85]] : f32
// CHECK:               %[[VAL_87:.*]] = krnl.load %[[VAL_26]]#2{{\[}}%[[VAL_72]]#1] : memref<4xf32>
// CHECK:               %[[VAL_88:.*]] = krnl.load %[[VAL_26]]#5{{\[}}%[[VAL_72]]#1] : memref<4xf32>
// CHECK:               %[[VAL_89:.*]] = addf %[[VAL_86]], %[[VAL_87]] : f32
// CHECK:               %[[VAL_90:.*]] = addf %[[VAL_89]], %[[VAL_88]] : f32
// CHECK:               %[[VAL_91:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_90]], %[[VAL_91]][] : memref<f32>
// CHECK:               %[[VAL_92:.*]] = "onnx.Tanh"(%[[VAL_91]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_93:.*]] = krnl.load %[[VAL_92]][] : memref<f32>
// CHECK:               %[[VAL_94:.*]] = subf %[[VAL_47]], %[[VAL_83]] : f32
// CHECK:               %[[VAL_95:.*]] = mulf %[[VAL_94]], %[[VAL_93]] : f32
// CHECK:               %[[VAL_96:.*]] = mulf %[[VAL_83]], %[[VAL_73]] : f32
// CHECK:               %[[VAL_97:.*]] = addf %[[VAL_95]], %[[VAL_96]] : f32
// CHECK:               krnl.store %[[VAL_97]], %[[VAL_5]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_30]] : memref<2x4xf32>
// CHECK:             memref.dealloc %[[VAL_29]] : memref<2x4xf32>
// CHECK:             memref.dealloc %[[VAL_31]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_98:.*]] = constant 32 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_6]], %[[VAL_5]], %[[VAL_98]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_5]] : memref<2x4xf32>
// CHECK:           return %[[VAL_6]] : memref<1x2x4xf32>
// CHECK:         }

}

// -----

func private @test_gru_forward_mode_linear_before_reset(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, linear_before_reset = 1 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_gru_forward_mode_linear_before_reset(
// CHECK-SAME:                                                            %[[VAL_0:.*]]: memref<7x2x3xf32>,
// CHECK-SAME:                                                            %[[VAL_1:.*]]: memref<1x12x3xf32>,
// CHECK-SAME:                                                            %[[VAL_2:.*]]: memref<1x12x4xf32>,
// CHECK-SAME:                                                            %[[VAL_3:.*]]: memref<1x24xf32>,
// CHECK-SAME:                                                            %[[VAL_4:.*]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
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
// CHECK:           %[[VAL_15:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK:           %[[VAL_16:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK:           %[[VAL_17:.*]]:3 = "onnx.Split"(%[[VAL_15]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Transpose"(%[[VAL_17]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Transpose"(%[[VAL_17]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_21:.*]]:3 = "onnx.Split"(%[[VAL_16]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK:           %[[VAL_22:.*]] = "onnx.Transpose"(%[[VAL_21]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Transpose"(%[[VAL_21]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Transpose"(%[[VAL_21]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Squeeze"(%[[VAL_3]]) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK:           %[[VAL_26:.*]]:6 = "onnx.Split"(%[[VAL_25]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_27:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_27]]) with (%[[VAL_27]] -> %[[VAL_28:.*]] = 0 to 7) {
// CHECK:             %[[VAL_29:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_30:.*]] = constant 0 : index
// CHECK:             %[[VAL_31:.*]] = constant 2 : index
// CHECK:             %[[VAL_32:.*]] = constant 3 : index
// CHECK:             %[[VAL_33:.*]] = constant 0 : index
// CHECK:             %[[VAL_34:.*]] = constant 0 : index
// CHECK:             %[[VAL_35:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_35]]#0, %[[VAL_35]]#1) with (%[[VAL_35]]#0 -> %[[VAL_36:.*]] = %[[VAL_33]] to %[[VAL_31]], %[[VAL_35]]#1 -> %[[VAL_37:.*]] = %[[VAL_34]] to %[[VAL_32]]) {
// CHECK:               %[[VAL_38:.*]]:2 = krnl.get_induction_var_value(%[[VAL_35]]#0, %[[VAL_35]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_39:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_28]], %[[VAL_38]]#0, %[[VAL_38]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_39]], %[[VAL_29]]{{\[}}%[[VAL_38]]#0, %[[VAL_38]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_40:.*]] = "onnx.MatMul"(%[[VAL_29]], %[[VAL_18]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_41:.*]] = "onnx.MatMul"(%[[VAL_5]], %[[VAL_22]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_42:.*]] = "onnx.MatMul"(%[[VAL_29]], %[[VAL_19]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_43:.*]] = "onnx.MatMul"(%[[VAL_5]], %[[VAL_23]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_44:.*]] = "onnx.MatMul"(%[[VAL_29]], %[[VAL_20]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_45:.*]] = constant 1.000000e+00 : f32
// CHECK:             %[[VAL_46:.*]] = "onnx.MatMul"(%[[VAL_5]], %[[VAL_24]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_47:.*]] = constant 2 : index
// CHECK:             %[[VAL_48:.*]] = constant 4 : index
// CHECK:             %[[VAL_49:.*]] = constant 0 : index
// CHECK:             %[[VAL_50:.*]] = constant 0 : index
// CHECK:             %[[VAL_51:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_51]]#0, %[[VAL_51]]#1) with (%[[VAL_51]]#0 -> %[[VAL_52:.*]] = %[[VAL_49]] to %[[VAL_47]], %[[VAL_51]]#1 -> %[[VAL_53:.*]] = %[[VAL_50]] to %[[VAL_48]]) {
// CHECK:               %[[VAL_54:.*]]:2 = krnl.get_induction_var_value(%[[VAL_51]]#0, %[[VAL_51]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_55:.*]] = krnl.load %[[VAL_5]]{{\[}}%[[VAL_54]]#0, %[[VAL_54]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_56:.*]] = krnl.load %[[VAL_40]]{{\[}}%[[VAL_54]]#0, %[[VAL_54]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_57:.*]] = krnl.load %[[VAL_41]]{{\[}}%[[VAL_54]]#0, %[[VAL_54]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_58:.*]] = addf %[[VAL_56]], %[[VAL_57]] : f32
// CHECK:               %[[VAL_59:.*]] = krnl.load %[[VAL_26]]#0{{\[}}%[[VAL_54]]#1] : memref<4xf32>
// CHECK:               %[[VAL_60:.*]] = krnl.load %[[VAL_26]]#3{{\[}}%[[VAL_54]]#1] : memref<4xf32>
// CHECK:               %[[VAL_61:.*]] = addf %[[VAL_58]], %[[VAL_59]] : f32
// CHECK:               %[[VAL_62:.*]] = addf %[[VAL_61]], %[[VAL_60]] : f32
// CHECK:               %[[VAL_63:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_62]], %[[VAL_63]][] : memref<f32>
// CHECK:               %[[VAL_64:.*]] = "onnx.Sigmoid"(%[[VAL_63]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_65:.*]] = krnl.load %[[VAL_64]][] : memref<f32>
// CHECK:               %[[VAL_66:.*]] = krnl.load %[[VAL_42]]{{\[}}%[[VAL_54]]#0, %[[VAL_54]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_67:.*]] = krnl.load %[[VAL_43]]{{\[}}%[[VAL_54]]#0, %[[VAL_54]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_68:.*]] = addf %[[VAL_66]], %[[VAL_67]] : f32
// CHECK:               %[[VAL_69:.*]] = krnl.load %[[VAL_26]]#1{{\[}}%[[VAL_54]]#1] : memref<4xf32>
// CHECK:               %[[VAL_70:.*]] = krnl.load %[[VAL_26]]#4{{\[}}%[[VAL_54]]#1] : memref<4xf32>
// CHECK:               %[[VAL_71:.*]] = addf %[[VAL_68]], %[[VAL_69]] : f32
// CHECK:               %[[VAL_72:.*]] = addf %[[VAL_71]], %[[VAL_70]] : f32
// CHECK:               %[[VAL_73:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_72]], %[[VAL_73]][] : memref<f32>
// CHECK:               %[[VAL_74:.*]] = "onnx.Sigmoid"(%[[VAL_73]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_75:.*]] = krnl.load %[[VAL_74]][] : memref<f32>
// CHECK:               %[[VAL_76:.*]] = krnl.load %[[VAL_44]]{{\[}}%[[VAL_54]]#0, %[[VAL_54]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_77:.*]] = krnl.load %[[VAL_46]]{{\[}}%[[VAL_54]]#0, %[[VAL_54]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_78:.*]] = krnl.load %[[VAL_26]]#5{{\[}}%[[VAL_54]]#1] : memref<4xf32>
// CHECK:               %[[VAL_79:.*]] = addf %[[VAL_77]], %[[VAL_78]] : f32
// CHECK:               %[[VAL_80:.*]] = mulf %[[VAL_75]], %[[VAL_79]] : f32
// CHECK:               %[[VAL_81:.*]] = addf %[[VAL_76]], %[[VAL_80]] : f32
// CHECK:               %[[VAL_82:.*]] = krnl.load %[[VAL_26]]#2{{\[}}%[[VAL_54]]#1] : memref<4xf32>
// CHECK:               %[[VAL_83:.*]] = addf %[[VAL_81]], %[[VAL_82]] : f32
// CHECK:               %[[VAL_84:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_83]], %[[VAL_84]][] : memref<f32>
// CHECK:               %[[VAL_85:.*]] = "onnx.Tanh"(%[[VAL_84]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_86:.*]] = krnl.load %[[VAL_85]][] : memref<f32>
// CHECK:               %[[VAL_87:.*]] = subf %[[VAL_45]], %[[VAL_65]] : f32
// CHECK:               %[[VAL_88:.*]] = mulf %[[VAL_87]], %[[VAL_86]] : f32
// CHECK:               %[[VAL_89:.*]] = mulf %[[VAL_65]], %[[VAL_55]] : f32
// CHECK:               %[[VAL_90:.*]] = addf %[[VAL_88]], %[[VAL_89]] : f32
// CHECK:               krnl.store %[[VAL_90]], %[[VAL_5]]{{\[}}%[[VAL_54]]#0, %[[VAL_54]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_29]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_91:.*]] = constant 32 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_6]], %[[VAL_5]], %[[VAL_91]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_5]] : memref<2x4xf32>
// CHECK:           return %[[VAL_6]] : memref<1x2x4xf32>
// CHECK:         }

}

// -----

func private @test_gru_forward_mode_constant_weight_and_bias(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %w = "onnx.Constant"() {value = dense<[[[1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.], [1., 1., 1.]]]> : tensor<1x12x3xf32>} : () -> tensor<1x12x3xf32> 
  %r = "onnx.Constant"() {value = dense<[[[2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.], [2., 2., 2., 2.]]]> : tensor<1x12x4xf32>} : () -> tensor<1x12x4xf32> 
  %b = "onnx.Constant"() {value = dense<[[1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18., 19., 20., 21., 22., 23., 24.]]> : tensor<1x24xf32>} : () -> tensor<1x24xf32> 

  %Y, %Y_h = "onnx.GRU"(%arg0, %w, %r, %b, %cst, %arg1) {hidden_size = 4 : si64} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_gru_forward_mode_constant_weight_and_bias(
// CHECK-SAME:                                                                 %[[VAL_0:.*]]: memref<7x2x3xf32>,
// CHECK-SAME:                                                                 %[[VAL_1:.*]]: memref<1x2x4xf32>) -> memref<1x2x4xf32> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:           %[[VAL_3:.*]] = memref.alloc() : memref<1x2x4xf32>
// CHECK:           %[[VAL_4:.*]] = constant unit
// CHECK:           %[[VAL_5:.*]] = "krnl.global"() {name = "constant_0", shape = [1, 12, 3], value = dense<1.000000e+00> : tensor<1x12x3xf32>} : () -> memref<1x12x3xf32>
// CHECK:           %[[VAL_6:.*]] = "krnl.global"() {name = "constant_1", shape = [1, 12, 4], value = dense<2.000000e+00> : tensor<1x12x4xf32>} : () -> memref<1x12x4xf32>
// CHECK:           %[[VAL_7:.*]] = "krnl.global"() {name = "constant_2", shape = [1, 24], value = dense<{{\[\[}}1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]]> : tensor<1x24xf32>} : () -> memref<1x24xf32>
// CHECK:           %[[VAL_8:.*]] = constant 0.000000e+00 : f32
// CHECK:           %[[VAL_9:.*]] = constant 0 : index
// CHECK:           %[[VAL_10:.*]] = constant 1 : index
// CHECK:           %[[VAL_11:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_11]]#0, %[[VAL_11]]#1) with (%[[VAL_11]]#0 -> %[[VAL_12:.*]] = 0 to 2, %[[VAL_11]]#1 -> %[[VAL_13:.*]] = 0 to 4) {
// CHECK:             %[[VAL_14:.*]] = krnl.load %[[VAL_1]]{{\[}}%[[VAL_9]], %[[VAL_12]], %[[VAL_13]]] : memref<1x2x4xf32>
// CHECK:             krnl.store %[[VAL_14]], %[[VAL_2]]{{\[}}%[[VAL_12]], %[[VAL_13]]] : memref<2x4xf32>
// CHECK:           }
// CHECK:           %[[VAL_15:.*]] = "krnl.global"() {name = "constant_3", shape = [12, 3], value = dense<1.000000e+00> : tensor<12x3xf32>} : () -> memref<12x3xf32>
// CHECK:           %[[VAL_16:.*]] = "krnl.global"() {name = "constant_4", shape = [12, 4], value = dense<2.000000e+00> : tensor<12x4xf32>} : () -> memref<12x4xf32>
// CHECK:           %[[VAL_17:.*]] = "krnl.global"() {name = "constant_5", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK:           %[[VAL_18:.*]] = "krnl.global"() {name = "constant_6", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK:           %[[VAL_19:.*]] = "krnl.global"() {name = "constant_7", shape = [4, 3], value = dense<1.000000e+00> : tensor<4x3xf32>} : () -> memref<4x3xf32>
// CHECK:           %[[VAL_20:.*]] = "krnl.global"() {name = "constant_8", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK:           %[[VAL_21:.*]] = "krnl.global"() {name = "constant_9", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK:           %[[VAL_22:.*]] = "krnl.global"() {name = "constant_10", shape = [3, 4], value = dense<1.000000e+00> : tensor<3x4xf32>} : () -> memref<3x4xf32>
// CHECK:           %[[VAL_23:.*]] = "krnl.global"() {name = "constant_11", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_24:.*]] = "krnl.global"() {name = "constant_12", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_25:.*]] = "krnl.global"() {name = "constant_13", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_26:.*]] = "krnl.global"() {name = "constant_14", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_27:.*]] = "krnl.global"() {name = "constant_15", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_28:.*]] = "krnl.global"() {name = "constant_16", shape = [4, 4], value = dense<2.000000e+00> : tensor<4x4xf32>} : () -> memref<4x4xf32>
// CHECK:           %[[VAL_29:.*]] = "krnl.global"() {name = "constant_17", shape = [24], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00, 9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01, 1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01, 1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01, 2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]> : tensor<24xf32>} : () -> memref<24xf32>
// CHECK:           %[[VAL_30:.*]] = "krnl.global"() {name = "constant_18", shape = [4], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_31:.*]] = "krnl.global"() {name = "constant_19", shape = [4], value = dense<[5.000000e+00, 6.000000e+00, 7.000000e+00, 8.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_32:.*]] = "krnl.global"() {name = "constant_20", shape = [4], value = dense<[9.000000e+00, 1.000000e+01, 1.100000e+01, 1.200000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_33:.*]] = "krnl.global"() {name = "constant_21", shape = [4], value = dense<[1.300000e+01, 1.400000e+01, 1.500000e+01, 1.600000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_34:.*]] = "krnl.global"() {name = "constant_22", shape = [4], value = dense<[1.700000e+01, 1.800000e+01, 1.900000e+01, 2.000000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_35:.*]] = "krnl.global"() {name = "constant_23", shape = [4], value = dense<[2.100000e+01, 2.200000e+01, 2.300000e+01, 2.400000e+01]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK:           %[[VAL_36:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_36]]) with (%[[VAL_36]] -> %[[VAL_37:.*]] = 0 to 7) {
// CHECK:             %[[VAL_38:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:             %[[VAL_39:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:             %[[VAL_40:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_41:.*]] = constant 0 : index
// CHECK:             %[[VAL_42:.*]] = constant 2 : index
// CHECK:             %[[VAL_43:.*]] = constant 3 : index
// CHECK:             %[[VAL_44:.*]] = constant 0 : index
// CHECK:             %[[VAL_45:.*]] = constant 0 : index
// CHECK:             %[[VAL_46:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_46]]#0, %[[VAL_46]]#1) with (%[[VAL_46]]#0 -> %[[VAL_47:.*]] = %[[VAL_44]] to %[[VAL_42]], %[[VAL_46]]#1 -> %[[VAL_48:.*]] = %[[VAL_45]] to %[[VAL_43]]) {
// CHECK:               %[[VAL_49:.*]]:2 = krnl.get_induction_var_value(%[[VAL_46]]#0, %[[VAL_46]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_50:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_37]], %[[VAL_49]]#0, %[[VAL_49]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_50]], %[[VAL_40]]{{\[}}%[[VAL_49]]#0, %[[VAL_49]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_51:.*]] = "onnx.MatMul"(%[[VAL_40]], %[[VAL_20]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_52:.*]] = "onnx.MatMul"(%[[VAL_2]], %[[VAL_26]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_53:.*]] = "onnx.MatMul"(%[[VAL_40]], %[[VAL_21]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_54:.*]] = "onnx.MatMul"(%[[VAL_2]], %[[VAL_27]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.MatMul"(%[[VAL_40]], %[[VAL_22]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_56:.*]] = constant 1.000000e+00 : f32
// CHECK:             %[[VAL_57:.*]] = constant 2 : index
// CHECK:             %[[VAL_58:.*]] = constant 4 : index
// CHECK:             %[[VAL_59:.*]] = constant 0 : index
// CHECK:             %[[VAL_60:.*]] = constant 0 : index
// CHECK:             %[[VAL_61:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_61]]#0, %[[VAL_61]]#1) with (%[[VAL_61]]#0 -> %[[VAL_62:.*]] = %[[VAL_59]] to %[[VAL_57]], %[[VAL_61]]#1 -> %[[VAL_63:.*]] = %[[VAL_60]] to %[[VAL_58]]) {
// CHECK:               %[[VAL_64:.*]]:2 = krnl.get_induction_var_value(%[[VAL_61]]#0, %[[VAL_61]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_65:.*]] = krnl.load %[[VAL_2]]{{\[}}%[[VAL_64]]#0, %[[VAL_64]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_66:.*]] = krnl.load %[[VAL_53]]{{\[}}%[[VAL_64]]#0, %[[VAL_64]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_67:.*]] = krnl.load %[[VAL_54]]{{\[}}%[[VAL_64]]#0, %[[VAL_64]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_68:.*]] = addf %[[VAL_66]], %[[VAL_67]] : f32
// CHECK:               %[[VAL_69:.*]] = krnl.load %[[VAL_31]]{{\[}}%[[VAL_64]]#1] : memref<4xf32>
// CHECK:               %[[VAL_70:.*]] = krnl.load %[[VAL_34]]{{\[}}%[[VAL_64]]#1] : memref<4xf32>
// CHECK:               %[[VAL_71:.*]] = addf %[[VAL_68]], %[[VAL_69]] : f32
// CHECK:               %[[VAL_72:.*]] = addf %[[VAL_71]], %[[VAL_70]] : f32
// CHECK:               %[[VAL_73:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_72]], %[[VAL_73]][] : memref<f32>
// CHECK:               %[[VAL_74:.*]] = "onnx.Sigmoid"(%[[VAL_73]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_75:.*]] = krnl.load %[[VAL_74]][] : memref<f32>
// CHECK:               krnl.store %[[VAL_75]], %[[VAL_39]]{{\[}}%[[VAL_64]]#0, %[[VAL_64]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_76:.*]] = mulf %[[VAL_75]], %[[VAL_65]] : f32
// CHECK:               krnl.store %[[VAL_76]], %[[VAL_38]]{{\[}}%[[VAL_64]]#0, %[[VAL_64]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             %[[VAL_77:.*]] = "onnx.MatMul"(%[[VAL_38]], %[[VAL_28]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_78:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_78]]#0, %[[VAL_78]]#1) with (%[[VAL_78]]#0 -> %[[VAL_79:.*]] = %[[VAL_59]] to %[[VAL_57]], %[[VAL_78]]#1 -> %[[VAL_80:.*]] = %[[VAL_60]] to %[[VAL_58]]) {
// CHECK:               %[[VAL_81:.*]]:2 = krnl.get_induction_var_value(%[[VAL_78]]#0, %[[VAL_78]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_82:.*]] = krnl.load %[[VAL_2]]{{\[}}%[[VAL_81]]#0, %[[VAL_81]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_83:.*]] = krnl.load %[[VAL_51]]{{\[}}%[[VAL_81]]#0, %[[VAL_81]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_84:.*]] = krnl.load %[[VAL_52]]{{\[}}%[[VAL_81]]#0, %[[VAL_81]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_85:.*]] = addf %[[VAL_83]], %[[VAL_84]] : f32
// CHECK:               %[[VAL_86:.*]] = krnl.load %[[VAL_30]]{{\[}}%[[VAL_81]]#1] : memref<4xf32>
// CHECK:               %[[VAL_87:.*]] = krnl.load %[[VAL_33]]{{\[}}%[[VAL_81]]#1] : memref<4xf32>
// CHECK:               %[[VAL_88:.*]] = addf %[[VAL_85]], %[[VAL_86]] : f32
// CHECK:               %[[VAL_89:.*]] = addf %[[VAL_88]], %[[VAL_87]] : f32
// CHECK:               %[[VAL_90:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_89]], %[[VAL_90]][] : memref<f32>
// CHECK:               %[[VAL_91:.*]] = "onnx.Sigmoid"(%[[VAL_90]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_92:.*]] = krnl.load %[[VAL_91]][] : memref<f32>
// CHECK:               %[[VAL_93:.*]] = krnl.load %[[VAL_55]]{{\[}}%[[VAL_81]]#0, %[[VAL_81]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_94:.*]] = krnl.load %[[VAL_77]]{{\[}}%[[VAL_81]]#0, %[[VAL_81]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_95:.*]] = addf %[[VAL_93]], %[[VAL_94]] : f32
// CHECK:               %[[VAL_96:.*]] = krnl.load %[[VAL_32]]{{\[}}%[[VAL_81]]#1] : memref<4xf32>
// CHECK:               %[[VAL_97:.*]] = krnl.load %[[VAL_35]]{{\[}}%[[VAL_81]]#1] : memref<4xf32>
// CHECK:               %[[VAL_98:.*]] = addf %[[VAL_95]], %[[VAL_96]] : f32
// CHECK:               %[[VAL_99:.*]] = addf %[[VAL_98]], %[[VAL_97]] : f32
// CHECK:               %[[VAL_100:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_99]], %[[VAL_100]][] : memref<f32>
// CHECK:               %[[VAL_101:.*]] = "onnx.Tanh"(%[[VAL_100]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_102:.*]] = krnl.load %[[VAL_101]][] : memref<f32>
// CHECK:               %[[VAL_103:.*]] = subf %[[VAL_56]], %[[VAL_92]] : f32
// CHECK:               %[[VAL_104:.*]] = mulf %[[VAL_103]], %[[VAL_102]] : f32
// CHECK:               %[[VAL_105:.*]] = mulf %[[VAL_92]], %[[VAL_82]] : f32
// CHECK:               %[[VAL_106:.*]] = addf %[[VAL_104]], %[[VAL_105]] : f32
// CHECK:               krnl.store %[[VAL_106]], %[[VAL_2]]{{\[}}%[[VAL_81]]#0, %[[VAL_81]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_39]] : memref<2x4xf32>
// CHECK:             memref.dealloc %[[VAL_38]] : memref<2x4xf32>
// CHECK:             memref.dealloc %[[VAL_40]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_107:.*]] = constant 32 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_3]], %[[VAL_2]], %[[VAL_107]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_2]] : memref<2x4xf32>
// CHECK:           return %[[VAL_3]] : memref<1x2x4xf32>
// CHECK:         }

}

// -----

func private @test_gru_reverse_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<1x12x3xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "reverse"} : (tensor<7x2x3xf32>, tensor<1x12x3xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_gru_reverse_mode(
// CHECK-SAME:                                        %[[VAL_0:.*]]: memref<7x2x3xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: memref<1x12x3xf32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: memref<1x12x4xf32>,
// CHECK-SAME:                                        %[[VAL_3:.*]]: memref<1x24xf32>,
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
// CHECK:           %[[VAL_15:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK:           %[[VAL_16:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK:           %[[VAL_17:.*]]:3 = "onnx.Split"(%[[VAL_15]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK:           %[[VAL_18:.*]] = "onnx.Transpose"(%[[VAL_17]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Transpose"(%[[VAL_17]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_20:.*]] = "onnx.Transpose"(%[[VAL_17]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_21:.*]]:3 = "onnx.Split"(%[[VAL_16]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK:           %[[VAL_22:.*]] = "onnx.Transpose"(%[[VAL_21]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_23:.*]] = "onnx.Transpose"(%[[VAL_21]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_24:.*]] = "onnx.Transpose"(%[[VAL_21]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Squeeze"(%[[VAL_3]]) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK:           %[[VAL_26:.*]]:6 = "onnx.Split"(%[[VAL_25]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_27:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_27]]) with (%[[VAL_27]] -> %[[VAL_28:.*]] = 0 to 7) {
// CHECK:             %[[VAL_29:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:             %[[VAL_30:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:             %[[VAL_31:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_32:.*]] = constant 0 : index
// CHECK:             %[[VAL_33:.*]] = constant 7 : index
// CHECK:             %[[VAL_34:.*]] = affine.apply #map(%[[VAL_28]]){{\[}}%[[VAL_33]]]
// CHECK:             %[[VAL_35:.*]] = constant 2 : index
// CHECK:             %[[VAL_36:.*]] = constant 3 : index
// CHECK:             %[[VAL_37:.*]] = constant 0 : index
// CHECK:             %[[VAL_38:.*]] = constant 0 : index
// CHECK:             %[[VAL_39:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_39]]#0, %[[VAL_39]]#1) with (%[[VAL_39]]#0 -> %[[VAL_40:.*]] = %[[VAL_37]] to %[[VAL_35]], %[[VAL_39]]#1 -> %[[VAL_41:.*]] = %[[VAL_38]] to %[[VAL_36]]) {
// CHECK:               %[[VAL_42:.*]]:2 = krnl.get_induction_var_value(%[[VAL_39]]#0, %[[VAL_39]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_43:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_34]], %[[VAL_42]]#0, %[[VAL_42]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_43]], %[[VAL_31]]{{\[}}%[[VAL_42]]#0, %[[VAL_42]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_44:.*]] = "onnx.MatMul"(%[[VAL_31]], %[[VAL_18]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_45:.*]] = "onnx.MatMul"(%[[VAL_5]], %[[VAL_22]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_46:.*]] = "onnx.MatMul"(%[[VAL_31]], %[[VAL_19]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_47:.*]] = "onnx.MatMul"(%[[VAL_5]], %[[VAL_23]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_48:.*]] = "onnx.MatMul"(%[[VAL_31]], %[[VAL_20]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_49:.*]] = constant 1.000000e+00 : f32
// CHECK:             %[[VAL_50:.*]] = constant 2 : index
// CHECK:             %[[VAL_51:.*]] = constant 4 : index
// CHECK:             %[[VAL_52:.*]] = constant 0 : index
// CHECK:             %[[VAL_53:.*]] = constant 0 : index
// CHECK:             %[[VAL_54:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_54]]#0, %[[VAL_54]]#1) with (%[[VAL_54]]#0 -> %[[VAL_55:.*]] = %[[VAL_52]] to %[[VAL_50]], %[[VAL_54]]#1 -> %[[VAL_56:.*]] = %[[VAL_53]] to %[[VAL_51]]) {
// CHECK:               %[[VAL_57:.*]]:2 = krnl.get_induction_var_value(%[[VAL_54]]#0, %[[VAL_54]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_58:.*]] = krnl.load %[[VAL_5]]{{\[}}%[[VAL_57]]#0, %[[VAL_57]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_59:.*]] = krnl.load %[[VAL_46]]{{\[}}%[[VAL_57]]#0, %[[VAL_57]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_60:.*]] = krnl.load %[[VAL_47]]{{\[}}%[[VAL_57]]#0, %[[VAL_57]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_61:.*]] = addf %[[VAL_59]], %[[VAL_60]] : f32
// CHECK:               %[[VAL_62:.*]] = krnl.load %[[VAL_26]]#1{{\[}}%[[VAL_57]]#1] : memref<4xf32>
// CHECK:               %[[VAL_63:.*]] = krnl.load %[[VAL_26]]#4{{\[}}%[[VAL_57]]#1] : memref<4xf32>
// CHECK:               %[[VAL_64:.*]] = addf %[[VAL_61]], %[[VAL_62]] : f32
// CHECK:               %[[VAL_65:.*]] = addf %[[VAL_64]], %[[VAL_63]] : f32
// CHECK:               %[[VAL_66:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_65]], %[[VAL_66]][] : memref<f32>
// CHECK:               %[[VAL_67:.*]] = "onnx.Sigmoid"(%[[VAL_66]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_68:.*]] = krnl.load %[[VAL_67]][] : memref<f32>
// CHECK:               krnl.store %[[VAL_68]], %[[VAL_30]]{{\[}}%[[VAL_57]]#0, %[[VAL_57]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_69:.*]] = mulf %[[VAL_68]], %[[VAL_58]] : f32
// CHECK:               krnl.store %[[VAL_69]], %[[VAL_29]]{{\[}}%[[VAL_57]]#0, %[[VAL_57]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             %[[VAL_70:.*]] = "onnx.MatMul"(%[[VAL_29]], %[[VAL_24]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_71:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_71]]#0, %[[VAL_71]]#1) with (%[[VAL_71]]#0 -> %[[VAL_72:.*]] = %[[VAL_52]] to %[[VAL_50]], %[[VAL_71]]#1 -> %[[VAL_73:.*]] = %[[VAL_53]] to %[[VAL_51]]) {
// CHECK:               %[[VAL_74:.*]]:2 = krnl.get_induction_var_value(%[[VAL_71]]#0, %[[VAL_71]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_75:.*]] = krnl.load %[[VAL_5]]{{\[}}%[[VAL_74]]#0, %[[VAL_74]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_76:.*]] = krnl.load %[[VAL_44]]{{\[}}%[[VAL_74]]#0, %[[VAL_74]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_77:.*]] = krnl.load %[[VAL_45]]{{\[}}%[[VAL_74]]#0, %[[VAL_74]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_78:.*]] = addf %[[VAL_76]], %[[VAL_77]] : f32
// CHECK:               %[[VAL_79:.*]] = krnl.load %[[VAL_26]]#0{{\[}}%[[VAL_74]]#1] : memref<4xf32>
// CHECK:               %[[VAL_80:.*]] = krnl.load %[[VAL_26]]#3{{\[}}%[[VAL_74]]#1] : memref<4xf32>
// CHECK:               %[[VAL_81:.*]] = addf %[[VAL_78]], %[[VAL_79]] : f32
// CHECK:               %[[VAL_82:.*]] = addf %[[VAL_81]], %[[VAL_80]] : f32
// CHECK:               %[[VAL_83:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_82]], %[[VAL_83]][] : memref<f32>
// CHECK:               %[[VAL_84:.*]] = "onnx.Sigmoid"(%[[VAL_83]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_85:.*]] = krnl.load %[[VAL_84]][] : memref<f32>
// CHECK:               %[[VAL_86:.*]] = krnl.load %[[VAL_48]]{{\[}}%[[VAL_74]]#0, %[[VAL_74]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_87:.*]] = krnl.load %[[VAL_70]]{{\[}}%[[VAL_74]]#0, %[[VAL_74]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_88:.*]] = addf %[[VAL_86]], %[[VAL_87]] : f32
// CHECK:               %[[VAL_89:.*]] = krnl.load %[[VAL_26]]#2{{\[}}%[[VAL_74]]#1] : memref<4xf32>
// CHECK:               %[[VAL_90:.*]] = krnl.load %[[VAL_26]]#5{{\[}}%[[VAL_74]]#1] : memref<4xf32>
// CHECK:               %[[VAL_91:.*]] = addf %[[VAL_88]], %[[VAL_89]] : f32
// CHECK:               %[[VAL_92:.*]] = addf %[[VAL_91]], %[[VAL_90]] : f32
// CHECK:               %[[VAL_93:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_92]], %[[VAL_93]][] : memref<f32>
// CHECK:               %[[VAL_94:.*]] = "onnx.Tanh"(%[[VAL_93]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_95:.*]] = krnl.load %[[VAL_94]][] : memref<f32>
// CHECK:               %[[VAL_96:.*]] = subf %[[VAL_49]], %[[VAL_85]] : f32
// CHECK:               %[[VAL_97:.*]] = mulf %[[VAL_96]], %[[VAL_95]] : f32
// CHECK:               %[[VAL_98:.*]] = mulf %[[VAL_85]], %[[VAL_75]] : f32
// CHECK:               %[[VAL_99:.*]] = addf %[[VAL_97]], %[[VAL_98]] : f32
// CHECK:               krnl.store %[[VAL_99]], %[[VAL_5]]{{\[}}%[[VAL_74]]#0, %[[VAL_74]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_30]] : memref<2x4xf32>
// CHECK:             memref.dealloc %[[VAL_29]] : memref<2x4xf32>
// CHECK:             memref.dealloc %[[VAL_31]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_100:.*]] = constant 32 : i64
// CHECK:           "krnl.memcpy"(%[[VAL_6]], %[[VAL_5]], %[[VAL_100]]) : (memref<1x2x4xf32>, memref<2x4xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_5]] : memref<2x4xf32>
// CHECK:           return %[[VAL_6]] : memref<1x2x4xf32>
// CHECK:         }

}

// -----

func private @test_gru_bidirectional_mode(%arg0: tensor<7x2x3xf32>, %arg1: tensor<2x12x3xf32>, %arg2: tensor<2x12x4xf32>, %arg3: tensor<2x24xf32>, %arg4: tensor<2x2x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64, direction = "bidirectional"} : (tensor<7x2x3xf32>, tensor<2x12x3xf32>, tensor<2x12x4xf32>, tensor<2x24xf32>, none, tensor<2x2x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_gru_bidirectional_mode(
// CHECK-SAME:                                              %[[VAL_0:.*]]: memref<7x2x3xf32>,
// CHECK-SAME:                                              %[[VAL_1:.*]]: memref<2x12x3xf32>,
// CHECK-SAME:                                              %[[VAL_2:.*]]: memref<2x12x4xf32>,
// CHECK-SAME:                                              %[[VAL_3:.*]]: memref<2x24xf32>,
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
// CHECK:           %[[VAL_17:.*]]:2 = "onnx.Split"(%[[VAL_1]]) {axis = 0 : si64} : (memref<2x12x3xf32>) -> (memref<1x12x3xf32>, memref<1x12x3xf32>)
// CHECK:           %[[VAL_18:.*]] = "onnx.Squeeze"(%[[VAL_17]]#0) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK:           %[[VAL_19:.*]] = "onnx.Squeeze"(%[[VAL_17]]#1) {axes = [0]} : (memref<1x12x3xf32>) -> memref<12x3xf32>
// CHECK:           %[[VAL_20:.*]]:2 = "onnx.Split"(%[[VAL_2]]) {axis = 0 : si64} : (memref<2x12x4xf32>) -> (memref<1x12x4xf32>, memref<1x12x4xf32>)
// CHECK:           %[[VAL_21:.*]] = "onnx.Squeeze"(%[[VAL_20]]#0) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Squeeze"(%[[VAL_20]]#1) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK:           %[[VAL_23:.*]]:3 = "onnx.Split"(%[[VAL_18]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK:           %[[VAL_24:.*]] = "onnx.Transpose"(%[[VAL_23]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Transpose"(%[[VAL_23]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.Transpose"(%[[VAL_23]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_27:.*]]:3 = "onnx.Split"(%[[VAL_21]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK:           %[[VAL_28:.*]] = "onnx.Transpose"(%[[VAL_27]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Transpose"(%[[VAL_27]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.Transpose"(%[[VAL_27]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_31:.*]]:3 = "onnx.Split"(%[[VAL_19]]) {axis = 0 : si64} : (memref<12x3xf32>) -> (memref<4x3xf32>, memref<4x3xf32>, memref<4x3xf32>)
// CHECK:           %[[VAL_32:.*]] = "onnx.Transpose"(%[[VAL_31]]#0) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_33:.*]] = "onnx.Transpose"(%[[VAL_31]]#1) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_34:.*]] = "onnx.Transpose"(%[[VAL_31]]#2) {perm = [1, 0]} : (memref<4x3xf32>) -> memref<3x4xf32>
// CHECK:           %[[VAL_35:.*]]:3 = "onnx.Split"(%[[VAL_22]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK:           %[[VAL_36:.*]] = "onnx.Transpose"(%[[VAL_35]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_37:.*]] = "onnx.Transpose"(%[[VAL_35]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_38:.*]] = "onnx.Transpose"(%[[VAL_35]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_39:.*]]:2 = "onnx.Split"(%[[VAL_3]]) {axis = 0 : si64} : (memref<2x24xf32>) -> (memref<1x24xf32>, memref<1x24xf32>)
// CHECK:           %[[VAL_40:.*]] = "onnx.Squeeze"(%[[VAL_39]]#0) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK:           %[[VAL_41:.*]] = "onnx.Squeeze"(%[[VAL_39]]#1) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK:           %[[VAL_42:.*]]:6 = "onnx.Split"(%[[VAL_40]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_43:.*]]:6 = "onnx.Split"(%[[VAL_41]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
// CHECK:           %[[VAL_44:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_44]]) with (%[[VAL_44]] -> %[[VAL_45:.*]] = 0 to 7) {
// CHECK:             %[[VAL_46:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:             %[[VAL_47:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:             %[[VAL_48:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_49:.*]] = constant 0 : index
// CHECK:             %[[VAL_50:.*]] = constant 2 : index
// CHECK:             %[[VAL_51:.*]] = constant 3 : index
// CHECK:             %[[VAL_52:.*]] = constant 0 : index
// CHECK:             %[[VAL_53:.*]] = constant 0 : index
// CHECK:             %[[VAL_54:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_54]]#0, %[[VAL_54]]#1) with (%[[VAL_54]]#0 -> %[[VAL_55:.*]] = %[[VAL_52]] to %[[VAL_50]], %[[VAL_54]]#1 -> %[[VAL_56:.*]] = %[[VAL_53]] to %[[VAL_51]]) {
// CHECK:               %[[VAL_57:.*]]:2 = krnl.get_induction_var_value(%[[VAL_54]]#0, %[[VAL_54]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_58:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_45]], %[[VAL_57]]#0, %[[VAL_57]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_58]], %[[VAL_48]]{{\[}}%[[VAL_57]]#0, %[[VAL_57]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_59:.*]] = "onnx.MatMul"(%[[VAL_48]], %[[VAL_24]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_60:.*]] = "onnx.MatMul"(%[[VAL_6]], %[[VAL_28]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_61:.*]] = "onnx.MatMul"(%[[VAL_48]], %[[VAL_25]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_62:.*]] = "onnx.MatMul"(%[[VAL_6]], %[[VAL_29]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_63:.*]] = "onnx.MatMul"(%[[VAL_48]], %[[VAL_26]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_64:.*]] = constant 1.000000e+00 : f32
// CHECK:             %[[VAL_65:.*]] = constant 2 : index
// CHECK:             %[[VAL_66:.*]] = constant 4 : index
// CHECK:             %[[VAL_67:.*]] = constant 0 : index
// CHECK:             %[[VAL_68:.*]] = constant 0 : index
// CHECK:             %[[VAL_69:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_69]]#0, %[[VAL_69]]#1) with (%[[VAL_69]]#0 -> %[[VAL_70:.*]] = %[[VAL_67]] to %[[VAL_65]], %[[VAL_69]]#1 -> %[[VAL_71:.*]] = %[[VAL_68]] to %[[VAL_66]]) {
// CHECK:               %[[VAL_72:.*]]:2 = krnl.get_induction_var_value(%[[VAL_69]]#0, %[[VAL_69]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_73:.*]] = krnl.load %[[VAL_6]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_74:.*]] = krnl.load %[[VAL_61]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_75:.*]] = krnl.load %[[VAL_62]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_76:.*]] = addf %[[VAL_74]], %[[VAL_75]] : f32
// CHECK:               %[[VAL_77:.*]] = krnl.load %[[VAL_42]]#1{{\[}}%[[VAL_72]]#1] : memref<4xf32>
// CHECK:               %[[VAL_78:.*]] = krnl.load %[[VAL_42]]#4{{\[}}%[[VAL_72]]#1] : memref<4xf32>
// CHECK:               %[[VAL_79:.*]] = addf %[[VAL_76]], %[[VAL_77]] : f32
// CHECK:               %[[VAL_80:.*]] = addf %[[VAL_79]], %[[VAL_78]] : f32
// CHECK:               %[[VAL_81:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_80]], %[[VAL_81]][] : memref<f32>
// CHECK:               %[[VAL_82:.*]] = "onnx.Sigmoid"(%[[VAL_81]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_83:.*]] = krnl.load %[[VAL_82]][] : memref<f32>
// CHECK:               krnl.store %[[VAL_83]], %[[VAL_47]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_84:.*]] = mulf %[[VAL_83]], %[[VAL_73]] : f32
// CHECK:               krnl.store %[[VAL_84]], %[[VAL_46]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             %[[VAL_85:.*]] = "onnx.MatMul"(%[[VAL_46]], %[[VAL_30]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_86:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_86]]#0, %[[VAL_86]]#1) with (%[[VAL_86]]#0 -> %[[VAL_87:.*]] = %[[VAL_67]] to %[[VAL_65]], %[[VAL_86]]#1 -> %[[VAL_88:.*]] = %[[VAL_68]] to %[[VAL_66]]) {
// CHECK:               %[[VAL_89:.*]]:2 = krnl.get_induction_var_value(%[[VAL_86]]#0, %[[VAL_86]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_90:.*]] = krnl.load %[[VAL_6]]{{\[}}%[[VAL_89]]#0, %[[VAL_89]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_91:.*]] = krnl.load %[[VAL_59]]{{\[}}%[[VAL_89]]#0, %[[VAL_89]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_92:.*]] = krnl.load %[[VAL_60]]{{\[}}%[[VAL_89]]#0, %[[VAL_89]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_93:.*]] = addf %[[VAL_91]], %[[VAL_92]] : f32
// CHECK:               %[[VAL_94:.*]] = krnl.load %[[VAL_42]]#0{{\[}}%[[VAL_89]]#1] : memref<4xf32>
// CHECK:               %[[VAL_95:.*]] = krnl.load %[[VAL_42]]#3{{\[}}%[[VAL_89]]#1] : memref<4xf32>
// CHECK:               %[[VAL_96:.*]] = addf %[[VAL_93]], %[[VAL_94]] : f32
// CHECK:               %[[VAL_97:.*]] = addf %[[VAL_96]], %[[VAL_95]] : f32
// CHECK:               %[[VAL_98:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_97]], %[[VAL_98]][] : memref<f32>
// CHECK:               %[[VAL_99:.*]] = "onnx.Sigmoid"(%[[VAL_98]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_100:.*]] = krnl.load %[[VAL_99]][] : memref<f32>
// CHECK:               %[[VAL_101:.*]] = krnl.load %[[VAL_63]]{{\[}}%[[VAL_89]]#0, %[[VAL_89]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_102:.*]] = krnl.load %[[VAL_85]]{{\[}}%[[VAL_89]]#0, %[[VAL_89]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_103:.*]] = addf %[[VAL_101]], %[[VAL_102]] : f32
// CHECK:               %[[VAL_104:.*]] = krnl.load %[[VAL_42]]#2{{\[}}%[[VAL_89]]#1] : memref<4xf32>
// CHECK:               %[[VAL_105:.*]] = krnl.load %[[VAL_42]]#5{{\[}}%[[VAL_89]]#1] : memref<4xf32>
// CHECK:               %[[VAL_106:.*]] = addf %[[VAL_103]], %[[VAL_104]] : f32
// CHECK:               %[[VAL_107:.*]] = addf %[[VAL_106]], %[[VAL_105]] : f32
// CHECK:               %[[VAL_108:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_107]], %[[VAL_108]][] : memref<f32>
// CHECK:               %[[VAL_109:.*]] = "onnx.Tanh"(%[[VAL_108]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_110:.*]] = krnl.load %[[VAL_109]][] : memref<f32>
// CHECK:               %[[VAL_111:.*]] = subf %[[VAL_64]], %[[VAL_100]] : f32
// CHECK:               %[[VAL_112:.*]] = mulf %[[VAL_111]], %[[VAL_110]] : f32
// CHECK:               %[[VAL_113:.*]] = mulf %[[VAL_100]], %[[VAL_90]] : f32
// CHECK:               %[[VAL_114:.*]] = addf %[[VAL_112]], %[[VAL_113]] : f32
// CHECK:               krnl.store %[[VAL_114]], %[[VAL_6]]{{\[}}%[[VAL_89]]#0, %[[VAL_89]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_47]] : memref<2x4xf32>
// CHECK:             memref.dealloc %[[VAL_46]] : memref<2x4xf32>
// CHECK:             memref.dealloc %[[VAL_48]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_115:.*]] = krnl.define_loops 1
// CHECK:           krnl.iterate(%[[VAL_115]]) with (%[[VAL_115]] -> %[[VAL_116:.*]] = 0 to 7) {
// CHECK:             %[[VAL_117:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:             %[[VAL_118:.*]] = memref.alloc() : memref<2x4xf32>
// CHECK:             %[[VAL_119:.*]] = memref.alloc() : memref<2x3xf32>
// CHECK:             %[[VAL_120:.*]] = constant 1 : index
// CHECK:             %[[VAL_121:.*]] = constant 7 : index
// CHECK:             %[[VAL_122:.*]] = affine.apply #map(%[[VAL_116]]){{\[}}%[[VAL_121]]]
// CHECK:             %[[VAL_123:.*]] = constant 2 : index
// CHECK:             %[[VAL_124:.*]] = constant 3 : index
// CHECK:             %[[VAL_125:.*]] = constant 0 : index
// CHECK:             %[[VAL_126:.*]] = constant 0 : index
// CHECK:             %[[VAL_127:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_127]]#0, %[[VAL_127]]#1) with (%[[VAL_127]]#0 -> %[[VAL_128:.*]] = %[[VAL_125]] to %[[VAL_123]], %[[VAL_127]]#1 -> %[[VAL_129:.*]] = %[[VAL_126]] to %[[VAL_124]]) {
// CHECK:               %[[VAL_130:.*]]:2 = krnl.get_induction_var_value(%[[VAL_127]]#0, %[[VAL_127]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_131:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_122]], %[[VAL_130]]#0, %[[VAL_130]]#1] : memref<7x2x3xf32>
// CHECK:               krnl.store %[[VAL_131]], %[[VAL_119]]{{\[}}%[[VAL_130]]#0, %[[VAL_130]]#1] : memref<2x3xf32>
// CHECK:             }
// CHECK:             %[[VAL_132:.*]] = "onnx.MatMul"(%[[VAL_119]], %[[VAL_32]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_133:.*]] = "onnx.MatMul"(%[[VAL_5]], %[[VAL_36]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_134:.*]] = "onnx.MatMul"(%[[VAL_119]], %[[VAL_33]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_135:.*]] = "onnx.MatMul"(%[[VAL_5]], %[[VAL_37]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_136:.*]] = "onnx.MatMul"(%[[VAL_119]], %[[VAL_34]]) : (memref<2x3xf32>, memref<3x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_137:.*]] = constant 1.000000e+00 : f32
// CHECK:             %[[VAL_138:.*]] = constant 2 : index
// CHECK:             %[[VAL_139:.*]] = constant 4 : index
// CHECK:             %[[VAL_140:.*]] = constant 0 : index
// CHECK:             %[[VAL_141:.*]] = constant 0 : index
// CHECK:             %[[VAL_142:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_142]]#0, %[[VAL_142]]#1) with (%[[VAL_142]]#0 -> %[[VAL_143:.*]] = %[[VAL_140]] to %[[VAL_138]], %[[VAL_142]]#1 -> %[[VAL_144:.*]] = %[[VAL_141]] to %[[VAL_139]]) {
// CHECK:               %[[VAL_145:.*]]:2 = krnl.get_induction_var_value(%[[VAL_142]]#0, %[[VAL_142]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_146:.*]] = krnl.load %[[VAL_5]]{{\[}}%[[VAL_145]]#0, %[[VAL_145]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_147:.*]] = krnl.load %[[VAL_134]]{{\[}}%[[VAL_145]]#0, %[[VAL_145]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_148:.*]] = krnl.load %[[VAL_135]]{{\[}}%[[VAL_145]]#0, %[[VAL_145]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_149:.*]] = addf %[[VAL_147]], %[[VAL_148]] : f32
// CHECK:               %[[VAL_150:.*]] = krnl.load %[[VAL_43]]#1{{\[}}%[[VAL_145]]#1] : memref<4xf32>
// CHECK:               %[[VAL_151:.*]] = krnl.load %[[VAL_43]]#4{{\[}}%[[VAL_145]]#1] : memref<4xf32>
// CHECK:               %[[VAL_152:.*]] = addf %[[VAL_149]], %[[VAL_150]] : f32
// CHECK:               %[[VAL_153:.*]] = addf %[[VAL_152]], %[[VAL_151]] : f32
// CHECK:               %[[VAL_154:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_153]], %[[VAL_154]][] : memref<f32>
// CHECK:               %[[VAL_155:.*]] = "onnx.Sigmoid"(%[[VAL_154]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_156:.*]] = krnl.load %[[VAL_155]][] : memref<f32>
// CHECK:               krnl.store %[[VAL_156]], %[[VAL_118]]{{\[}}%[[VAL_145]]#0, %[[VAL_145]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_157:.*]] = mulf %[[VAL_156]], %[[VAL_146]] : f32
// CHECK:               krnl.store %[[VAL_157]], %[[VAL_117]]{{\[}}%[[VAL_145]]#0, %[[VAL_145]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             %[[VAL_158:.*]] = "onnx.MatMul"(%[[VAL_117]], %[[VAL_38]]) : (memref<2x4xf32>, memref<4x4xf32>) -> memref<2x4xf32>
// CHECK:             %[[VAL_159:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_159]]#0, %[[VAL_159]]#1) with (%[[VAL_159]]#0 -> %[[VAL_160:.*]] = %[[VAL_140]] to %[[VAL_138]], %[[VAL_159]]#1 -> %[[VAL_161:.*]] = %[[VAL_141]] to %[[VAL_139]]) {
// CHECK:               %[[VAL_162:.*]]:2 = krnl.get_induction_var_value(%[[VAL_159]]#0, %[[VAL_159]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_163:.*]] = krnl.load %[[VAL_5]]{{\[}}%[[VAL_162]]#0, %[[VAL_162]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_164:.*]] = krnl.load %[[VAL_132]]{{\[}}%[[VAL_162]]#0, %[[VAL_162]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_165:.*]] = krnl.load %[[VAL_133]]{{\[}}%[[VAL_162]]#0, %[[VAL_162]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_166:.*]] = addf %[[VAL_164]], %[[VAL_165]] : f32
// CHECK:               %[[VAL_167:.*]] = krnl.load %[[VAL_43]]#0{{\[}}%[[VAL_162]]#1] : memref<4xf32>
// CHECK:               %[[VAL_168:.*]] = krnl.load %[[VAL_43]]#3{{\[}}%[[VAL_162]]#1] : memref<4xf32>
// CHECK:               %[[VAL_169:.*]] = addf %[[VAL_166]], %[[VAL_167]] : f32
// CHECK:               %[[VAL_170:.*]] = addf %[[VAL_169]], %[[VAL_168]] : f32
// CHECK:               %[[VAL_171:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_170]], %[[VAL_171]][] : memref<f32>
// CHECK:               %[[VAL_172:.*]] = "onnx.Sigmoid"(%[[VAL_171]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_173:.*]] = krnl.load %[[VAL_172]][] : memref<f32>
// CHECK:               %[[VAL_174:.*]] = krnl.load %[[VAL_136]]{{\[}}%[[VAL_162]]#0, %[[VAL_162]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_175:.*]] = krnl.load %[[VAL_158]]{{\[}}%[[VAL_162]]#0, %[[VAL_162]]#1] : memref<2x4xf32>
// CHECK:               %[[VAL_176:.*]] = addf %[[VAL_174]], %[[VAL_175]] : f32
// CHECK:               %[[VAL_177:.*]] = krnl.load %[[VAL_43]]#2{{\[}}%[[VAL_162]]#1] : memref<4xf32>
// CHECK:               %[[VAL_178:.*]] = krnl.load %[[VAL_43]]#5{{\[}}%[[VAL_162]]#1] : memref<4xf32>
// CHECK:               %[[VAL_179:.*]] = addf %[[VAL_176]], %[[VAL_177]] : f32
// CHECK:               %[[VAL_180:.*]] = addf %[[VAL_179]], %[[VAL_178]] : f32
// CHECK:               %[[VAL_181:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_180]], %[[VAL_181]][] : memref<f32>
// CHECK:               %[[VAL_182:.*]] = "onnx.Tanh"(%[[VAL_181]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_183:.*]] = krnl.load %[[VAL_182]][] : memref<f32>
// CHECK:               %[[VAL_184:.*]] = subf %[[VAL_137]], %[[VAL_173]] : f32
// CHECK:               %[[VAL_185:.*]] = mulf %[[VAL_184]], %[[VAL_183]] : f32
// CHECK:               %[[VAL_186:.*]] = mulf %[[VAL_173]], %[[VAL_163]] : f32
// CHECK:               %[[VAL_187:.*]] = addf %[[VAL_185]], %[[VAL_186]] : f32
// CHECK:               krnl.store %[[VAL_187]], %[[VAL_5]]{{\[}}%[[VAL_162]]#0, %[[VAL_162]]#1] : memref<2x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_118]] : memref<2x4xf32>
// CHECK:             memref.dealloc %[[VAL_117]] : memref<2x4xf32>
// CHECK:             memref.dealloc %[[VAL_119]] : memref<2x3xf32>
// CHECK:           }
// CHECK:           %[[VAL_188:.*]] = constant 2 : index
// CHECK:           %[[VAL_189:.*]] = constant 4 : index
// CHECK:           %[[VAL_190:.*]] = constant 0 : index
// CHECK:           %[[VAL_191:.*]] = constant 0 : index
// CHECK:           %[[VAL_192:.*]] = constant 0 : index
// CHECK:           %[[VAL_193:.*]] = constant 1 : index
// CHECK:           %[[VAL_194:.*]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate(%[[VAL_194]]#0, %[[VAL_194]]#1) with (%[[VAL_194]]#0 -> %[[VAL_195:.*]] = %[[VAL_190]] to %[[VAL_188]], %[[VAL_194]]#1 -> %[[VAL_196:.*]] = %[[VAL_191]] to %[[VAL_189]]) {
// CHECK:             %[[VAL_197:.*]]:2 = krnl.get_induction_var_value(%[[VAL_194]]#0, %[[VAL_194]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             %[[VAL_198:.*]] = krnl.load %[[VAL_6]]{{\[}}%[[VAL_197]]#0, %[[VAL_197]]#1] : memref<2x4xf32>
// CHECK:             krnl.store %[[VAL_198]], %[[VAL_7]]{{\[}}%[[VAL_192]], %[[VAL_197]]#0, %[[VAL_197]]#1] : memref<2x2x4xf32>
// CHECK:             %[[VAL_199:.*]] = krnl.load %[[VAL_5]]{{\[}}%[[VAL_197]]#0, %[[VAL_197]]#1] : memref<2x4xf32>
// CHECK:             krnl.store %[[VAL_199]], %[[VAL_7]]{{\[}}%[[VAL_193]], %[[VAL_197]]#0, %[[VAL_197]]#1] : memref<2x2x4xf32>
// CHECK:           }
// CHECK:           memref.dealloc %[[VAL_6]] : memref<2x4xf32>
// CHECK:           memref.dealloc %[[VAL_5]] : memref<2x4xf32>
// CHECK:           return %[[VAL_7]] : memref<2x2x4xf32>
// CHECK:         }

}

// -----

func private @test_gru_unknown_dims(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x12x?xf32>, %arg2: tensor<1x12x4xf32>, %arg3: tensor<1x24xf32>, %arg4: tensor<1x?x4xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %arg4) {hidden_size = 4 : si64} : (tensor<?x?x?xf32>, tensor<1x12x?xf32>, tensor<1x12x4xf32>, tensor<1x24xf32>, none, tensor<1x?x4xf32>) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

// CHECK-LABEL:   func private @test_gru_unknown_dims(
// CHECK-SAME:                                        %[[VAL_0:.*]]: memref<?x?x?xf32>,
// CHECK-SAME:                                        %[[VAL_1:.*]]: memref<1x12x?xf32>,
// CHECK-SAME:                                        %[[VAL_2:.*]]: memref<1x12x4xf32>,
// CHECK-SAME:                                        %[[VAL_3:.*]]: memref<1x24xf32>,
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
// CHECK:           %[[VAL_21:.*]] = "onnx.Squeeze"(%[[VAL_1]]) {axes = [0]} : (memref<1x12x?xf32>) -> memref<12x?xf32>
// CHECK:           %[[VAL_22:.*]] = "onnx.Squeeze"(%[[VAL_2]]) {axes = [0]} : (memref<1x12x4xf32>) -> memref<12x4xf32>
// CHECK:           %[[VAL_23:.*]]:3 = "onnx.Split"(%[[VAL_21]]) {axis = 0 : si64} : (memref<12x?xf32>) -> (memref<4x?xf32>, memref<4x?xf32>, memref<4x?xf32>)
// CHECK:           %[[VAL_24:.*]] = "onnx.Transpose"(%[[VAL_23]]#0) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK:           %[[VAL_25:.*]] = "onnx.Transpose"(%[[VAL_23]]#1) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK:           %[[VAL_26:.*]] = "onnx.Transpose"(%[[VAL_23]]#2) {perm = [1, 0]} : (memref<4x?xf32>) -> memref<?x4xf32>
// CHECK:           %[[VAL_27:.*]]:3 = "onnx.Split"(%[[VAL_22]]) {axis = 0 : si64} : (memref<12x4xf32>) -> (memref<4x4xf32>, memref<4x4xf32>, memref<4x4xf32>)
// CHECK:           %[[VAL_28:.*]] = "onnx.Transpose"(%[[VAL_27]]#0) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_29:.*]] = "onnx.Transpose"(%[[VAL_27]]#1) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_30:.*]] = "onnx.Transpose"(%[[VAL_27]]#2) {perm = [1, 0]} : (memref<4x4xf32>) -> memref<4x4xf32>
// CHECK:           %[[VAL_31:.*]] = "onnx.Squeeze"(%[[VAL_3]]) {axes = [0]} : (memref<1x24xf32>) -> memref<24xf32>
// CHECK:           %[[VAL_32:.*]]:6 = "onnx.Split"(%[[VAL_31]]) {axis = 0 : si64} : (memref<24xf32>) -> (memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>, memref<4xf32>)
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
// CHECK:             %[[VAL_47:.*]] = constant 0 : index
// CHECK:             %[[VAL_48:.*]] = constant 0 : index
// CHECK:             %[[VAL_49:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_49]]#0, %[[VAL_49]]#1) with (%[[VAL_49]]#0 -> %[[VAL_50:.*]] = %[[VAL_47]] to %[[VAL_44]], %[[VAL_49]]#1 -> %[[VAL_51:.*]] = %[[VAL_48]] to %[[VAL_46]]) {
// CHECK:               %[[VAL_52:.*]]:2 = krnl.get_induction_var_value(%[[VAL_49]]#0, %[[VAL_49]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_53:.*]] = krnl.load %[[VAL_0]]{{\[}}%[[VAL_36]], %[[VAL_52]]#0, %[[VAL_52]]#1] : memref<?x?x?xf32>
// CHECK:               krnl.store %[[VAL_53]], %[[VAL_42]]{{\[}}%[[VAL_52]]#0, %[[VAL_52]]#1] : memref<?x?xf32>
// CHECK:             }
// CHECK:             %[[VAL_54:.*]] = "onnx.MatMul"(%[[VAL_42]], %[[VAL_24]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_55:.*]] = "onnx.MatMul"(%[[VAL_11]], %[[VAL_28]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_56:.*]] = "onnx.MatMul"(%[[VAL_42]], %[[VAL_25]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_57:.*]] = "onnx.MatMul"(%[[VAL_11]], %[[VAL_29]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_58:.*]] = "onnx.MatMul"(%[[VAL_42]], %[[VAL_26]]) : (memref<?x?xf32>, memref<?x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_59:.*]] = constant 1.000000e+00 : f32
// CHECK:             %[[VAL_60:.*]] = constant 0 : index
// CHECK:             %[[VAL_61:.*]] = memref.dim %[[VAL_11]], %[[VAL_60]] : memref<?x4xf32>
// CHECK:             %[[VAL_62:.*]] = memref.alloc(%[[VAL_61]]) : memref<?x4xf32>
// CHECK:             %[[VAL_63:.*]] = memref.alloc(%[[VAL_61]]) : memref<?x4xf32>
// CHECK:             %[[VAL_64:.*]] = constant 0 : index
// CHECK:             %[[VAL_65:.*]] = memref.dim %[[VAL_11]], %[[VAL_64]] : memref<?x4xf32>
// CHECK:             %[[VAL_66:.*]] = constant 4 : index
// CHECK:             %[[VAL_67:.*]] = constant 0 : index
// CHECK:             %[[VAL_68:.*]] = constant 0 : index
// CHECK:             %[[VAL_69:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_69]]#0, %[[VAL_69]]#1) with (%[[VAL_69]]#0 -> %[[VAL_70:.*]] = %[[VAL_67]] to %[[VAL_65]], %[[VAL_69]]#1 -> %[[VAL_71:.*]] = %[[VAL_68]] to %[[VAL_66]]) {
// CHECK:               %[[VAL_72:.*]]:2 = krnl.get_induction_var_value(%[[VAL_69]]#0, %[[VAL_69]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_73:.*]] = krnl.load %[[VAL_11]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_74:.*]] = krnl.load %[[VAL_56]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_75:.*]] = krnl.load %[[VAL_57]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_76:.*]] = addf %[[VAL_74]], %[[VAL_75]] : f32
// CHECK:               %[[VAL_77:.*]] = krnl.load %[[VAL_32]]#1{{\[}}%[[VAL_72]]#1] : memref<4xf32>
// CHECK:               %[[VAL_78:.*]] = krnl.load %[[VAL_32]]#4{{\[}}%[[VAL_72]]#1] : memref<4xf32>
// CHECK:               %[[VAL_79:.*]] = addf %[[VAL_76]], %[[VAL_77]] : f32
// CHECK:               %[[VAL_80:.*]] = addf %[[VAL_79]], %[[VAL_78]] : f32
// CHECK:               %[[VAL_81:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_80]], %[[VAL_81]][] : memref<f32>
// CHECK:               %[[VAL_82:.*]] = "onnx.Sigmoid"(%[[VAL_81]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_83:.*]] = krnl.load %[[VAL_82]][] : memref<f32>
// CHECK:               krnl.store %[[VAL_83]], %[[VAL_62]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_84:.*]] = mulf %[[VAL_83]], %[[VAL_73]] : f32
// CHECK:               krnl.store %[[VAL_84]], %[[VAL_63]]{{\[}}%[[VAL_72]]#0, %[[VAL_72]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:             %[[VAL_85:.*]] = "onnx.MatMul"(%[[VAL_63]], %[[VAL_30]]) : (memref<?x4xf32>, memref<4x4xf32>) -> memref<?x4xf32>
// CHECK:             %[[VAL_86:.*]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate(%[[VAL_86]]#0, %[[VAL_86]]#1) with (%[[VAL_86]]#0 -> %[[VAL_87:.*]] = %[[VAL_67]] to %[[VAL_65]], %[[VAL_86]]#1 -> %[[VAL_88:.*]] = %[[VAL_68]] to %[[VAL_66]]) {
// CHECK:               %[[VAL_89:.*]]:2 = krnl.get_induction_var_value(%[[VAL_86]]#0, %[[VAL_86]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               %[[VAL_90:.*]] = krnl.load %[[VAL_11]]{{\[}}%[[VAL_89]]#0, %[[VAL_89]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_91:.*]] = krnl.load %[[VAL_54]]{{\[}}%[[VAL_89]]#0, %[[VAL_89]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_92:.*]] = krnl.load %[[VAL_55]]{{\[}}%[[VAL_89]]#0, %[[VAL_89]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_93:.*]] = addf %[[VAL_91]], %[[VAL_92]] : f32
// CHECK:               %[[VAL_94:.*]] = krnl.load %[[VAL_32]]#0{{\[}}%[[VAL_89]]#1] : memref<4xf32>
// CHECK:               %[[VAL_95:.*]] = krnl.load %[[VAL_32]]#3{{\[}}%[[VAL_89]]#1] : memref<4xf32>
// CHECK:               %[[VAL_96:.*]] = addf %[[VAL_93]], %[[VAL_94]] : f32
// CHECK:               %[[VAL_97:.*]] = addf %[[VAL_96]], %[[VAL_95]] : f32
// CHECK:               %[[VAL_98:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_97]], %[[VAL_98]][] : memref<f32>
// CHECK:               %[[VAL_99:.*]] = "onnx.Sigmoid"(%[[VAL_98]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_100:.*]] = krnl.load %[[VAL_99]][] : memref<f32>
// CHECK:               %[[VAL_101:.*]] = krnl.load %[[VAL_58]]{{\[}}%[[VAL_89]]#0, %[[VAL_89]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_102:.*]] = krnl.load %[[VAL_85]]{{\[}}%[[VAL_89]]#0, %[[VAL_89]]#1] : memref<?x4xf32>
// CHECK:               %[[VAL_103:.*]] = addf %[[VAL_101]], %[[VAL_102]] : f32
// CHECK:               %[[VAL_104:.*]] = krnl.load %[[VAL_32]]#2{{\[}}%[[VAL_89]]#1] : memref<4xf32>
// CHECK:               %[[VAL_105:.*]] = krnl.load %[[VAL_32]]#5{{\[}}%[[VAL_89]]#1] : memref<4xf32>
// CHECK:               %[[VAL_106:.*]] = addf %[[VAL_103]], %[[VAL_104]] : f32
// CHECK:               %[[VAL_107:.*]] = addf %[[VAL_106]], %[[VAL_105]] : f32
// CHECK:               %[[VAL_108:.*]] = memref.alloca() : memref<f32>
// CHECK:               krnl.store %[[VAL_107]], %[[VAL_108]][] : memref<f32>
// CHECK:               %[[VAL_109:.*]] = "onnx.Tanh"(%[[VAL_108]]) : (memref<f32>) -> memref<f32>
// CHECK:               %[[VAL_110:.*]] = krnl.load %[[VAL_109]][] : memref<f32>
// CHECK:               %[[VAL_111:.*]] = subf %[[VAL_59]], %[[VAL_100]] : f32
// CHECK:               %[[VAL_112:.*]] = mulf %[[VAL_111]], %[[VAL_110]] : f32
// CHECK:               %[[VAL_113:.*]] = mulf %[[VAL_100]], %[[VAL_90]] : f32
// CHECK:               %[[VAL_114:.*]] = addf %[[VAL_112]], %[[VAL_113]] : f32
// CHECK:               krnl.store %[[VAL_114]], %[[VAL_11]]{{\[}}%[[VAL_89]]#0, %[[VAL_89]]#1] : memref<?x4xf32>
// CHECK:             }
// CHECK:             memref.dealloc %[[VAL_62]] : memref<?x4xf32>
// CHECK:             memref.dealloc %[[VAL_63]] : memref<?x4xf32>
// CHECK:             memref.dealloc %[[VAL_42]] : memref<?x?xf32>
// CHECK:           }
// CHECK:           %[[VAL_115:.*]] = constant 16 : i64
// CHECK:           %[[VAL_116:.*]] = constant 0 : index
// CHECK:           %[[VAL_117:.*]] = memref.dim %[[VAL_11]], %[[VAL_116]] : memref<?x4xf32>
// CHECK:           %[[VAL_118:.*]] = index_cast %[[VAL_117]] : index to i64
// CHECK:           %[[VAL_119:.*]] = muli %[[VAL_115]], %[[VAL_118]] : i64
// CHECK:           "krnl.memcpy"(%[[VAL_8]], %[[VAL_11]], %[[VAL_119]]) : (memref<1x?x4xf32>, memref<?x4xf32>, i64) -> ()
// CHECK:           memref.dealloc %[[VAL_11]] : memref<?x4xf32>
// CHECK:           return %[[VAL_8]] : memref<1x?x4xf32>
// CHECK:         }

}
