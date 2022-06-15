// RUN: onnx-mlir-opt --convert-krnl-to-llvm="verify-input-tensors" --canonicalize %s -split-input-file | FileCheck %s

// COM: Check verification code at the beginning of the entry point function.
module { 
  func @main_graph(%arg0: memref<3x4x5xf32>, %arg1: memref<?x4x5xf32>) -> memref<3x4x5xf32> {
    return %arg0 : memref<3x4x5xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [3 , 4 , 5] , \22name\22 : \22input0\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [3 , 4 , 5] , \22name\22 : \22input1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [? , 4 , 5] , \22name\22 : \22output0\22 }\0A\0A]\00"} : () -> ()

// CHECK-LABEL:   llvm.func @run_main_graph(
// CHECK-SAME:                              %[[VAL_0:.*]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       %[[VAL_1:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       %[[VAL_2:.*]] = llvm.call @omTensorListGetSize(%[[VAL_0]]) : (!llvm.ptr<i8>) -> i64
// CHECK:           %[[VAL_3:.*]] = llvm.icmp "ne" %[[VAL_1]], %[[VAL_2]] : i64
// CHECK:           llvm.cond_br %[[VAL_3]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK-DAG:       %[[VAL_4:.*]] = llvm.mlir.addressof @"Wrong number of input tensors: expect 2, but got {{\%}}lld\0A" : !llvm.ptr<array<54 x i8>>
// CHECK-DAG:       %[[VAL_5:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:       %[[VAL_6:.*]] = llvm.getelementptr %[[VAL_4]]{{\[}}%[[VAL_5]], %[[VAL_5]]] : (!llvm.ptr<array<54 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf(%[[VAL_6]], %[[VAL_2]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK-DAG:       %[[VAL_7:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr<i32>
// CHECK-DAG:       %[[VAL_8:.*]] = llvm.mlir.constant(22 : i32) : i32
// CHECK-DAG:       llvm.store %[[VAL_8]], %[[VAL_7]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return %[[VAL_9]] : !llvm.ptr<i8>
//
// CHECK:         ^bb2:
// CHECK-DAG:       %[[VAL_10:.*]] = llvm.call @omTensorListGetOmtArray(%[[VAL_0]]) : (!llvm.ptr<i8>) -> !llvm.ptr<ptr<i8>>
// CHECK-DAG:       %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr<ptr<i8>>
// CHECK-DAG:       %[[VAL_12:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       %[[VAL_13:.*]] = llvm.call @omTensorGetDataType(%[[VAL_11]]) : (!llvm.ptr<i8>) -> i64
// CHECK:           %[[VAL_14:.*]] = llvm.icmp "ne" %[[VAL_12]], %[[VAL_13]] : i64
// CHECK:           llvm.cond_br %[[VAL_14]], ^bb3, ^bb4
// CHECK:         ^bb3:
// CHECK-DAG:       %[[VAL_15:.*]] = llvm.mlir.addressof @"Wrong data type for the input 0: expect f32\0A" : !llvm.ptr<array<44 x i8>>
// CHECK-DAG:       %[[VAL_16:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:       %[[VAL_17:.*]] = llvm.getelementptr %[[VAL_15]]{{\[}}%[[VAL_16]], %[[VAL_16]]] : (!llvm.ptr<array<44 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf(%[[VAL_17]]) : (!llvm.ptr<i8>) -> ()
// CHECK-DAG:       %[[VAL_18:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr<i32>
// CHECK-DAG:       %[[VAL_19:.*]] = llvm.mlir.constant(22 : i32) : i32
// CHECK-DAG:       llvm.store %[[VAL_19]], %[[VAL_18]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_20:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return %[[VAL_20]] : !llvm.ptr<i8>
//
// CHECK:         ^bb4:
// CHECK-DAG:       %[[VAL_21:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG:       %[[VAL_22:.*]] = llvm.call @omTensorGetRank(%[[VAL_11]]) : (!llvm.ptr<i8>) -> i64
// CHECK:           %[[VAL_23:.*]] = llvm.icmp "ne" %[[VAL_21]], %[[VAL_22]] : i64
// CHECK:           llvm.cond_br %[[VAL_23]], ^bb5, ^bb6
// CHECK:         ^bb5:
// CHECK-DAG:       %[[VAL_24:.*]] = llvm.mlir.addressof @"Wrong rank for the input 0: expect 3, but got {{\%}}lld\0A" : !llvm.ptr<array<51 x i8>>
// CHECK-DAG:       %[[VAL_25:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:       %[[VAL_26:.*]] = llvm.getelementptr %[[VAL_24]]{{\[}}%[[VAL_25]], %[[VAL_25]]] : (!llvm.ptr<array<51 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf(%[[VAL_26]], %[[VAL_22]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK-DAG:       %[[VAL_27:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr<i32>
// CHECK-DAG:       %[[VAL_28:.*]] = llvm.mlir.constant(22 : i32) : i32
// CHECK-DAG:       llvm.store %[[VAL_28]], %[[VAL_27]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_29:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return %[[VAL_29]] : !llvm.ptr<i8>
//
// CHECK:         ^bb6:
// CHECK-DAG:       %[[VAL_30:.*]] = llvm.call @omTensorGetShape(%[[VAL_11]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i64>
// CHECK-DAG:       %[[VAL_31:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG:       %[[VAL_32:.*]] = llvm.load %[[VAL_30]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_33:.*]] = llvm.icmp "ne" %[[VAL_31]], %[[VAL_32]] : i64
// CHECK:           llvm.cond_br %[[VAL_33]], ^bb7, ^bb8
// CHECK:         ^bb7:
// CHECK-DAG:       %[[VAL_34:.*]] = llvm.mlir.addressof @"Wrong size for the dimension 0 of the input 0: expect 3, but got {{\%}}lld\0A" : !llvm.ptr<array<70 x i8>>
// CHECK-DAG:       %[[VAL_35:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:       %[[VAL_36:.*]] = llvm.getelementptr %[[VAL_34]]{{\[}}%[[VAL_35]], %[[VAL_35]]] : (!llvm.ptr<array<70 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf(%[[VAL_36]], %[[VAL_32]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK-DAG:       %[[VAL_37:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr<i32>
// CHECK-DAG:       %[[VAL_38:.*]] = llvm.mlir.constant(22 : i32) : i32
// CHECK-DAG:       llvm.store %[[VAL_38]], %[[VAL_37]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_39:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return %[[VAL_39]] : !llvm.ptr<i8>
//
// CHECK:         ^bb8:
// CHECK-DAG:       %[[VAL_40:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       %[[VAL_41:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       %[[VAL_42:.*]] = llvm.getelementptr %[[VAL_30]]{{\[}}%[[VAL_40]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
// CHECK-DAG:       %[[VAL_43:.*]] = llvm.load %[[VAL_42]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_44:.*]] = llvm.icmp "ne" %[[VAL_41]], %[[VAL_43]] : i64
// CHECK:           llvm.cond_br %[[VAL_44]], ^bb9, ^bb10
// CHECK:         ^bb9:
// CHECK-DAG:       %[[VAL_45:.*]] = llvm.mlir.addressof @"Wrong size for the dimension 1 of the input 0: expect 4, but got {{\%}}lld\0A" : !llvm.ptr<array<70 x i8>>
// CHECK-DAG:       %[[VAL_46:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:       %[[VAL_47:.*]] = llvm.getelementptr %[[VAL_45]]{{\[}}%[[VAL_46]], %[[VAL_46]]] : (!llvm.ptr<array<70 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf(%[[VAL_47]], %[[VAL_43]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK-DAG:       %[[VAL_48:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr<i32>
// CHECK-DAG:       %[[VAL_49:.*]] = llvm.mlir.constant(22 : i32) : i32
// CHECK-DAG:       llvm.store %[[VAL_49]], %[[VAL_48]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_50:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return %[[VAL_50]] : !llvm.ptr<i8>
//
// CHECK:         ^bb10:
// CHECK-DAG:       %[[VAL_51:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       %[[VAL_52:.*]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG:       %[[VAL_53:.*]] = llvm.getelementptr %[[VAL_30]]{{\[}}%[[VAL_51]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
// CHECK-DAG:       %[[VAL_54:.*]] = llvm.load %[[VAL_53]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_55:.*]] = llvm.icmp "ne" %[[VAL_52]], %[[VAL_54]] : i64
// CHECK:           llvm.cond_br %[[VAL_55]], ^bb11, ^bb12
// CHECK:         ^bb11:
// CHECK-DAG:       %[[VAL_56:.*]] = llvm.mlir.addressof @"Wrong size for the dimension 2 of the input 0: expect 5, but got {{\%}}lld\0A" : !llvm.ptr<array<70 x i8>>
// CHECK-DAG:       %[[VAL_57:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:       %[[VAL_58:.*]] = llvm.getelementptr %[[VAL_56]]{{\[}}%[[VAL_57]], %[[VAL_57]]] : (!llvm.ptr<array<70 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf(%[[VAL_58]], %[[VAL_54]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK-DAG:       %[[VAL_59:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr<i32>
// CHECK-DAG:       %[[VAL_60:.*]] = llvm.mlir.constant(22 : i32) : i32
// CHECK-DAG:       llvm.store %[[VAL_60]], %[[VAL_59]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_61:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return %[[VAL_61]] : !llvm.ptr<i8>
//
// CHECK:         ^bb12:
// CHECK-DAG:       %[[VAL_62:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       %[[VAL_63:.*]] = llvm.getelementptr %[[VAL_10]]{{\[}}%[[VAL_62]]] : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
// CHECK-DAG:       %[[VAL_64:.*]] = llvm.load %[[VAL_63]] : !llvm.ptr<ptr<i8>>
// CHECK-DAG:       %[[VAL_65:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       %[[VAL_66:.*]] = llvm.call @omTensorGetDataType(%[[VAL_64]]) : (!llvm.ptr<i8>) -> i64
// CHECK:           %[[VAL_67:.*]] = llvm.icmp "ne" %[[VAL_65]], %[[VAL_66]] : i64
// CHECK:           llvm.cond_br %[[VAL_67]], ^bb13, ^bb14
// CHECK:         ^bb13:
// CHECK-DAG:       %[[VAL_68:.*]] = llvm.mlir.addressof @"Wrong data type for the input 1: expect f32\0A" : !llvm.ptr<array<44 x i8>>
// CHECK-DAG:       %[[VAL_69:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:       %[[VAL_70:.*]] = llvm.getelementptr %[[VAL_68]]{{\[}}%[[VAL_69]], %[[VAL_69]]] : (!llvm.ptr<array<44 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf(%[[VAL_70]]) : (!llvm.ptr<i8>) -> ()
// CHECK-DAG:       %[[VAL_71:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr<i32>
// CHECK-DAG:       %[[VAL_72:.*]] = llvm.mlir.constant(22 : i32) : i32
// CHECK-DAG:       llvm.store %[[VAL_72]], %[[VAL_71]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_73:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return %[[VAL_73]] : !llvm.ptr<i8>
//
// CHECK:         ^bb14:
// CHECK-DAG:       %[[VAL_74:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG:       %[[VAL_75:.*]] = llvm.call @omTensorGetRank(%[[VAL_64]]) : (!llvm.ptr<i8>) -> i64
// CHECK:           %[[VAL_76:.*]] = llvm.icmp "ne" %[[VAL_74]], %[[VAL_75]] : i64
// CHECK:           llvm.cond_br %[[VAL_76]], ^bb15, ^bb16
// CHECK:         ^bb15:
// CHECK-DAG:       %[[VAL_77:.*]] = llvm.mlir.addressof @"Wrong rank for the input 1: expect 3, but got {{\%}}lld\0A" : !llvm.ptr<array<51 x i8>>
// CHECK-DAG:       %[[VAL_78:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:       %[[VAL_79:.*]] = llvm.getelementptr %[[VAL_77]]{{\[}}%[[VAL_78]], %[[VAL_78]]] : (!llvm.ptr<array<51 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf(%[[VAL_79]], %[[VAL_75]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK-DAG:       %[[VAL_80:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr<i32>
// CHECK-DAG:       %[[VAL_81:.*]] = llvm.mlir.constant(22 : i32) : i32
// CHECK-DAG:       llvm.store %[[VAL_81]], %[[VAL_80]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_82:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return %[[VAL_82]] : !llvm.ptr<i8>
//
// CHECK:         ^bb16:
// CHECK-DAG:       %[[VAL_83:.*]] = llvm.call @omTensorGetShape(%[[VAL_64]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i64>
// CHECK-DAG:       %[[VAL_84:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG:       %[[VAL_85:.*]] = llvm.load %[[VAL_83]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_86:.*]] = llvm.icmp "ne" %[[VAL_84]], %[[VAL_85]] : i64
// CHECK:           llvm.cond_br %[[VAL_86]], ^bb17, ^bb18
// CHECK:         ^bb17:
// CHECK-DAG:       %[[VAL_87:.*]] = llvm.mlir.addressof @"Wrong size for the dimension 0 of the input 1: expect 3, but got {{\%}}lld\0A" : !llvm.ptr<array<70 x i8>>
// CHECK-DAG:       %[[VAL_88:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:       %[[VAL_89:.*]] = llvm.getelementptr %[[VAL_87]]{{\[}}%[[VAL_88]], %[[VAL_88]]] : (!llvm.ptr<array<70 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf(%[[VAL_89]], %[[VAL_85]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK-DAG:       %[[VAL_90:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr<i32>
// CHECK-DAG:       %[[VAL_91:.*]] = llvm.mlir.constant(22 : i32) : i32
// CHECK-DAG:       llvm.store %[[VAL_91]], %[[VAL_90]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_92:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return %[[VAL_92]] : !llvm.ptr<i8>
//
// CHECK:         ^bb18:
// CHECK-DAG:       %[[VAL_93:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       %[[VAL_94:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       %[[VAL_95:.*]] = llvm.getelementptr %[[VAL_83]]{{\[}}%[[VAL_93]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
// CHECK-DAG:       %[[VAL_96:.*]] = llvm.load %[[VAL_95]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_97:.*]] = llvm.icmp "ne" %[[VAL_94]], %[[VAL_96]] : i64
// CHECK:           llvm.cond_br %[[VAL_97]], ^bb19, ^bb20
// CHECK:         ^bb19:
// CHECK-DAG:       %[[VAL_98:.*]] = llvm.mlir.addressof @"Wrong size for the dimension 1 of the input 1: expect 4, but got {{\%}}lld\0A" : !llvm.ptr<array<70 x i8>>
// CHECK-DAG:       %[[VAL_99:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:       %[[VAL_100:.*]] = llvm.getelementptr %[[VAL_98]]{{\[}}%[[VAL_99]], %[[VAL_99]]] : (!llvm.ptr<array<70 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf(%[[VAL_100]], %[[VAL_96]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK-DAG:       %[[VAL_101:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr<i32>
// CHECK-DAG:       %[[VAL_102:.*]] = llvm.mlir.constant(22 : i32) : i32
// CHECK-DAG:       llvm.store %[[VAL_102]], %[[VAL_101]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_103:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return %[[VAL_103]] : !llvm.ptr<i8>
//
// CHECK:         ^bb20:
// CHECK-DAG:       %[[VAL_104:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       %[[VAL_105:.*]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG:       %[[VAL_106:.*]] = llvm.getelementptr %[[VAL_83]]{{\[}}%[[VAL_104]]] : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
// CHECK-DAG:       %[[VAL_107:.*]] = llvm.load %[[VAL_106]] : !llvm.ptr<i64>
// CHECK:           %[[VAL_108:.*]] = llvm.icmp "ne" %[[VAL_105]], %[[VAL_107]] : i64
// CHECK:           llvm.cond_br %[[VAL_108]], ^bb21, ^bb22
// CHECK:         ^bb21:
// CHECK-DAG:       %[[VAL_109:.*]] = llvm.mlir.addressof @"Wrong size for the dimension 2 of the input 1: expect 5, but got {{\%}}lld\0A" : !llvm.ptr<array<70 x i8>>
// CHECK-DAG:       %[[VAL_110:.*]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:       %[[VAL_111:.*]] = llvm.getelementptr %[[VAL_109]]{{\[}}%[[VAL_110]], %[[VAL_110]]] : (!llvm.ptr<array<70 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf(%[[VAL_111]], %[[VAL_107]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK-DAG:       %[[VAL_112:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr<i32>
// CHECK-DAG:       %[[VAL_113:.*]] = llvm.mlir.constant(22 : i32) : i32
// CHECK-DAG:       llvm.store %[[VAL_113]], %[[VAL_112]] : !llvm.ptr<i32>
// CHECK:           %[[VAL_114:.*]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return %[[VAL_114]] : !llvm.ptr<i8>
//
// CHECK:         ^bb22:
// CHECK:           %[[VAL_115:.*]] = llvm.call @omTensorListGetOmtArray(%[[VAL_0]]) : (!llvm.ptr<i8>) -> !llvm.ptr<ptr<i8>>
}
