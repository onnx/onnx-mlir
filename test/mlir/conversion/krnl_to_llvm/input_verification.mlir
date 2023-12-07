// RUN: onnx-mlir-opt --convert-krnl-to-llvm="verify-input-tensors=true" --canonicalize %s -split-input-file | FileCheck %s

// -----

// COM: Check verification code at the beginning of the entry point function.

module {
  func.func @main_graph(%arg0: memref<3x4x5xf32>, %arg1: memref<?x4x5xf32>) -> memref<3x4x5xf32> {
    return %arg0 : memref<3x4x5xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [3 , 4 , 5] , \22name\22 : \22input0\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [-1 , 4 , 5] , \22name\22 : \22input1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [3 , 4 , 5] , \22name\22 : \22output0\22 }\0A\0A]\00"} : () -> ()

// CHECK-LABEL:   llvm.func @run_main_graph(
// CHECK-SAME:                              %[[VAL_0:.*]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       %[[CONST_2:.*]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       %[[CONST_0:.*]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       %[[CONST_22:.*]] = llvm.mlir.constant(22 : i32) : i32
// CHECK-DAG:       %[[CONST_1:.*]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       %[[CONST_3:.*]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG:       %[[CONST_4:.*]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       %[[CONST_5:.*]] = llvm.mlir.constant(5 : i64) : i64
// CHECK:           %[[VAL_2:.*]] = llvm.call @omTensorListGetSize(%[[VAL_0]]) : (!llvm.ptr) -> i64
// CHECK:           %[[VAL_3:.*]] = llvm.icmp "ne" %[[CONST_2]], %[[VAL_2]] : i64
// CHECK:           llvm.cond_br %[[VAL_3]], ^bb1, ^bb2
// CHECK:         ^bb1:
// CHECK:           %[[VAL_4:.*]] = llvm.mlir.addressof @"om_Wrong number of input tensors: expect 2, but got {{\%}}lld\0A" : !llvm.ptr<array<54 x i8>>
// CHECK:           llvm.call @printf(%[[VAL_4]], %[[VAL_2]]) : (!llvm.ptr, i64) -> ()
// CHECK-DAG:       %[[VAL_7:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr
// CHECK-DAG:       llvm.store %[[CONST_22]], %[[VAL_7]] : i32, !llvm.ptr
// CHECK:           %[[VAL_9:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return %[[VAL_9]] : !llvm.ptr
//
// CHECK:         ^bb2:
// CHECK-DAG:       %[[VAL_10:.*]] = llvm.call @omTensorListGetOmtArray(%[[VAL_0]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-DAG:       %[[VAL_11:.*]] = llvm.load %[[VAL_10]] : !llvm.ptr
// CHECK-DAG:       %[[VAL_13:.*]] = llvm.call @omTensorGetDataType(%[[VAL_11]]) : (!llvm.ptr) -> i64
// CHECK:           %[[VAL_14:.*]] = llvm.icmp "ne" %[[CONST_1]], %[[VAL_13]] : i64
// CHECK:           llvm.cond_br %[[VAL_14]], ^bb3, ^bb4
// CHECK:         ^bb3:
// CHECK-DAG:       %[[VAL_15:.*]] = llvm.mlir.addressof @"om_Wrong data type for the input 0: expect f32\0A" : !llvm.ptr
// CHECK:           llvm.call @printf(%[[VAL_15]]) : (!llvm.ptr) -> ()
// CHECK-DAG:       %[[VAL_18:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr
// CHECK-DAG:       llvm.store %[[CONST_22]], %[[VAL_18]] : i32, !llvm.ptr
// CHECK:           %[[VAL_20:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return %[[VAL_20]] : !llvm.ptr
//
// CHECK:         ^bb4:
// CHECK-DAG:       %[[VAL_22:.*]] = llvm.call @omTensorGetRank(%[[VAL_11]]) : (!llvm.ptr) -> i64
// CHECK:           %[[VAL_23:.*]] = llvm.icmp "ne" %[[CONST_3]], %[[VAL_22]] : i64
// CHECK:           llvm.cond_br %[[VAL_23]], ^bb5, ^bb6
// CHECK:         ^bb5:
// CHECK-DAG:       %[[VAL_24:.*]] = llvm.mlir.addressof @"om_Wrong rank for the input 0: expect 3, but got {{\%}}lld\0A" : !llvm.ptr
// CHECK:           llvm.call @printf(%[[VAL_24]], %[[VAL_22]]) : (!llvm.ptr, i64) -> ()
// CHECK-DAG:       %[[VAL_27:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr
// CHECK-DAG:       llvm.store %[[CONST_22]], %[[VAL_27]] : i32, !llvm.ptr
// CHECK:           %[[VAL_29:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return %[[VAL_29]] : !llvm.ptr
//
// CHECK:         ^bb6:
// CHECK-DAG:       %[[VAL_30:.*]] = llvm.call @omTensorGetShape(%[[VAL_11]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-DAG:       %[[VAL_31:.*]] = llvm.load %[[VAL_30]] : !llvm.ptr
// CHECK-DAG:       %[[VAL_33:.*]] = llvm.icmp "ne" %[[CONST_3]], %[[VAL_31]] : i64
// CHECK:           llvm.cond_br %[[VAL_33]], ^bb7, ^bb8
// CHECK:         ^bb7:
// CHECK-DAG:       %[[VAL_34:.*]] = llvm.mlir.addressof @"om_Wrong size for the dimension 0 of the input 0: expect 3, but got {{\%}}lld\0A" : !llvm.ptr
// CHECK:           llvm.call @printf(%[[VAL_34]], %[[VAL_31]]) : (!llvm.ptr, i64) -> ()
// CHECK-DAG:       %[[VAL_37:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr
// CHECK-DAG:       llvm.store %[[CONST_22]], %[[VAL_37]] : i32, !llvm.ptr
// CHECK:           %[[VAL_39:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return %[[VAL_39]] : !llvm.ptr
//
// CHECK:         ^bb8:
// CHECK-DAG:       %[[VAL_41:.*]] = llvm.getelementptr %[[VAL_30]][1] : (!llvm.ptr) -> !llvm.ptr
// CHECK-DAG:       %[[VAL_42:.*]] = llvm.load %[[VAL_41]] : !llvm.ptr
// CHECK-DAG:       %[[VAL_44:.*]] = llvm.icmp "ne" %[[CONST_4]], %[[VAL_42]] : i64
// CHECK:           llvm.cond_br %[[VAL_44]], ^bb9, ^bb10
// CHECK:         ^bb9:
// CHECK-DAG:       %[[VAL_45:.*]] = llvm.mlir.addressof @"om_Wrong size for the dimension 1 of the input 0: expect 4, but got {{\%}}lld\0A" : !llvm.ptr
// CHECK:           llvm.call @printf(%[[VAL_45]], %[[VAL_42]]) : (!llvm.ptr, i64) -> ()
// CHECK-DAG:       %[[VAL_48:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr
// CHECK-DAG:       llvm.store %[[CONST_22]], %[[VAL_48]] : i32, !llvm.ptr
// CHECK:           %[[VAL_50:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return %[[VAL_50]] : !llvm.ptr
//
// CHECK:         ^bb10:
// CHECK-DAG:       %[[VAL_52:.*]] = llvm.getelementptr %[[VAL_30]][2] : (!llvm.ptr) -> !llvm.ptr
// CHECK-DAG:       %[[VAL_53:.*]] = llvm.load %[[VAL_52]] : !llvm.ptr
// CHECK-DAG:       %[[VAL_55:.*]] = llvm.icmp "ne" %[[CONST_5]], %[[VAL_53]] : i64
// CHECK:           llvm.cond_br %[[VAL_55]], ^bb11, ^bb12
// CHECK:         ^bb11:
// CHECK-DAG:       %[[VAL_56:.*]] = llvm.mlir.addressof @"om_Wrong size for the dimension 2 of the input 0: expect 5, but got {{\%}}lld\0A" : !llvm.ptr
// CHECK:           llvm.call @printf(%[[VAL_56]], %[[VAL_53]]) : (!llvm.ptr, i64) -> ()
// CHECK-DAG:       %[[VAL_59:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr
// CHECK-DAG:       llvm.store %[[CONST_22]], %[[VAL_59]] : i32, !llvm.ptr
// CHECK:           %[[VAL_61:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return %[[VAL_61]] : !llvm.ptr
//
// CHECK:         ^bb12:
// CHECK-DAG:       %[[VAL_63:.*]] = llvm.getelementptr %[[VAL_10]][1] : (!llvm.ptr) -> !llvm.ptr
// CHECK-DAG:       %[[VAL_64:.*]] = llvm.load %[[VAL_63]] : !llvm.ptr
// CHECK-DAG:       %[[VAL_66:.*]] = llvm.call @omTensorGetDataType(%[[VAL_64]]) : (!llvm.ptr) -> i64
// CHECK:           %[[VAL_67:.*]] = llvm.icmp "ne" %[[CONST_1]], %[[VAL_66]] : i64
// CHECK:           llvm.cond_br %[[VAL_67]], ^bb13, ^bb14
// CHECK:         ^bb13:
// CHECK-DAG:       %[[VAL_68:.*]] = llvm.mlir.addressof @"om_Wrong data type for the input 1: expect f32\0A" : !llvm.ptr
// CHECK:           llvm.call @printf(%[[VAL_68]]) : (!llvm.ptr) -> ()
// CHECK-DAG:       %[[VAL_71:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr
// CHECK-DAG:       llvm.store %[[CONST_22]], %[[VAL_71]] : i32, !llvm.ptr
// CHECK:           %[[VAL_73:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return %[[VAL_73]] : !llvm.ptr
//
// CHECK:         ^bb14:
// CHECK-DAG:       %[[VAL_75:.*]] = llvm.call @omTensorGetRank(%[[VAL_64]]) : (!llvm.ptr) -> i64
// CHECK-DAG:       %[[VAL_76:.*]] = llvm.icmp "ne" %[[CONST_3]], %[[VAL_75]] : i64
// CHECK:           llvm.cond_br %[[VAL_76]], ^bb15, ^bb16
// CHECK:         ^bb15:
// CHECK-DAG:       %[[VAL_77:.*]] = llvm.mlir.addressof @"om_Wrong rank for the input 1: expect 3, but got {{\%}}lld\0A" : !llvm.ptr
// CHECK:           llvm.call @printf(%[[VAL_77]], %[[VAL_75]]) : (!llvm.ptr, i64) -> ()
// CHECK-DAG:       %[[VAL_80:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr
// CHECK-DAG:       llvm.store %[[CONST_22]], %[[VAL_80]] : i32, !llvm.ptr
// CHECK:           %[[VAL_82:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return %[[VAL_82]] : !llvm.ptr
//
// CHECK:         ^bb16:
// CHECK:           %[[VAL_83:.*]] = llvm.call @omTensorGetShape(%[[VAL_64]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-DAG:       %[[VAL_84:.*]] = llvm.load %[[VAL_83]] : !llvm.ptr
// CHECK-DAG:       %[[VAL_86:.*]] = llvm.icmp "slt" %[[VAL_84]], %[[CONST_0]] : i64
// CHECK:           llvm.cond_br %[[VAL_86]], ^bb17, ^bb18
// CHECK:         ^bb17:
// CHECK-DAG:       %[[VAL_87:.*]] = llvm.mlir.addressof @"om_Wrong size for the dimension 0 of the input 1: expect a non-negative value\0A" : !llvm.ptr
// CHECK:           llvm.call @printf(%[[VAL_87]]) : (!llvm.ptr) -> ()
// CHECK-DAG:       %[[VAL_90:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr
// CHECK-DAG:       llvm.store %[[CONST_22]], %[[VAL_90]] : i32, !llvm.ptr
// CHECK:           %[[VAL_92:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return %[[VAL_92]] : !llvm.ptr
//
// CHECK:         ^bb18:
// CHECK-DAG:       %[[VAL_94:.*]] = llvm.getelementptr %[[VAL_83]][1] : (!llvm.ptr) -> !llvm.ptr
// CHECK-DAG:       %[[VAL_95:.*]] = llvm.load %[[VAL_94]] : !llvm.ptr
// CHECK-DAG:       %[[VAL_97:.*]] = llvm.icmp "ne" %[[CONST_4]], %[[VAL_95]] : i64
// CHECK:           llvm.cond_br %[[VAL_97]], ^bb19, ^bb20
// CHECK:         ^bb19:
// CHECK-DAG:       %[[VAL_98:.*]] = llvm.mlir.addressof @"om_Wrong size for the dimension 1 of the input 1: expect 4, but got {{\%}}lld\0A" : !llvm.ptr
// CHECK:           llvm.call @printf(%[[VAL_98]], %[[VAL_95]]) : (!llvm.ptr, i64) -> ()
// CHECK-DAG:       %[[VAL_101:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr
// CHECK-DAG:       llvm.store %[[CONST_22]], %[[VAL_101]] : i32, !llvm.ptr
// CHECK:           %[[VAL_103:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return %[[VAL_103]] : !llvm.ptr
//
// CHECK:         ^bb20:
// CHECK-DAG:      %[[VAL_105:.*]] = llvm.getelementptr %[[VAL_83]][2] : (!llvm.ptr) -> !llvm.ptr
// CHECK-DAG:      %[[VAL_106:.*]] = llvm.load %[[VAL_105]] : !llvm.ptr
// CHECK-DAG:      %[[VAL_108:.*]] = llvm.icmp "ne" %[[CONST_5]], %[[VAL_106]] : i64
// CHECK:           llvm.cond_br %[[VAL_108]], ^bb21, ^bb22
// CHECK:         ^bb21:
// CHECK-DAG:       %[[VAL_109:.*]] = llvm.mlir.addressof @"om_Wrong size for the dimension 2 of the input 1: expect 5, but got {{\%}}lld\0A" : !llvm.ptr
// CHECK:           llvm.call @printf(%[[VAL_109]], %[[VAL_106]]) : (!llvm.ptr, i64) -> ()
// CHECK-DAG:       %[[VAL_112:.*]] = llvm.call @__errno_location() : () -> !llvm.ptr
// CHECK-DAG:       llvm.store %[[CONST_22]], %[[VAL_112]] : i32, !llvm.ptr
// CHECK:           %[[VAL_114:.*]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return %[[VAL_114]] : !llvm.ptr
//
// CHECK:         ^bb22:
// CHECK:           %[[VAL_115:.*]] = llvm.call @omTensorListGetOmtArray(%[[VAL_0]]) : (!llvm.ptr) -> !llvm.ptr
}
