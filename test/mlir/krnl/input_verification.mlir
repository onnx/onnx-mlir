// RUN: onnx-mlir-opt --convert-krnl-to-llvm="verify-input-tensors" --canonicalize %s -split-input-file | FileCheck %s

// COM: Check verification code at the beginning of the entry point function.
module { 
  func @main_graph(%arg0: memref<3x4x5xf32>, %arg1: memref<?x4x5xf32>) -> memref<3x4x5xf32> {
    return %arg0 : memref<3x4x5xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [3 , 4 , 5] , \22name\22 : \22input0\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [3 , 4 , 5] , \22name\22 : \22input1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [? , 4 , 5] , \22name\22 : \22output0\22 }\0A\0A]\00"} : () -> ()

// CHECK:         llvm.func @run_main_graph([[arg0_:%.+]]: !llvm.ptr<i8>) -> !llvm.ptr<i8> {
// CHECK-DAG:       [[VAR_0_2_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_1_2_:%.+]] = llvm.call @omTensorListGetSize([[arg0_]]) : (!llvm.ptr<i8>) -> i64
// CHECK:           [[VAR_2_2_:%.+]] = llvm.icmp "ne" [[VAR_0_2_]], [[VAR_1_2_]] : i64
// CHECK:           llvm.cond_br [[VAR_2_2_]], ^bb1, ^bb2
// CHECK:         ^bb1:  // pred: ^bb0
// CHECK-DAG:       [[VAR_3_2_:%.+]] = llvm.mlir.addressof @"Wrong number of input tensors: expect 2, but got {{\%}}lld\0A" : !llvm.ptr<array<54 x i8>>
// CHECK-DAG:       [[VAR_4_2_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_5_2_:%.+]] = llvm.getelementptr [[VAR_3_2_]]{{.}}[[VAR_4_2_]], [[VAR_4_2_]]{{.}} : (!llvm.ptr<array<54 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf([[VAR_5_2_]], [[VAR_1_2_]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK:           [[VAR_6_2_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return [[VAR_6_2_]] : !llvm.ptr<i8>

// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_7_2_:%.+]] = llvm.call @omTensorListGetOmtArray([[arg0_]]) : (!llvm.ptr<i8>) -> !llvm.ptr<ptr<i8>>
// CHECK-DAG:       [[VAR_8_1_:%.+]] = llvm.load [[VAR_7_2_]] : !llvm.ptr<ptr<i8>>
// CHECK-DAG:       [[VAR_9_2_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[LOAD_arg2_MEM_1_:%.+]] = llvm.call @omTensorGetDataType([[VAR_8_1_]]) : (!llvm.ptr<i8>) -> i64
// CHECK:           [[VAR_11_1_:%.+]] = llvm.icmp "ne" [[VAR_9_2_]], [[LOAD_arg2_MEM_1_]] : i64
// CHECK:           llvm.cond_br [[VAR_11_1_]], ^bb3, ^bb4
// CHECK:         ^bb3:  // pred: ^bb2
// CHECK-DAG:       [[VAR_12_1_:%.+]] = llvm.mlir.addressof @"Wrong data type for the input 0: expect f32\0A" : !llvm.ptr<array<44 x i8>>
// CHECK-DAG:       [[VAR_13_1_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_14_1_:%.+]] = llvm.getelementptr [[VAR_12_1_]]{{.}}[[VAR_13_1_]], [[VAR_13_1_]]{{.}} : (!llvm.ptr<array<44 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf([[VAR_14_1_]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           [[VAR_15_1_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return [[VAR_15_1_]] : !llvm.ptr<i8>

// CHECK:         ^bb4:  // pred: ^bb2
// CHECK-DAG:       [[VAR_16_1_:%.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG:       [[VAR_17_1_:%.+]] = llvm.call @omTensorGetRank([[VAR_8_1_]]) : (!llvm.ptr<i8>) -> i64
// CHECK:           [[VAR_18_1_:%.+]] = llvm.icmp "ne" [[VAR_16_1_]], [[VAR_17_1_]] : i64
// CHECK:           llvm.cond_br [[VAR_18_1_]], ^bb5, ^bb6
// CHECK:         ^bb5:  // pred: ^bb4
// CHECK-DAG:       [[VAR_19_1_:%.+]] = llvm.mlir.addressof @"Wrong rank for the input 0: expect 3, but got {{\%}}lld\0A" : !llvm.ptr<array<51 x i8>>
// CHECK-DAG:       [[VAR_20_1_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_21_:%.+]] = llvm.getelementptr [[VAR_19_1_]]{{.}}[[VAR_20_1_]], [[VAR_20_1_]]{{.}} : (!llvm.ptr<array<51 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf([[VAR_21_]], [[VAR_17_1_]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK:           [[VAR_22_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return [[VAR_22_]] : !llvm.ptr<i8>

// CHECK:         ^bb6:  // pred: ^bb4
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.call @omTensorGetShape([[VAR_8_1_]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i64>
// CHECK-DAG:       [[VAR_24_:%.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG:       [[LOAD_VAR_23_MEM_:%.+]] = llvm.load [[VAR_23_]] : !llvm.ptr<i64>
// CHECK:           [[VAR_26_:%.+]] = llvm.icmp "ne" [[VAR_24_]], [[LOAD_VAR_23_MEM_]] : i64
// CHECK:           llvm.cond_br [[VAR_26_]], ^bb7, ^bb8
// CHECK:         ^bb7:  // pred: ^bb6
// CHECK-DAG:       [[VAR_27_:%.+]] = llvm.mlir.addressof @"Wrong size for the dimension 0 of the input 0: expect 3, but got {{\%}}lld\0A" : !llvm.ptr<array<70 x i8>>
// CHECK-DAG:       [[VAR_28_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_29_:%.+]] = llvm.getelementptr [[VAR_27_]]{{.}}[[VAR_28_]], [[VAR_28_]]{{.}} : (!llvm.ptr<array<70 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf([[VAR_29_]], [[LOAD_VAR_23_MEM_]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK:           [[VAR_30_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return [[VAR_30_]] : !llvm.ptr<i8>

// CHECK:         ^bb8:  // pred: ^bb6
// CHECK-DAG:       [[VAR_31_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_32_:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       [[VAR_33_:%.+]] = llvm.getelementptr [[VAR_23_]]{{.}}[[VAR_31_]]{{.}} : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
// CHECK-DAG:       [[LOAD_VAR_33_MEM_:%.+]] = llvm.load [[VAR_33_]] : !llvm.ptr<i64>
// CHECK:           [[VAR_35_:%.+]] = llvm.icmp "ne" [[VAR_32_]], [[LOAD_VAR_33_MEM_]] : i64
// CHECK:           llvm.cond_br [[VAR_35_]], ^bb9, ^bb10
// CHECK:         ^bb9:  // pred: ^bb8
// CHECK-DAG:       [[VAR_36_:%.+]] = llvm.mlir.addressof @"Wrong size for the dimension 1 of the input 0: expect 4, but got {{\%}}lld\0A" : !llvm.ptr<array<70 x i8>>
// CHECK-DAG:       [[VAR_37_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_38_:%.+]] = llvm.getelementptr [[VAR_36_]]{{.}}[[VAR_37_]], [[VAR_37_]]{{.}} : (!llvm.ptr<array<70 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf([[VAR_38_]], [[LOAD_VAR_33_MEM_]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK:           [[VAR_39_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return [[VAR_39_]] : !llvm.ptr<i8>

// CHECK:         ^bb10:  // pred: ^bb8
// CHECK-DAG:       [[VAR_40_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_41_:%.+]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG:       [[VAR_42_:%.+]] = llvm.getelementptr [[VAR_23_]]{{.}}[[VAR_40_]]{{.}} : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
// CHECK-DAG:       [[LOAD_VAR_42_MEM_:%.+]] = llvm.load [[VAR_42_]] : !llvm.ptr<i64>
// CHECK:           [[VAR_44_:%.+]] = llvm.icmp "ne" [[VAR_41_]], [[LOAD_VAR_42_MEM_]] : i64
// CHECK:           llvm.cond_br [[VAR_44_]], ^bb11, ^bb12
// CHECK:         ^bb11:  // pred: ^bb10
// CHECK-DAG:       [[VAR_45_:%.+]] = llvm.mlir.addressof @"Wrong size for the dimension 2 of the input 0: expect 5, but got {{\%}}lld\0A" : !llvm.ptr<array<70 x i8>>
// CHECK-DAG:       [[VAR_46_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_47_:%.+]] = llvm.getelementptr [[VAR_45_]]{{.}}[[VAR_46_]], [[VAR_46_]]{{.}} : (!llvm.ptr<array<70 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf([[VAR_47_]], [[LOAD_VAR_42_MEM_]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK:           [[VAR_48_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return [[VAR_48_]] : !llvm.ptr<i8>

// CHECK:         ^bb12:  // pred: ^bb10
// CHECK:           [[VAR_49_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK:           [[VAR_50_:%.+]] = llvm.getelementptr [[VAR_7_2_]]{{.}}[[VAR_49_]]{{.}} : (!llvm.ptr<ptr<i8>>, i64) -> !llvm.ptr<ptr<i8>>
// CHECK-DAG:       [[LOAD_VAR_50_MEM_:%.+]] = llvm.load [[VAR_50_]] : !llvm.ptr<ptr<i8>>
// CHECK-DAG:       [[VAR_52_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_53_:%.+]] = llvm.call @omTensorGetDataType([[LOAD_VAR_50_MEM_]]) : (!llvm.ptr<i8>) -> i64
// CHECK:           [[VAR_54_:%.+]] = llvm.icmp "ne" [[VAR_52_]], [[VAR_53_]] : i64
// CHECK:           llvm.cond_br [[VAR_54_]], ^bb13, ^bb14
// CHECK:         ^bb13:  // pred: ^bb12
// CHECK-DAG:       [[VAR_55_:%.+]] = llvm.mlir.addressof @"Wrong data type for the input 1: expect f32\0A" : !llvm.ptr<array<44 x i8>>
// CHECK-DAG:       [[VAR_56_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_57_:%.+]] = llvm.getelementptr [[VAR_55_]]{{.}}[[VAR_56_]], [[VAR_56_]]{{.}} : (!llvm.ptr<array<44 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf([[VAR_57_]]) : (!llvm.ptr<i8>) -> ()
// CHECK:           [[VAR_58_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return [[VAR_58_]] : !llvm.ptr<i8>

// CHECK:         ^bb14:  // pred: ^bb12
// CHECK-DAG:       [[VAR_59_:%.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG:       [[VAR_60_:%.+]] = llvm.call @omTensorGetRank([[LOAD_VAR_50_MEM_]]) : (!llvm.ptr<i8>) -> i64
// CHECK:           [[VAR_61_:%.+]] = llvm.icmp "ne" [[VAR_59_]], [[VAR_60_]] : i64
// CHECK:           llvm.cond_br [[VAR_61_]], ^bb15, ^bb16
// CHECK:         ^bb15:  // pred: ^bb14
// CHECK-DAG:       [[VAR_62_:%.+]] = llvm.mlir.addressof @"Wrong rank for the input 1: expect 3, but got {{\%}}lld\0A" : !llvm.ptr<array<51 x i8>>
// CHECK-DAG:       [[VAR_63_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_64_:%.+]] = llvm.getelementptr [[VAR_62_]]{{.}}[[VAR_63_]], [[VAR_63_]]{{.}} : (!llvm.ptr<array<51 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf([[VAR_64_]], [[VAR_60_]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK:           [[VAR_65_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return [[VAR_65_]] : !llvm.ptr<i8>

// CHECK:         ^bb16:  // pred: ^bb14
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.call @omTensorGetShape([[LOAD_VAR_50_MEM_]]) : (!llvm.ptr<i8>) -> !llvm.ptr<i64>
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG:       [[LOAD_VAR_66_MEM_:%.+]] = llvm.load [[VAR_66_]] : !llvm.ptr<i64>
// CHECK:           [[VAR_69_:%.+]] = llvm.icmp "ne" [[VAR_67_]], [[LOAD_VAR_66_MEM_]] : i64
// CHECK:           llvm.cond_br [[VAR_69_]], ^bb17, ^bb18
// CHECK:         ^bb17:  // pred: ^bb16
// CHECK-DAG:       [[VAR_70_:%.+]] = llvm.mlir.addressof @"Wrong size for the dimension 0 of the input 1: expect 3, but got {{\%}}lld\0A" : !llvm.ptr<array<70 x i8>>
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_72_:%.+]] = llvm.getelementptr [[VAR_70_]]{{.}}[[VAR_71_]], [[VAR_71_]]{{.}} : (!llvm.ptr<array<70 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf([[VAR_72_]], [[LOAD_VAR_66_MEM_]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK:           [[VAR_73_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return [[VAR_73_]] : !llvm.ptr<i8>

// CHECK:         ^bb18:  // pred: ^bb16
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       [[VAR_76_:%.+]] = llvm.getelementptr [[VAR_66_]]{{.}}[[VAR_74_]]{{.}} : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
// CHECK-DAG:       [[LOAD_VAR_76_MEM_:%.+]] = llvm.load [[VAR_76_]] : !llvm.ptr<i64>
// CHECK:           [[VAR_78_:%.+]] = llvm.icmp "ne" [[VAR_75_]], [[LOAD_VAR_76_MEM_]] : i64
// CHECK:           llvm.cond_br [[VAR_78_]], ^bb19, ^bb20
// CHECK:         ^bb19:  // pred: ^bb18
// CHECK-DAG:       [[VAR_79_:%.+]] = llvm.mlir.addressof @"Wrong size for the dimension 1 of the input 1: expect 4, but got {{\%}}lld\0A" : !llvm.ptr<array<70 x i8>>
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_81_:%.+]] = llvm.getelementptr [[VAR_79_]]{{.}}[[VAR_80_]], [[VAR_80_]]{{.}} : (!llvm.ptr<array<70 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf([[VAR_81_]], [[LOAD_VAR_76_MEM_]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK:           [[VAR_82_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return [[VAR_82_]] : !llvm.ptr<i8>

// CHECK:         ^bb20:  // pred: ^bb18
// CHECK-DAG:       [[VAR_83_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_84_:%.+]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG:       [[VAR_85_:%.+]] = llvm.getelementptr [[VAR_66_]]{{.}}[[VAR_83_]]{{.}} : (!llvm.ptr<i64>, i64) -> !llvm.ptr<i64>
// CHECK-DAG:       [[LOAD_VAR_85_MEM_:%.+]] = llvm.load [[VAR_85_]] : !llvm.ptr<i64>
// CHECK:           [[VAR_87_:%.+]] = llvm.icmp "ne" [[VAR_84_]], [[LOAD_VAR_85_MEM_]] : i64
// CHECK:           llvm.cond_br [[VAR_87_]], ^bb21, ^bb22
// CHECK:         ^bb21:  // pred: ^bb20
// CHECK-DAG:       [[VAR_88_:%.+]] = llvm.mlir.addressof @"Wrong size for the dimension 2 of the input 1: expect 5, but got {{\%}}lld\0A" : !llvm.ptr<array<70 x i8>>
// CHECK-DAG:       [[VAR_89_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:           [[VAR_90_:%.+]] = llvm.getelementptr [[VAR_88_]]{{.}}[[VAR_89_]], [[VAR_89_]]{{.}} : (!llvm.ptr<array<70 x i8>>, i64, i64) -> !llvm.ptr<i8>
// CHECK:           llvm.call @printf([[VAR_90_]], [[LOAD_VAR_85_MEM_]]) : (!llvm.ptr<i8>, i64) -> ()
// CHECK:           [[VAR_91_:%.+]] = llvm.mlir.null : !llvm.ptr<i8>
// CHECK:           llvm.return [[VAR_91_]] : !llvm.ptr<i8>

// CHECK:         ^bb22:  // pred: ^bb20
// CHECK-DAG:       [[VAR_92_:%.+]] = llvm.call @omTensorListGetOmtArray([[arg0_]]) : (!llvm.ptr<i8>) -> !llvm.ptr<ptr<i8>>
}
