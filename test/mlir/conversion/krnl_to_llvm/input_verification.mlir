// RUN: onnx-mlir-opt --convert-krnl-to-llvm="verify-input-tensors=true" --canonicalize %s -split-input-file | FileCheck %s

// -----

// COM: Check verification code at the beginning of the entry point function.

module {
  func.func @main_graph(%arg0: memref<3x4x5xf32>, %arg1: memref<?x4x5xf32>) -> memref<3x4x5xf32> {
    return %arg0 : memref<3x4x5xf32>
  }
  "krnl.entry_point"() {func = @main_graph, numInputs = 2 : i32, numOutputs = 1 : i32, signature = "[    { \22type\22 : \22f32\22 , \22dims\22 : [3 , 4 , 5] , \22name\22 : \22input0\22 }\0A ,    { \22type\22 : \22f32\22 , \22dims\22 : [-1 , 4 , 5] , \22name\22 : \22input1\22 }\0A\0A]\00@[   { \22type\22 : \22f32\22 , \22dims\22 : [3 , 4 , 5] , \22name\22 : \22output0\22 }\0A\0A]\00"} : () -> ()

// CHECK:         llvm.func @run_main_graph([[arg0_:.*]]: !llvm.ptr) -> !llvm.ptr {
// CHECK-DAG:       [[VAR_0_2_:%.+]] = llvm.mlir.addressof @"om_Wrong size for the dimension 2 of the input 1: expect 5, but got {{\%}}lld\0A" : !llvm.ptr
// CHECK-DAG:       [[VAR_1_2_:%.+]] = llvm.mlir.addressof @"om_Wrong size for the dimension 1 of the input 1: expect 4, but got {{\%}}lld\0A" : !llvm.ptr
// CHECK-DAG:       [[VAR_2_2_:%.+]] = llvm.mlir.addressof @"om_Wrong size for the dimension 0 of the input 1: expect a non-negative value\0A" : !llvm.ptr
// CHECK-DAG:       [[VAR_3_2_:%.+]] = llvm.mlir.constant(0 : i64) : i64
// CHECK-DAG:       [[VAR_4_2_:%.+]] = llvm.mlir.addressof @"om_Wrong rank for the input 1: expect 3, but got {{\%}}lld\0A" : !llvm.ptr
// CHECK-DAG:       [[VAR_5_2_:%.+]] = llvm.mlir.addressof @"om_Wrong data type for the input 1: expect f32\0A" : !llvm.ptr
// CHECK-DAG:       [[VAR_6_2_:%.+]] = llvm.mlir.addressof @"om_Wrong size for the dimension 2 of the input 0: expect 5, but got {{\%}}lld\0A" : !llvm.ptr
// CHECK-DAG:       [[VAR_7_2_:%.+]] = llvm.mlir.constant(5 : i64) : i64
// CHECK-DAG:       [[VAR_8_2_:%.+]] = llvm.mlir.addressof @"om_Wrong size for the dimension 1 of the input 0: expect 4, but got {{\%}}lld\0A" : !llvm.ptr
// CHECK-DAG:       [[VAR_9_2_:%.+]] = llvm.mlir.constant(4 : i64) : i64
// CHECK-DAG:       [[LOAD_arg2_MEM_1_:%.+]] = llvm.mlir.addressof @"om_Wrong size for the dimension 0 of the input 0: expect 3, but got {{\%}}lld\0A" : !llvm.ptr
// CHECK-DAG:       [[VAR_11_1_:%.+]] = llvm.mlir.addressof @"om_Wrong rank for the input 0: expect 3, but got {{\%}}lld\0A" : !llvm.ptr
// CHECK-DAG:       [[VAR_12_1_:%.+]] = llvm.mlir.constant(3 : i64) : i64
// CHECK-DAG:       [[VAR_13_1_:%.+]] = llvm.mlir.addressof @"om_Wrong data type for the input 0: expect f32\0A" : !llvm.ptr
// CHECK-DAG:       [[VAR_14_1_:%.+]] = llvm.mlir.constant(1 : i64) : i64
// CHECK-DAG:       [[VAR_15_1_:%.+]] = llvm.mlir.constant(22 : i32) : i32
// CHECK-DAG:       [[VAR_16_1_:%.+]] = llvm.mlir.addressof @"om_Wrong number of input tensors: expect 2, but got {{\%}}lld\0A" : !llvm.ptr
// CHECK-DAG:       [[VAR_17_1_:%.+]] = llvm.mlir.constant(2 : i64) : i64
// CHECK-DAG:       [[VAR_18_1_:%.+]] = llvm.call @omTensorListGetSize([[arg0_]]) : (!llvm.ptr) -> i64
// CHECK:           [[VAR_19_1_:%.+]] = llvm.icmp "ne" [[VAR_17_1_]], [[VAR_18_1_]] : i64
// CHECK:           llvm.cond_br [[VAR_19_1_]], ^bb1([[VAR_16_1_]], [[VAR_18_1_]] : !llvm.ptr, i64), ^bb2
// CHECK:         ^bb1([[VAR_20_:%.+]]: !llvm.ptr, [[VAR_21_:%.+]]: i64):  // 8 preds: ^bb0, ^bb4, ^bb5, ^bb6, ^bb7, ^bb9, ^bb11, ^bb12
// CHECK:           llvm.call @printf([[VAR_20_]], [[VAR_21_]]) : (!llvm.ptr, i64) -> ()
// CHECK:           [[VAR_22_:%.+]] = llvm.call @__errno_location() : () -> !llvm.ptr
// CHECK:           llvm.store [[VAR_15_1_]], [[VAR_22_]] : i32, !llvm.ptr
// CHECK:           [[VAR_23_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return [[VAR_23_]] : !llvm.ptr
// CHECK:         ^bb2:  // pred: ^bb0
// CHECK:           [[VAR_24_:%.+]] = llvm.call @omTensorListGetOmtArray([[arg0_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           [[LOAD_VAR_24_MEM_:%.+]] = llvm.load [[VAR_24_]] : !llvm.ptr -> !llvm.ptr
// CHECK:           [[VAR_26_:%.+]] = llvm.call @omTensorGetDataType([[LOAD_VAR_24_MEM_]]) : (!llvm.ptr) -> i64
// CHECK:           [[VAR_27_:%.+]] = llvm.icmp "ne" [[VAR_14_1_]], [[VAR_26_]] : i64
// CHECK:           llvm.cond_br [[VAR_27_]], ^bb3([[VAR_13_1_]] : !llvm.ptr), ^bb4
// CHECK:         ^bb3([[VAR_28_:%.+]]: !llvm.ptr):  // 3 preds: ^bb2, ^bb8, ^bb10
// CHECK:           llvm.call @printf([[VAR_28_]]) : (!llvm.ptr) -> ()
// CHECK:           [[VAR_29_:%.+]] = llvm.call @__errno_location() : () -> !llvm.ptr
// CHECK:           llvm.store [[VAR_15_1_]], [[VAR_29_]] : i32, !llvm.ptr
// CHECK:           [[VAR_30_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           llvm.return [[VAR_30_]] : !llvm.ptr
// CHECK:         ^bb4:  // pred: ^bb2
// CHECK:           [[VAR_31_:%.+]] = llvm.call @omTensorGetRank([[LOAD_VAR_24_MEM_]]) : (!llvm.ptr) -> i64
// CHECK:           [[VAR_32_:%.+]] = llvm.icmp "ne" [[VAR_12_1_]], [[VAR_31_]] : i64
// CHECK:           llvm.cond_br [[VAR_32_]], ^bb1([[VAR_11_1_]], [[VAR_31_]] : !llvm.ptr, i64), ^bb5
// CHECK:         ^bb5:  // pred: ^bb4
// CHECK:           [[VAR_33_:%.+]] = llvm.call @omTensorGetShape([[LOAD_VAR_24_MEM_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           [[LOAD_VAR_33_MEM_:%.+]] = llvm.load [[VAR_33_]] : !llvm.ptr -> i64
// CHECK:           [[VAR_35_:%.+]] = llvm.icmp "ne" [[VAR_12_1_]], [[LOAD_VAR_33_MEM_]] : i64
// CHECK:           llvm.cond_br [[VAR_35_]], ^bb1([[LOAD_arg2_MEM_1_]], [[LOAD_VAR_33_MEM_]] : !llvm.ptr, i64), ^bb6
// CHECK:         ^bb6:  // pred: ^bb5
// CHECK:           [[VAR_36_:%.+]] = llvm.getelementptr [[VAR_33_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           [[LOAD_VAR_36_MEM_:%.+]] = llvm.load [[VAR_36_]] : !llvm.ptr -> i64
// CHECK:           [[VAR_38_:%.+]] = llvm.icmp "ne" [[VAR_9_2_]], [[LOAD_VAR_36_MEM_]] : i64
// CHECK:           llvm.cond_br [[VAR_38_]], ^bb1([[VAR_8_2_]], [[LOAD_VAR_36_MEM_]] : !llvm.ptr, i64), ^bb7
// CHECK:         ^bb7:  // pred: ^bb6
// CHECK:           [[VAR_39_:%.+]] = llvm.getelementptr [[VAR_33_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           [[LOAD_VAR_39_MEM_:%.+]] = llvm.load [[VAR_39_]] : !llvm.ptr -> i64
// CHECK:           [[VAR_41_:%.+]] = llvm.icmp "ne" [[VAR_7_2_]], [[LOAD_VAR_39_MEM_]] : i64
// CHECK:           llvm.cond_br [[VAR_41_]], ^bb1([[VAR_6_2_]], [[LOAD_VAR_39_MEM_]] : !llvm.ptr, i64), ^bb8
// CHECK:         ^bb8:  // pred: ^bb7
// CHECK:           [[VAR_42_:%.+]] = llvm.getelementptr [[VAR_24_]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK:           [[LOAD_VAR_42_MEM_:%.+]] = llvm.load [[VAR_42_]] : !llvm.ptr -> !llvm.ptr
// CHECK:           [[VAR_44_:%.+]] = llvm.call @omTensorGetDataType([[LOAD_VAR_42_MEM_]]) : (!llvm.ptr) -> i64
// CHECK:           [[VAR_45_:%.+]] = llvm.icmp "ne" [[VAR_14_1_]], [[VAR_44_]] : i64
// CHECK:           llvm.cond_br [[VAR_45_]], ^bb3([[VAR_5_2_]] : !llvm.ptr), ^bb9
// CHECK:         ^bb9:  // pred: ^bb8
// CHECK:           [[VAR_46_:%.+]] = llvm.call @omTensorGetRank([[LOAD_VAR_42_MEM_]]) : (!llvm.ptr) -> i64
// CHECK:           [[VAR_47_:%.+]] = llvm.icmp "ne" [[VAR_12_1_]], [[VAR_46_]] : i64
// CHECK:           llvm.cond_br [[VAR_47_]], ^bb1([[VAR_4_2_]], [[VAR_4_2_]]6 : !llvm.ptr, i64), ^bb10
// CHECK:         ^bb10:  // pred: ^bb9
// CHECK:           [[VAR_48_:%.+]] = llvm.call @omTensorGetShape([[LOAD_VAR_42_MEM_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           [[LOAD_VAR_48_MEM_:%.+]] = llvm.load [[VAR_48_]] : !llvm.ptr -> i64
// CHECK:           [[VAR_50_:%.+]] = llvm.icmp "slt" [[LOAD_VAR_48_MEM_]], [[VAR_3_2_]] : i64
// CHECK:           llvm.cond_br [[VAR_50_]], ^bb3([[VAR_2_2_]] : !llvm.ptr), ^bb11
// CHECK:         ^bb11:  // pred: ^bb10
// CHECK:           [[VAR_51_:%.+]] = llvm.getelementptr [[VAR_48_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           [[LOAD_VAR_51_MEM_:%.+]] = llvm.load [[VAR_51_]] : !llvm.ptr -> i64
// CHECK:           [[VAR_53_:%.+]] = llvm.icmp "ne" [[VAR_9_2_]], [[LOAD_VAR_51_MEM_]] : i64
// CHECK:           llvm.cond_br [[VAR_53_]], ^bb1([[VAR_1_2_]], [[LOAD_VAR_51_MEM_]] : !llvm.ptr, i64), ^bb12
// CHECK:         ^bb12:  // pred: ^bb11
// CHECK:           [[VAR_54_:%.+]] = llvm.getelementptr [[VAR_48_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           [[LOAD_VAR_54_MEM_:%.+]] = llvm.load [[VAR_54_]] : !llvm.ptr -> i64
// CHECK:           [[VAR_56_:%.+]] = llvm.icmp "ne" [[VAR_7_2_]], [[LOAD_VAR_54_MEM_]] : i64
// CHECK:           llvm.cond_br [[VAR_56_]], ^bb1([[VAR_0_2_]], [[LOAD_VAR_54_MEM_]] : !llvm.ptr, i64), ^bb13
// CHECK:         ^bb13:  // pred: ^bb12
// CHECK-DAG:       [[VAR_57_:%.+]] = llvm.call @omTensorListGetOmtArray([[arg0_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-DAG:       [[VAR_58_:%.+]] = llvm.alloca [[VAR_14_1_]] x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_VAR_57_MEM_:%.+]] = llvm.load [[VAR_57_]] : !llvm.ptr -> !llvm.ptr
// CHECK-DAG:       [[VAR_60_:%.+]] = llvm.alloca [[VAR_14_1_]] x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_61_:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_62_:%.+]] = llvm.call @omTensorGetDataPtr([[LOAD_VAR_57_MEM_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           [[VAR_63_:%.+]] = llvm.insertvalue [[VAR_62_]], [[VAR_61_]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_64_:%.+]] = llvm.insertvalue [[VAR_62_]], [[VAR_63_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_65_:%.+]] = llvm.insertvalue [[VAR_3_2_]], [[VAR_64_]][2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_66_:%.+]] = llvm.call @omTensorGetShape([[LOAD_VAR_57_MEM_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-DAG:       [[VAR_67_:%.+]] = llvm.call @omTensorGetStrides([[LOAD_VAR_57_MEM_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           [[LOAD_VAR_66_MEM_:%.+]] = llvm.load [[VAR_66_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_69_:%.+]] = llvm.insertvalue [[LOAD_VAR_66_MEM_]], [[VAR_65_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[LOAD_VAR_67_MEM_:%.+]] = llvm.load [[VAR_67_]] : !llvm.ptr -> i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_71_:%.+]] = llvm.insertvalue [[LOAD_VAR_67_MEM_]], [[VAR_69_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_72_:%.+]] = llvm.getelementptr [[VAR_66_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           [[LOAD_VAR_72_MEM_:%.+]] = llvm.load [[VAR_72_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_74_:%.+]] = llvm.insertvalue [[LOAD_VAR_72_MEM_]], [[VAR_71_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_75_:%.+]] = llvm.getelementptr [[VAR_67_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           [[LOAD_VAR_75_MEM_:%.+]] = llvm.load [[VAR_75_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_77_:%.+]] = llvm.insertvalue [[LOAD_VAR_75_MEM_]], [[VAR_74_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_78_:%.+]] = llvm.getelementptr [[VAR_66_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           [[LOAD_VAR_78_MEM_:%.+]] = llvm.load [[VAR_78_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_80_:%.+]] = llvm.insertvalue [[LOAD_VAR_78_MEM_]], [[VAR_77_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_81_:%.+]] = llvm.getelementptr [[VAR_67_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           [[LOAD_VAR_81_MEM_:%.+]] = llvm.load [[VAR_81_]] : !llvm.ptr -> i64
// CHECK:           [[VAR_83_:%.+]] = llvm.insertvalue [[LOAD_VAR_81_MEM_]], [[VAR_80_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.store [[VAR_83_]], [[VAR_60_]] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
// CHECK:           [[VAR_84_:%.+]] = llvm.getelementptr [[VAR_57_]][1] : (!llvm.ptr) -> !llvm.ptr, !llvm.ptr
// CHECK-DAG:       [[LOAD_VAR_84_MEM_:%.+]] = llvm.load [[VAR_84_]] : !llvm.ptr -> !llvm.ptr
// CHECK-DAG:       [[VAR_86_:%.+]] = llvm.alloca [[VAR_14_1_]] x !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_87_:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_88_:%.+]] = llvm.call @omTensorGetDataPtr([[LOAD_VAR_84_MEM_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           [[VAR_89_:%.+]] = llvm.insertvalue [[VAR_88_]], [[VAR_87_]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_90_:%.+]] = llvm.insertvalue [[VAR_88_]], [[VAR_89_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_91_:%.+]] = llvm.insertvalue [[VAR_3_2_]], [[VAR_90_]][2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_92_:%.+]] = llvm.call @omTensorGetShape([[LOAD_VAR_84_MEM_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-DAG:       [[VAR_93_:%.+]] = llvm.call @omTensorGetStrides([[LOAD_VAR_84_MEM_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK:           [[LOAD_VAR_92_MEM_:%.+]] = llvm.load [[VAR_92_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_95_:%.+]] = llvm.insertvalue [[LOAD_VAR_92_MEM_]], [[VAR_91_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[LOAD_VAR_93_MEM_:%.+]] = llvm.load [[VAR_93_]] : !llvm.ptr -> i64
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_97_:%.+]] = llvm.insertvalue [[LOAD_VAR_93_MEM_]], [[VAR_95_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_98_:%.+]] = llvm.getelementptr [[VAR_92_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           [[LOAD_VAR_98_MEM_:%.+]] = llvm.load [[VAR_98_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_100_:%.+]] = llvm.insertvalue [[LOAD_VAR_98_MEM_]], [[VAR_97_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_101_:%.+]] = llvm.getelementptr [[VAR_93_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           [[LOAD_VAR_101_MEM_:%.+]] = llvm.load [[VAR_101_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_103_:%.+]] = llvm.insertvalue [[LOAD_VAR_101_MEM_]], [[VAR_100_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_104_:%.+]] = llvm.getelementptr [[VAR_92_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           [[LOAD_VAR_104_MEM_:%.+]] = llvm.load [[VAR_104_]] : !llvm.ptr -> i64
// CHECK-DAG:       [[VAR_106_:%.+]] = llvm.insertvalue [[LOAD_VAR_104_MEM_]], [[VAR_103_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_107_:%.+]] = llvm.getelementptr [[VAR_93_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           [[LOAD_VAR_107_MEM_:%.+]] = llvm.load [[VAR_107_]] : !llvm.ptr -> i64
// CHECK:           [[VAR_109_:%.+]] = llvm.insertvalue [[LOAD_VAR_107_MEM_]], [[VAR_106_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.store [[VAR_109_]], [[VAR_86_]] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>, !llvm.ptr
// CHECK:           llvm.call @_mlir_ciface_main_graph([[VAR_58_]], [[VAR_60_]], [[VAR_86_]]) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> ()
// CHECK-DAG:       [[LOAD_VAR_58_MEM_:%.+]] = llvm.load [[VAR_58_]] : !llvm.ptr -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_111_:%.+]] = llvm.alloca [[VAR_14_1_]] x !llvm.ptr : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_112_:%.+]] = llvm.call @omTensorCreateUntyped([[VAR_12_1_]]) : (i64) -> !llvm.ptr
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_113_:%.+]] = llvm.extractvalue [[LOAD_VAR_58_MEM_]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_114_:%.+]] = llvm.extractvalue [[LOAD_VAR_58_MEM_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.call @omTensorSetDataPtr([[VAR_112_]], [[VAR_3_2_]], [[VAR_113_]], [[VAR_114_]]) : (!llvm.ptr, i64, !llvm.ptr, !llvm.ptr) -> ()
// CHECK:           llvm.call @omTensorSetDataType([[VAR_112_]], [[VAR_14_1_]]) : (!llvm.ptr, i64) -> ()
// CHECK-DAG:       [[VAR_115_:%.+]] = llvm.call @omTensorGetShape([[VAR_112_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-DAG:       [[VAR_116_:%.+]] = llvm.call @omTensorGetStrides([[VAR_112_]]) : (!llvm.ptr) -> !llvm.ptr
// CHECK-DAG:       [[VAR_117_:%.+]] = llvm.extractvalue [[LOAD_VAR_58_MEM_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.store [[VAR_117_]], [[VAR_115_]] : i64, !llvm.ptr
// CHECK:           [[VAR_118_:%.+]] = llvm.extractvalue [[LOAD_VAR_58_MEM_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.store [[VAR_118_]], [[VAR_116_]] : i64, !llvm.ptr
// CHECK-DAG:       [[VAR_119_:%.+]] = llvm.extractvalue [[LOAD_VAR_58_MEM_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_120_:%.+]] = llvm.getelementptr [[VAR_115_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           llvm.store [[VAR_119_]], [[VAR_120_]] : i64, !llvm.ptr
// CHECK-DAG:       [[VAR_121_:%.+]] = llvm.extractvalue [[LOAD_VAR_58_MEM_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_122_:%.+]] = llvm.getelementptr [[VAR_116_]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           llvm.store [[VAR_121_]], [[VAR_122_]] : i64, !llvm.ptr
// CHECK-DAG:       [[VAR_123_:%.+]] = llvm.extractvalue [[LOAD_VAR_58_MEM_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_124_:%.+]] = llvm.getelementptr [[VAR_115_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           llvm.store [[VAR_123_]], [[VAR_124_]] : i64, !llvm.ptr
// CHECK-DAG:       [[VAR_125_:%.+]] = llvm.extractvalue [[LOAD_VAR_58_MEM_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_126_:%.+]] = llvm.getelementptr [[VAR_116_]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK:           llvm.store [[VAR_125_]], [[VAR_126_]] : i64, !llvm.ptr
// CHECK:           llvm.store [[VAR_112_]], [[VAR_111_]] : !llvm.ptr, !llvm.ptr
// CHECK:           [[VAR_127_:%.+]] = llvm.call @omTensorListCreate([[VAR_111_]], [[VAR_14_1_]]) : (!llvm.ptr, i64) -> !llvm.ptr
// CHECK:           llvm.return [[VAR_127_]] : !llvm.ptr
// CHECK:         }
}
