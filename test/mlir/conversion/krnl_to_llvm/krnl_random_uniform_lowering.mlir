// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine --convert-krnl-to-llvm --canonicalize %s -split-input-file | FileCheck %s


func.func @test_random_uniform_lowering() -> memref<3x4x5xf32> {
  %0 = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
  %c60 = arith.constant 60 : index
  %cst = arith.constant 0.000000e+00 : f32
  %cst_0 = arith.constant 1.000000e+00 : f32
  %cst_1 = arith.constant 2.000000e+00 : f32
  "krnl.random_uniform"(%0, %c60, %cst, %cst_0, %cst_1) : (memref<3x4x5xf32>, index, f32, f32, f32) -> ()
  return %0 : memref<3x4x5xf32>

// mlir2FileCheck.py
// CHECK:         llvm.func @get_uniform_random_value_f32(!llvm.ptr, i64, f32, f32, f32)
// CHECK:         llvm.func @malloc(i64) -> !llvm.ptr
// CHECK:         llvm.func @test_random_uniform_lowering() -> !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)> attributes {llvm.emit_c_interface} {
// CHECK-DAG:       [[VAR_0_:%.+]] = llvm.mlir.constant(2.000000e+00 : f32) : f32
// CHECK-DAG:       [[VAR_1_:%.+]] = llvm.mlir.constant(1.000000e+00 : f32) : f32
// CHECK-DAG:       [[VAR_2_:%.+]] = llvm.mlir.constant(0.000000e+00 : f32) : f32
// CHECK-DAG:       [[VAR_3_:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-DAG:       [[VAR_4_:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK-DAG:       [[VAR_5_:%.+]] = llvm.mlir.constant(16 : index) : i64
// CHECK-DAG:       [[VAR_6_:%.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-DAG:       [[VAR_7_:%.+]] = llvm.mlir.constant(4 : index) : i64
// CHECK-DAG:       [[VAR_8_:%.+]] = llvm.mlir.constant(5 : index) : i64
// CHECK-DAG:       [[VAR_9_:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-DAG:       [[VAR_10_:%.+]] = llvm.mlir.constant(20 : index) : i64
// CHECK-DAG:       [[VAR_11_:%.+]] = llvm.mlir.constant(60 : index) : i64
// CHECK-DAG:       [[VAR_12_:%.+]] = llvm.mlir.zero : !llvm.ptr
// CHECK:           [[VAR_13_:%.+]] = llvm.getelementptr [[VAR_12_]][60] : (!llvm.ptr) -> !llvm.ptr, f32
// CHECK:           [[VAR_14_:%.+]] = llvm.ptrtoint [[VAR_13_]] : !llvm.ptr to i64
// CHECK:           [[VAR_15_:%.+]] = llvm.add [[VAR_14_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_16_:%.+]] = llvm.call @malloc([[VAR_15_]]) : (i64) -> !llvm.ptr
// CHECK-DAG:       [[VAR_17_:%.+]] = llvm.ptrtoint [[VAR_16_]] : !llvm.ptr to i64
// CHECK-DAG:       [[VAR_18_:%.+]] = llvm.sub [[VAR_5_]], [[VAR_9_]] : i64
// CHECK:           [[VAR_19_:%.+]] = llvm.add [[VAR_17_]], [[VAR_18_]] : i64
// CHECK:           [[VAR_20_:%.+]] = llvm.urem [[VAR_19_]], [[VAR_5_]] : i64
// CHECK:           [[VAR_21_:%.+]] = llvm.sub [[VAR_19_]], [[VAR_20_]] : i64
// CHECK-DAG:       [[VAR_22_:%.+]] = llvm.inttoptr [[VAR_21_]] : i64 to !llvm.ptr
// CHECK-DAG:       [[VAR_23_:%.+]] = llvm.insertvalue [[VAR_16_]], [[VAR_4_]][0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_24_:%.+]] = llvm.insertvalue [[VAR_22_]], [[VAR_23_]][1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_25_:%.+]] = llvm.insertvalue [[VAR_3_]], [[VAR_24_]][2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_26_:%.+]] = llvm.insertvalue [[VAR_6_]], [[VAR_25_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_27_:%.+]] = llvm.insertvalue [[VAR_7_]], [[VAR_26_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_28_:%.+]] = llvm.insertvalue [[VAR_8_]], [[VAR_27_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_29_:%.+]] = llvm.insertvalue [[VAR_10_]], [[VAR_28_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_30_:%.+]] = llvm.insertvalue [[VAR_8_]], [[VAR_29_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           [[VAR_31_:%.+]] = llvm.insertvalue [[VAR_9_]], [[VAR_30_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:           llvm.call @get_uniform_random_value_f32([[VAR_22_]], [[VAR_11_]], [[VAR_2_]], [[VAR_1_]], [[VAR_0_]]) : (!llvm.ptr, i64, f32, f32, f32) -> ()
// CHECK:           llvm.return [[VAR_31_]] : !llvm.struct<(ptr, ptr, i64, array<3 x i64>, array<3 x i64>)>
// CHECK:         }

}