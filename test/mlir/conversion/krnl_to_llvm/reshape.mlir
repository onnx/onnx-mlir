// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl --convert-krnl-to-affine --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// -----

func.func @test_reshape(%arg0 : tensor<?x10xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<4xi64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL: llvm.func @test_reshape
// CHECK:    [[OLD_MEMREF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:    [[INSERT_1_:%.+]] = llvm.insertvalue {{.*}}, [[OLD_MEMREF]][0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:    [[INSERT_2_:%.+]] = llvm.insertvalue {{.*}}, [[INSERT_1_]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:    [[INSERT_3_:%.+]] = llvm.insertvalue {{.*}}, [[INSERT_2_]][2] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:    [[INSERT_4_:%.+]] = llvm.insertvalue {{.*}}, [[INSERT_3_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:    [[INSERT_5_:%.+]] = llvm.insertvalue {{.*}}, [[INSERT_4_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK:    [[INSERT_6_:%.+]] = llvm.insertvalue {{.*}}, [[INSERT_5_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
// CHECK-DAG:[[INSERT_7_:%.+]] = llvm.insertvalue {{.*}}, [[INSERT_6_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

// COM: Check that there is no copy but only a new MemRef with a new view, i.e. new sizes and strides.
// CHECK-DAG:  [[NEW_MEMREF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:      [[INSERT_8_:%.+]] = llvm.insertvalue {{.*}}, [[NEW_MEMREF]][0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:  [[INSERT_9_:%.+]] = llvm.insertvalue {{.*}}, [[INSERT_8_]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:  [[C0:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK:      [[INSERT_10_:%.+]] = llvm.insertvalue [[C0]], [[INSERT_9_]][2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:      [[INSERT_11_:%.+]] = llvm.insertvalue {{.*}}, [[INSERT_10_]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:      [[INSERT_12_:%.+]] = llvm.insertvalue {{.*}}, [[INSERT_11_]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:      [[INSERT_13_:%.+]] = llvm.insertvalue {{.*}}, [[INSERT_12_]][3, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:      [[INSERT_14_:%.+]] = llvm.insertvalue {{.*}}, [[INSERT_13_]][4, 1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:      [[INSERT_15_:%.+]] = llvm.insertvalue {{.*}}, [[INSERT_14_]][3, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:      [[INSERT_16_:%.+]] = llvm.insertvalue {{.*}}, [[INSERT_15_]][4, 2] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:  [[INSERT_17_:%.+]] = llvm.insertvalue {{.*}}, [[INSERT_16_]][3, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK-DAG:  [[C1:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK:      [[INSERT_18_:%.+]] = llvm.insertvalue [[C1]], [[INSERT_17_]][4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
// CHECK:      llvm.return [[INSERT_18_]] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
}
