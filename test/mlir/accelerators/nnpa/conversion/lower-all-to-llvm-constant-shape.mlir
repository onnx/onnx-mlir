// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-krnl-to-llvm -cse %s -split-input-file | FileCheck %s

// -----

// COM: Check the lowering of an zlow operation when its shape includes constant dims.
// COM: In this case, the constant values will be passed directly to
// COM: 'zdnn_init_pre_transformed_desc' that initializes a zTensor descriptor.
// COM: Using zlow.softmax as an example.
func.func @test_zlow_softmax_constant_shape() -> () {
  // %0 = "onnx.Softmax"(%arg0) : (memref<5x10xf32>) -> memref<5x10xf32>
  // "func.return"(%0) : (memref<5x10xf32>) -> ()
  %shape = "krnl.global"() {name = "constant_fold_std_alloc_0", shape = [3], value = dense<[1, 5, 10]> : tensor<3xi64>} : () -> memref<3xi64>
  %res = memref.alloc() {alignment = 4096 : i64} : memref<1x1x1x1x32x64xf16>
  %input = memref.alloc() {alignment = 4096 : i64} : memref<1x1x1x1x32x64xf16>
  %work_area = memref.alloc() {alignment = 4096 : i64} : memref<8192xi8>
  "zlow.softmax"(%input, %work_area, %shape, %res) {act_func = "ACT_NONE"} : (memref<1x1x1x1x32x64xf16>, memref<8192xi8>, memref<3xi64>, memref<1x1x1x1x32x64xf16>) -> ()
  return
}
// CHECK:       llvm.mlir.global internal constant @[[SHAPE_CONST_GLOBAL:.*]](dense<[1, 5, 10]> : tensor<3xi64>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<3 x i64>
// CHECK-LABEL: llvm.func @test_zlow_softmax_constant_shape
// CHECK-DAG:       [[SHAPE_MEMREF_0:%.+]] = llvm.mlir.addressof @[[SHAPE_CONST_GLOBAL]] : !llvm.ptr
// CHECK-DAG:       [[SHAPE_MEMREF_1:%.+]] = llvm.bitcast [[SHAPE_MEMREF_0]] : !llvm.ptr to !llvm.ptr
// CHECK-DAG:       [[SHAPE_MEMREF_2:%.+]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:      [[SHAPE_MEMREF_3:%.+]] = llvm.insertvalue [[SHAPE_MEMREF_1]], [[SHAPE_MEMREF_2]][0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:      [[SHAPE_MEMREF_4:%.+]] = llvm.insertvalue [[SHAPE_MEMREF_1]], [[SHAPE_MEMREF_3]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:      [[SHAPE_MEMREF_5:%.+]] = llvm.mlir.constant(0 : index) : i64
// CHECK-NEXT:      [[SHAPE_MEMREF_6:%.+]] = llvm.insertvalue [[SHAPE_MEMREF_5]], [[SHAPE_MEMREF_4]][2] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:      [[SHAPE_MEMREF_7:%.+]] = llvm.mlir.constant(3 : index) : i64
// CHECK-NEXT:      [[SHAPE_MEMREF_8:%.+]] = llvm.insertvalue [[SHAPE_MEMREF_7]], [[SHAPE_MEMREF_6]][3, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>
// CHECK-NEXT:      [[SHAPE_MEMREF_9:%.+]] = llvm.mlir.constant(1 : index) : i64
// CHECK-NEXT:      [[SHAPE_MEMREF_10:%.+]] = llvm.insertvalue [[SHAPE_MEMREF_9]], [[SHAPE_MEMREF_8]][4, 0] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)>

// ...

// CHECK:           %[[SHAPE:.*]] = llvm.extractvalue [[SHAPE_MEMREF_10]][1] : !llvm.struct<(ptr, ptr, i64, array<1 x i64>, array<1 x i64>)> 
// CHECK-NEXT:      %[[DIM0_0:.*]] = llvm.getelementptr %[[SHAPE]][0] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:      %[[DIM0_1:.*]] = llvm.load %[[DIM0_0]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[DIM1_0:.*]] = llvm.getelementptr %[[SHAPE]][1] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:      %[[DIM1_1:.*]] = llvm.load %[[DIM1_0]] : !llvm.ptr -> i64
// CHECK-NEXT:      %[[DIM2_0:.*]] = llvm.getelementptr %[[SHAPE]][2] : (!llvm.ptr) -> !llvm.ptr, i64
// CHECK-NEXT:      %[[DIM2_1:.*]] = llvm.load %[[DIM2_0]] : !llvm.ptr -> i64

// ...

// CHECK:           llvm.call @zdnn_init_pre_transformed_desc({{.*}}, {{.*}}, {{.*}}, %[[DIM0_1]], %[[DIM1_1]], %[[DIM2_1]]) vararg(!llvm.func<void (i64, i64, ptr, i64, i64, i64, ...)>) : (i64, i64, !llvm.ptr, i64, i64, i64) -> ()
