// RUN: onnx-mlir-opt --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// COM: Check lowering a krnl constant with a dense attribute.
// COM: The dense attribute's data pointer will be used in the returned MemRef
// COM: directly.
func @test_krnl_dense_constant() -> memref<3x2xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [3, 2], value = dense<[[0.0, 0.0], [1.0, 1.1], [2.0, 2.1]]> : tensor<3x2xf32>} : () -> memref<3x2xf32>
  return %0 : memref<3x2xf32>

  /// Put constant value to global.
  // CHECK: llvm.mlir.global internal constant [[GLOBAL_CONST:@.+]]("\00\00\00\00\00\00\00\00?\80\00\00?\8C\CC\CD@\00\00\00@\06ff")
  // CHECK: llvm.func @test_krnl_dense_constant({{.*}}) -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> {

  // CHECK: [[GLOBAL_ADDR:%.+]] = llvm.mlir.addressof [[GLOBAL_CONST]] : !llvm.ptr<array<24 x i8>>
  // CHECK: [[TYPED_GLOBAL:%.+]] = llvm.bitcast [[GLOBAL_ADDR]] : !llvm.ptr<array<24 x i8>> to !llvm.ptr<f32>

  /// Insert the constant value in the returned MemRef.
  // CHECK: [[LOCAL_MEMREF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[LOCAL_MEMREF0:%.+]] = llvm.insertvalue [[TYPED_GLOBAL]], [[LOCAL_MEMREF]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[LOCAL_MEMREF1:%.+]] = llvm.insertvalue [[TYPED_GLOBAL]], [[LOCAL_MEMREF0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  /// Insert offset.
  // CHECK: [[CONST00:%.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: [[MEMREF1:%.+]] = llvm.insertvalue [[CONST00]], [[LOCAL_MEMREF1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
  /// Insert sizes and strides.
  // CHECK: [[CONST3:%.+]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: [[MEMREF2:%.+]] = llvm.insertvalue [[CONST3]], [[MEMREF1]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[CONST1:%.+]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: [[MEMREF3:%.+]] = llvm.insertvalue [[CONST1]], [[MEMREF2]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[CONST2:%.+]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: [[MEMREF4:%.+]] = llvm.insertvalue [[CONST2]], [[MEMREF3]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST1:%.+]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[MEMREF5:%.+]] = llvm.insertvalue [[CONST1]], [[MEMREF4]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

  // CHECK: llvm.return [[MEMREF5]] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
}

// -----

// COM: Check lowering a krnl constant with alignment.
// COM: In this case, we need to allocate an aligned buffer and do data copy.
func @test_krnl_aligned_constant() -> memref<3x2xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [3, 2], alignment = 1024, value = dense<[[0.0, 0.0], [1.0, 1.1], [2.0, 2.1]]> : tensor<3x2xf32>} : () -> memref<3x2xf32>
  return %0 : memref<3x2xf32>

  /// Put constant value to global.
  // CHECK: llvm.mlir.global internal constant [[GLOBAL_CONST:@.+]]("\00\00\00\00\00\00\00\00?\80\00\00?\8C\CC\CD@\00\00\00@\06ff")
  // CHECK: llvm.func @test_krnl_aligned_constant({{.*}}) -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> {

  // CHECK: [[GLOBAL_ADDR:%.+]] = llvm.mlir.addressof [[GLOBAL_CONST]] : !llvm.ptr<array<24 x i8>>

  /// Allocate an aligned buffer.
  // CHECK: [[ALLOC:%.+]] = llvm.alloca {{.*}} x !llvm.array<3 x array<2 x f32>> {alignment = 1024 : i64} : (i64) -> !llvm.ptr<array<3 x array<2 x f32>>>

  /// Copy data from the global to the aligned buffer.
  // CHECK: [[ALLOC_PTR:%.+]] = llvm.bitcast [[ALLOC]] : !llvm.ptr<array<3 x array<2 x f32>>> to !llvm.ptr<i8>
  // CHECK: [[I8GLOBAL_PTR:%.+]] = llvm.bitcast [[GLOBAL_ADDR]] : !llvm.ptr<array<24 x i8>> to !llvm.ptr<i8>
  // CHECK: llvm.call @llvm.memcpy.p0i8.p0i8.i64([[ALLOC_PTR]], [[I8GLOBAL_PTR]], %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()

  /// Insert the constant value in the returned MemRef.
  // CHECK: [[TYPED_ALLOC:%.+]] = llvm.bitcast [[ALLOC]] : !llvm.ptr<array<3 x array<2 x f32>>> to !llvm.ptr<f32>
  // CHECK: [[LOCAL_MEMREF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[LOCAL_MEMREF0:%.+]] = llvm.insertvalue [[TYPED_ALLOC]], [[LOCAL_MEMREF]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[LOCAL_MEMREF1:%.+]] = llvm.insertvalue [[TYPED_ALLOC]], [[LOCAL_MEMREF0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  /// Insert offset.
  // CHECK: [[CONST00:%.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: [[MEMREF1:%.+]] = llvm.insertvalue [[CONST00]], [[LOCAL_MEMREF1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
  /// Insert sizes and strides.
  // CHECK: [[CONST3:%.+]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: [[MEMREF2:%.+]] = llvm.insertvalue [[CONST3]], [[MEMREF1]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[CONST1:%.+]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: [[MEMREF3:%.+]] = llvm.insertvalue [[CONST1]], [[MEMREF2]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[CONST2:%.+]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: [[MEMREF4:%.+]] = llvm.insertvalue [[CONST2]], [[MEMREF3]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST1:%.+]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[MEMREF5:%.+]] = llvm.insertvalue [[CONST1]], [[MEMREF4]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

  // CHECK: llvm.return [[MEMREF5]] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
}

// -----

// COM: Check lowering a krnl constant with an opaque attribute
// COM: Data for the opaque attribute: [0.5, 1., 1.5, 2.]
func @test_krnl_opaque_constant() -> memref<1x4xf32> {
  %0 = "krnl.global"() {name = "constant_0", shape = [1, 4], value = opaque<"krnl", "0x3F0000003F8000003FC0000040000000"> : tensor<1x4xf32>} : () -> memref<1x4xf32>
  return %0 : memref<1x4xf32>

  /// Put constant value to global.
  // CHECK: llvm.mlir.global internal constant [[GLOBAL_CONST:@.+]]("?\00\00\00?\80\00\00?\C0\00\00@\00\00\00")
  // CHECK: llvm.func @test_krnl_opaque_constant({{.*}}) -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> {

  // CHECK: [[GLOBAL_ADDR:%.+]] = llvm.mlir.addressof [[GLOBAL_CONST]] : !llvm.ptr<array<16 x i8>>
  // CHECK: [[TYPED_GLOBAL:%.+]] = llvm.bitcast [[GLOBAL_ADDR]] : !llvm.ptr<array<16 x i8>> to !llvm.ptr<f32>

  /// Insert the constant value in the returned MemRef.
  // CHECK: [[LOCAL_MEMREF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[LOCAL_MEMREF0:%.+]] = llvm.insertvalue [[TYPED_GLOBAL]], [[LOCAL_MEMREF]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[LOCAL_MEMREF1:%.+]] = llvm.insertvalue [[TYPED_GLOBAL]], [[LOCAL_MEMREF0]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  /// Insert offset.
  // CHECK: [[CONST0:%.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: [[MEMREF1:%.+]] = llvm.insertvalue [[CONST0]], [[LOCAL_MEMREF1]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
  /// Insert sizes and strides.
  // CHECK: [[CONST1:%.+]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[MEMREF2:%.+]] = llvm.insertvalue [[CONST1]], [[MEMREF1]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[CONST4:%.+]] = llvm.mlir.constant(4 : index) : i64
  // CHECK: [[MEMREF3:%.+]] = llvm.insertvalue [[CONST4]], [[MEMREF2]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[CONST4_:%.+]] = llvm.mlir.constant(4 : index) : i64
  // CHECK: [[MEMREF4:%.+]] = llvm.insertvalue [[CONST4_]], [[MEMREF3]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST1_:%.+]] = llvm.mlir.constant(1 : index) : i64
  // CHECK: [[MEMREF5:%.+]] = llvm.insertvalue [[CONST1_]], [[MEMREF4]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

  // CHECK: llvm.return [[MEMREF5]] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
}

