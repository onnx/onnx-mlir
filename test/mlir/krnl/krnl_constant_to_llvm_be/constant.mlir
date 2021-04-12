// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --convert-krnl-to-affine --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// -----

func @test_constant(%arg0 : tensor<3x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[[0.0, 0.0], [1.0, 1.1], [2.0, 2.1]]> : tensor<3x2xf32>} : () -> tensor<*xf32>
  %1 = "onnx.Relu"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK: llvm.mlir.global internal constant [[GLOBAL_CONST:@.+]]("\00\00\00\00\00\00\00\00?\80\00\00?\8C\CC\CD@\00\00\00@\06ff")
  // CHECK: llvm.func @test_constant({{.*}}) -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> {

  // CHECK: [[CONST_3:%.+]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: [[CONST_2:%.+]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: [[CONST_1:%.+]] = llvm.mlir.constant(1 : index) : i64

  /// This is the result MemRef:
  // CHECK: [[MALLOC_FOR_RES:%.+]] = llvm.call @malloc
  // CHECK: [[CAST_MALLOC_FOR_RES:%.+]] = llvm.bitcast [[MALLOC_FOR_RES]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK: [[RES_MEMREF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_1:%.+]] = llvm.insertvalue [[CAST_MALLOC_FOR_RES]], [[RES_MEMREF]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_2:%.+]] = llvm.insertvalue [[CAST_MALLOC_FOR_RES]], [[RES_MEMREF_1]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST_0:%.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: [[RES_MEMREF_3:%.+]] = llvm.insertvalue [[CONST_0]], [[RES_MEMREF_2]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_4:%.+]] = llvm.insertvalue [[CONST_3]], [[RES_MEMREF_3]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_5:%.+]] = llvm.insertvalue [[CONST_2]], [[RES_MEMREF_4]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_6:%.+]] = llvm.insertvalue [[CONST_2]], [[RES_MEMREF_5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_7:%.+]] = llvm.insertvalue [[CONST_1]], [[RES_MEMREF_6]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

  // CHECK: [[GLOBAL_ADDR:%.+]] = llvm.mlir.addressof [[GLOBAL_CONST]] : !llvm.ptr<array<24 x i8>>
  // CHECK: [[TYPED_GLOBAL:%.+]] = llvm.bitcast [[GLOBAL_ADDR]] : !llvm.ptr<array<24 x i8>> to !llvm.ptr<f32>

  /// Insert the constant value in the local MemRef.
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

  // CHECK: llvm.return [[RES_MEMREF_7]] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
}

// -----

func @test_constant(%arg0 : tensor<3x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[[0.0, 0.0], [1.0, 1.1], [2.0, 2.1]]> : tensor<3x2xf32>} : () -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK: llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1)
  // CHECK: llvm.mlir.global internal constant [[GLOBAL_CONST:@.+]]("\00\00\00\00\00\00\00\00?\80\00\00?\8C\CC\CD@\00\00\00@\06ff")
  // CHECK: llvm.func @test_constant({{.*}}) -> !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)> {

  // CHECK-DAG: [[CONST_5:%.+]] = llvm.mlir.constant(24 : i64) : i64
  // CHECK: [[GLOBAL_ADDR:%.+]] = llvm.mlir.addressof [[GLOBAL_CONST]] : !llvm.ptr<array<24 x i8>>
  // CHECK: [[TYPED_GLOBAL:%.+]] = llvm.bitcast [[GLOBAL_ADDR]] : !llvm.ptr<array<24 x i8>> to !llvm.ptr<f32>

  /// Insert the constant value in the local MemRef.
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

  // CHECK: [[CONST_3:%.+]] = llvm.mlir.constant(3 : index) : i64
  // CHECK: [[CONST_2:%.+]] = llvm.mlir.constant(2 : index) : i64
  // CHECK: [[CONST_1:%.+]] = llvm.mlir.constant(1 : index) : i64

  /// This is the result MemRef:
  // CHECK: [[MALLOC_FOR_RES:%.+]] = llvm.call @malloc
  // CHECK: [[CAST_MALLOC_FOR_RES:%.+]] = llvm.bitcast [[MALLOC_FOR_RES]] : !llvm.ptr<i8> to !llvm.ptr<f32>
  // CHECK: [[RES_MEMREF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_1:%.+]] = llvm.insertvalue [[CAST_MALLOC_FOR_RES]], [[RES_MEMREF]][0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_2:%.+]] = llvm.insertvalue [[CAST_MALLOC_FOR_RES]], [[RES_MEMREF_1]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST_0:%.+]] = llvm.mlir.constant(0 : index) : i64
  // CHECK: [[RES_MEMREF_3:%.+]] = llvm.insertvalue [[CONST_0]], [[RES_MEMREF_2]][2] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_4:%.+]] = llvm.insertvalue [[CONST_3]], [[RES_MEMREF_3]][3, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_5:%.+]] = llvm.insertvalue [[CONST_2]], [[RES_MEMREF_4]][3, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_6:%.+]] = llvm.insertvalue [[CONST_2]], [[RES_MEMREF_5]][4, 0] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_7:%.+]] = llvm.insertvalue [[CONST_1]], [[RES_MEMREF_6]][4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

  /// Copy result in a MemRef:
  // CHECK: [[OUT_DATA:%.+]] = llvm.extractvalue [[RES_MEMREF_7]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[TYPED_OUT_DATA:%.+]] = llvm.bitcast [[OUT_DATA]] : !llvm.ptr<f32> to !llvm.ptr<i8>
  // CHECK: [[GLOBAL_DATA:%.+]] = llvm.extractvalue [[MEMREF5]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[TYPED_GLOBAL_DATA:%.+]] = llvm.bitcast [[GLOBAL_DATA]] : !llvm.ptr<f32> to !llvm.ptr<i8>
  // CHECK: [[EXTENDED_CONST_5:%.+]] = llvm.sext [[CONST_5]] : i64 to i64
  // CHECK: [[FALSE:%.+]] = llvm.mlir.constant(false) : i1
  // CHECK: llvm.call @llvm.memcpy.p0i8.p0i8.i64([[TYPED_OUT_DATA]], [[TYPED_GLOBAL_DATA]], [[EXTENDED_CONST_5]], [[FALSE]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, i1) -> ()
  // CHECK: llvm.return [[RES_MEMREF_7]] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
}
