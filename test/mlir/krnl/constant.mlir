// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --convert-krnl-to-affine --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// -----

func @test_constant(%arg0 : tensor<3x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[[0.0, 0.0], [1.0, 1.1], [2.0, 2.1]]> : tensor<3x2xf32>} : () -> tensor<*xf32>
  %1 = "onnx.Relu"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK: llvm.func @llvm.memcpy.p0i8.p0i8.i64(!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i1)
  // CHECK: llvm.mlir.global internal constant [[GLOBAL_CONST:@.+]](dense<{{.*}}[0.000000e+00, 0.000000e+00], [1.000000e+00, 1.100000e+00], [2.000000e+00, 2.100000e+00]{{.*}}> : tensor<3x2xf32>) : !llvm.array<3 x array<2 x float>>
  // CHECK: llvm.func @test_constant({{.*}}) -> !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)> {

  // CHECK: [[CONST_3:%.+]] = llvm.mlir.constant(3 : index) : !llvm.i64
  // CHECK: [[CONST_4:%.+]] = llvm.mlir.constant(2 : index) : !llvm.i64

  /// This is the result MemRef:
  // CHECK: [[MALLOC_FOR_RES:%.+]] = llvm.call @malloc
  // CHECK: [[CAST_MALLOC_FOR_RES:%.+]] = llvm.bitcast [[MALLOC_FOR_RES]] : !llvm.ptr<i8> to !llvm.ptr<float>
  // CHECK: [[RES_MEMREF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_1:%.+]] = llvm.insertvalue [[CAST_MALLOC_FOR_RES]], [[RES_MEMREF]][0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_2:%.+]] = llvm.insertvalue [[CAST_MALLOC_FOR_RES]], [[RES_MEMREF_1]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST_0:%.+]] = llvm.mlir.constant(0 : index) : !llvm.i64
  // CHECK: [[RES_MEMREF_3:%.+]] = llvm.insertvalue [[CONST_0]], [[RES_MEMREF_2]][2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST_1:%.+]] = llvm.mlir.constant(1 : index) : !llvm.i64
  // CHECK: [[CONST_2:%.+]] = llvm.mlir.constant(2 : index) : !llvm.i64
  // CHECK: [[RES_MEMREF_4:%.+]] = llvm.insertvalue [[CONST_3]], [[RES_MEMREF_3]][3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_5:%.+]] = llvm.insertvalue [[CONST_2]], [[RES_MEMREF_4]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_6:%.+]] = llvm.insertvalue [[CONST_4]], [[RES_MEMREF_5]][3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_7:%.+]] = llvm.insertvalue [[CONST_1]], [[RES_MEMREF_6]][4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>

  // CHECK: [[CONST1:%.+]] = llvm.mlir.constant(1 : i64) : !llvm.i64
  // CHECK: [[ALLOCA:%.+]] = llvm.alloca [[CONST1]] x !llvm.array<3 x array<2 x float>> : (!llvm.i64) -> !llvm.ptr<array<3 x array<2 x float>>> 
  // CHECK: [[I8ALLOCA:%.+]] = llvm.bitcast [[ALLOCA]] : !llvm.ptr<array<3 x array<2 x float>>> to !llvm.ptr<i8> 
  
  // CHECK: [[GLOBAL_ADDR:%.+]] = llvm.mlir.addressof [[GLOBAL_CONST]] : !llvm.ptr<array<3 x array<2 x float>>>  
  // CHECK: [[I8GLOBAL:%.+]] = llvm.bitcast [[GLOBAL_ADDR]] : !llvm.ptr<array<3 x array<2 x float>>> to !llvm.ptr<i8> 

  /// Size of the constant tensor in bytes.
  // CHECK: [[CONST4:%.+]] = llvm.mlir.constant(4 : i64) : !llvm.i64
  // CHECK: [[CONST6:%.+]] = llvm.mlir.constant(6 : i64) : !llvm.i64
  // CHECK: [[CONST_MUL1:%.+]] = llvm.mul [[CONST4]], [[CONST6]] : !llvm.i64
  // CHECK: [[GLOBAL_SIZE_BYTES:%.+]] = llvm.sext [[CONST_MUL1]] : !llvm.i64 to !llvm.i64

  /// Volatile flag
  // CHECK: [[CONST0:%.+]] = llvm.mlir.constant(false) : !llvm.i1

  // CHECK: llvm.call @llvm.memcpy.p0i8.p0i8.i64([[I8ALLOCA]], [[I8GLOBAL]], [[GLOBAL_SIZE_BYTES]], [[CONST0]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i1) -> ()

  /// Prepare data for MemRef insertion.
  // CHECK: [[TYPED_ALLOCA:%.+]] = llvm.bitcast [[ALLOCA]] : !llvm.ptr<array<3 x array<2 x float>>> to !llvm.ptr<float>

  /// Insert the constant value in the local MemRef.
  // CHECK: [[LOCAL_MEMREF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[LOCAL_MEMREF0:%.+]] = llvm.insertvalue [[TYPED_ALLOCA]], [[LOCAL_MEMREF]][0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[LOCAL_MEMREF1:%.+]] = llvm.insertvalue [[TYPED_ALLOCA]], [[LOCAL_MEMREF0]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>

  /// Insert offset.
  // CHECK: [[CONST00:%.+]] = llvm.mlir.constant(0 : index) : !llvm.i64
  // CHECK: [[MEMREF1:%.+]] = llvm.insertvalue [[CONST00]], [[LOCAL_MEMREF1]][2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)> 

  /// Insert sizes and strides.
  // CHECK: [[CONST3:%.+]] = llvm.mlir.constant(3 : index) : !llvm.i64
  // CHECK: [[MEMREF2:%.+]] = llvm.insertvalue [[CONST3]], [[MEMREF1]][3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)> 
  // CHECK: [[CONST1:%.+]] = llvm.mlir.constant(2 : index) : !llvm.i64
  // CHECK: [[MEMREF3:%.+]] = llvm.insertvalue [[CONST1]], [[MEMREF2]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)> 

  // CHECK: [[CONST2:%.+]] = llvm.mlir.constant(2 : index) : !llvm.i64
  // CHECK: [[MEMREF4:%.+]] = llvm.insertvalue [[CONST2]], [[MEMREF3]][3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST1:%.+]] = llvm.mlir.constant(1 : index) : !llvm.i64
  // CHECK: [[MEMREF5:%.+]] = llvm.insertvalue [[CONST1]], [[MEMREF4]][4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>

  // CHECK: llvm.return [[RES_MEMREF_7]] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
}

// -----

func @test_constant(%arg0 : tensor<3x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[[0.0, 0.0], [1.0, 1.1], [2.0, 2.1]]> : tensor<3x2xf32>} : () -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK: [[CONST1:%.+]] = llvm.mlir.constant(1 : i64) : !llvm.i64
  // CHECK: [[ALLOCA:%.+]] = llvm.alloca [[CONST1]] x !llvm.array<3 x array<2 x float>> : (!llvm.i64) -> !llvm.ptr<array<3 x array<2 x float>>>
  // CHECK: [[I8ALLOCA:%.+]] = llvm.bitcast [[ALLOCA]] : !llvm.ptr<array<3 x array<2 x float>>> to !llvm.ptr<i8>

  // CHECK: [[GLOBAL_ADDR:%.+]] = llvm.mlir.addressof [[GLOBAL_CONST]] : !llvm.ptr<array<3 x array<2 x float>>>
  // CHECK: [[I8GLOBAL:%.+]] = llvm.bitcast [[GLOBAL_ADDR]] : !llvm.ptr<array<3 x array<2 x float>>> to !llvm.ptr<i8>

  /// Size of the constant tensor in bytes.
  // CHECK: [[CONST4:%.+]] = llvm.mlir.constant(4 : i64) : !llvm.i64
  // CHECK: [[CONST6:%.+]] = llvm.mlir.constant(6 : i64) : !llvm.i64
  // CHECK: [[CONST_MUL1:%.+]] = llvm.mul [[CONST4]], [[CONST6]] : !llvm.i64
  // CHECK: [[GLOBAL_SIZE_BYTES:%.+]] = llvm.sext [[CONST_MUL1]] : !llvm.i64 to !llvm.i64

  /// Volatile flag
  // CHECK: [[CONST0:%.+]] = llvm.mlir.constant(false) : !llvm.i1

  // CHECK: llvm.call @llvm.memcpy.p0i8.p0i8.i64([[I8ALLOCA]], [[I8GLOBAL]], [[GLOBAL_SIZE_BYTES]], [[CONST0]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i1) -> ()

  /// Prepare data for MemRef insertion.
  // CHECK: [[TYPED_ALLOCA:%.+]] = llvm.bitcast [[ALLOCA]] : !llvm.ptr<array<3 x array<2 x float>>> to !llvm.ptr<float>

  /// Insert the constant value in the local MemRef.
  // CHECK: [[LOCAL_MEMREF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[LOCAL_MEMREF0:%.+]] = llvm.insertvalue [[TYPED_ALLOCA]], [[LOCAL_MEMREF]][0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[LOCAL_MEMREF1:%.+]] = llvm.insertvalue [[TYPED_ALLOCA]], [[LOCAL_MEMREF0]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>

  /// Insert offset.
  // CHECK: [[CONST00:%.+]] = llvm.mlir.constant(0 : index) : !llvm.i64
  // CHECK: [[MEMREF1:%.+]] = llvm.insertvalue [[CONST00]], [[LOCAL_MEMREF1]][2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>

  /// Insert sizes and strides.
  // CHECK: [[CONST3:%.+]] = llvm.mlir.constant(3 : index) : !llvm.i64
  // CHECK: [[MEMREF2:%.+]] = llvm.insertvalue [[CONST3]], [[MEMREF1]][3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST1:%.+]] = llvm.mlir.constant(2 : index) : !llvm.i64
  // CHECK: [[MEMREF3:%.+]] = llvm.insertvalue [[CONST1]], [[MEMREF2]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>

  // CHECK: [[CONST2:%.+]] = llvm.mlir.constant(2 : index) : !llvm.i64
  // CHECK: [[MEMREF4:%.+]] = llvm.insertvalue [[CONST2]], [[MEMREF3]][3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST1:%.+]] = llvm.mlir.constant(1 : index) : !llvm.i64
  // CHECK: [[MEMREF5:%.+]] = llvm.insertvalue [[CONST1]], [[MEMREF4]][4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>

  // CHECK: [[CONST_3:%.+]] = llvm.mlir.constant(3 : index) : !llvm.i64
  // CHECK: [[CONST_4:%.+]] = llvm.mlir.constant(2 : index) : !llvm.i64

  /// This is the result MemRef:
  // CHECK: [[MALLOC_FOR_RES:%.+]] = llvm.call @malloc
  // CHECK: [[CAST_MALLOC_FOR_RES:%.+]] = llvm.bitcast [[MALLOC_FOR_RES]] : !llvm.ptr<i8> to !llvm.ptr<float>
  // CHECK: [[RES_MEMREF:%.+]] = llvm.mlir.undef : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_1:%.+]] = llvm.insertvalue [[CAST_MALLOC_FOR_RES]], [[RES_MEMREF]][0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_2:%.+]] = llvm.insertvalue [[CAST_MALLOC_FOR_RES]], [[RES_MEMREF_1]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST_0:%.+]] = llvm.mlir.constant(0 : index) : !llvm.i64
  // CHECK: [[RES_MEMREF_3:%.+]] = llvm.insertvalue [[CONST_0]], [[RES_MEMREF_2]][2] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[CONST_1:%.+]] = llvm.mlir.constant(1 : index) : !llvm.i64
  // CHECK: [[CONST_2:%.+]] = llvm.mlir.constant(2 : index) : !llvm.i64
  // CHECK: [[RES_MEMREF_4:%.+]] = llvm.insertvalue [[CONST_3]], [[RES_MEMREF_3]][3, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_5:%.+]] = llvm.insertvalue [[CONST_2]], [[RES_MEMREF_4]][4, 0] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_6:%.+]] = llvm.insertvalue [[CONST_4]], [[RES_MEMREF_5]][3, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[RES_MEMREF_7:%.+]] = llvm.insertvalue [[CONST_1]], [[RES_MEMREF_6]][4, 1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>

  /// Copy result in a MemRef:
  // CHECK: [[CONST_5:%.+]] = llvm.mlir.constant(24 : i64) : !llvm.i64
  // CHECK: [[OUT_DATA:%.+]] = llvm.extractvalue [[RES_MEMREF_7]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[TYPED_OUT_DATA:%.+]] = llvm.bitcast [[OUT_DATA]] : !llvm.ptr<float> to !llvm.ptr<i8>
  // CHECK: [[GLOBAL_DATA:%.+]] = llvm.extractvalue [[MEMREF5]][1] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[TYPED_GLOBAL_DATA:%.+]] = llvm.bitcast [[GLOBAL_DATA]] : !llvm.ptr<float> to !llvm.ptr<i8>
  // CHECK: [[EXTENDED_CONST_5:%.+]] = llvm.sext [[CONST_5]] : !llvm.i64 to !llvm.i64
  // CHECK: [[FALSE:%.+]] = llvm.mlir.constant(false) : !llvm.i1
  // CHECK: llvm.call @llvm.memcpy.p0i8.p0i8.i64([[TYPED_OUT_DATA]], [[TYPED_GLOBAL_DATA]], [[EXTENDED_CONST_5]], [[FALSE]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.i64, !llvm.i1) -> ()
  // CHECK: llvm.return [[RES_MEMREF_7]] : !llvm.struct<(ptr<float>, ptr<float>, i64, array<2 x i64>, array<2 x i64>)>
}
