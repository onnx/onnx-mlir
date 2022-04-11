// RUN: onnx-mlir-opt --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// -----

func @test_lower_both_zlow_and_krnl() -> () {
  %0 = memref.alloc() : memref<10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %2 = "krnl.global"() {name = "constant_0", shape = [1, 2], value = dense<[[0., 1.0]]> : tensor<1x2xf32>} : () -> memref<1x2xf32>
  "zlow.stick"(%0, %1) : (memref<10x10xf32>, memref<1x1x32x64xf32>) -> ()
  return

  // CHECK-DAG: llvm.mlir.global internal constant @{{.*}}(dense<{{\[}}[0.000000e+00, 1.000000e+00]{{\]}}> : tensor<1x2xf32>) {alignment = 16 : i64} : !llvm.array<1 x array<2 x f32>>
}

// -----

func @test_stick() -> () {
  %0 = memref.alloc() : memref<10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  "zlow.stick"(%0, %1) : (memref<10x10xf32>, memref<1x1x32x64xf32>) -> ()
  return

  // CHECK-LABEL: test_stick
  // CHECK: [[UNSTICKIFIED_MEMREF:%.+]] = llvm.insertvalue {{.*}}, {{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[STICKIFIED_MEMREF:%.+]] = llvm.insertvalue {{.*}}, {{.*}}[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>

  // CHECK: [[ALIGNED_BUFFER:%.+]] = llvm.extractvalue [[STICKIFIED_MEMREF]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
  // CHECK: [[ALIGNED_BUFFER_I8PTR:%.+]] = llvm.bitcast [[ALIGNED_BUFFER]] : !llvm.ptr<f32> to !llvm.ptr<i8>

  // CHECK: [[PRE_TRANSFORMED_DESC:%.+]] = llvm.alloca {{.*}} x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>
  // CHECK: [[DATA_LAYOUT:%.+]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: [[DATA_TYPE:%.+]] = llvm.mlir.constant(255 : i64) : i64
  // CHECK: [[PRE_TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[PRE_TRANSFORMED_DESC]] : !llvm.ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>> to !llvm.ptr<i8>
  // CHECK: llvm.call @zdnn_init_pre_transformed_desc([[DATA_LAYOUT]], [[DATA_TYPE]], [[PRE_TRANSFORMED_DESC_I8PTR]], {{.*}}, {{.*}}) : (i64, i64, !llvm.ptr<i8>, i64, i64) -> ()

  // CHECK: [[TRANSFORMED_DESC:%.+]] = llvm.alloca {{.*}} x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>
  // CHECK: [[PRE_TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[PRE_TRANSFORMED_DESC]] : !llvm.ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>> to !llvm.ptr<i8>
  // CHECK: [[TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[TRANSFORMED_DESC]] : !llvm.ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>> to !llvm.ptr<i8>
  // CHECK: {{.*}} = llvm.call @zdnn_generate_transformed_desc([[PRE_TRANSFORMED_DESC_I8PTR]], [[TRANSFORMED_DESC_I8PTR]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32

  // CHECK: [[ZTENSOR:%.+]] = llvm.alloca {{.*}} x !llvm.struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)> : (i64) -> !llvm.ptr<struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)>>
  // CHECK: [[TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[TRANSFORMED_DESC]] : !llvm.ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>> to !llvm.ptr<i8>
  // CHECK: [[BUFFER_SIZE:%.+]] = llvm.call @zdnn_getsize_ztensor([[TRANSFORMED_DESC_I8PTR]]) : (!llvm.ptr<i8>) -> i64
  // CHECK: [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: [[C2:%.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: [[C3:%.+]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: [[C4:%.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: [[ZTENSOR_PRE_TRANSFORMED_DESC:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}[[C0]], 0] : (!llvm.ptr<struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)>>, i32) -> !llvm.ptr<ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>>
  // CHECK: llvm.store [[PRE_TRANSFORMED_DESC]], [[ZTENSOR_PRE_TRANSFORMED_DESC]] : !llvm.ptr<ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>>

  // CHECK: [[ZTENSOR_TRANSFORMED_DESC:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}[[C0]], 1] : (!llvm.ptr<struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)>>, i32) -> !llvm.ptr<ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>>
  // CHECK: llvm.store [[TRANSFORMED_DESC]], [[ZTENSOR_TRANSFORMED_DESC]] : !llvm.ptr<ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>>

  // CHECK: [[ZTENSOR_BUFFER_SIZE:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}[[C0]], 2] :  (!llvm.ptr<struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)>>, i32) -> !llvm.ptr<i64>
  // CHECK: llvm.store [[BUFFER_SIZE]], [[ZTENSOR_BUFFER_SIZE]] : !llvm.ptr<i64>

  // CHECK: [[ZTENSOR_BUFFER:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}[[C0]], 3] :  (!llvm.ptr<struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)>>, i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: llvm.store [[ALIGNED_BUFFER_I8PTR]], [[ZTENSOR_BUFFER]] : !llvm.ptr<ptr<i8>>

  // CHECK: [[FALSE:%.+]] = llvm.mlir.constant(false) : i1

  // CHECK: [[IS_TRANSFORMED:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}[[C0]], 4] : (!llvm.ptr<struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)>>, i32) -> !llvm.ptr<i1>
  // CHECK: llvm.store [[FALSE]], [[IS_TRANSFORMED]] : !llvm.ptr<i1>

  // CHECK: [[UNSTICKIFIED:%.+]] = llvm.extractvalue [[UNSTICKIFIED_MEMREF]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[UNSTICKIFIED_I8PTR:%.+]] = llvm.bitcast [[UNSTICKIFIED]] : !llvm.ptr<f32> to !llvm.ptr<i8>
  // CHECK: [[ZTENSOR_I8PTR:%.+]] = llvm.bitcast [[ZTENSOR]] : !llvm.ptr<struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)>> to !llvm.ptr<i8>
  // CHECK: {{.*}} = llvm.call @zdnn_transform_ztensor([[ZTENSOR_I8PTR]], [[UNSTICKIFIED_I8PTR]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32

  // CHECK: llvm.return
}

// -----

func @test_unstick() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<10x10xf32>
  "zlow.unstick"(%0, %1) : (memref<1x1x32x64xf32>, memref<10x10xf32>) -> ()
  return

  // CHECK-LABEL: test_unstick
  // CHECK: [[STICKIFIED_MEMREF:%.+]] = llvm.insertvalue {{.*}}, {{.*}}[4, 3] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
  // CHECK: [[UNSTICKIFIED_MEMREF:%.+]] = llvm.insertvalue {{.*}}, {{.*}}[4, 1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>

  // CHECK: [[ALIGNED_BUFFER:%.+]] = llvm.extractvalue [[STICKIFIED_MEMREF]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<4 x i64>, array<4 x i64>)>
  // CHECK: [[ALIGNED_BUFFER_I8PTR:%.+]] = llvm.bitcast [[ALIGNED_BUFFER]] : !llvm.ptr<f32> to !llvm.ptr<i8>

  // CHECK: [[PRE_TRANSFORMED_DESC:%.+]] = llvm.alloca {{.*}} x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>
  // CHECK: [[DATA_LAYOUT:%.+]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: [[DATA_TYPE:%.+]] = llvm.mlir.constant(255 : i64) : i64
  // CHECK: [[PRE_TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[PRE_TRANSFORMED_DESC]] : !llvm.ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>> to !llvm.ptr<i8>
  // CHECK: llvm.call @zdnn_init_pre_transformed_desc([[DATA_LAYOUT]], [[DATA_TYPE]], [[PRE_TRANSFORMED_DESC_I8PTR]], {{.*}}, {{.*}}) : (i64, i64, !llvm.ptr<i8>, i64, i64) -> ()

  // CHECK: [[TRANSFORMED_DESC:%.+]] = llvm.alloca {{.*}} x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>
  // CHECK: [[PRE_TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[PRE_TRANSFORMED_DESC]] : !llvm.ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>> to !llvm.ptr<i8>
  // CHECK: [[TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[TRANSFORMED_DESC]] : !llvm.ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>> to !llvm.ptr<i8>
  // CHECK: {{.*}} = llvm.call @zdnn_generate_transformed_desc([[PRE_TRANSFORMED_DESC_I8PTR]], [[TRANSFORMED_DESC_I8PTR]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32

  // CHECK: [[ZTENSOR:%.+]] = llvm.alloca {{.*}} x !llvm.struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)> : (i64) -> !llvm.ptr<struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)>>
  // CHECK: [[TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[TRANSFORMED_DESC]] : !llvm.ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>> to !llvm.ptr<i8>
  // CHECK: [[BUFFER_SIZE:%.+]] = llvm.call @zdnn_getsize_ztensor([[TRANSFORMED_DESC_I8PTR]]) : (!llvm.ptr<i8>) -> i64
  // CHECK: [[C0:%.+]] = llvm.mlir.constant(0 : i32) : i32
  // CHECK: [[C1:%.+]] = llvm.mlir.constant(1 : i32) : i32
  // CHECK: [[C2:%.+]] = llvm.mlir.constant(2 : i32) : i32
  // CHECK: [[C3:%.+]] = llvm.mlir.constant(3 : i32) : i32
  // CHECK: [[C4:%.+]] = llvm.mlir.constant(4 : i32) : i32
  // CHECK: [[ZTENSOR_PRE_TRANSFORMED_DESC:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}[[C0]], 0] : (!llvm.ptr<struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)>>, i32) -> !llvm.ptr<ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>>
  // CHECK: llvm.store [[PRE_TRANSFORMED_DESC]], [[ZTENSOR_PRE_TRANSFORMED_DESC]] : !llvm.ptr<ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>>

  // CHECK: [[ZTENSOR_TRANSFORMED_DESC:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}[[C0]], 1] : (!llvm.ptr<struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)>>, i32) -> !llvm.ptr<ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>>
  // CHECK: llvm.store [[TRANSFORMED_DESC]], [[ZTENSOR_TRANSFORMED_DESC]] : !llvm.ptr<ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>>

  // CHECK: [[ZTENSOR_BUFFER_SIZE:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}[[C0]], 2] : (!llvm.ptr<struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)>>, i32) -> !llvm.ptr<i64>
  // CHECK: llvm.store [[BUFFER_SIZE]], [[ZTENSOR_BUFFER_SIZE]] : !llvm.ptr<i64>

  // CHECK: [[ZTENSOR_BUFFER:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}[[C0]], 3] :  (!llvm.ptr<struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)>>, i32) -> !llvm.ptr<ptr<i8>>
  // CHECK: llvm.store [[ALIGNED_BUFFER_I8PTR]], [[ZTENSOR_BUFFER]] : !llvm.ptr<ptr<i8>>

  // CHECK: [[TRUE:%.+]] = llvm.mlir.constant(true) : i1

  // CHECK: [[IS_TRANSFORMED:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}[[C0]], 4] : (!llvm.ptr<struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)>>, i32) -> !llvm.ptr<i1>
  // CHECK: llvm.store [[TRUE]], [[IS_TRANSFORMED]] : !llvm.ptr<i1>

  // CHECK: [[UNSTICKIFIED:%.+]] = llvm.extractvalue [[UNSTICKIFIED_MEMREF]][1] : !llvm.struct<(ptr<f32>, ptr<f32>, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[UNSTICKIFIED_I8PTR:%.+]] = llvm.bitcast [[UNSTICKIFIED]] : !llvm.ptr<f32> to !llvm.ptr<i8>
  // CHECK: [[ZTENSOR_I8PTR:%.+]] = llvm.bitcast [[ZTENSOR]] : !llvm.ptr<struct<(ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, ptr<struct<(i32, i32, i32, i32, i32, i32, i32)>>, i64, ptr<i8>, i1, array<32 x i8>)>> to !llvm.ptr<i8>
  // CHECK: {{.*}} = llvm.call @zdnn_transform_origtensor([[ZTENSOR_I8PTR]], [[UNSTICKIFIED_I8PTR]]) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32

  // CHECK: llvm.return
}

// -----

// Check whether the lowering of zlow.relu calls the correct zDNN API or not.
func @test_call_zdnn_relu() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.relu"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_relu
  // CHECK: {{.*}} = llvm.call @zdnn_relu({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether the lowering of zlow.tanh calls the correct zDNN API or not.
func @test_call_zdnn_tanh() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.tanh"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_tanh
  // CHECK: {{.*}} = llvm.call @zdnn_tanh({{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether the lowering of zlow.sigmoid calls the correct zDNN API or not.
func @test_call_zdnn_sigmoid() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.sigmoid"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_sigmoid
  // CHECK: {{.*}} = llvm.call @zdnn_sigmoid({{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether the lowering of zlow.add calls the correct zDNN API or not.
func @test_call_zdnn_add() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %2 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.add"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_add
  // CHECK: {{.*}} = llvm.call @zdnn_add({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether the lowering of zlow.sub calls the correct zDNN API or not.
func @test_call_zdnn_sub() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %2 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.sub"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_sub
  // CHECK: {{.*}} = llvm.call @zdnn_sub({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether the lowering of zlow.mul calls the correct zDNN API or not.
func @test_call_zdnn_mul() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %2 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.mul"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_mul
  // CHECK: {{.*}} = llvm.call @zdnn_mul({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether the lowering of zlow.div calls the correct zDNN API or not.
func @test_call_zdnn_div() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %2 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.div"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_div
  // CHECK: {{.*}} = llvm.call @zdnn_div({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether the lowering of zlow.softmax calls the correct zDNN API or not.
func @test_call_zdnn_softmax() -> () {
  %0 = memref.alloc() : memref<1x1x1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x1x1x32x64xf32>
  %work_area = memref.alloc() {alignment = 4096 : i64} : memref<8192xi8>
  %shape = memref.alloc() : memref<3xi64>
  "zlow.softmax"(%0, %work_area, %shape, %1) {act_func = "ACT_NONE"} : (memref<1x1x1x1x32x64xf32>, memref<8192xi8>, memref<3xi64>, memref<1x1x1x1x32x64xf32>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_softmax
  // CHECK: {{.*}} = llvm.call @zdnn_softmax({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, i64, !llvm.ptr<i8>) -> i32
}

// -----

// COM: Check whether the lowering of zlow.stickForLSTM calls the correct zDNN API or not.
// COM: We should call zdnn_transform_ztensor with zTensor and four pointers to the buffers fori F, I, C, and O gates. 
func @test_stick_for_lstm() -> () {
  %f = memref.alloc() : memref<1x10x10xf32>
  %i = memref.alloc() : memref<1x10x10xf32>
  %c = memref.alloc() : memref<1x10x10xf32>
  %o = memref.alloc() : memref<1x10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  "zlow.stickForLSTM"(%f, %i, %c, %o, %1) : (memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x1x32x64xf32>) -> ()
  return

  // CHECK-LABEL: test_stick_for_lstm
  // CHECK: call @zdnn_transform_ztensor({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
}

// -----

// COM: Check whether the lowering of zlow.stickForGRU calls the correct zDNN API or not.
// COM: We should call zdnn_transform_ztensor with zTensor and three pointers to the buffers for Z, R, and H gates. 
func @test_stick_for_gru() -> () {
  %g = memref.alloc() : memref<1x10x10xf32>
  %r = memref.alloc() : memref<1x10x10xf32>
  %h = memref.alloc() : memref<1x10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  "zlow.stickForGRU"(%g, %r, %h, %1) : (memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x1x32x64xf32>) -> ()
  return

  // CHECK-LABEL: test_stick_for_gru
  // CHECK: call @zdnn_transform_ztensor({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether the lowering of zlow.min calls the correct zDNN API or not.
func @test_call_zdnn_min() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %2 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.min"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_min
  // CHECK: {{.*}} = llvm.call @zdnn_min({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether the lowering of zlow.max calls the correct zDNN API or not.
func @test_call_zdnn_max() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %2 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.max"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_max
  // CHECK: {{.*}} = llvm.call @zdnn_max({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether the lowering of zlow.exp calls the correct zDNN API or not.
func @test_call_zdnn_exp() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.exp"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_exp
  // CHECK: {{.*}} = llvm.call @zdnn_exp({{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether the lowering of zlow.log calls the correct zDNN API or not.
func @test_call_zdnn_log() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf32>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.log"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf32>, memref<2xi64>, memref<1x1x32x64xf32>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_log
  // CHECK: {{.*}} = llvm.call @zdnn_log({{.*}}, {{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether the lowering of zlow.matmul calls the correct zDNN API or not.
func @test_matmul_no_bcast_unstacked(%x: memref<2048xf32>,%y: memref<2048xf32>,%bias: memref<2048xf32>, %shape: memref<3xi64>) -> memref<2048xf32> {
  %res = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  "zlow.matmul"(%x, %y, %bias, %shape, %res) {is_bcast = 0 : si64, is_stacked = 0 : si64} : (memref<2048xf32>, memref<2048xf32>, memref<2048xf32>, memref<3xi64>, memref<2048xf32>) -> ()
  return %res : memref<2048xf32>
  // CHECK-LABEL: test_matmul_no_bcast_unstacked
  // CHECK: %{{.*}} = llvm.call @zdnn_matmul_op(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, i64, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether the lowering of zlow.matmul calls the correct zDNN API or not.
func @test_matmul_no_bcast_stacked(%x: memref<2048xf32>,%y: memref<2048xf32>,%bias: memref<2048xf32>, %shape: memref<3xi64>) -> memref<2048xf32> {
  %res = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  "zlow.matmul"(%x, %y, %bias, %shape, %res) {is_bcast = 0 : si64, is_stacked = -1 : si64} : (memref<2048xf32>, memref<2048xf32>, memref<2048xf32>, memref<3xi64>, memref<2048xf32>) -> ()
  return %res : memref<2048xf32>
  // CHECK-LABEL: test_matmul_no_bcast_stacked
  // CHECK: %{{.*}} = llvm.call @zdnn_matmul_op(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, i64, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether the lowering of zlow.matmul calls the correct zDNN API or not.
func @test_matmul_bcast_stacked(%x: memref<2048xf32>,%y: memref<2048xf32>,%bias: memref<2048xf32>, %shape: memref<3xi64>) -> memref<2048xf32> {
  %res = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  "zlow.matmul"(%x, %y, %bias, %shape, %res) {is_bcast = -1 : si64, is_stacked = -1 : si64} : (memref<2048xf32>, memref<2048xf32>, memref<2048xf32>, memref<3xi64>, memref<2048xf32>) -> ()
  return %res : memref<2048xf32>
  // CHECK-LABEL: test_matmul_bcast_stacked
  // CHECK: %{{.*}} = llvm.call @zdnn_matmul_bcast_op(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, i64, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether the lowering of zlow.matmul calls the correct zDNN API or not.
func @test_matmul_bcast_unstacked(%x: memref<2048xf32>,%y: memref<2048xf32>,%bias: memref<2048xf32>, %shape: memref<3xi64>) -> memref<2048xf32> {
  %res = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  "zlow.matmul"(%x, %y, %bias, %shape, %res) {is_bcast = -1 : si64, is_stacked = 0 : si64} : (memref<2048xf32>, memref<2048xf32>, memref<2048xf32>, memref<3xi64>, memref<2048xf32>) -> ()
  return %res : memref<2048xf32>
  // CHECK-LABEL: test_matmul_bcast_unstacked
  // CHECK: %{{.*}} = llvm.call @zdnn_matmul_bcast_op(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, i64, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether conv2d calls the correct zDNN API or not.
func @test_call_zdnn_cond2d() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %kernel = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %bias = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %shape = memref.alloc() : memref<7xi64>
  "zlow.conv2d"(%input, %kernel, %bias, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "SAME_PADDING", act_func = "ACT_NONE" } : (memref<2048xf32>, memref<2048xf32>, memref<2048xf32>, memref<7xi64>, memref<2048xf32>)-> ()
  return

  // CHECK-LABEL: test_call_zdnn_cond2d
  // CHECK: {{.*}} = llvm.call @zdnn_conv2d(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, i64, i64, i64, i64, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32 
}

// -----

// Check whether conv2d calls the correct zDNN API or not.
func @test_call_zdnn_cond2d_valid_padding() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %kernel = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %bias = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %shape = memref.alloc() : memref<7xi64>
  "zlow.conv2d"(%input, %kernel, %bias, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "VALID_PADDING", act_func = "ACT_NONE" } : (memref<2048xf32>, memref<2048xf32>, memref<2048xf32>, memref<7xi64>, memref<2048xf32>)-> ()
  return

  // CHECK-LABEL: test_call_zdnn_cond2d_valid_padding
  // CHECK: {{.*}} = llvm.call @zdnn_conv2d(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, i64, i64, i64, i64, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32 
}

// -----

// Check whether conv2d calls the correct zDNN API or not.
func @test_call_zdnn_cond2d_relu_act() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %kernel = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %bias = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %shape = memref.alloc() : memref<7xi64>
  "zlow.conv2d"(%input, %kernel, %bias, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "SAME_PADDING", act_func = "ACT_RELU" } : (memref<2048xf32>, memref<2048xf32>, memref<2048xf32>, memref<7xi64>, memref<2048xf32>)-> ()
  return

  // CHECK-LABEL: test_call_zdnn_cond2d_relu_act
  // CHECK: {{.*}} = llvm.call @zdnn_conv2d(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, i64, i64, i64, i64, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32 
}

// -----

// Check whether avgpool2d calls the correct zDNN API or not.
func @test_call_zdnn_avgpool2d() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %shape = memref.alloc() : memref<6xi64>
  "zlow.avgpool2d"(%input, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "SAME_PADDING" } : (memref<2048xf32>, memref<6xi64>, memref<2048xf32>)-> ()
  return

  // CHECK-LABEL: test_call_zdnn_avgpool2d
  // CHECK: {{.*}} = llvm.call @zdnn_avgpool2d(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, i64, i64, i64, i64, i64, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether maxpool2d calls the correct zDNN API or not.
func @test_call_zdnn_maxpool2d() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32> 
  %shape = memref.alloc() : memref<6xi64>
  "zlow.maxpool2d"(%input, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "SAME_PADDING" } : (memref<2048xf32>, memref<6xi64>, memref<2048xf32>)-> ()
  return

  // CHECK-LABEL: test_call_zdnn_maxpool2d
  // CHECK: {{.*}} = llvm.call @zdnn_maxpool2d(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, i64, i64, i64, i64, i64, !llvm.ptr<i8>) -> i32
}


// -----

// Check whether meanreduce2d calls the correct zDNN API or not.
func @test_call_zdnn_meanreduce2d() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32>
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32>
  %shape = memref.alloc() : memref<4xi64>
  "zlow.meanreduce2d"(%input, %shape, %output) : (memref<2048xf32>, memref<4xi64>, memref<2048xf32>)-> ()
  return

  // CHECK-LABEL: test_call_zdnn_meanreduce2d
  // CHECK: {{.*}} = llvm.call @zdnn_meanreduce2d(%{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
}

// -----

// Check whether batchnorm calls the correct zDNN API or not.
func @test_call_zdnn_batchnorm() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32>
  %a = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32>
  %b = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32>
  %shape = memref.alloc() : memref<4xi64>
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf32>
  "zlow.batchnorm"(%input, %a, %b, %shape, %output) : (memref<2048xf32>, memref<2048xf32>, memref<2048xf32>, memref<4xi64>, memref<2048xf32>)-> ()
  return

  // CHECK-LABEL: test_call_zdnn_batchnorm
  // CHECK: {{.*}} = llvm.call @zdnn_batchnorm(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>, !llvm.ptr<i8>) -> i32
}
