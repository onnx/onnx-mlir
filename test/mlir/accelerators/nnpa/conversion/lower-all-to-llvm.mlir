// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-krnl-to-llvm %s -split-input-file | FileCheck %s

// -----

func.func @test_lower_both_zlow_and_krnl() -> () {
  %0 = memref.alloc() : memref<10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %2 = "krnl.global"() {name = "constant_0", shape = [1, 2], value = dense<[[0., 1.0]]> : tensor<1x2xf32>} : () -> memref<1x2xf32>
  "zlow.stick"(%0, %1) : (memref<10x10xf32>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-DAG: llvm.mlir.global internal constant @{{.*}}(dense<{{\[}}[0.000000e+00, 1.000000e+00]{{\]}}> : tensor<1x2xf32>) {addr_space = 0 : i32, alignment = 16 : i64} : !llvm.array<1 x array<2 x f32>>
}

// -----

func.func @test_stick() -> () {
  %0 = memref.alloc() : memref<10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  "zlow.stick"(%0, %1) : (memref<10x10xf32>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_stick
  // CHECK: [[UNSTICKIFIED_MEMREF:%.+]] = llvm.insertvalue {{.*}}, {{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[STICKIFIED_MEMREF:%.+]] = llvm.insertvalue {{.*}}, {{.*}}[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>

  // CHECK: [[ALIGNED_BUFFER:%.+]] = llvm.extractvalue [[STICKIFIED_MEMREF]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  // CHECK: [[ALIGNED_BUFFER_I8PTR:%.+]] = llvm.bitcast [[ALIGNED_BUFFER]] : !llvm.ptr to !llvm.ptr

  // CHECK: [[PRE_TRANSFORMED_DESC:%.+]] = llvm.alloca {{.*}} x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
  // CHECK: [[DATA_LAYOUT:%.+]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: [[DATA_TYPE:%.+]] = llvm.mlir.constant(255 : i64) : i64
  // CHECK: [[PRE_TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[PRE_TRANSFORMED_DESC]] : !llvm.ptr to !llvm.ptr
  // CHECK: llvm.call @zdnn_init_pre_transformed_desc([[DATA_LAYOUT]], [[DATA_TYPE]], [[PRE_TRANSFORMED_DESC_I8PTR]], {{.*}}, {{.*}}) : (i64, i64, !llvm.ptr, i64, i64) -> ()

  // CHECK: [[TRANSFORMED_DESC:%.+]] = llvm.alloca {{.*}} x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
  // CHECK: [[PRE_TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[PRE_TRANSFORMED_DESC]] : !llvm.ptr to !llvm.ptr
  // CHECK: [[TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[TRANSFORMED_DESC]] : !llvm.ptr to !llvm.ptr
  // CHECK: {{.*}} = llvm.call @zdnn_generate_transformed_desc([[PRE_TRANSFORMED_DESC_I8PTR]], [[TRANSFORMED_DESC_I8PTR]]) : (!llvm.ptr, !llvm.ptr) -> i32

  // CHECK: [[ZTENSOR:%.+]] = llvm.alloca {{.*}} x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
  // CHECK: [[TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[TRANSFORMED_DESC]] : !llvm.ptr to !llvm.ptr
  // CHECK: [[BUFFER_SIZE:%.+]] = llvm.call @zdnn_getsize_ztensor([[TRANSFORMED_DESC_I8PTR]]) : (!llvm.ptr) -> i64
  // CHECK: [[ZTENSOR_PRE_TRANSFORMED_DESC:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
  // CHECK: llvm.store [[PRE_TRANSFORMED_DESC]], [[ZTENSOR_PRE_TRANSFORMED_DESC]] : !llvm.ptr, !llvm.ptr

  // CHECK: [[ZTENSOR_TRANSFORMED_DESC:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
  // CHECK: llvm.store [[TRANSFORMED_DESC]], [[ZTENSOR_TRANSFORMED_DESC]] : !llvm.ptr, !llvm.ptr 

  // CHECK: [[ZTENSOR_BUFFER_SIZE:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
  // CHECK: llvm.store [[BUFFER_SIZE]], [[ZTENSOR_BUFFER_SIZE]] : i64, !llvm.ptr 

  // CHECK: [[ZTENSOR_BUFFER:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
  // CHECK: llvm.store [[ALIGNED_BUFFER_I8PTR]], [[ZTENSOR_BUFFER]] : !llvm.ptr, !llvm.ptr 

  // CHECK: [[FALSE:%.+]] = llvm.mlir.constant(false) : i1

  // CHECK: [[IS_TRANSFORMED:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
  // CHECK: llvm.store [[FALSE]], [[IS_TRANSFORMED]] : i1, !llvm.ptr 

  // CHECK: [[UNSTICKIFIED:%.+]] = llvm.extractvalue [[UNSTICKIFIED_MEMREF]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[UNSTICKIFIED_I8PTR:%.+]] = llvm.bitcast [[UNSTICKIFIED]] : !llvm.ptr to !llvm.ptr
  // CHECK: [[ZTENSOR_I8PTR:%.+]] = llvm.bitcast [[ZTENSOR]] : !llvm.ptr to !llvm.ptr
  // CHECK: {{.*}} = llvm.call @zdnn_transform_ztensor([[ZTENSOR_I8PTR]], [[UNSTICKIFIED_I8PTR]]) vararg(!llvm.func<i32 (ptr, ptr, ...)>) : (!llvm.ptr, !llvm.ptr) -> i32

  // CHECK: llvm.return
}

// -----

func.func @test_unstick() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<10x10xf32>
  "zlow.unstick"(%0, %1) : (memref<1x1x32x64xf16>, memref<10x10xf32>) -> ()
  return

  // CHECK-LABEL: test_unstick
  // CHECK: [[STICKIFIED_MEMREF:%.+]] = llvm.insertvalue {{.*}}, {{.*}}[4, 3] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  // CHECK: [[UNSTICKIFIED_MEMREF:%.+]] = llvm.insertvalue {{.*}}, {{.*}}[4, 1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>

  // CHECK: [[ALIGNED_BUFFER:%.+]] = llvm.extractvalue [[STICKIFIED_MEMREF]][1] : !llvm.struct<(ptr, ptr, i64, array<4 x i64>, array<4 x i64>)>
  // CHECK: [[ALIGNED_BUFFER_I8PTR:%.+]] = llvm.bitcast [[ALIGNED_BUFFER]] : !llvm.ptr to !llvm.ptr

  // CHECK: [[PRE_TRANSFORMED_DESC:%.+]] = llvm.alloca {{.*}} x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
  // CHECK: [[DATA_LAYOUT:%.+]] = llvm.mlir.constant(1 : i64) : i64
  // CHECK: [[DATA_TYPE:%.+]] = llvm.mlir.constant(255 : i64) : i64
  // CHECK: [[PRE_TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[PRE_TRANSFORMED_DESC]] : !llvm.ptr to !llvm.ptr
  // CHECK: llvm.call @zdnn_init_pre_transformed_desc([[DATA_LAYOUT]], [[DATA_TYPE]], [[PRE_TRANSFORMED_DESC_I8PTR]], {{.*}}, {{.*}}) : (i64, i64, !llvm.ptr, i64, i64) -> ()

  // CHECK: [[TRANSFORMED_DESC:%.+]] = llvm.alloca {{.*}} x !llvm.struct<(i32, i32, i32, i32, i32, i32, i32)> : (i64) -> !llvm.ptr
  // CHECK: [[PRE_TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[PRE_TRANSFORMED_DESC]] : !llvm.ptr to !llvm.ptr
  // CHECK: [[TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[TRANSFORMED_DESC]] : !llvm.ptr to !llvm.ptr
  // CHECK: {{.*}} = llvm.call @zdnn_generate_transformed_desc([[PRE_TRANSFORMED_DESC_I8PTR]], [[TRANSFORMED_DESC_I8PTR]]) : (!llvm.ptr, !llvm.ptr) -> i32

  // CHECK: [[ZTENSOR:%.+]] = llvm.alloca {{.*}} x !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)> : (i64) -> !llvm.ptr
  // CHECK: [[TRANSFORMED_DESC_I8PTR:%.+]] = llvm.bitcast [[TRANSFORMED_DESC]] : !llvm.ptr to !llvm.ptr
  // CHECK: [[BUFFER_SIZE:%.+]] = llvm.call @zdnn_getsize_ztensor([[TRANSFORMED_DESC_I8PTR]]) : (!llvm.ptr) -> i64
  // CHECK: [[ZTENSOR_PRE_TRANSFORMED_DESC:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}0, 0] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
  // CHECK: llvm.store [[PRE_TRANSFORMED_DESC]], [[ZTENSOR_PRE_TRANSFORMED_DESC]] : !llvm.ptr, !llvm.ptr

  // CHECK: [[ZTENSOR_TRANSFORMED_DESC:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}0, 1] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
  // CHECK: llvm.store [[TRANSFORMED_DESC]], [[ZTENSOR_TRANSFORMED_DESC]] : !llvm.ptr, !llvm.ptr

  // CHECK: [[ZTENSOR_BUFFER_SIZE:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}0, 2] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
  // CHECK: llvm.store [[BUFFER_SIZE]], [[ZTENSOR_BUFFER_SIZE]] : i64, !llvm.ptr

  // CHECK: [[ZTENSOR_BUFFER:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}0, 3] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
  // CHECK: llvm.store [[ALIGNED_BUFFER_I8PTR]], [[ZTENSOR_BUFFER]] : !llvm.ptr, !llvm.ptr

  // CHECK: [[TRUE:%.+]] = llvm.mlir.constant(true) : i1

  // CHECK: [[IS_TRANSFORMED:%.+]] = llvm.getelementptr [[ZTENSOR]]{{\[}}0, 4] : (!llvm.ptr) -> !llvm.ptr, !llvm.struct<(ptr, ptr, i64, ptr, i1, array<3 x i8>, f32, f32, array<20 x i8>)>
  // CHECK: llvm.store [[TRUE]], [[IS_TRANSFORMED]] : i1, !llvm.ptr

  // CHECK: [[UNSTICKIFIED:%.+]] = llvm.extractvalue [[UNSTICKIFIED_MEMREF]][1] : !llvm.struct<(ptr, ptr, i64, array<2 x i64>, array<2 x i64>)>
  // CHECK: [[UNSTICKIFIED_I8PTR:%.+]] = llvm.bitcast [[UNSTICKIFIED]] : !llvm.ptr to !llvm.ptr
  // CHECK: [[ZTENSOR_I8PTR:%.+]] = llvm.bitcast [[ZTENSOR]] : !llvm.ptr to !llvm.ptr
  // CHECK: {{.*}} = llvm.call @zdnn_transform_origtensor([[ZTENSOR_I8PTR]], [[UNSTICKIFIED_I8PTR]]) : (!llvm.ptr, !llvm.ptr) -> i32

  // CHECK: llvm.return
}

// -----

// Check whether the lowering of zlow.relu calls the correct zDNN API or not.
func.func @test_call_zdnn_relu() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.relu"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_relu
  // CHECK: {{.*}} = llvm.call @zdnn_relu_ext({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.tanh calls the correct zDNN API or not.
func.func @test_call_zdnn_tanh() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.tanh"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_tanh
  // CHECK: {{.*}} = llvm.call @zdnn_tanh_ext({{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.sigmoid calls the correct zDNN API or not.
func.func @test_call_zdnn_sigmoid() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.sigmoid"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_sigmoid
  // CHECK: {{.*}} = llvm.call @zdnn_sigmoid_ext({{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.add calls the correct zDNN API or not.
func.func @test_call_zdnn_add() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %2 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.add"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_add
  // CHECK: {{.*}} = llvm.call @zdnn_add_ext({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.sub calls the correct zDNN API or not.
func.func @test_call_zdnn_sub() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %2 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.sub"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_sub
  // CHECK: {{.*}} = llvm.call @zdnn_sub_ext({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.mul calls the correct zDNN API or not.
func.func @test_call_zdnn_mul() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %2 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.mul"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_mul
  // CHECK: {{.*}} = llvm.call @zdnn_mul_ext({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.div calls the correct zDNN API or not.
func.func @test_call_zdnn_div() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %2 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.div"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_div
  // CHECK: {{.*}} = llvm.call @zdnn_div_ext({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.softmax calls the correct zDNN API or not.
func.func @test_call_zdnn_softmax() -> () {
  %0 = memref.alloc() : memref<1x1x1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x1x1x32x64xf16>
  %work_area = memref.alloc() {alignment = 4096 : i64} : memref<8192xi8>
  %shape = memref.alloc() : memref<3xi64>
  "zlow.softmax"(%0, %work_area, %shape, %1) {act_func = "ACT_NONE"} : (memref<1x1x1x1x32x64xf16>, memref<8192xi8>, memref<3xi64>, memref<1x1x1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_softmax
  // CHECK: {{.*}} = llvm.call @zdnn_softmax_ext({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
}

// -----

// COM: Check whether the lowering of zlow.stickForLSTM calls the correct zDNN API or not.
// COM: We should call zdnn_transform_ztensor with zTensor and four pointers to the buffers fori F, I, C, and O gates. 
func.func @test_stick_for_lstm() -> () {
  %f = memref.alloc() : memref<1x10x10xf32>
  %i = memref.alloc() : memref<1x10x10xf32>
  %c = memref.alloc() : memref<1x10x10xf32>
  %o = memref.alloc() : memref<1x10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  "zlow.stickForLSTM"(%f, %i, %c, %o, %1) : (memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_stick_for_lstm
  // CHECK: call @zdnn_transform_ztensor({{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
}

// -----

// COM: Check whether the lowering of zlow.stickForGRU calls the correct zDNN API or not.
// COM: We should call zdnn_transform_ztensor with zTensor and three pointers to the buffers for Z, R, and H gates. 
func.func @test_stick_for_gru() -> () {
  %g = memref.alloc() : memref<1x10x10xf32>
  %r = memref.alloc() : memref<1x10x10xf32>
  %h = memref.alloc() : memref<1x10x10xf32>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  "zlow.stickForGRU"(%g, %r, %h, %1) : (memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x10x10xf32>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_stick_for_gru
  // CHECK: call @zdnn_transform_ztensor({{.*}}, {{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.min calls the correct zDNN API or not.
func.func @test_call_zdnn_min() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %2 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.min"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_min
  // CHECK: {{.*}} = llvm.call @zdnn_min_ext({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.max calls the correct zDNN API or not.
func.func @test_call_zdnn_max() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %2 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.max"(%0, %1, %shape, %2) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_max
  // CHECK: {{.*}} = llvm.call @zdnn_max_ext({{.*}}, {{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.exp calls the correct zDNN API or not.
func.func @test_call_zdnn_exp() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.exp"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_exp
  // CHECK: {{.*}} = llvm.call @zdnn_exp_ext({{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.log calls the correct zDNN API or not.
func.func @test_call_zdnn_log() -> () {
  %0 = memref.alloc() : memref<1x1x32x64xf16>
  %1 = memref.alloc() : memref<1x1x32x64xf16>
  %shape = memref.alloc() : memref<2xi64>
  "zlow.log"(%0, %shape, %1) {layout = "2D"} : (memref<1x1x32x64xf16>, memref<2xi64>, memref<1x1x32x64xf16>) -> ()
  return

  // CHECK-LABEL: test_call_zdnn_log
  // CHECK: {{.*}} = llvm.call @zdnn_log_ext({{.*}}, {{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.matmul calls the correct zDNN API or not.
func.func @test_matmul_no_bcast_unstacked(%x: memref<2048xf16>,%y: memref<2048xf16>,%bias: memref<2048xf16>, %shape: memref<3xi64>) -> memref<2048xf16> {
  %res = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  "zlow.matmul"(%x, %y, %bias, %shape, %res) {is_bcast1 = 0 : si64, is_bcast23 = 0 : si64, is_stacked = 0 : si64} : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<3xi64>, memref<2048xf16>) -> ()
  return %res : memref<2048xf16>
  // CHECK-LABEL: test_matmul_no_bcast_unstacked
  // CHECK: %{{.*}} = llvm.call @zdnn_matmul_op_ext(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.matmul calls the correct zDNN API or not.
func.func @test_matmul_no_bcast_stacked(%x: memref<2048xf16>,%y: memref<2048xf16>,%bias: memref<2048xf16>, %shape: memref<3xi64>) -> memref<2048xf16> {
  %res = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  "zlow.matmul"(%x, %y, %bias, %shape, %res) {is_bcast1 = 0 : si64, is_bcast23 = 0 : si64, is_stacked = -1 : si64} : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<3xi64>, memref<2048xf16>) -> ()
  return %res : memref<2048xf16>
  // CHECK-LABEL: test_matmul_no_bcast_stacked
  // CHECK: %{{.*}} = llvm.call @zdnn_matmul_op_ext(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.matmul calls the correct zDNN API or not.
func.func @test_matmul_bcast_stacked(%x: memref<2048xf16>,%y: memref<2048xf16>,%bias: memref<2048xf16>, %shape: memref<3xi64>) -> memref<2048xf16> {
  %res = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  "zlow.matmul"(%x, %y, %bias, %shape, %res) {is_bcast1 = 0 : si64, is_bcast23 = -1 : si64, is_stacked = -1 : si64} : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<3xi64>, memref<2048xf16>) -> ()
  return %res : memref<2048xf16>
  // CHECK-LABEL: test_matmul_bcast_stacked
  // CHECK: %{{.*}} = llvm.call @zdnn_matmul_bcast_op_ext(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
}

// -----

// Check whether the lowering of zlow.matmul calls the correct zDNN API or not.
func.func @test_matmul_bcast_unstacked(%x: memref<2048xf16>,%y: memref<2048xf16>,%bias: memref<2048xf16>, %shape: memref<3xi64>) -> memref<2048xf16> {
  %res = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  "zlow.matmul"(%x, %y, %bias, %shape, %res) {is_bcast1 = 0 : si64, is_bcast23 = -1 : si64, is_stacked = 0 : si64} : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<3xi64>, memref<2048xf16>) -> ()
  return %res : memref<2048xf16>
  // CHECK-LABEL: test_matmul_bcast_unstacked
  // CHECK: %{{.*}} = llvm.call @zdnn_matmul_bcast_op_ext(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, !llvm.ptr) -> i32
}

// -----

// Check whether conv2d calls the correct zDNN API or not.
func.func @test_call_zdnn_cond2d() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %kernel = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %bias = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %shape = memref.alloc() : memref<7xi64>
  "zlow.conv2d"(%input, %kernel, %bias, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "SAME_PADDING", act_func = "ACT_NONE" } : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<7xi64>, memref<2048xf16>)-> ()
  return

  // CHECK-LABEL: test_call_zdnn_cond2d
  // CHECK: {{.*}} = llvm.call @zdnn_conv2d(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr) -> i32 
}

// -----

// Check whether conv2d calls the correct zDNN API or not.
func.func @test_call_zdnn_cond2d_valid_padding() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %kernel = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %bias = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %shape = memref.alloc() : memref<7xi64>
  "zlow.conv2d"(%input, %kernel, %bias, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "VALID_PADDING", act_func = "ACT_NONE" } : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<7xi64>, memref<2048xf16>)-> ()
  return

  // CHECK-LABEL: test_call_zdnn_cond2d_valid_padding
  // CHECK: {{.*}} = llvm.call @zdnn_conv2d(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr) -> i32 
}

// -----

// Check whether conv2d calls the correct zDNN API or not.
func.func @test_call_zdnn_cond2d_relu_act() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %kernel = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %bias = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %shape = memref.alloc() : memref<7xi64>
  "zlow.conv2d"(%input, %kernel, %bias, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "SAME_PADDING", act_func = "ACT_RELU" } : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<7xi64>, memref<2048xf16>)-> ()
  return

  // CHECK-LABEL: test_call_zdnn_cond2d_relu_act
  // CHECK: {{.*}} = llvm.call @zdnn_conv2d(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, i64, i64, i64, i64, !llvm.ptr, !llvm.ptr) -> i32 
}

// -----

// Check whether avgpool2d calls the correct zDNN API or not.
func.func @test_call_zdnn_avgpool2d() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %shape = memref.alloc() : memref<6xi64>
  "zlow.avgpool2d"(%input, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "SAME_PADDING" } : (memref<2048xf16>, memref<6xi64>, memref<2048xf16>)-> ()
  return

  // CHECK-LABEL: test_call_zdnn_avgpool2d
  // CHECK: {{.*}} = llvm.call @zdnn_avgpool2d(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> i32
}

// -----

// Check whether maxpool2d calls the correct zDNN API or not.
func.func @test_call_zdnn_maxpool2d() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16> 
  %shape = memref.alloc() : memref<6xi64>
  "zlow.maxpool2d"(%input, %shape, %output) {kernel_shape = [5, 5], strides = [2, 2], padding_type = "SAME_PADDING" } : (memref<2048xf16>, memref<6xi64>, memref<2048xf16>)-> ()
  return

  // CHECK-LABEL: test_call_zdnn_maxpool2d
  // CHECK: {{.*}} = llvm.call @zdnn_maxpool2d(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, i64, i64, i64, i64, i64, !llvm.ptr) -> i32
}


// -----

// Check whether meanreduce2d calls the correct zDNN API or not.
func.func @test_call_zdnn_meanreduce2d() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16>
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16>
  %shape = memref.alloc() : memref<4xi64>
  "zlow.meanreduce2d"(%input, %shape, %output) : (memref<2048xf16>, memref<4xi64>, memref<2048xf16>)-> ()
  return

  // CHECK-LABEL: test_call_zdnn_meanreduce2d
  // CHECK: {{.*}} = llvm.call @zdnn_meanreduce2d(%{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr) -> i32
}

// -----

// Check whether batchnorm calls the correct zDNN API or not.
func.func @test_call_zdnn_batchnorm() -> () {
  %input = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16>
  %a = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16>
  %b = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16>
  %shape = memref.alloc() : memref<4xi64>
  %output = memref.alloc() {alignment = 4096 : i64} : memref<2048xf16>
  "zlow.batchnorm"(%input, %a, %b, %shape, %output) : (memref<2048xf16>, memref<2048xf16>, memref<2048xf16>, memref<4xi64>, memref<2048xf16>)-> ()
  return

  // CHECK-LABEL: test_call_zdnn_batchnorm
  // CHECK: {{.*}} = llvm.call @zdnn_batchnorm(%{{.*}}, %{{.*}}, %{{.*}}, %{{.*}}) : (!llvm.ptr, !llvm.ptr, !llvm.ptr, !llvm.ptr) -> i32
}
