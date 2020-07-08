// RUN: onnx-mlir-opt --shape-inference --lower-frontend %s -split-input-file | FileCheck %s

// -----

func @test_add_add(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg1) : (tensor<*xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_add_add
  /// First Add
  // CHECK: [[RET_RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[ADDF:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[ADDF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>

  /// Second Add
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[ADDF:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[ADDF]], [[RET_RES]][%arg2, %arg3] : memref<10x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<10x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<10x10xf32>

  // CHECK: return [[RET_RES]] : memref<10x10xf32>
}

// -----

func @test_mul_mul(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Mul"(%0, %arg1) : (tensor<*xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_mul_mul
  /// First Mul
  // CHECK: [[RET_RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MULF:%.+]] = mulf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[MULF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>

  /// Second Mul
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MULF:%.+]] = mulf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[MULF]], [[RET_RES]][%arg2, %arg3] : memref<10x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<10x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<10x10xf32>

  // CHECK: return [[RET_RES]] : memref<10x10xf32>
}

// -----

func @test_div_div(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Div"(%0, %arg1) : (tensor<*xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_div_div
  /// First Div
  // CHECK: [[RET_RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[DIVF:%.+]] = divf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[DIVF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>

  /// Second Div
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[DIVF:%.+]] = divf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[DIVF]], [[RET_RES]][%arg2, %arg3] : memref<10x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<10x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<10x10xf32>

  // CHECK: return [[RET_RES]] : memref<10x10xf32>
}

// -----

func @test_sub_sub(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Sub"(%0, %arg1) : (tensor<*xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sub_sub
  /// First Sub
  // CHECK: [[RET_RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[SUBF:%.+]] = subf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[SUBF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>

  /// Second Sub
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[SUBF:%.+]] = subf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[SUBF]], [[RET_RES]][%arg2, %arg3] : memref<10x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<10x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<10x10xf32>

  // CHECK: return [[RET_RES]] : memref<10x10xf32>
}

// -----

func @test_and_and(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<*xi1> {
  %0 = "onnx.And"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  %1 = "onnx.And"(%0, %arg1) : (tensor<*xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  "std.return"(%1) : (tensor<*xi1>) -> ()

  // CHECK-LABEL: test_and_and
  /// First And
  // CHECK: [[RET_RES:%.+]] = alloc() : memref<10x10xi1>
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xi1>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[AND:%.+]] = and [[LOAD1]], [[LOAD2]] : i1
  // CHECK: affine.store [[AND]], [[RES]][%arg2, %arg3] : memref<10x10xi1>

  /// Second And
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load [[RES]][%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[AND:%.+]] = and [[LOAD1]], [[LOAD2]] : i1
  // CHECK: affine.store [[AND]], [[RET_RES]][%arg2, %arg3] : memref<10x10xi1>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<10x10xi1>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<10x10xi1>

  // CHECK: return [[RET_RES]] : memref<10x10xi1>
}

// -----

func @test_or_or(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<*xi1> {
  %0 = "onnx.Or"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  %1 = "onnx.Or"(%0, %arg1) : (tensor<*xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  "std.return"(%1) : (tensor<*xi1>) -> ()

  // CHECK-LABEL: test_or_or
  /// First Or
  // CHECK: [[RET_RES:%.+]] = alloc() : memref<10x10xi1>
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xi1>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[OR:%.+]] = or [[LOAD1]], [[LOAD2]] : i1
  // CHECK: affine.store [[OR]], [[RES]][%arg2, %arg3] : memref<10x10xi1>

  /// Second Or
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load [[RES]][%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[OR:%.+]] = or [[LOAD1]], [[LOAD2]] : i1
  // CHECK: affine.store [[OR]], [[RET_RES]][%arg2, %arg3] : memref<10x10xi1>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<10x10xi1>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<10x10xi1>

  // CHECK: return [[RET_RES]] : memref<10x10xi1>
}

// -----

func @test_xor_xor(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<*xi1> {
  %0 = "onnx.Xor"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  %1 = "onnx.Xor"(%0, %arg1) : (tensor<*xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  "std.return"(%1) : (tensor<*xi1>) -> ()

  // CHECK-LABEL: test_xor_xor
  /// First Xor
  // CHECK: [[RET_RES:%.+]] = alloc() : memref<10x10xi1>
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xi1>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[XOR:%.+]] = xor [[LOAD1]], [[LOAD2]] : i1
  // CHECK: affine.store [[XOR]], [[RES]][%arg2, %arg3] : memref<10x10xi1>

  /// Second Xor
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load [[RES]][%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[XOR:%.+]] = xor [[LOAD1]], [[LOAD2]] : i1
  // CHECK: affine.store [[XOR]], [[RET_RES]][%arg2, %arg3] : memref<10x10xi1>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<10x10xi1>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<10x10xi1>

  // CHECK: return [[RET_RES]] : memref<10x10xi1>
}

// -----

func @test_exp_exp(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Exp"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Exp"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_exp_exp
  /// First Exp
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: affine.store [[EXP]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  
  /// Second Exp
  // CHECK: [[C0_1:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], [[C0_1]] : memref<?x10xf32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_2:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], [[C0_2]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: affine.store [[EXP]], [[RET_RES]][%arg1, %arg2] : memref<?x10xf32>
  
  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xf32>

  // CHECK: return [[RET_RES]] : memref<?x10xf32>
}

// -----

func @test_tanh_tanh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Tanh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Tanh"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_tanh_tanh
  /// First Tanh
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[NLOAD:%.+]] = subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = exp [[NLOAD]] : f32
  // CHECK: [[DIVIDEND:%.+]] = subf [[EXP]], [[NEXP]] : f32
  // CHECK: [[DIVISOR:%.+]] = addf [[EXP]], [[NEXP]] : f32
  // CHECK: [[TANH:%.+]] = divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK: affine.store [[TANH]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  
  /// Second Tanh
  // CHECK: [[C0_1:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], [[C0_1]] : memref<?x10xf32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_2:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], [[C0_2]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[NLOAD:%.+]] = subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = exp [[NLOAD]] : f32
  // CHECK: [[DIVIDEND:%.+]] = subf [[EXP]], [[NEXP]] : f32
  // CHECK: [[DIVISOR:%.+]] = addf [[EXP]], [[NEXP]] : f32
  // CHECK: [[TANH_RES:%.+]] = divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK: affine.store [[TANH_RES]], [[RET_RES]][%arg1, %arg2] : memref<?x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xf32>

  // CHECK: return [[RET_RES]] : memref<?x10xf32>
}

// -----

func @test_sinh_sinh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sinh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Sinh"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sinh_sinh
  /// First Sinh
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[TWO:%.+]] = constant {{2.+}} : f32
  // CHECK: [[NLOAD:%.+]] = subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = exp [[NLOAD]] : f32
  // CHECK: [[DIVIDEND:%.+]] = subf [[EXP]], [[NEXP]] : f32
  // CHECK: [[SINH_RES:%.+]] = divf [[DIVIDEND]], [[TWO]] : f32
  // CHECK: affine.store [[SINH_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  
  /// Second Sinh
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], [[C0_0]] : memref<?x10xf32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_2:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], [[C0_2]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[TWO:%.+]] = constant {{2.+}} : f32
  // CHECK: [[NLOAD:%.+]] = subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = exp [[NLOAD]] : f32
  // CHECK: [[DIVIDEND:%.+]] = subf [[EXP]], [[NEXP]] : f32
  // CHECK: [[SINH_RES:%.+]] = divf [[DIVIDEND]], [[TWO]] : f32
  // CHECK: affine.store [[SINH_RES]], [[RET_RES]][%arg1, %arg2] : memref<?x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xf32>

  // CHECK: return [[RET_RES]] : memref<?x10xf32>
}

// -----

func @test_cosh_cosh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Cosh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Cosh"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_cosh_cosh
  /// First Cosh
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[TWO:%.+]] = constant {{2.+}} : f32
  // CHECK: [[NLOAD:%.+]] = subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = exp [[NLOAD]] : f32
  // CHECK: [[DIVIDEND:%.+]] = addf [[EXP]], [[NEXP]] : f32
  // CHECK: [[COSH_RES:%.+]] = divf [[DIVIDEND]], [[TWO]] : f32
  // CHECK: affine.store [[COSH_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  
  /// Second Cosh
  // CHECK: [[C0_1:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], [[C0_1]] : memref<?x10xf32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_2:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], [[C0_2]] : memref<?x10xf32>
  // CHECK: [[LOAD:%.+]] = affine.load [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[TWO:%.+]] = constant {{2.+}} : f32
  // CHECK: [[NLOAD:%.+]] = subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = exp [[NLOAD]] : f32
  // CHECK: [[DIVIDEND:%.+]] = addf [[EXP]], [[NEXP]] : f32
  // CHECK: [[COSH_RES:%.+]] = divf [[DIVIDEND]], [[TWO]] : f32
  // CHECK: affine.store [[COSH_RES]], [[RET_RES]][%arg1, %arg2] : memref<?x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xf32>

  // CHECK: return [[RET_RES]] : memref<?x10xf32>
}

// -----

func @test_sigmoid_sigmoid(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Sigmoid"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sigmoid_sigmoid
  /// First Sigmoid
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[NLOAD:%.+]] = subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = exp [[NLOAD]] : f32
  // CHECK: [[DIVISOR:%.+]] = addf [[ONE]], [[NEXP]] : f32
  // CHECK: [[SIGMOID_RES:%.+]] = divf [[ONE]], [[DIVISOR]] : f32
  // CHECK: affine.store [[SIGMOID_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  
  /// Second Sigmoid
  // CHECK: [[C0_1:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], [[C0_1]] : memref<?x10xf32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_2:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], [[C0_2]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[NLOAD:%.+]] = subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = exp [[NLOAD]] : f32
  // CHECK: [[DIVISOR:%.+]] = addf [[ONE]], [[NEXP]] : f32
  // CHECK: [[SIGMOID_RES:%.+]] = divf [[ONE]], [[DIVISOR]] : f32
  // CHECK: affine.store [[SIGMOID_RES]], [[RET_RES]][%arg1, %arg2] : memref<?x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xf32>

  // CHECK: return [[RET_RES]] : memref<?x10xf32>
}

// -----

func @test_relu_relu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Relu"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_relu_relu
  /// First Relu
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[LTZERO:%.+]] = cmpf "olt", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[RELU_RES:%.+]] = select [[LTZERO]], [[ZERO]], [[LOAD]] : f32
  // CHECK: affine.store [[RELU_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>

  /// Second Relu
  // CHECK: [[C0_1:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], [[C0_1]] : memref<?x10xf32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_2:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], [[C0_2]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[LTZERO:%.+]] = cmpf "olt", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[RELU_RES:%.+]] = select [[LTZERO]], [[ZERO]], [[LOAD]] : f32
  // CHECK: affine.store [[RELU_RES]], [[RET_RES]][%arg1, %arg2] : memref<?x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xf32>

  // CHECK: return [[RET_RES]] : memref<?x10xf32>
}

// -----

func @test_sum_sum(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Sum"(%0, %arg1) : (tensor<*xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sum_sum
  /// First Sum
  // CHECK: [[RET_RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[ADD:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[ADD]], [[RES]][%arg2, %arg3] : memref<10x10xf32>

  /// Second Sum
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[ADD:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[ADD]], [[RET_RES]][%arg2, %arg3] : memref<10x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<10x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<10x10xf32>

  // CHECK: return [[RET_RES]] : memref<10x10xf32>
}

// -----

func @test_max_max(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Max"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Max"(%0, %arg1) : (tensor<*xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_max_max
  /// First Max
  // CHECK: [[RET_RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MAX:%.+]] = cmpf "ogt", [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[RELU_RES:%.+]] = select [[MAX]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[RELU_RES]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  
  /// Second Max
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MAX:%.+]] = cmpf "ogt", [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[RELU_RES:%.+]] = select [[MAX]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[RELU_RES]], [[RET_RES]][%arg2, %arg3] : memref<10x10xf32>
  
  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<10x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<10x10xf32>
  
  // CHECK: return [[RET_RES]] : memref<10x10xf32>
}

// -----

func @test_min_min(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Min"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Min"(%0, %arg1) : (tensor<*xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_min_min
  /// First Min
  // CHECK: [[RET_RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MIN:%.+]] = cmpf "olt", [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[RELU_RES:%.+]] = select [[MIN]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[RELU_RES]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  
  /// Second Min
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MIN:%.+]] = cmpf "olt", [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[RELU_RES:%.+]] = select [[MIN]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[RELU_RES]], [[RET_RES]][%arg2, %arg3] : memref<10x10xf32>
  
  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<10x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<10x10xf32>
  
  // CHECK: return [[RET_RES]] : memref<10x10xf32>
}

// -----

func @test_elu_elu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Elu"(%arg0) {alpha=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Elu"(%0) {alpha=2.0:f32} : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_elu_elu
  /// First Elu
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32 
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32 
  // CHECK: [[ALPHA:%.+]] = constant {{2.+}} : f32 
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[CMP:%.+]] = cmpf "olt", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[SUB:%.+]] = subf [[EXP]], [[ONE]] : f32
  // CHECK: [[MUL:%.+]] = mulf [[ALPHA]], [[SUB]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[MUL]], [[LOAD]] : f32
  // CHECK: affine.store [[SELECT]], [[RES]][%arg1, %arg2] : memref<?x10xf32>

  /// Second Elu
  // CHECK: [[C0_1:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], [[C0_1]] : memref<?x10xf32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_2:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], [[C0_2]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32 
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32 
  // CHECK: [[ALPHA:%.+]] = constant {{2.+}} : f32 
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[CMP:%.+]] = cmpf "olt", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[SUB:%.+]] = subf [[EXP]], [[ONE]] : f32
  // CHECK: [[MUL:%.+]] = mulf [[ALPHA]], [[SUB]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[MUL]], [[LOAD]] : f32
  // CHECK: affine.store [[SELECT]], [[RET_RES]][%arg1, %arg2] : memref<?x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xf32>

  // CHECK: return [[RET_RES]] : memref<?x10xf32>
}

// -----

func @test_leakyrelu_leakyrelu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha=1.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.LeakyRelu"(%0) {alpha=1.0:f32} : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_leakyrelu_leakyrelu
  /// First LeakyRelu
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32 
  // CHECK: [[ALPHA:%.+]] = constant {{1.+}} : f32 
  // CHECK: [[CMP:%.+]] = cmpf "olt", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[MUL:%.+]] = mulf [[ALPHA]], [[LOAD]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[MUL]], [[LOAD]] : f32
  // CHECK: affine.store [[SELECT]], [[RES]][%arg1, %arg2] : memref<?x10xf32>

  /// Second LeakyRelu
  // CHECK: [[C0_1:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], [[C0_1]] : memref<?x10xf32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_2:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], [[C0_2]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32 
  // CHECK: [[ALPHA:%.+]] = constant {{1.+}} : f32 
  // CHECK: [[CMP:%.+]] = cmpf "olt", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[MUL:%.+]] = mulf [[ALPHA]], [[LOAD]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[MUL]], [[LOAD]] : f32
  // CHECK: affine.store [[SELECT]], [[RET_RES]][%arg1, %arg2] : memref<?x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xf32>

  // CHECK: return [[RET_RES]] : memref<?x10xf32>
}

// -----

func @test_selu_selu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Selu"(%arg0) {alpha=1.0:f32, gamma=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Selu"(%0) {alpha=1.0:f32, gamma=2.0:f32} : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_selu_selu
  /// First Selu
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32 
  // CHECK: [[ALPHA:%.+]] = constant {{1.+}} : f32 
  // CHECK: [[GAMMA:%.+]] = constant {{2.+}} : f32 
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[CMP:%.+]] = cmpf "ogt", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[MUL:%.+]] = mulf [[ALPHA]], [[EXP]] : f32
  // CHECK: [[SUB:%.+]] = subf [[MUL]], [[ALPHA]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[LOAD]], [[SUB]] : f32
  // CHECK: [[SELU_RES:%.+]] = mulf [[GAMMA]], [[SELECT]] : f32
  // CHECK: affine.store [[SELU_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>

  /// Second Selu
  // CHECK: [[C0_1:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], [[C0_1]] : memref<?x10xf32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_2:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], [[C0_2]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32 
  // CHECK: [[ALPHA:%.+]] = constant {{1.+}} : f32 
  // CHECK: [[GAMMA:%.+]] = constant {{2.+}} : f32 
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[CMP:%.+]] = cmpf "ogt", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[MUL:%.+]] = mulf [[ALPHA]], [[EXP]] : f32
  // CHECK: [[SUB:%.+]] = subf [[MUL]], [[ALPHA]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[LOAD]], [[SUB]] : f32
  // CHECK: [[SELU_RES:%.+]] = mulf [[GAMMA]], [[SELECT]] : f32
  // CHECK: affine.store [[SELU_RES]], [[RET_RES]][%arg1, %arg2] : memref<?x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xf32>

  // CHECK: return [[RET_RES]] : memref<?x10xf32>
}

// -----

func @test_hardsigmoid_hardsigmoid(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha=1.0:f32, beta=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.HardSigmoid"(%0) {alpha=1.0:f32, beta=2.0:f32} : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_hardsigmoid_hardsigmoid
  /// First HardSigmoid
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32 
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32 
  // CHECK: [[ALPHA:%.+]] = constant {{1.+}} : f32 
  // CHECK: [[BETA:%.+]] = constant {{2.+}} : f32 
  // CHECK: [[MUL:%.+]] = mulf [[ALPHA]], [[LOAD]] : f32
  // CHECK: [[ADD:%.+]] = addf [[MUL]], [[BETA]] : f32
  // CHECK: [[CMP1:%.+]] = cmpf "ogt", [[ADD]], [[ZERO]] : f32
  // CHECK: [[SELECT1:%.+]] = select [[CMP1]], [[ADD]], [[ZERO]] : f32
  // CHECK: [[CMP2:%.+]] = cmpf "olt", [[SELECT1]], [[ONE]] : f32
  // CHECK: [[SELECT2:%.+]] = select [[CMP2]], [[SELECT1]], [[ONE]] : f32
  // CHECK: affine.store [[SELECT2]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  
  /// Second HardSigmoid
  // CHECK: [[C0_1:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], [[C0_1]] : memref<?x10xf32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_2:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], [[C0_2]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32 
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32 
  // CHECK: [[ALPHA:%.+]] = constant {{1.+}} : f32 
  // CHECK: [[BETA:%.+]] = constant {{2.+}} : f32 
  // CHECK: [[MUL:%.+]] = mulf [[ALPHA]], [[LOAD]] : f32
  // CHECK: [[ADD:%.+]] = addf [[MUL]], [[BETA]] : f32
  // CHECK: [[CMP1:%.+]] = cmpf "ogt", [[ADD]], [[ZERO]] : f32
  // CHECK: [[SELECT1:%.+]] = select [[CMP1]], [[ADD]], [[ZERO]] : f32
  // CHECK: [[CMP2:%.+]] = cmpf "olt", [[SELECT1]], [[ONE]] : f32
  // CHECK: [[SELECT2:%.+]] = select [[CMP2]], [[SELECT1]], [[ONE]] : f32
  // CHECK: affine.store [[SELECT2]], [[RET_RES]][%arg1, %arg2] : memref<?x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xf32>

  // CHECK: return [[RET_RES]] : memref<?x10xf32>
}

// -----

func @test_reciprocal_reciprocal(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Reciprocal"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Reciprocal"(%0) : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reciprocal_reciprocal
  /// First Reciprocal
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[RECIPROCAL_RES:%.+]] = divf [[ONE]], [[LOAD]] : f32
  // CHECK: affine.store [[RECIPROCAL_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>

  /// Second Reciprocal
  // CHECK: [[C0_1:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], [[C0_1]] : memref<?x10xf32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_2:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], [[C0_2]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[RECIPROCAL_RES:%.+]] = divf [[ONE]], [[LOAD]] : f32
  // CHECK: affine.store [[RECIPROCAL_RES]], [[RET_RES]][%arg1, %arg2] : memref<?x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xf32>

  // CHECK: return [[RET_RES]] : memref<?x10xf32>
}
