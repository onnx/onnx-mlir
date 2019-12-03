// RUN: dlc-opt --shape-inference --lower-frontend %s -split-input-file | FileCheck %s

func @test_add_add(%arg0 : tensor<?x10xf32>, %arg1 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg1) : (tensor<*xf32>, tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_add_add
  /// First Add
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[ADDF:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[ADDF]], [[RES]][%arg2, %arg3] : memref<?x10xf32>

  /// Second Add
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], 0 : memref<?x10xf32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[ADDF:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[ADDF]], [[RET_RES]][%arg2, %arg3] : memref<?x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xf32>

  // CHECK: return [[RET_RES]] : memref<?x10xf32>
}


func @test_mul_mul(%arg0 : tensor<?x10xf32>, %arg1 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Mul"(%0, %arg1) : (tensor<*xf32>, tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_mul_mul
  /// First Mul
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[MULF:%.+]] = mulf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[MULF]], [[RES]][%arg2, %arg3] : memref<?x10xf32>

  /// Second Mul
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], 0 : memref<?x10xf32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[MULF:%.+]] = mulf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[MULF]], [[RET_RES]][%arg2, %arg3] : memref<?x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xf32>

  // CHECK: return [[RET_RES]] : memref<?x10xf32>
}

func @test_div_div(%arg0 : tensor<?x10xf32>, %arg1 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Div"(%0, %arg1) : (tensor<*xf32>, tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_div_div
  /// First Div
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[DIVF:%.+]] = divf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[DIVF]], [[RES]][%arg2, %arg3] : memref<?x10xf32>

  /// Second Div
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], 0 : memref<?x10xf32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[DIVF:%.+]] = divf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[DIVF]], [[RET_RES]][%arg2, %arg3] : memref<?x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xf32>

  // CHECK: return [[RET_RES]] : memref<?x10xf32>
}

func @test_sub_sub(%arg0 : tensor<?x10xf32>, %arg1 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<*xf32>
  %1 = "onnx.Sub"(%0, %arg1) : (tensor<*xf32>, tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sub_sub
  /// First Sub
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[SUBF:%.+]] = subf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[SUBF]], [[RES]][%arg2, %arg3] : memref<?x10xf32>

  /// Second Sub
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], 0 : memref<?x10xf32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[SUBF:%.+]] = subf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[SUBF]], [[RET_RES]][%arg2, %arg3] : memref<?x10xf32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xf32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xf32>

  // CHECK: return [[RET_RES]] : memref<?x10xf32>
}

func @test_and_and(%arg0 : tensor<?x10xi32>, %arg1 : tensor<?x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.And"(%arg0, %arg1) : (tensor<?x10xi32>, tensor<?x10xi32>) -> tensor<*xi32>
  %1 = "onnx.And"(%0, %arg1) : (tensor<*xi32>, tensor<?x10xi32>) -> tensor<*xi32>
  "std.return"(%1) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_and_and
  /// First And
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xi32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xi32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<?x10xi32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xi32>
  // CHECK: [[AND:%.+]] = and [[LOAD1]], [[LOAD2]] : i32
  // CHECK: store [[AND]], [[RES]][%arg2, %arg3] : memref<?x10xi32>

  /// Second And
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], 0 : memref<?x10xi32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], 0 : memref<?x10xi32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load [[RES]][%arg2, %arg3] : memref<?x10xi32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xi32>
  // CHECK: [[AND:%.+]] = and [[LOAD1]], [[LOAD2]] : i32
  // CHECK: store [[AND]], [[RET_RES]][%arg2, %arg3] : memref<?x10xi32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xi32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xi32>

  // CHECK: return [[RET_RES]] : memref<?x10xi32>
}

func @test_or_or(%arg0 : tensor<?x10xi32>, %arg1 : tensor<?x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.Or"(%arg0, %arg1) : (tensor<?x10xi32>, tensor<?x10xi32>) -> tensor<*xi32>
  %1 = "onnx.Or"(%0, %arg1) : (tensor<*xi32>, tensor<?x10xi32>) -> tensor<*xi32>
  "std.return"(%1) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_or_or
  /// First Or
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xi32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xi32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<?x10xi32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xi32>
  // CHECK: [[OR:%.+]] = or [[LOAD1]], [[LOAD2]] : i32
  // CHECK: store [[OR]], [[RES]][%arg2, %arg3] : memref<?x10xi32>

  /// Second Or
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], 0 : memref<?x10xi32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], 0 : memref<?x10xi32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load [[RES]][%arg2, %arg3] : memref<?x10xi32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xi32>
  // CHECK: [[OR:%.+]] = or [[LOAD1]], [[LOAD2]] : i32
  // CHECK: store [[OR]], [[RET_RES]][%arg2, %arg3] : memref<?x10xi32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xi32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xi32>

  // CHECK: return [[RET_RES]] : memref<?x10xi32>
}

func @test_xor_xor(%arg0 : tensor<?x10xi32>, %arg1 : tensor<?x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.Xor"(%arg0, %arg1) : (tensor<?x10xi32>, tensor<?x10xi32>) -> tensor<*xi32>
  %1 = "onnx.Xor"(%0, %arg1) : (tensor<*xi32>, tensor<?x10xi32>) -> tensor<*xi32>
  "std.return"(%1) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_xor_xor
  /// First Xor
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xi32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xi32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<?x10xi32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xi32>
  // CHECK: [[XOR:%.+]] = xor [[LOAD1]], [[LOAD2]] : i32
  // CHECK: store [[XOR]], [[RES]][%arg2, %arg3] : memref<?x10xi32>

  /// Second Xor
  // CHECK: [[DIM_0:%.+]] = dim [[RES]], 0 : memref<?x10xi32>
  // CHECK: [[RET_RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim [[RES]], 0 : memref<?x10xi32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load [[RES]][%arg2, %arg3] : memref<?x10xi32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xi32>
  // CHECK: [[XOR:%.+]] = xor [[LOAD1]], [[LOAD2]] : i32
  // CHECK: store [[XOR]], [[RET_RES]][%arg2, %arg3] : memref<?x10xi32>

  /// Dealloc of first result.
  // CHECK: dealloc [[RES]] : memref<?x10xi32>
  // CHECK-NOT: dealloc [[RET_RES]] : memref<?x10xi32>

  // CHECK: return [[RET_RES]] : memref<?x10xi32>
}
