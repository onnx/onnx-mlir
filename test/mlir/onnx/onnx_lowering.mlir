// RUN: onnx-mlir-opt --shape-inference --lower-frontend %s -split-input-file | FileCheck %s

// -----

func @test_add(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_add
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[ADDF:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[ADDF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_mul(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_mul
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MULF:%.+]] = mulf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[MULF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_div(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_div
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[DIVF:%.+]] = divf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[DIVF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_sub(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sub
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[SUBF:%.+]] = subf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[SUBF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_and(%arg0 : tensor<10x10xi32>, %arg1 : tensor<10x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.And"(%arg0, %arg1) : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_and
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<10x10xi32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<10x10xi32>
  // CHECK: [[AND:%.+]] = and [[LOAD1]], [[LOAD2]] : i32
  // CHECK: store [[AND]], [[RES]][%arg2, %arg3] : memref<10x10xi32>
  // CHECK: return [[RES]] : memref<10x10xi32>
}

// -----

func @test_or(%arg0 : tensor<10x10xi32>, %arg1 : tensor<10x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.Or"(%arg0, %arg1) : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_or
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<10x10xi32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<10x10xi32>
  // CHECK: [[OR:%.+]] = or [[LOAD1]], [[LOAD2]] : i32
  // CHECK: store [[OR]], [[RES]][%arg2, %arg3] : memref<10x10xi32>
  // CHECK: return [[RES]] : memref<10x10xi32>
}

// -----

func @test_xor(%arg0 : tensor<10x10xi32>, %arg1 : tensor<10x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.Xor"(%arg0, %arg1) : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_xor
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<10x10xi32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<10x10xi32>
  // CHECK: [[XOR:%.+]] = xor [[LOAD1]], [[LOAD2]] : i32
  // CHECK: store [[XOR]], [[RES]][%arg2, %arg3] : memref<10x10xi32>
  // CHECK: return [[RES]] : memref<10x10xi32>
}

// -----

func @test_exp(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Exp"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_exp
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: store [[EXP]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_tanh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Tanh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_tanh
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[NLOAD:%.+]] = subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = exp [[NLOAD]] : f32
  // CHECK: [[DIVIDEND:%.+]] = subf [[EXP]], [[NEXP]] : f32
  // CHECK: [[DIVISOR:%.+]] = addf [[EXP]], [[NEXP]] : f32
  // CHECK: [[TANH:%.+]] = divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK: store [[TANH]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_sinh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sinh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sinh
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[TWO:%.+]] = constant {{2.+}} : f32
  // CHECK: [[NLOAD:%.+]] = subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = exp [[NLOAD]] : f32
  // CHECK: [[DIVIDEND:%.+]] = subf [[EXP]], [[NEXP]] : f32
  // CHECK: [[SINH_RES:%.+]] = divf [[DIVIDEND]], [[TWO]] : f32
  // CHECK: store [[SINH_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_cosh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Cosh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_cosh
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[TWO:%.+]] = constant {{2.+}} : f32
  // CHECK: [[NLOAD:%.+]] = subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = exp [[NLOAD]] : f32
  // CHECK: [[DIVIDEND:%.+]] = addf [[EXP]], [[NEXP]] : f32
  // CHECK: [[COSH_RES:%.+]] = divf [[DIVIDEND]], [[TWO]] : f32
  // CHECK: store [[COSH_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_cos(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Cos"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_cos
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[COS:%.+]] = cos [[LOAD]] : f32
  // CHECK: store [[COS]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_log(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Log"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_log
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[LOG:%.+]] = log [[LOAD]] : f32
  // CHECK: store [[LOG]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_sigmoid(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sigmoid
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[NLOAD:%.+]] = subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = exp [[NLOAD]] : f32
  // CHECK: [[DIVISOR:%.+]] = addf [[ONE]], [[NEXP]] : f32
  // CHECK: [[SIGMOID_RES:%.+]] = divf [[ONE]], [[DIVISOR]] : f32
  // CHECK: store [[SIGMOID_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_relu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_relu
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[LTZERO:%.+]] = cmpf "olt", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[RELU_RES:%.+]] = select [[LTZERO]], [[ZERO]], [[LOAD]] : f32
  // CHECK: store [[RELU_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_reshape(%arg0 : tensor<?x10xf32>, %arg1 : tensor<4xi32>) -> tensor<*xf32> {
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<4xi32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reshape
  // CHECK: [[TYPE_IN_BYTES_0:%.+]] = constant 4 : i64
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[DIM_0_CAST:%.+]] = index_cast [[DIM_0]] : index to i64
  // CHECK: [[MUL_0:%.+]] = muli [[TYPE_IN_BYTES_0]], [[DIM_0_CAST]] : i64
  // CHECK: [[CONSTANT_0:%.+]] = constant 10 : i64
  // CHECK: [[TENSOR_SIZE:%.+]] = muli [[MUL_0]], [[CONSTANT_0]] : i64

  // CHECK: [[TYPE_IN_BYTES_1:%.+]] = constant 4 : i64
  // CHECK: %[[CONSTANT_1:.+]] = constant 0 : index
  // CHECK: [[LOAD_0:%.+]] = load %arg1[%[[CONSTANT_1]]] : memref<4xi32>
  // CHECK: [[DIM_1:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[DIM_1_CAST:%.+]] = index_cast [[DIM_1]] : index to i32
  // CHECK: [[CONSTANT_2:%.+]] = constant 0 : i32
  // CHECK: [[CMP_0:%.+]] = cmpi "eq", [[LOAD_0]], [[CONSTANT_2]] : i32
  // CHECK: [[SELECT_0:%.+]] = select [[CMP_0]], [[DIM_1_CAST]], [[LOAD_0]] : i32
  // CHECK: [[ZEXTI_0:%.+]] = zexti [[SELECT_0]] : i32 to i64
  // CHECK: [[MUL_1:%.+]] = muli [[TYPE_IN_BYTES_1]], [[ZEXTI_0]] : i64

  // CHECK: %[[CONSTANT_3:.+]] = constant 1 : index
  // CHECK: [[LOAD_1:%.+]] = load %arg1[%[[CONSTANT_3]]] : memref<4xi32>
  // CHECK: [[CONSTANT_3:%.+]] = constant 10 : i32
  // CHECK: [[CONSTANT_4:%.+]] = constant 0 : i32
  // CHECK: [[CMP_1:%.+]] = cmpi "eq", [[LOAD_1]], [[CONSTANT_4]] : i32
  // CHECK: [[SELECT_1:%.+]] = select [[CMP_1]], [[CONSTANT_3]], [[LOAD_1]] : i32
  // CHECK: [[ZEXTI_1:%.+]] = zexti [[SELECT_1]] : i32 to i64
  // CHECK: [[MUL_2:%.+]] = muli [[MUL_1]], [[ZEXTI_1]] : i64

  // CHECK: %[[CONSTANT_5:.+]] = constant 2 : index
  // CHECK: [[LOAD_2:%.+]] = load %arg1[%[[CONSTANT_5]]] : memref<4xi32>
  // CHECK: [[ZEXTI_2:%.+]] = zexti [[LOAD_2]] : i32 to i64
  // CHECK: [[MUL_3:%.+]] = muli [[MUL_2]], [[ZEXTI_2]] : i64

  // CHECK: %[[CONSTANT_6:.+]] = constant 3 : index
  // CHECK: [[LOAD_3:%.+]] = load %arg1[%[[CONSTANT_6]]] : memref<4xi32>
  // CHECK: [[ZEXTI_3:%.+]] = zexti [[LOAD_3]] : i32 to i64
  // CHECK: [[MUL_4:%.+]] = muli [[MUL_3]], [[ZEXTI_3]] : i64

  // CHECK: [[CONSTANT_7:%.+]] = constant 0 : i64
  // CHECK: [[SUB_0:%.+]] = subi [[CONSTANT_7]], [[MUL_4]] : i64

  // CHECK: [[CONSTANT_8:%.+]] = constant -1 : i64
  // CHECK: [[CMP_2:%.+]] = cmpi "eq", [[ZEXTI_0]], [[CONSTANT_8]] : i64
  // CHECK: [[DIVISIGNED_0:%.+]] = divi_signed [[TENSOR_SIZE]], [[SUB_0]] : i64
  // CHECK: [[SELECT_2:%.+]] = select [[CMP_2]], [[DIVISIGNED_0]], [[ZEXTI_0]] : i64
  // CHECK: [[CAST_0:%.+]] = index_cast [[SELECT_2]] : i64 to index

  // CHECK: [[CMP_3:%.+]] = cmpi "eq", [[ZEXTI_1]], [[CONSTANT_8]] : i64
  // CHECK: [[DIVISIGNED_1:%.+]] = divi_signed [[TENSOR_SIZE]], [[SUB_0]] : i64
  // CHECK: [[SELECT_3:%.+]] = select [[CMP_3]], [[DIVISIGNED_1]], [[ZEXTI_1]] : i64
  // CHECK: [[CAST_1:%.+]] = index_cast [[SELECT_3]] : i64 to index

  // CHECK: [[CMP_4:%.+]] = cmpi "eq", [[ZEXTI_2]], [[CONSTANT_8]] : i64
  // CHECK: [[DIVISIGNED_2:%.+]] = divi_signed [[TENSOR_SIZE]], [[SUB_0]] : i64
  // CHECK: [[SELECT_4:%.+]] = select [[CMP_4]], [[DIVISIGNED_2]], [[ZEXTI_2]] : i64
  // CHECK: [[CAST_2:%.+]] = index_cast [[SELECT_4]] : i64 to index

  // CHECK: [[CMP_5:%.+]] = cmpi "eq", [[ZEXTI_3]], [[CONSTANT_8]] : i64
  // CHECK: [[DIVISIGNED_3:%.+]] = divi_signed [[TENSOR_SIZE]], [[SUB_0]] : i64
  // CHECK: [[SELECT_5:%.+]] = select [[CMP_5]], [[DIVISIGNED_3]], [[ZEXTI_3]] : i64
  // CHECK: [[CAST_3:%.+]] = index_cast [[SELECT_5]] : i64 to index

  // CHECK: [[ALLOC:%.+]] = alloc([[CAST_0]], [[CAST_1]], [[CAST_2]], [[CAST_3]]) : memref<?x?x?x?xf32>
  // CHECK: "krnl.memcpy"([[ALLOC]], %arg0, [[TENSOR_SIZE]]) : (memref<?x?x?x?xf32>, memref<?x10xf32>, i64) -> ()
  // CHECK: return [[ALLOC]] : memref<?x?x?x?xf32>
}

// -----

func @test_sum(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sum
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[ADD:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[ADD]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_max(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Max"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_max
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MAX:%.+]] = cmpf "ogt", [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[RELU_RES:%.+]] = select [[MAX]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[RELU_RES]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_min(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Min"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_min
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MIN:%.+]] = cmpf "olt", [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[RELU_RES:%.+]] = select [[MIN]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[RELU_RES]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_elu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Elu"(%arg0) {alpha=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_elu
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32 
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32 
  // CHECK: [[ALPHA:%.+]] = constant {{2.+}} : f32 
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[CMP:%.+]] = cmpf "olt", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[SUB:%.+]] = subf [[EXP]], [[ONE]] : f32
  // CHECK: [[MUL:%.+]] = mulf [[ALPHA]], [[SUB]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[MUL]], [[LOAD]] : f32
  // CHECK: store [[SELECT]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_leakyrelu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha=1.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_leakyrelu
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32 
  // CHECK: [[ALPHA:%.+]] = constant {{1.+}} : f32 
  // CHECK: [[CMP:%.+]] = cmpf "olt", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[MUL:%.+]] = mulf [[ALPHA]], [[LOAD]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[MUL]], [[LOAD]] : f32
  // CHECK: store [[SELECT]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_selu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Selu"(%arg0) {alpha=1.0:f32, gamma=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_selu
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32 
  // CHECK: [[ALPHA:%.+]] = constant {{1.+}} : f32 
  // CHECK: [[GAMMA:%.+]] = constant {{2.+}} : f32 
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[CMP:%.+]] = cmpf "ogt", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[MUL:%.+]] = mulf [[ALPHA]], [[EXP]] : f32
  // CHECK: [[SUB:%.+]] = subf [[MUL]], [[ALPHA]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[LOAD]], [[SUB]] : f32
  // CHECK: [[SELU_RES:%.+]] = mulf [[GAMMA]], [[SELECT]] : f32
  // CHECK: store [[SELU_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_hardsigmoid(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha=1.0:f32, beta=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_hardsigmoid
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
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
  // CHECK: store [[SELECT2]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_reciprocal(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Reciprocal"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reciprocal
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[RECIPROCAL_RES:%.+]] = divf [[ONE]], [[LOAD]] : f32
  // CHECK: store [[RECIPROCAL_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_softplus(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softplus"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_softplus
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[ADD:%.+]] = addf [[EXP]], [[ONE]] : f32
  // CHECK: [[SOFTPLUS_RES:%.+]] = log [[ADD]] : f32
  // CHECK: store [[SOFTPLUS_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_softsign(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softsign"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_softsign
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ABS:%.+]] = absf [[LOAD]] : f32
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[ADD:%.+]] = addf [[ABS]], [[ONE]] : f32
  // CHECK: [[SOFTSIGN_RES:%.+]] = divf [[LOAD]], [[ADD]] : f32
  // CHECK: store [[SOFTSIGN_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_add_with_broadcasting(%arg0 : tensor<?xf32>, %arg1 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?xf32>, tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_add_with_broadcasting
  // CHECK: [[DIM1:%.+]] = dim %arg1, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM1]]) : memref<?x10xf32>
  // CHECK: [[DIM2:%.+]] = dim %arg0, 0 : memref<?xf32>
  // CHECK: [[ONE:%.+]] = constant 1 : index
  // CHECK: [[IS_ONE:%.+]] = cmpi "eq", [[DIM2]], [[ONE]] : index
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM3:%.+]] = dim [[RES]], 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM3]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[ZERO:%.+]] = constant 0 : index
  // CHECK: %[[SELECT1:.+]] = select [[IS_ONE]], [[ZERO]], %arg3 : index
  // CHECK: [[LOAD1:%.+]] = load %arg0[%[[SELECT1]]] : memref<?xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[ADD:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: store [[ADD]], [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_reducemax(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMax"(%arg0) {axes=[1], keepdims = 0 : i64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducemax
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS1:%.+]]:2 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS1]]#0, [[OPT_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 0xFF800000 : f32
  // CHECK: store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[OPT_LOOPS2:%.+]]:3 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS2]]#0, [[OPT_LOOPS2]]#1, [[OPT_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = load %0[%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[CMP:%.+]] = cmpf "ogt", [[LOAD2]], [[LOAD1]] : f32
  // CHECK: [[SELECT:%.+]] = select %7, %6, %5 : f32
  // CHECK: store [[SELECT]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func @test_reducemin(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMin"(%arg0) {axes=[1], keepdims = 0 : i64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducemin
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS1:%.+]]:2 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS1]]#0, [[OPT_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 0x7F800000 : f32
  // CHECK: store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[OPT_LOOPS2:%.+]]:3 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS2]]#0, [[OPT_LOOPS2]]#1, [[OPT_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = load %0[%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[CMP:%.+]] = cmpf "olt", [[LOAD2]], [[LOAD1]] : f32
  // CHECK: [[SELECT:%.+]] = select %7, %6, %5 : f32
  // CHECK: store [[SELECT]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func @test_reduceprod(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceProd"(%arg0) {axes=[1], keepdims = 0 : i64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduceprod
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS1:%.+]]:2 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS1]]#0, [[OPT_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 1.000000e+00 : f32
  // CHECK: store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[OPT_LOOPS2:%.+]]:3 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS2]]#0, [[OPT_LOOPS2]]#1, [[OPT_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = load %0[%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[REDUCE:%.+]] = mulf %6, %5 : f32
  // CHECK: store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func @test_reducesum(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceSum"(%arg0) {axes=[1], keepdims = 0 : i64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducesum
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS1:%.+]]:2 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS1]]#0, [[OPT_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 0.000000e+00 : f32
  // CHECK: store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[OPT_LOOPS2:%.+]]:3 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS2]]#0, [[OPT_LOOPS2]]#1, [[OPT_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = load %0[%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[REDUCE:%.+]] = addf %6, %5 : f32
  // CHECK: store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}
  
// -----

func @test_softmax(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis=1:i64} : (tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_softmax
  // CHECK: [[MAX:%.+]] = alloc() : memref<f32>
  // CHECK: [[SUM:%.+]] = alloc() : memref<f32>
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[CST:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[CST_0:%.+]] = constant 0xFF800000 : f32
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:  krnl.return_loops [[DEF_LOOPS]]#0, %3#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to 10) {
  // CHECK: store [[CST]], [[SUM]][] : memref<f32>
  // CHECK: store [[CST_0]], [[MAX]][] : memref<f32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK:   [[LOAD1:%.+]] = load [[MAX]][] : memref<f32>
  // CHECK:   [[LOAD2:%.+]] = load %arg0[%arg1, %arg2] : memref<10x10xf32>
  // CHECK:   [[COND:%.+]] = cmpf "ogt", [[LOAD1]], [[LOAD2]] : f32
  // CHECK:   [[SELECT:%.+]] = select [[COND]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK:   store [[SELECT]], [[MAX]][] : memref<f32>
  // CHECK: }
  // CHECK: %5 = load [[MAX]][] : memref<f32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK:   [[LOAD1]] = load [[SUM]][] : memref<f32>
  // CHECK:   [[LOAD2]] = load %arg0[%arg1, %arg2] : memref<10x10xf32>
  // CHECK:   [[SUB:%.+]] = subf [[LOAD2]], %5 : f32
  // CHECK:   [[EXP:%.+]] = exp [[SUB]] : f32
  // CHECK:   [[ADD:%.+]] = addf [[LOAD1]], [[EXP]] : f32
  // CHECK:   store [[ADD]], [[SUM]][] : memref<f32>
  // CHECK:   store %10, [[RES]][%arg1, %arg2] : memref<10x10xf32>
  // CHECK: }
  // CHECK: %6 = load [[SUM]][] : memref<f32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK:   [[LOAD1]] = load [[RES]][%arg1, %arg2] : memref<10x10xf32>
  // CHECK:   [[DIV:%.+]] = divf [[LOAD1]], %6 : f32
  // CHECK:   store [[DIV]], [[RES]][%arg1, %arg2] : memref<10x10xf32>
  // CHECK: }
  // CHECK: }
  // CHECK: dealloc [[SUM]] : memref<f32>
  // CHECK: dealloc [[MAX]] : memref<f32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_gemm(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1, transB = 0} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_gemm
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[ALPHA:%.+]] = constant 1.000000e+00 : f32
  // CHECK: [[BETA:%.+]] = constant 5.000000e+00 : f32
  // CHECK: [[DEF_LOOPS:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[OPT_LOOPS:%.+]]:3 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg3 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg4 = 0 to 10) {
  // CHECK: krnl.iterate([[OPT_LOOPS]]#2) with ([[DEF_LOOPS]]#2 -> %arg5 = 0 to 5) {
  // CHECK: [[A:%.+]] = load %arg0[%arg5, %arg3] : memref<5x10xf32>
  // CHECK: [[B:%.+]] = load %arg1[%arg5, %arg4] : memref<5x10xf32>
  // CHECK: [[Y:%.+]] = load [[RES]][%arg3, %arg4] : memref<10x10xf32>
  // CHECK: [[AB:%.+]] = mulf [[A]], [[B]] : f32
  // CHECK: [[SUM:%.+]] = addf [[Y]], [[AB]] : f32
  // CHECK: store [[SUM]], [[RES]][%arg3, %arg4] : memref<10x10xf32>
  // CHECK: }
  // CHECK: [[LOAD_Y:%.+]] = load [[RES]][%arg3, %arg4] : memref<10x10xf32>
  // CHECK: [[ALPHA_AB:%.+]] = mulf [[ALPHA]], [[LOAD_Y]] : f32
  // CHECK: [[C:%.+]] = load %arg2[%arg4] : memref<10xf32>
  // CHECK: [[BETA_C:%.+]] = mulf [[BETA]], [[C]] : f32
  // CHECK: [[Y_RES:%.+]] = addf [[ALPHA_AB]], [[BETA_C]] : f32
  // CHECK: store [[Y_RES]], [[RES]][%arg3, %arg4] : memref<10x10xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<10x10xf32>
  // CHECK: }
}

// -----

func @test_sqrt(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sqrt
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[SQRT:%.+]] = sqrt [[LOAD]] : f32
  // CHECK: store [[SQRT]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_unsqueeze(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Unsqueeze"(%arg0) {axes=[0,3]} : (tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_unsqueeze
  // CHECK: [[RES:%.+]] = alloc() : memref<1x10x10x1xf32>
  // CHECK: [[INBYTES:%.+]] = constant 4 : i64
  // CHECK: [[DIM1:%.+]] = constant 1 : i64
  // CHECK: [[SIZE1:%.+]] = muli [[INBYTES]], [[DIM1]] : i64
  // CHECK: [[DIM2:%.+]] = constant 10 : i64
  // CHECK: [[SIZE2:%.+]] = muli [[SIZE1]], [[DIM2]] : i64
  // CHECK: [[DIM3:%.+]] = constant 10 : i64
  // CHECK: [[SIZE3:%.+]] = muli [[SIZE2]], [[DIM3]] : i64
  // CHECK: [[DIM4:%.+]] = constant 1 : i64
  // CHECK: [[SIZE4:%.+]] = muli [[SIZE3]], [[DIM4]] : i64
  // CHECK: "krnl.memcpy"([[RES]], %arg0, [[SIZE4]]) : (memref<1x10x10x1xf32>, memref<10x10xf32>, i64) -> ()
  // CHECK: return [[RES]] : memref<1x10x10x1xf32>
}

// -----

func @test_transpose(%arg0 : tensor<10x20x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<10x20x30x40xf32>) -> tensor<*xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_transpose
  // CHECK: [[RES0:%.+]] = alloc() : memref<40x10x30x20xf32>
  // CHECK: [[RES1:%.+]] = alloc() : memref<40x30x20x10xf32>

  // CHECK: [[LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: [[OPT_LOOPS:%.+]]:4 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[LOOPS]]#0, [[LOOPS]]#1, [[LOOPS]]#2, [[LOOPS]]#3
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1, [[OPT_LOOPS]]#2, [[OPT_LOOPS]]#3) with ([[LOOPS]]#0 -> %arg1 = 0 to 10, [[LOOPS]]#1 -> %arg2 = 0 to 20, [[LOOPS]]#2 -> %arg3 = 0 to 30, [[LOOPS]]#3 -> %arg4 = 0 to 40) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2, %arg3, %arg4] : memref<10x20x30x40xf32>
  // CHECK: store [[LOAD]], [[RES1]][%arg4, %arg3, %arg2, %arg1] : memref<40x30x20x10xf32>

  // CHECK: [[LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: [[OPT_LOOPS:%.+]]:4 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[LOOPS]]#0, [[LOOPS]]#1, [[LOOPS]]#2, [[LOOPS]]#3
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1, [[OPT_LOOPS]]#2, [[OPT_LOOPS]]#3) with ([[LOOPS]]#0 -> %arg1 = 0 to 40, [[LOOPS]]#1 -> %arg2 = 0 to 30, [[LOOPS]]#2 -> %arg3 = 0 to 20, [[LOOPS]]#3 -> %arg4 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load [[RES1]][%arg1, %arg2, %arg3, %arg4] : memref<40x30x20x10xf32>
  // CHECK: store [[LOAD]], [[RES0]][%arg1, %arg4, %arg2, %arg3] : memref<40x10x30x20xf32>

  // CHECK: dealloc [[RES1]] : memref<40x30x20x10xf32>
  // CHECK: return [[RES0]] : memref<40x10x30x20xf32>
}

// -----

func @test_identity(%arg0 : tensor<10x20x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Identity"(%arg0) : (tensor<10x20x30x40xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_identity
  // CHECK: return %arg0 : memref<10x20x30x40xf32>
}

// -----

func @test_sign_f(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sign"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sign_f
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[MINUS_ONE:%.+]] = constant {{-1.+}} : f32
  // CHECK: [[GTZERO:%.+]] = cmpf "ogt", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[SELECT_PLUS:%.+]] = select [[GTZERO]], [[ONE]], [[MINUS_ONE]] : f32
  // CHECK: [[EQZERO:%.+]] = cmpf "oeq", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[SIGN_RES:%.+]] = select [[EQZERO]], [[ZERO]], [[SELECT_PLUS]] : f32
  // CHECK: store [[SIGN_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_sign_i(%arg0 : tensor<?x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.Sign"(%arg0) : (tensor<?x10xi32>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_sign_i
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xi32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xi32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xi32>
  // CHECK: [[ZERO:%.+]] = constant 0 : i32
  // CHECK: [[ONE:%.+]] = constant 1 : i32
  // CHECK: [[MINUS_ONE:%.+]] = constant -1 : i32
  // CHECK: [[GTZERO:%.+]] = cmpi "sgt", [[LOAD]], [[ZERO]] : i32
  // CHECK: [[SELECT_PLUS:%.+]] = select [[GTZERO]], [[ONE]], [[MINUS_ONE]] : i32
  // CHECK: [[EQZERO:%.+]] = cmpi "eq", [[LOAD]], [[ZERO]] : i32
  // CHECK: [[SIGN_RES:%.+]] = select [[EQZERO]], [[ZERO]], [[SELECT_PLUS]] : i32
  // CHECK: store [[SIGN_RES]], [[RES]][%arg1, %arg2] : memref<?x10xi32>
  // CHECK: return [[RES]] : memref<?x10xi32>
}

// -----

// 2-D x 2-D
func @test_matmul1(%arg0 : tensor<10x5xf32>, %arg1 : tensor<5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<10x5xf32>, tensor<5x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul1
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[CONSTANT:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[LOOPS]]#0, [[LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[LOOPS]]#0 -> %arg2 = 0 to 10, [[LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK:   store [[CONSTANT]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK:   [[LOOPS_REDUCE:%.+]] = krnl.define_loops 1
  // CHECK:   [[OPT_LOOPS_REDUCE:%.+]] = krnl.optimize_loops  {
  // CHECK:     krnl.return_loops [[LOOPS_REDUCE]]
  // CHECK:   } : () -> !krnl.loop
  // CHECK:   krnl.iterate([[OPT_LOOPS_REDUCE]]) with ([[LOOPS_REDUCE]] -> %arg4 = 0 to 5) {
  // CHECK:     [[LOAD_0:%.+]] = load %arg0[%arg2, %arg4] : memref<10x5xf32>
  // CHECK:     [[LOAD_1:%.+]] = load %arg1[%arg4, %arg3] : memref<5x10xf32>
  // CHECK:     [[LOAD_RES:%.+]] = load [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK:     [[MUL:%.+]] = mulf [[LOAD_0]], [[LOAD_1]] : f32
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_RES]], [[MUL]] : f32
  // CHECK:     store [[ADD]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK:   }
  // CHECK: }
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

// 2-D x N-D
func @test_matmul2(%arg0 : tensor<10x5xf32>, %arg1 : tensor<2x3x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<10x5xf32>, tensor<2x3x5x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul2
  // CHECK: [[RES:%.+]] = alloc() : memref<2x3x10x10xf32>
  // CHECK: [[CONSTANT:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: [[OPT_LOOPS:%.+]]:4 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[LOOPS]]#0, [[LOOPS]]#1, [[LOOPS]]#2, [[LOOPS]]#3
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[LOOPS]]#0 -> %arg2 = 0 to 2, [[LOOPS]]#1 -> %arg3 = 0 to 3) {
  // CHECK:   krnl.iterate([[OPT_LOOPS]]#2, [[OPT_LOOPS]]#3) with ([[LOOPS]]#2 -> %arg4 = 0 to 10, [[LOOPS]]#3 -> %arg5 = 0 to 10) {
  // CHECK:     store [[CONSTANT]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<2x3x10x10xf32>
  // CHECK:     [[LOOPS_REDUCE:%.+]] = krnl.define_loops 1
  // CHECK:     [[OPT_LOOPS_REDUCE:%.+]] = krnl.optimize_loops  {
  // CHECK:       krnl.return_loops [[LOOPS_REDUCE]]
  // CHECK:     } : () -> !krnl.loop
  // CHECK:     krnl.iterate([[OPT_LOOPS_REDUCE]]) with ([[LOOPS_REDUCE]] -> %arg6 = 0 to 5) {
  // CHECK:       [[LOAD_0:%.+]] = load %arg0[%arg4, %arg6] : memref<10x5xf32>
  // CHECK:       [[LOAD_1:%.+]] = load %arg1[%arg2, %arg3, %arg6, %arg5] : memref<2x3x5x10xf32>
  // CHECK:       [[LOAD_RES:%.+]] = load [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<2x3x10x10xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_0]], [[LOAD_1]] : f32
  // CHECK:       [[ADD:%.+]] = addf [[LOAD_RES]], [[MUL]] : f32
  // CHECK:       store [[ADD]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<2x3x10x10xf32>
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  // CHECK: return [[RES]] : memref<2x3x10x10xf32>
}

// -----

// N-D x N-D
func @test_matmul3(%arg0 : tensor<2x3x10x5xf32>, %arg1 : tensor<2x3x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<2x3x10x5xf32>, tensor<2x3x5x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul3
  // CHECK: [[RES:%.+]] = alloc() : memref<2x3x10x10xf32>
  // CHECK: [[CONSTANT:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: [[OPT_LOOPS:%.+]]:4 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[LOOPS]]#0, [[LOOPS]]#1, [[LOOPS]]#2, [[LOOPS]]#3
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[LOOPS]]#0 -> %arg2 = 0 to 2, [[LOOPS]]#1 -> %arg3 = 0 to 3) {
  // CHECK:   krnl.iterate([[OPT_LOOPS]]#2, [[OPT_LOOPS]]#3) with ([[LOOPS]]#2 -> %arg4 = 0 to 10, [[LOOPS]]#3 -> %arg5 = 0 to 10) {
  // CHECK:     store [[CONSTANT]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<2x3x10x10xf32>
  // CHECK:     [[LOOPS_REDUCE:%.+]] = krnl.define_loops 1
  // CHECK:     [[OPT_LOOPS_REDUCE:%.+]] = krnl.optimize_loops  {
  // CHECK:       krnl.return_loops [[LOOPS_REDUCE]]
  // CHECK:     } : () -> !krnl.loop
  // CHECK:     krnl.iterate([[OPT_LOOPS_REDUCE]]) with ([[LOOPS_REDUCE]] -> %arg6 = 0 to 5) {
  // CHECK:       [[LOAD_0:%.+]] = load %arg0[%arg2, %arg3, %arg4, %arg6] : memref<2x3x10x5xf32>
  // CHECK:       [[LOAD_1:%.+]] = load %arg1[%arg2, %arg3, %arg6, %arg5] : memref<2x3x5x10xf32>
  // CHECK:       [[LOAD_RES:%.+]] = load [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<2x3x10x10xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_0]], [[LOAD_1]] : f32
  // CHECK:       [[ADD:%.+]] = addf [[LOAD_RES]], [[MUL]] : f32
  // CHECK:       store [[ADD]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<2x3x10x10xf32>
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  // CHECK: return [[RES]] : memref<2x3x10x10xf32>
}

// -----

// 1-D x 2-D
func @test_matmul4(%arg0 : tensor<5xf32>, %arg1 : tensor<5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<5x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul4
  // CHECK: [[RES:%.+]] = alloc() : memref<10xf32>
  // CHECK: [[CONSTANT:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: [[OPT_LOOPS:%.+]] = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[LOOPS]]
  // CHECK: } : () -> !krnl.loop
  // CHECK: krnl.iterate([[OPT_LOOPS]]) with ([[LOOPS]] -> %arg2 = 0 to 10) {
  // CHECK:   store [[CONSTANT]], [[RES]][%arg2] : memref<10xf32>
  // CHECK:   [[LOOPS_REDUCE:%.+]] = krnl.define_loops 1
  // CHECK:   [[OPT_LOOPS_REDUCE:%.+]] = krnl.optimize_loops  {
  // CHECK:     krnl.return_loops [[LOOPS_REDUCE]]
  // CHECK:   } : () -> !krnl.loop
  // CHECK:   krnl.iterate([[OPT_LOOPS_REDUCE]]) with ([[LOOPS_REDUCE]] -> %arg3 = 0 to 5) {
  // CHECK:     [[LOAD_0:%.+]] = load %arg0[%arg3] : memref<5xf32>
  // CHECK:     [[LOAD_1:%.+]] = load %arg1[%arg3, %arg2] : memref<5x10xf32>
  // CHECK:     [[LOAD_RES:%.+]] = load [[RES]][%arg2] : memref<10xf32>
  // CHECK:     [[MUL:%.+]] = mulf [[LOAD_0]], [[LOAD_1]] : f32
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_RES]], [[MUL]] : f32
  // CHECK:     store [[ADD]], [[RES]][%arg2] : memref<10xf32>
  // CHECK:   }
  // CHECK: }
  // CHECK: return [[RES]] : memref<10xf32>
}

// -----

// 1-D x N-D
func @test_matmul5(%arg0 : tensor<5xf32>, %arg1 : tensor<?x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<?x5x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul5
  // CHECK: [[CONSTANT:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[DIM_0:%.+]] = dim %arg1, 0 : memref<?x5x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[LOOPS]]#0, [[LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_1:%.+]] = dim [[RES]], 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0) with ([[LOOPS]]#0 -> %arg2 = 0 to [[DIM_1]]) {
  // CHECK:   krnl.iterate([[OPT_LOOPS]]#1) with ([[LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK:     store [[CONSTANT]], [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK:     [[LOOPS_REDUCE:%.+]] = krnl.define_loops 1
  // CHECK:     [[OPT_LOOPS_REDUCE:%.+]] = krnl.optimize_loops  {
  // CHECK:       krnl.return_loops [[LOOPS_REDUCE]]
  // CHECK:     } : () -> !krnl.loop
  // CHECK:     krnl.iterate([[OPT_LOOPS_REDUCE]]) with ([[LOOPS_REDUCE]] -> %arg4 = 0 to 5) {
  // CHECK:       [[LOAD_0:%.+]] = load %arg0[%arg4] : memref<5xf32>
  // CHECK:       [[LOAD_1:%.+]] = load %arg1[%arg2, %arg4, %arg3] : memref<?x5x10xf32>
  // CHECK:       [[LOAD_RES:%.+]] = load [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_0]], [[LOAD_1]] : f32
  // CHECK:       [[ADD:%.+]] = addf [[LOAD_RES]], [[MUL]] : f32
  // CHECK:       store [[ADD]], [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

// N-D x 1-D
func @test_matmul6(%arg0 : tensor<?x10x5xf32>, %arg1 : tensor<5xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<?x10x5xf32>, tensor<5xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul6
  // CHECK: [[CONSTANT:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10x5xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[LOOPS]]#0, [[LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_1:%.+]] = dim [[RES]], 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0) with ([[LOOPS]]#0 -> %arg2 = 0 to [[DIM_1]]) {
  // CHECK:   krnl.iterate([[OPT_LOOPS]]#1) with ([[LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK:     store [[CONSTANT]], [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK:     [[LOOPS_REDUCE:%.+]] = krnl.define_loops 1
  // CHECK:     [[OPT_LOOPS_REDUCE:%.+]] = krnl.optimize_loops  {
  // CHECK:       krnl.return_loops [[LOOPS_REDUCE]]
  // CHECK:     } : () -> !krnl.loop
  // CHECK:     krnl.iterate([[OPT_LOOPS_REDUCE]]) with ([[LOOPS_REDUCE]] -> %arg4 = 0 to 5) {
  // CHECK:       [[LOAD_0:%.+]] = load %arg0[%arg2, %arg3, %arg4] : memref<?x10x5xf32>
  // CHECK:       [[LOAD_1:%.+]] = load %arg1[%arg4] : memref<5xf32>
  // CHECK:       [[LOAD_RES:%.+]] = load [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_0]], [[LOAD_1]] : f32
  // CHECK:       [[ADD:%.+]] = addf [[LOAD_RES]], [[MUL]] : f32
  // CHECK:       store [[ADD]], [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK:     }
  // CHECK:   }
  // CHECK: }
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

// 1-D x 1-D
func @test_matmul7(%arg0 : tensor<5xf32>, %arg1 : tensor<5xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<5xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_matmul7
  // CHECK: [[RES:%.+]] = alloc() : memref<1xf32>
  // CHECK: [[CONSTANT:%.+]] = constant 0.000000e+00 : f32
  // CHECK: %[[CONSTANT_INDEX:.+]] = constant 0 : index
  // CHECK: store [[CONSTANT]], [[RES]][%[[CONSTANT_INDEX]]] : memref<1xf32>
  // CHECK: [[LOOPS_REDUCE:%.+]] = krnl.define_loops 1
  // CHECK: [[OPT_LOOPS_REDUCE:%.+]] = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[LOOPS_REDUCE]]
  // CHECK: } : () -> !krnl.loop
  // CHECK: krnl.iterate([[OPT_LOOPS_REDUCE]]) with ([[LOOPS_REDUCE]] -> %arg2 = 0 to 5) {
  // CHECK:   [[LOAD_0:%.+]] = load %arg0[%arg2] : memref<5xf32>
  // CHECK:   [[LOAD_1:%.+]] = load %arg1[%arg2] : memref<5xf32>
  // CHECK:   [[LOAD_RES:%.+]] = load [[RES]][%[[CONSTANT_INDEX]]] : memref<1xf32>
  // CHECK:   [[MUL:%.+]] = mulf [[LOAD_0]], [[LOAD_1]] : f32
  // CHECK:   [[ADD:%.+]] = addf [[LOAD_RES]], [[MUL]] : f32
  // CHECK:   store [[ADD]], [[RES]][%[[CONSTANT_INDEX]]] : memref<1xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<1xf32>
}

// -----

func @test_conv_no_bias_no_pad(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_no_pad
  // CHECK: [[RES:%.+]] = alloc() : memref<1x5x27x58xf32>
  // CHECK: [[CONST0:%.+]] = constant 5 : index
  // CHECK: [[CONST1:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[CONST2:%.+]] = constant 2 : index
  // CHECK: [[OUTER_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_OUTER_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[OUTER_LOOPS]]#0, [[OUTER_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)

  // CHECK: krnl.iterate([[OPT_OUTER_LOOPS]]#0, [[OPT_OUTER_LOOPS]]#1) with ([[OUTER_LOOPS]]#0 -> %arg2 = 0 to 1, [[OUTER_LOOPS]]#1 -> %arg3 = 0 to 5) {
  // CHECK: [[SPATIAL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_SPATIAL_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[SPATIAL_LOOPS]]#0, [[SPATIAL_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)

  // CHECK: krnl.iterate([[OPT_SPATIAL_LOOPS]]#0, [[OPT_SPATIAL_LOOPS]]#1) with ([[SPATIAL_LOOPS]]#0 -> %arg4 = 0 to 27, [[SPATIAL_LOOPS]]#1 -> %arg5 = 0 to 58) {
  // CHECK: store [[CONST1]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<1x5x27x58xf32>
  // CHECK: [[INNER_LOOPS:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[OPT_INNER_LOOPS:%.+]]:3 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[INNER_LOOPS]]#0, [[INNER_LOOPS]]#1, [[INNER_LOOPS]]#2
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop)

  // CHECK: krnl.iterate([[OPT_INNER_LOOPS]]#0, [[OPT_INNER_LOOPS]]#1, [[OPT_INNER_LOOPS]]#2) with ([[INNER_LOOPS]]#0 -> %arg6 = 0 to 2, [[INNER_LOOPS]]#1 -> %arg7 = 0 to 6, [[INNER_LOOPS]]#2 -> %arg8 = 0 to 7) {
  // CHECK: [[R1PLUSK1:%.+]] = addi %arg4, %arg7 : index
  // CHECK: [[R2PLUSK2:%.+]] = addi %arg5, %arg8 : index
  // CHECK: [[DATA:%.+]] = load %arg0[%arg2, %arg6, [[R1PLUSK1]], [[R2PLUSK2]]] : memref<1x2x32x64xf32>
  // CHECK: [[KERNEL:%.+]] = load %arg1[%arg3, %arg6, %arg7, %arg8] : memref<5x2x6x7xf32>
  // CHECK: [[ACC_RES:%.+]] = load %0[%arg2, %arg3, %arg4, %arg5] : memref<1x5x27x58xf32>
  // CHECK: [[MUL:%.+]] = mulf [[DATA]], [[KERNEL]] : f32
  // CHECK: [[ADD:%.+]] = addf [[ACC_RES]], [[MUL]] : f32
  // CHECK: store [[ADD]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<1x5x27x58xf32>
  // CHECK: }
  // CHECK: }
  // CHECK: }

  // CHECK: return [[RES]] : memref<1x5x27x58xf32>
}

// -----

func @test_conv_bias_no_pad(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 1 : i64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, tensor<5xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_bias_no_pad
  // CHECK: [[RES:%.+]] = alloc() : memref<1x5x27x58xf32>
  // CHECK: [[CONST0:%.+]] = constant 5 : index
  // CHECK: [[CONST1:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[CONST2:%.+]] = constant 2 : index
  // CHECK: [[OUTER_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_OUTER_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[OUTER_LOOPS]]#0, [[OUTER_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)

  // CHECK: krnl.iterate([[OPT_OUTER_LOOPS]]#0, [[OPT_OUTER_LOOPS]]#1) with ([[OUTER_LOOPS]]#0 -> %arg3 = 0 to 1, [[OUTER_LOOPS]]#1 -> %arg4 = 0 to 5) {
  // CHECK: [[SPATIAL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_SPATIAL_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[SPATIAL_LOOPS]]#0, [[SPATIAL_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)

  // CHECK: krnl.iterate([[OPT_SPATIAL_LOOPS]]#0, [[OPT_SPATIAL_LOOPS]]#1) with ([[SPATIAL_LOOPS]]#0 -> %arg5 = 0 to 27, [[SPATIAL_LOOPS]]#1 -> %arg6 = 0 to 58) {
  // CHECK: store [[CONST1]], [[RES]][%arg3, %arg4, %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: [[INNER_LOOPS:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[OPT_INNER_LOOPS:%.+]]:3 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[INNER_LOOPS]]#0, [[INNER_LOOPS]]#1, [[INNER_LOOPS]]#2
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop)

  // CHECK: krnl.iterate([[OPT_INNER_LOOPS]]#0, [[OPT_INNER_LOOPS]]#1, [[OPT_INNER_LOOPS]]#2) with ([[INNER_LOOPS]]#0 -> %arg7 = 0 to 2, [[INNER_LOOPS]]#1 -> %arg8 = 0 to 6, [[INNER_LOOPS]]#2 -> %arg9 = 0 to 7) {
  // CHECK: [[R1PLUSK1:%.+]] = addi %arg5, %arg8 : index
  // CHECK: [[R2PLUSK2:%.+]] = addi %arg6, %arg9 : index
  // CHECK: [[DATA:%.+]] = load %arg0[%arg3, %arg7, [[R1PLUSK1]], [[R2PLUSK2]]] : memref<1x2x32x64xf32>
  // CHECK: [[KERNEL:%.+]] = load %arg1[%arg4, %arg7, %arg8, %arg9] : memref<5x2x6x7xf32>
  // CHECK: [[ACC_RES:%.+]] = load %0[%arg3, %arg4, %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: [[MUL:%.+]] = mulf [[DATA]], [[KERNEL]] : f32
  // CHECK: [[ADD:%.+]] = addf [[ACC_RES]], [[MUL]] : f32
  // CHECK: store [[ADD]], [[RES]][%arg3, %arg4, %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: }
  // CHECK: [[BIAS1:%.+]] = load [[RES]][%arg3, %arg4, %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: [[BIAS2:%.+]] = load %arg2[%arg4] : memref<5xf32>
  // CHECK: [[BIAS3:%.+]] = mulf [[BIAS1]], [[BIAS2]] : f32
  // CHECK: store [[BIAS3]], [[RES]][%arg3, %arg4, %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: }
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x5x27x58xf32>
}

// -----

func @test_conv_no_bias_no_pad_w_group(%arg0 : tensor<1x9x32x64xf32>, %arg1 : tensor<5x3x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 3 : i64} : (tensor<1x9x32x64xf32>, tensor<5x3x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_no_pad_w_group
  // CHECK: [[RES:%.+]] = alloc() : memref<1x5x27x58xf32>
  // CHECK: [[CONST0:%.+]] = constant 1 : index
  // CHECK: [[CONST1:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[CONST2:%.+]] = constant 3 : index
  // CHECK: [[OUTER_LOOPS:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[OPT_OUTER_LOOPS:%.+]]:3 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[OUTER_LOOPS]]#0, [[OUTER_LOOPS]]#1, [[OUTER_LOOPS]]#2
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop)

  // CHECK: krnl.iterate([[OPT_OUTER_LOOPS]]#0, [[OPT_OUTER_LOOPS]]#1, [[OPT_OUTER_LOOPS]]#2) with ([[OUTER_LOOPS]]#0 -> %arg2 = 0 to 1, [[OUTER_LOOPS]]#1 -> %arg3 = 0 to 3, [[OUTER_LOOPS]]#2 -> %arg4 = 0 to 1) {
  // CHECK: [[MUL1:%.+]] = muli %arg3, [[CONST0]] : index
  // CHECK: %[[ADD1:.+]] = addi [[MUL1]], %arg4 : index
  // CHECK: [[SPATIAL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_SPATIAL_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[SPATIAL_LOOPS]]#0, [[SPATIAL_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)

  // CHECK: krnl.iterate([[OPT_SPATIAL_LOOPS]]#0, [[OPT_SPATIAL_LOOPS]]#1) with ([[SPATIAL_LOOPS]]#0 -> %arg5 = 0 to 27, [[SPATIAL_LOOPS]]#1 -> %arg6 = 0 to 58) {
  // CHECK: store [[CONST1]], [[RES]][%arg2, %[[ADD1]], %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: [[INNER_LOOPS:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[OPT_INNER_LOOPS:%.+]]:3 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[INNER_LOOPS]]#0, [[INNER_LOOPS]]#1, [[INNER_LOOPS]]#2
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop)

  // CHECK: krnl.iterate([[OPT_INNER_LOOPS]]#0, [[OPT_INNER_LOOPS]]#1, [[OPT_INNER_LOOPS]]#2) with ([[INNER_LOOPS]]#0 -> %arg7 = 0 to 3, [[INNER_LOOPS]]#1 -> %arg8 = 0 to 6, [[INNER_LOOPS]]#2 -> %arg9 = 0 to 7) {
  // CHECK: [[MUL2:%.+]] = muli [[CONST2]], %arg3 : index
  // CHECK: [[ADD2:%.+]] = addi %arg7, [[MUL2]] : index
  // CHECK: [[R1PLUSK1:%.+]] = addi %arg5, %arg8 : index
  // CHECK: [[R2PLUSK2:%.+]] = addi %arg6, %arg9 : index
  // CHECK: [[DATA:%.+]] = load %arg0[%arg2, [[ADD2]], [[R1PLUSK1]], [[R2PLUSK2]]] : memref<1x9x32x64xf32>
  // CHECK: [[KERNEL:%.+]] = load %arg1[%[[ADD1]], %arg7, %arg8, %arg9] : memref<5x3x6x7xf32>
  // CHECK: [[ACC_RES:%.+]] = load %0[%arg2, %[[ADD1]], %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: [[MUL:%.+]] = mulf [[DATA]], [[KERNEL]] : f32
  // CHECK: [[ADD:%.+]] = addf [[ACC_RES]], [[MUL]] : f32
  // CHECK: store [[ADD]], [[RES]][%arg2, %[[ADD1]], %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: }
  // CHECK: }
  // CHECK: }

  // CHECK: return [[RES]] : memref<1x5x27x58xf32>
}

// -----

func @test_conv_no_bias_no_pad_w_strides(%arg0 : tensor<1x9x32x64xf32>, %arg1 : tensor<5x9x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : i64, strides = [2, 2]} : (tensor<1x9x32x64xf32>, tensor<5x9x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_no_pad_w_strides
  // CHECK: [[RES:%.+]] = alloc() : memref<1x5x14x29xf32>
  // CHECK: [[CONST0:%.+]] = constant 5 : index
  // CHECK: [[CONST1:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[CONST2:%.+]] = constant 9 : index
  // CHECK: [[OUTER_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_OUTER_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[OUTER_LOOPS]]#0, [[OUTER_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)

  // CHECK: krnl.iterate([[OPT_OUTER_LOOPS]]#0, [[OPT_OUTER_LOOPS]]#1) with ([[OUTER_LOOPS]]#0 -> %arg2 = 0 to 1, [[OUTER_LOOPS]]#1 -> %arg3 = 0 to 5) {
  // CHECK: [[SPATIAL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_SPATIAL_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[SPATIAL_LOOPS]]#0, [[SPATIAL_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)

  // CHECK: krnl.iterate([[OPT_SPATIAL_LOOPS]]#0, [[OPT_SPATIAL_LOOPS]]#1) with ([[SPATIAL_LOOPS]]#0 -> %arg4 = 0 to 14, [[SPATIAL_LOOPS]]#1 -> %arg5 = 0 to 29) {
  // CHECK: store [[CONST1]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<1x5x14x29xf32>
  // CHECK: [[INNER_LOOPS:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[OPT_INNER_LOOPS:%.+]]:3 = krnl.optimize_loops  {
  // CHECK: krnl.return_loops [[INNER_LOOPS]]#0, [[INNER_LOOPS]]#1, [[INNER_LOOPS]]#2
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop)

  // CHECK: krnl.iterate([[OPT_INNER_LOOPS]]#0, [[OPT_INNER_LOOPS]]#1, [[OPT_INNER_LOOPS]]#2) with ([[INNER_LOOPS]]#0 -> %arg6 = 0 to 9, [[INNER_LOOPS]]#1 -> %arg7 = 0 to 6, [[INNER_LOOPS]]#2 -> %arg8 = 0 to 7) {
  // CHECK: [[CONST_STRIDE1:%.+]] = constant 2 : index
  // CHECK: [[MUL1:%.+]] = muli [[CONST_STRIDE1]], %arg4 : index
  // CHECK: [[R1PLUSK1:%.+]] = addi [[MUL1]], %arg7 : index
  // CHECK: [[CONST_STRIDE2:%.+]] = constant 2 : index
  // CHECK: [[MUL2:%.+]] = muli [[CONST_STRIDE2]], %arg5 : index
  // CHECK: [[R2PLUSK2:%.+]] = addi [[MUL2]], %arg8 : index
  // CHECK: [[DATA:%.+]] = load %arg0[%arg2, %arg6, [[R1PLUSK1]], [[R2PLUSK2]]] : memref<1x9x32x64xf32>
  // CHECK: [[KERNEL:%.+]] = load %arg1[%arg3, %arg6, %arg7, %arg8] : memref<5x9x6x7xf32>
  // CHECK: [[ACC_RES:%.+]] = load %0[%arg2, %arg3, %arg4, %arg5] : memref<1x5x14x29xf32>
  // CHECK: [[MUL:%.+]] = mulf [[DATA]], [[KERNEL]] : f32
  // CHECK: [[ADD:%.+]] = addf [[ACC_RES]], [[MUL]] : f32
  // CHECK: store [[ADD]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<1x5x14x29xf32>
  // CHECK: }
  // CHECK: }
  // CHECK: }

  // CHECK: return [[RES]] : memref<1x5x14x29xf32>
}

// -----

func @test_batchnorm_testmode_Nd(%arg0: tensor<1x2x1x3xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> tensor<1x2x1x3xf32> {
  %0 = "onnx.BatchNormalizationTestMode"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x2x1x3xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x1x3xf32>
  return %0 : tensor<1x2x1x3xf32>

  // CHECK-LABEL: test_batchnorm_testmode_Nd
  // CHECK: [[RES:%.+]] = alloc() : memref<1x2x1x3xf32>
  // CHECK: [[EPSILON:%.+]] = constant 9.99999974E-6 : f32
  // CHECK: [[DEF_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: [[OPT_LOOPS:%.+]]:4 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg5 = 0 to 2) {
  // CHECK:   [[SCALE:%.+]] = load %arg1[%arg5] : memref<2xf32>
  // CHECK:   [[BIAS:%.+]] = load %arg2[%arg5] : memref<2xf32>
  // CHECK:   [[MEAN:%.+]] = load %arg3[%arg5] : memref<2xf32>
  // CHECK:   [[VARIANCE:%.+]] = load %arg4[%arg5] : memref<2xf32>
  // CHECK:   krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#2, [[OPT_LOOPS]]#3) with ([[DEF_LOOPS]]#0 -> %arg6 = 0 to 1, [[DEF_LOOPS]]#2 -> %arg7 = 0 to 1, [[DEF_LOOPS]]#3 -> %arg8 = 0 to 3) {
  // CHECK:     [[LOADED_VAL:%.+]] = load %arg0[%arg6, %arg5, %arg7, %arg8] : memref<1x2x1x3xf32>
  // CHECK:     [[DIVIDEND:%.+]] = subf [[LOADED_VAL]], [[MEAN]] : f32
  // CHECK:     [[ADJUSTED_VARIANCE:%.+]] = addf [[VARIANCE]], [[EPSILON]] : f32
  // CHECK:     [[DIVISOR:%.+]] = sqrt [[ADJUSTED_VARIANCE]] : f32
  // CHECK:     [[NORM:%.+]] = divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK:     [[SCALE_NORM:%.+]] = mulf [[SCALE]], [[NORM]] : f32
  // CHECK:     [[SHIFT_SCALE_NORM:%.+]] = addf [[SCALE_NORM]], [[BIAS]] : f32
  // CHECK:     store [[SHIFT_SCALE_NORM]], [[RES]][%arg6, %arg5, %arg7, %arg8] : memref<1x2x1x3xf32>
  // CHECK:   }
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x2x1x3xf32>
}

// -----

func @test_batchnorm_testmode_1d(%arg0: tensor<10xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<10xf32> {
  %0 = "onnx.BatchNormalizationTestMode"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<10xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>

  // CHECK-LABEL: test_batchnorm_testmode_1d
  // CHECK: [[RES:%.+]] = alloc() : memref<10xf32>
  // CHECK: [[EPSILON:%.+]] = constant 9.99999974E-6 : f32
  // CHECK: [[DEF_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: [[OPT_LOOPS:%.+]] = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]
  // CHECK: } : () -> !krnl.loop
  // CHECK: %[[ZERO_INDEX:.+]] = constant 0 : index
  // CHECK: [[SCALE:%.+]] = load %arg1[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: [[BIAS:%.+]] = load %arg2[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: [[MEAN:%.+]] = load %arg3[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: [[VARIANCE:%.+]] = load %arg4[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]) with ([[DEF_LOOPS]] -> %arg5 = 0 to 10) {
  // CHECK:   [[LOADED_VAL:%.+]] = load %arg0[%arg5] : memref<10xf32>
  // CHECK:   [[DIVIDEND:%.+]] = subf [[LOADED_VAL]], [[MEAN]] : f32
  // CHECK:   [[ADJUSTED_VARIANCE:%.+]] = addf [[VARIANCE]], [[EPSILON]] : f32
  // CHECK:   [[DIVISOR:%.+]] = sqrt [[ADJUSTED_VARIANCE]] : f32
  // CHECK:   [[NORM:%.+]] = divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK:   [[SCALE_NORM:%.+]] = mulf [[SCALE]], [[NORM]] : f32
  // CHECK:   [[SHIFT_SCALE_NORM:%.+]] = addf [[SCALE_NORM]], [[BIAS]] : f32
  // CHECK:   store [[SHIFT_SCALE_NORM]], [[RES]][%arg5] : memref<10xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<10xf32>
}

// -----

func @test_abs_float(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Abs"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_abs_float
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xf32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ABS:%.+]] = absf [[LOAD]] : f32
  // CHECK: store [[ABS]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_abs_int(%arg0 : tensor<?x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.Abs"(%arg0) : (tensor<?x10xi32>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_abs_int
  // CHECK: [[DIM_0:%.+]] = dim %arg0, 0 : memref<?x10xi32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: [[DIM_2:%.+]] = dim %arg0, 0 : memref<?x10xi32>
  // CHECK: krnl.iterate([[OPT_LOOPS]]#0, [[OPT_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<?x10xi32>
  // CHECK: [[ZERO:%.+]] = constant 0 : i32
  // CHECK: [[LESS_THAN_ZERO:%.+]] = cmpi "slt", [[LOAD]], [[ZERO]] : i32
  // CHECK: [[NEGATIVE_LOAD:%.+]] = subi [[ZERO]], [[LOAD]] : i32
  // CHECK: [[SELECT:%.+]] = select [[LESS_THAN_ZERO]], [[NEGATIVE_LOAD]], [[LOAD]] : i32
  // CHECK: store [[SELECT]], [[RES]][%arg1, %arg2] : memref<?x10xi32>
  // CHECK: return [[RES]] : memref<?x10xi32>
}

// -----

func @test_constant_pad1(%arg0: tensor<16x16xf32>) -> tensor<18x20xf32> {
  %0 = "onnx.PadConstantValuePad"(%arg0) {constant_value = 0.000000e+00 : f32, mode = "constant", pads = [0, 3, 2, 1]} : (tensor<16x16xf32>) -> tensor<18x20xf32>
  return %0 : tensor<18x20xf32>
  // CHECK-LABEL: test_constant_pad1
  // CHECK: [[RES:%.+]] = alloc() : memref<18x20xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS1:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS1]]#0, [[OPT_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 18, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 20) {
  // CHECK: [[CST:%.+]] = constant 0.000000e+00 : f32
  // CHECK: store [[CST]], [[RES]][%arg1, %arg2] : memref<18x20xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOPS2:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[OPT_LOOPS2:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1
  // CHECK: } : () -> (!krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS2]]#0, [[OPT_LOOPS2]]#1) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 16, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 16) {
  // CHECK: [[CST1:%.+]] = constant 3 : index
  // CHECK: [[ADD:%.+]] = addi [[CST1]], %arg2 : index
  // CHECK: [[LOAD:%.+]] = load %arg0[%arg1, %arg2] : memref<16x16xf32>
  // CHECK: store [[LOAD]], [[RES]][%arg1, [[ADD]]] : memref<18x20xf32>
  // CHECK: }
}

// -----

func @test_constant_dense_2d_value(%arg0: tensor<1xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[[0.0, 0.0], [1.0, 1.1], [2.0, 2.1]]> : tensor<3x2xf32>} : () -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_constant_dense_2d_value
  // CHECK: [[RES:%.+]] = "krnl.global"() {name = "constant_0", shape = [3, 2], value = dense<{{.*}}[0.000000e+00, 0.000000e+00], [1.000000e+00, 1.100000e+00], [2.000000e+00, 2.100000e+00]{{.*}}> : tensor<3x2xf32>} : () -> memref<3x2xf32>
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func @test_concat_1(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>, %arg2 : tensor<5x5x5x32xf32>) -> tensor<5x5x9x32xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = 2 } : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>)  -> tensor<5x5x9x32xf32>
  "std.return"(%1) : (tensor<5x5x9x32xf32>) -> ()

  // CHECK-LABEL: test_concat_1
  // CHECK: [[RES:%.+]] = alloc() : memref<5x5x9x32xf32>
  // CHECK: [[DEF_LOOPS0:%.+]]:4 = krnl.define_loops 4
  // CHECK: [[OPT_LOOPS0:%.+]]:4 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS0]]#0, [[DEF_LOOPS0]]#1, [[DEF_LOOPS0]]#2, [[DEF_LOOPS0]]#3
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS0]]#0, [[OPT_LOOPS0]]#1, [[OPT_LOOPS0]]#2, [[OPT_LOOPS0]]#3) with ([[DEF_LOOPS0]]#0 -> %arg3 = 0 to 5, [[DEF_LOOPS0]]#1 -> %arg4 = 0 to 5, [[DEF_LOOPS0]]#2 -> %arg5 = 0 to 1, [[DEF_LOOPS0]]#3 -> %arg6 = 0 to 32) {
  // CHECK: [[LOAD0:%.+]] = load %arg0[%arg3, %arg4, %arg5, %arg6] :  memref<5x5x1x32xf32>
  // CHECK: store [[LOAD0]], [[RES]][%arg3, %arg4, %arg5, %arg6] : memref<5x5x9x32xf32>

  // CHECK: [[DEF_LOOPS1:%.+]]:4 = krnl.define_loops 4
  // CHECK: [[OPT_LOOPS1:%.+]]:4 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1, [[DEF_LOOPS1]]#2, [[DEF_LOOPS1]]#3
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS1]]#0, [[OPT_LOOPS1]]#1, [[OPT_LOOPS1]]#2, [[OPT_LOOPS1]]#3) with ([[DEF_LOOPS1]]#0 -> %arg3 = 0 to 5, [[DEF_LOOPS1]]#1 -> %arg4 = 0 to 5, [[DEF_LOOPS1]]#2 -> %arg5 = 0 to 3, [[DEF_LOOPS1]]#3 -> %arg6 = 0 to 32) {
  // CHECK: [[OFF1:%.+]] = constant 1 : index
  // CHECK: [[ADD1:%.+]] = addi [[OFF1]], %arg5 : index
  // CHECK: [[LOAD1:%.+]] = load %arg1[%arg3, %arg4, %arg5, %arg6] :  memref<5x5x3x32xf32>
  // CHECK: store [[LOAD1]], [[RES]][%arg3, %arg4, [[ADD1]], %arg6] : memref<5x5x9x32xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:4 = krnl.define_loops 4
  // CHECK: [[OPT_LOOPS2:%.+]]:4 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2, [[DEF_LOOPS2]]#3
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_LOOPS2]]#0, [[OPT_LOOPS2]]#1, [[OPT_LOOPS2]]#2, [[OPT_LOOPS2]]#3) with ([[DEF_LOOPS2]]#0 -> %arg3 = 0 to 5, [[DEF_LOOPS2]]#1 -> %arg4 = 0 to 5, [[DEF_LOOPS2]]#2 -> %arg5 = 0 to 5, [[DEF_LOOPS2]]#3 -> %arg6 = 0 to 32) {
  // CHECK: [[OFF2:%.+]] = constant 4 : index
  // CHECK: [[ADD2:%.+]] = addi [[OFF2]], %arg5 : index
  // CHECK: [[LOAD2:%.+]] = load %arg2[%arg3, %arg4, %arg5, %arg6] :  memref<5x5x5x32xf32>
  // CHECK: store [[LOAD2]], [[RES]][%arg3, %arg4, [[ADD2]], %arg6] : memref<5x5x9x32xf32>

  // CHECK: return [[RES]] :  memref<5x5x9x32xf32>
}

// -----

func @test_pool_general_computation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-DAG: #{{.*}} = affine_map<(d0)[s0, s1, s2, s3, s4] -> ((s2 ceildiv s4) * s4 - s2, d0 * s3 - s2)>
  // CHECK-DAG: #{{.*}} = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0, d0 * s3 + (s1 - 1) * s4 - s2 + 1)>
  // CHECK-DAG: #{{.*}} = affine_map<() -> (0)>
  // CHECK-DAG: #{{.*}} = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>

  // CHECK-LABEL: @test_pool_general_computation

  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = constant 0.000000e+00 : f32

  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: [[OPT_OUTPUT_LOOPS:%.+]]:4 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_OUTPUT_LOOPS]]#0, [[OPT_OUTPUT_LOOPS]]#1, [[OPT_OUTPUT_LOOPS]]#2, [[OPT_OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31) {

  // CHECK:   store [[IDENTITY]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   [[OPT_POOL_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:     krnl.return_loops [[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1
  // CHECK:   } : () -> (!krnl.loop, !krnl.loop)
  // CHECK:   krnl.iterate([[OPT_POOL_LOOPS]]#0, [[OPT_POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #map3(%arg3)[%c32, %c2, %c0, %c1, %c1_0], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #map3(%arg4)[%c32_1, %c2_2, %c0_3, %c1_4, %c1_5]) {
  // CHECK:     {{.*}} = load %arg0[%arg1, %arg2, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     {{.*}} = load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:     store {{.*}}, [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   }

  // CHECK:   {{.*}} = load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   store {{.*}}, [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK: }
}

// -----

func @test_averagepool_identity_value(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_averagepool_identity_value
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = constant 0.000000e+00 : f32
  // CHECK: store [[IDENTITY]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
}

// -----

func @test_maxpool_identity_value(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_maxpool_identity_value
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = constant 0xFF800000 : f32
  // CHECK: store [[IDENTITY]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
}

// -----

func @test_averagepool_pooling_operation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_averagepool_pooling_operation
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>

  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: [[OPT_OUTPUT_LOOPS:%.+]]:4 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_OUTPUT_LOOPS]]#0, [[OPT_OUTPUT_LOOPS]]#1, [[OPT_OUTPUT_LOOPS]]#2, [[OPT_OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31) {

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   [[OPT_POOL_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:     krnl.return_loops [[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1
  // CHECK:   } : () -> (!krnl.loop, !krnl.loop)
  // CHECK:   krnl.iterate([[OPT_POOL_LOOPS]]#0, [[OPT_POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #map3(%arg3)[%c32, %c2, %c0, %c1, %c1_0], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #map3(%arg4)[%c32_1, %c2_2, %c0_3, %c1_4, %c1_5]) {

  // CHECK:     [[INPUT_LOAD:%.+]] = load %arg0[%arg1, %arg2, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     [[OUTPUT_LOAD:%.+]] = load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:     [[SUM:%.+]] = addf [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     store [[SUM]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   }

  // CHECK:   [[NUMERATOR:%.+]] = load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   [[AVERAGE:%.+]] = divf [[NUMERATOR]], {{.*}} : f32
  // CHECK:   store [[AVERAGE]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK: }
}

// -----

func @test_maxpool_pooling_operation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_maxpool_pooling_operation
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>

  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: [[OPT_OUTPUT_LOOPS:%.+]]:4 = krnl.optimize_loops  {
  // CHECK:   krnl.return_loops [[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3
  // CHECK: } : () -> (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop)
  // CHECK: krnl.iterate([[OPT_OUTPUT_LOOPS]]#0, [[OPT_OUTPUT_LOOPS]]#1, [[OPT_OUTPUT_LOOPS]]#2, [[OPT_OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31) {

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   [[OPT_POOL_LOOPS:%.+]]:2 = krnl.optimize_loops  {
  // CHECK:     krnl.return_loops [[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1
  // CHECK:   } : () -> (!krnl.loop, !krnl.loop)
  // CHECK:   krnl.iterate([[OPT_POOL_LOOPS]]#0, [[OPT_POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #map3(%arg3)[%c32, %c2, %c0, %c1, %c1_0], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #map3(%arg4)[%c32_1, %c2_2, %c0_3, %c1_4, %c1_5]) {

  // CHECK:     [[INPUT_LOAD:%.+]] = load %arg0[%arg1, %arg2, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     [[OUTPUT_LOAD:%.+]] = load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:     [[GREATER:%.+]] = cmpf "ogt", [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     [[SELECT:%.+]] = select [[GREATER]], [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     store [[SELECT]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   }

  // CHECK-NOT:   {{.*}} = load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK-NOT:   store {{.*}}, [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK: }
}

