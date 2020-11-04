// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// ----

func @test_no_argument_1() -> () {
}

func @test_no_argument_2() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value =  dense<[[1.000000e+0, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>} : () -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

}

// CHECK: test_no_argument_1
// CHECK-NEXT: test_no_argument_2
// CHECK: [[RES:%.+]] = "{{.*}}"({{.*}}) {{.*}} : ({{.*}}) -> memref<2x2xf32>
// CHECK: return [[RES]] : memref<2x2xf32>

// -----

func @test_elementwise_op_with_scalar_values_1(%arg0 : tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Exp"(%arg0) : (tensor<f32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_elementwise_op_with_scalar_values_1
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[] : memref<f32>
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: affine.store [[EXP]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func @test_elementwise_op_with_scalar_values_2(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_elementwise_op_with_scalar_values_2
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[] : memref<f32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[] : memref<f32>
  // CHECK: [[ADD:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[ADD]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func @test_elementwise_op_with_scalar_values_3(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_elementwise_op_with_scalar_values_3
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[] : memref<f32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[] : memref<f32>
  // CHECK: [[ADD1:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[LOAD3:%.+]] = affine.load %arg2[] : memref<f32>
  // CHECK: [[ADD2:%.+]] = addf [[ADD1]], [[LOAD3]] : f32
  // CHECK: affine.store [[ADD2]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func @test_add(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_add
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[ADDF:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[ADDF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_mul(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_mul
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MULF:%.+]] = mulf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[MULF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_div(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_div
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[DIVF:%.+]] = divf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[DIVF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_sub(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sub
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[SUBF:%.+]] = subf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[SUBF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_and(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<*xi1> {
  %0 = "onnx.And"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  "std.return"(%0) : (tensor<*xi1>) -> ()

  // CHECK-LABEL: test_and
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xi1>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[AND:%.+]] = and [[LOAD1]], [[LOAD2]] : i1
  // CHECK: affine.store [[AND]], [[RES]][%arg2, %arg3] : memref<10x10xi1>
  // CHECK: return [[RES]] : memref<10x10xi1>
}

// -----

func @test_or(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<*xi1> {
  %0 = "onnx.Or"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  "std.return"(%0) : (tensor<*xi1>) -> ()

  // CHECK-LABEL: test_or
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xi1>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[OR:%.+]] = or [[LOAD1]], [[LOAD2]] : i1
  // CHECK: affine.store [[OR]], [[RES]][%arg2, %arg3] : memref<10x10xi1>
  // CHECK: return [[RES]] : memref<10x10xi1>
}

// -----

func @test_xor(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<*xi1> {
  %0 = "onnx.Xor"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  "std.return"(%0) : (tensor<*xi1>) -> ()

  // CHECK-LABEL: test_xor
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xi1>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[XOR:%.+]] = xor [[LOAD1]], [[LOAD2]] : i1
  // CHECK: affine.store [[XOR]], [[RES]][%arg2, %arg3] : memref<10x10xi1>
  // CHECK: return [[RES]] : memref<10x10xi1>
}

// -----

func @test_exp(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Exp"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_exp
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
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_tanh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Tanh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_tanh
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
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_sinh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sinh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sinh
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
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_cosh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Cosh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_cosh
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
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_cos(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Cos"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_cos
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[COS:%.+]] = cos [[LOAD]] : f32
  // CHECK: affine.store [[COS]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_log(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Log"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_log
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[LOG:%.+]] = log [[LOAD]] : f32
  // CHECK: affine.store [[LOG]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_sigmoid(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sigmoid
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
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_relu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_relu
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
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_reshape(%arg0 : tensor<?x10xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<4xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reshape
  // CHECK: [[TYPE_IN_BYTES_0:%.+]] = constant 4 : i64
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[DIM_0_CAST:%.+]] = index_cast [[DIM_0]] : index to i64
  // CHECK: [[MUL_0:%.+]] = muli [[TYPE_IN_BYTES_0]], [[DIM_0_CAST]] : i64
  // CHECK: [[CONSTANT_0:%.+]] = constant 10 : i64
  // CHECK: [[TENSOR_SIZE:%.+]] = muli [[MUL_0]], [[CONSTANT_0]] : i64

  // CHECK: [[TYPE_IN_BYTES_1:%.+]] = constant 4 : i64
  // CHECK: %[[CONSTANT_1:.+]] = constant 0 : index
  // CHECK: [[LOAD_0:%.+]] = affine.load %arg1[%[[CONSTANT_1]]] : memref<4xi64>
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_1:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: [[DIM_1_CAST:%.+]] = index_cast [[DIM_1]] : index to i64
  // CHECK: [[CONSTANT_2:%.+]] = constant 0 : i64
  // CHECK: [[CMP_0:%.+]] = cmpi "eq", [[LOAD_0]], [[CONSTANT_2]] : i64
  // CHECK: [[SELECT_0:%.+]] = select [[CMP_0]], [[DIM_1_CAST]], [[LOAD_0]] : i64
  // CHECK: [[MUL_1:%.+]] = muli [[TYPE_IN_BYTES_1]], [[SELECT_0]] : i64

  // CHECK: %[[CONSTANT_3:.+]] = constant 1 : index
  // CHECK: [[LOAD_1:%.+]] = affine.load %arg1[%[[CONSTANT_3]]] : memref<4xi64>
  // CHECK: [[CONSTANT_3:%.+]] = constant 10 : i64
  // CHECK: [[CONSTANT_4:%.+]] = constant 0 : i64
  // CHECK: [[CMP_1:%.+]] = cmpi "eq", [[LOAD_1]], [[CONSTANT_4]] : i64
  // CHECK: [[SELECT_1:%.+]] = select [[CMP_1]], [[CONSTANT_3]], [[LOAD_1]] : i64
  // CHECK: [[MUL_2:%.+]] = muli [[MUL_1]], [[SELECT_1]] : i64

  // CHECK: %[[CONSTANT_5:.+]] = constant 2 : index
  // CHECK: [[LOAD_2:%.+]] = affine.load %arg1[%[[CONSTANT_5]]] : memref<4xi64>
  // CHECK: [[MUL_3:%.+]] = muli [[MUL_2]], [[LOAD_2]] : i64

  // CHECK: %[[CONSTANT_6:.+]] = constant 3 : index
  // CHECK: [[LOAD_3:%.+]] = affine.load %arg1[%[[CONSTANT_6]]] : memref<4xi64>
  // CHECK: [[MUL_4:%.+]] = muli [[MUL_3]], [[LOAD_3]] : i64

  // CHECK: [[CONSTANT_7:%.+]] = constant 0 : i64
  // CHECK: [[SUB_0:%.+]] = subi [[CONSTANT_7]], [[MUL_4]] : i64

  // CHECK: [[CONSTANT_8:%.+]] = constant -1 : i64
  // CHECK: [[CMP_2:%.+]] = cmpi "eq", [[SELECT_0]], [[CONSTANT_8]] : i64
  // CHECK: [[DIVISIGNED_0:%.+]] = divi_signed [[TENSOR_SIZE]], [[SUB_0]] : i64
  // CHECK: [[SELECT_2:%.+]] = select [[CMP_2]], [[DIVISIGNED_0]], [[SELECT_0]] : i64
  // CHECK: [[CAST_0:%.+]] = index_cast [[SELECT_2]] : i64 to index

  // CHECK: [[CMP_3:%.+]] = cmpi "eq", [[SELECT_1]], [[CONSTANT_8]] : i64
  // CHECK: [[DIVISIGNED_1:%.+]] = divi_signed [[TENSOR_SIZE]], [[SUB_0]] : i64
  // CHECK: [[SELECT_3:%.+]] = select [[CMP_3]], [[DIVISIGNED_1]], [[SELECT_1]] : i64
  // CHECK: [[CAST_1:%.+]] = index_cast [[SELECT_3]] : i64 to index

  // CHECK: [[CMP_4:%.+]] = cmpi "eq", [[LOAD_2]], [[CONSTANT_8]] : i64
  // CHECK: [[DIVISIGNED_2:%.+]] = divi_signed [[TENSOR_SIZE]], [[SUB_0]] : i64
  // CHECK: [[SELECT_4:%.+]] = select [[CMP_4]], [[DIVISIGNED_2]], [[LOAD_2]] : i64
  // CHECK: [[CAST_2:%.+]] = index_cast [[SELECT_4]] : i64 to index

  // CHECK: [[CMP_5:%.+]] = cmpi "eq", [[LOAD_3]], [[CONSTANT_8]] : i64
  // CHECK: [[DIVISIGNED_3:%.+]] = divi_signed [[TENSOR_SIZE]], [[SUB_0]] : i64
  // CHECK: [[SELECT_5:%.+]] = select [[CMP_5]], [[DIVISIGNED_3]], [[LOAD_3]] : i64
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
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[ADD:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[ADD]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_max(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Max"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_max
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MAX:%.+]] = cmpf "ogt", [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[RELU_RES:%.+]] = select [[MAX]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[RELU_RES]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_min(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Min"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_min
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MIN:%.+]] = cmpf "olt", [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[RELU_RES:%.+]] = select [[MIN]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[RELU_RES]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_elu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Elu"(%arg0) {alpha=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_elu
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
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_leakyrelu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha=1.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_leakyrelu
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
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_selu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Selu"(%arg0) {alpha=1.0:f32, gamma=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_selu
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
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_hardsigmoid(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha=1.0:f32, beta=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_hardsigmoid
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
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_reciprocal(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Reciprocal"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reciprocal
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
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_softplus(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softplus"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_softplus
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[ADD:%.+]] = addf [[EXP]], [[ONE]] : f32
  // CHECK: [[SOFTPLUS_RES:%.+]] = log [[ADD]] : f32
  // CHECK: affine.store [[SOFTPLUS_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_softsign(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softsign"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_softsign
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ABS:%.+]] = absf [[LOAD]] : f32
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[ADD:%.+]] = addf [[ABS]], [[ONE]] : f32
  // CHECK: [[SOFTSIGN_RES:%.+]] = divf [[LOAD]], [[ADD]] : f32
  // CHECK: affine.store [[SOFTSIGN_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_add_with_broadcasting(%arg0 : tensor<?xf32>, %arg1 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?xf32>, tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_add_with_broadcasting
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM1:%.+]] = dim %arg1, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM1]]) : memref<?x10xf32>
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM2:%.+]] = dim %arg0, [[C0_0]] : memref<?xf32>
  // CHECK: [[ONE:%.+]] = constant 1 : index
  // CHECK: [[IS_ONE:%.+]] = cmpi "eq", [[DIM2]], [[ONE]] : index
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_1:%.+]] = constant 0 : index
  // CHECK: [[DIM3:%.+]] = dim [[RES]], [[C0_1]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM3]], [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[ZERO:%.+]] = constant 0 : index
  // CHECK: %[[SELECT1:.+]] = select [[IS_ONE]], [[ZERO]], %arg3 : index
  // CHECK: [[LOAD1:%.+]] = load %arg0[%[[SELECT1]]] : memref<?xf32>
  // CHECK: [[LOAD2:%.+]] = load %arg1[%arg2, %arg3] : memref<?x10xf32>
  // CHECK: [[ADD:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: affine.store [[ADD]], [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_reducemax(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMax"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducemax
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 0xFF800000 : f32
  // CHECK: affine.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[CMP:%.+]] = cmpf "ogt", [[LOAD2]], [[LOAD1]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[LOAD2]], [[LOAD1]] : f32
  // CHECK: store [[SELECT]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func @test_reducemin(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMin"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducemin
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 0x7F800000 : f32
  // CHECK: affine.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[CMP:%.+]] = cmpf "olt", [[LOAD2]], [[LOAD1]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[LOAD2]], [[LOAD1]] : f32
  // CHECK: affine.store [[SELECT]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func @test_reduceprod(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceProd"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduceprod
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 1.000000e+00 : f32
  // CHECK: affine.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[REDUCE:%.+]] = mulf [[LOAD2]], [[LOAD1]] : f32
  // CHECK: affine.store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func @test_reducesum(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceSum"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducesum
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 0.000000e+00 : f32
  // CHECK: affine.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[REDUCE:%.+]] = addf [[LOAD2]], [[LOAD1]] : f32
  // CHECK: affine.store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

/// Check ReduceMean with f32.
func @test_reducemean_f32(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMean"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducemean_f32
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 0.000000e+00 : f32
  // CHECK: affine.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = affine.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[REDUCE:%.+]] = addf [[LOAD2]], [[LOAD1]] : f32
  // CHECK: affine.store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }

  // CHECK: [[INPUT_SIZE:%.+]] = constant 1.200000e+01 : f32
  // CHECK: [[OUTPUT_SIZE:%.+]] = constant 6.000000e+00 : f32
  // CHECK: [[DIVISOR:%.+]] = divf [[INPUT_SIZE]], [[OUTPUT_SIZE]] : f32
  // CHECK: [[DEF_MEAN_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_MEAN_LOOPS]]#0, [[DEF_MEAN_LOOPS]]#1) with ([[DEF_MEAN_LOOPS]]#0 -> %arg1 = 0 to 3, [[DEF_MEAN_LOOPS]]#1 -> %arg2 = 0 to 2) {
  // CHECK:   [[LOAD3:%.+]] = affine.load [[RES]][%arg1, %arg2] : memref<3x2xf32>
  // CHECK:   [[MEAN:%.+]] = divf [[LOAD3]], [[DIVISOR]] : f32
  // CHECK:   affine.store [[MEAN]], [[RES]][%arg1, %arg2] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

/// Check ReduceMean with i32.
func @test_reducemean_i32(%arg0 : tensor<3x2x2xi32>) -> tensor<*xi32> {
  %0 ="onnx.ReduceMean"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xi32>)-> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_reducemean_i32
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xi32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 0 : i32
  // CHECK: affine.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xi32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xi32>
  // CHECK: [[LOAD2:%.+]] = affine.load [[RES]][%arg1, %arg3] : memref<3x2xi32>
  // CHECK: [[REDUCE:%.+]] = addi [[LOAD2]], [[LOAD1]] : i32
  // CHECK: affine.store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xi32>
  // CHECK: }

  // CHECK: [[INPUT_SIZE:%.+]] = constant 12 : i32
  // CHECK: [[OUTPUT_SIZE:%.+]] = constant 6 : i32
  // CHECK: [[DIVISOR:%.+]] = divi_signed [[INPUT_SIZE]], [[OUTPUT_SIZE]] : i32
  // CHECK: [[DEF_MEAN_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_MEAN_LOOPS]]#0, [[DEF_MEAN_LOOPS]]#1) with ([[DEF_MEAN_LOOPS]]#0 -> %arg1 = 0 to 3, [[DEF_MEAN_LOOPS]]#1 -> %arg2 = 0 to 2) {
  // CHECK:   [[LOAD3:%.+]] = affine.load [[RES]][%arg1, %arg2] : memref<3x2xi32>
  // CHECK:   [[MEAN:%.+]] = divi_signed [[LOAD3]], [[DIVISOR]] : i32
  // CHECK:   affine.store [[MEAN]], [[RES]][%arg1, %arg2] : memref<3x2xi32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xi32>
}

// -----

/// Check computing the divisor in ReduceMean
/// when the input has unknown dimensions and is of i32.
func @test_reducemean_i32_unknown_dims(%arg0 : tensor<3x?x2xi32>) -> tensor<*xi32> {
  %0 ="onnx.ReduceMean"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x?x2xi32>)-> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()
  // CHECK-LABEL: test_reducemean_i32_unknown_dims
  // CHECK: [[INPUT_SIZE_CONSTANT:%.+]] = constant 6 : i32
  // CHECK: [[ONE:%.+]] = constant 1 : index
  // CHECK: [[DIM:%.+]] = dim %arg0, [[ONE]] : memref<3x?x2xi32>
  // CHECK: [[UNKNOWN_DIM:%.+]] = index_cast [[DIM]] : index to i32
  // CHECK: [[INPUT_SIZE:%.+]] = muli [[INPUT_SIZE_CONSTANT]], [[UNKNOWN_DIM]] : i32
  // CHECK: [[OUTPUT_SIZE:%.+]] = constant 6 : i32
  // CHECK: [[DIVISOR:%.+]] = divi_signed [[INPUT_SIZE]], [[OUTPUT_SIZE]] : i32
}

// -----

/// Check computing the divisor in ReduceMean
/// when the input has unknown dimensions and is of f32.
func @test_reducemean_f32_unknown_dims(%arg0 : tensor<3x?x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMean"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x?x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_reducemean_f32_unknown_dims
  // CHECK: [[INPUT_SIZE_CONSTANT:%.+]] = constant 6.000000e+00 : f32
  // CHECK: [[ONE:%.+]] = constant 1 : index
  // CHECK: [[DIM:%.+]] = dim %arg0, [[ONE]] : memref<3x?x2xf32>
  // CHECK: [[UNKNOWN_DIM_i64:%.+]] = index_cast [[DIM]] : index to i64
  // CHECK: [[UNKNOWN_DIM:%.+]] = uitofp [[UNKNOWN_DIM_i64]] : i64 to f32
  // CHECK: [[INPUT_SIZE:%.+]] = mulf [[INPUT_SIZE_CONSTANT]], [[UNKNOWN_DIM]] : f32
  // CHECK: [[OUTPUT_SIZE:%.+]] = constant 6.000000e+00 : f32
  // CHECK: [[DIVISOR:%.+]] = divf [[INPUT_SIZE]], [[OUTPUT_SIZE]] : f32
}

// -----

func @test_softmax(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis=1: si64} : (tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_softmax
  // CHECK: [[MAX:%.+]] = alloc() : memref<f32>
  // CHECK: [[SUM:%.+]] = alloc() : memref<f32>
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[CST:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[CST_0:%.+]] = constant 0xFF800000 : f32
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to 10) {
  // CHECK: affine.store [[CST]], [[SUM]][] : memref<f32>
  // CHECK: affine.store [[CST_0]], [[MAX]][] : memref<f32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK:   [[LOAD1:%.+]] = affine.load [[MAX]][] : memref<f32>
  // CHECK:   [[LOAD2:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<10x10xf32>
  // CHECK:   [[COND:%.+]] = cmpf "ogt", [[LOAD1]], [[LOAD2]] : f32
  // CHECK:   [[SELECT:%.+]] = select [[COND]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK:   affine.store [[SELECT]], [[MAX]][] : memref<f32>
  // CHECK: }
  // CHECK: [[LOAD_MAX:%.+]] = affine.load [[MAX]][] : memref<f32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK:   [[LOAD1]] = affine.load [[SUM]][] : memref<f32>
  // CHECK:   [[LOAD2]] = affine.load %arg0[%arg1, %arg2] : memref<10x10xf32>
  // CHECK:   [[SUB:%.+]] = subf [[LOAD2]], [[LOAD_MAX]] : f32
  // CHECK:   [[EXP:%.+]] = exp [[SUB]] : f32
  // CHECK:   [[ADD:%.+]] = addf [[LOAD1]], [[EXP]] : f32
  // CHECK:   affine.store [[ADD]], [[SUM]][] : memref<f32>
  // CHECK:   affine.store [[EXP]], [[RES]][%arg1, %arg2] : memref<10x10xf32>
  // CHECK: }
  // CHECK: [[LOAD_SUM:%.+]] = affine.load [[SUM]][] : memref<f32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK:   [[LOAD1]] = affine.load [[RES]][%arg1, %arg2] : memref<10x10xf32>
  // CHECK:   [[DIV:%.+]] = divf [[LOAD1]], [[LOAD_SUM]] : f32
  // CHECK:   affine.store [[DIV]], [[RES]][%arg1, %arg2] : memref<10x10xf32>
  // CHECK: }
  // CHECK: }
  // CHECK: dealloc [[SUM]] : memref<f32>
  // CHECK: dealloc [[MAX]] : memref<f32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func @test_gemm(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_gemm
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[ALPHA:%.+]] = constant 1.000000e+00 : f32
  // CHECK: [[BETA:%.+]] = constant 5.000000e+00 : f32
  // CHECK: [[DEF_LOOPS:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg3 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg4 = 0 to 10) {
  // CHECK: krnl.iterate([[DEF_LOOPS]]#2) with ([[DEF_LOOPS]]#2 -> %arg5 = 0 to 5) {
  // CHECK: [[A:%.+]] = affine.load %arg0[%arg5, %arg3] : memref<5x10xf32>
  // CHECK: [[B:%.+]] = affine.load %arg1[%arg5, %arg4] : memref<5x10xf32>
  // CHECK: [[Y:%.+]] = affine.load [[RES]][%arg3, %arg4] : memref<10x10xf32>
  // CHECK: [[AB:%.+]] = mulf [[A]], [[B]] : f32
  // CHECK: [[SUM:%.+]] = addf [[Y]], [[AB]] : f32
  // CHECK: affine.store [[SUM]], [[RES]][%arg3, %arg4] : memref<10x10xf32>
  // CHECK: }
  // CHECK: [[LOAD_Y:%.+]] = affine.load [[RES]][%arg3, %arg4] : memref<10x10xf32>
  // CHECK: [[ALPHA_AB:%.+]] = mulf [[ALPHA]], [[LOAD_Y]] : f32
  // CHECK: [[C:%.+]] = affine.load %arg2[%arg4] : memref<10xf32>
  // CHECK: [[BETA_C:%.+]] = mulf [[BETA]], [[C]] : f32
  // CHECK: [[Y_RES:%.+]] = addf [[ALPHA_AB]], [[BETA_C]] : f32
  // CHECK: affine.store [[Y_RES]], [[RES]][%arg3, %arg4] : memref<10x10xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<10x10xf32>
  // CHECK: }
}

// -----

func @test_sqrt(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sqrt
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[SQRT:%.+]] = sqrt [[LOAD]] : f32
  // CHECK: affine.store [[SQRT]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
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

  // CHECK: [[DEF_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg2 = 0 to 20, [[DEF_LOOPS]]#2 -> %arg3 = 0 to 30, [[DEF_LOOPS]]#3 -> %arg4 = 0 to 40) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3, %arg4] : memref<10x20x30x40xf32>
  // CHECK: affine.store [[LOAD]], [[RES1]][%arg4, %arg3, %arg2, %arg1] : memref<40x30x20x10xf32>

  // CHECK: [[DEF_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to 40, [[DEF_LOOPS]]#1 -> %arg2 = 0 to 30, [[DEF_LOOPS]]#2 -> %arg3 = 0 to 20, [[DEF_LOOPS]]#3 -> %arg4 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load [[RES1]][%arg1, %arg2, %arg3, %arg4] : memref<40x30x20x10xf32>
  // CHECK: affine.store [[LOAD]], [[RES0]][%arg1, %arg4, %arg2, %arg3] : memref<40x10x30x20xf32>

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
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[MINUS_ONE:%.+]] = constant {{-1.+}} : f32
  // CHECK: [[GTZERO:%.+]] = cmpf "ogt", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[SELECT_PLUS:%.+]] = select [[GTZERO]], [[ONE]], [[MINUS_ONE]] : f32
  // CHECK: [[EQZERO:%.+]] = cmpf "oeq", [[LOAD]], [[ZERO]] : f32
  // CHECK: [[SIGN_RES:%.+]] = select [[EQZERO]], [[ZERO]], [[SELECT_PLUS]] : f32
  // CHECK: affine.store [[SIGN_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_sign_i(%arg0 : tensor<?x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.Sign"(%arg0) : (tensor<?x10xi32>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_sign_i
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xi32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xi32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xi32>
  // CHECK: [[ZERO:%.+]] = constant 0 : i32
  // CHECK: [[ONE:%.+]] = constant 1 : i32
  // CHECK: [[MINUS_ONE:%.+]] = constant -1 : i32
  // CHECK: [[GTZERO:%.+]] = cmpi "sgt", [[LOAD]], [[ZERO]] : i32
  // CHECK: [[SELECT_PLUS:%.+]] = select [[GTZERO]], [[ONE]], [[MINUS_ONE]] : i32
  // CHECK: [[EQZERO:%.+]] = cmpi "eq", [[LOAD]], [[ZERO]] : i32
  // CHECK: [[SIGN_RES:%.+]] = select [[EQZERO]], [[ZERO]], [[SELECT_PLUS]] : i32
  // CHECK: affine.store [[SIGN_RES]], [[RES]][%arg1, %arg2] : memref<?x10xi32>
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
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK:   affine.store [[CONSTANT]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK:   [[DEF_LOOPS_REDUCE:%.+]] = krnl.define_loops 1
  // CHECK:   krnl.iterate([[DEF_LOOPS_REDUCE]]) with ([[DEF_LOOPS_REDUCE]] -> %arg4 = 0 to 5) {
  // CHECK:     [[LOAD_0:%.+]] = affine.load %arg0[%arg2, %arg4] : memref<10x5xf32>
  // CHECK:     [[LOAD_1:%.+]] = affine.load %arg1[%arg4, %arg3] : memref<5x10xf32>
  // CHECK:     [[LOAD_RES:%.+]] = affine.load [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK:     [[MUL:%.+]] = mulf [[LOAD_0]], [[LOAD_1]] : f32
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_RES]], [[MUL]] : f32
  // CHECK:     affine.store [[ADD]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
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
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[LOOPS]]#0 -> %arg2 = 0 to 2, [[LOOPS]]#1 -> %arg3 = 0 to 3) {
  // CHECK:   krnl.iterate([[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) with ([[LOOPS]]#2 -> %arg4 = 0 to 10, [[LOOPS]]#3 -> %arg5 = 0 to 10) {
  // CHECK:     affine.store [[CONSTANT]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<2x3x10x10xf32>
  // CHECK:     [[LOOPS_REDUCE:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[DEF_LOOPS_REDUCE]]) with ([[LOOPS_REDUCE]] -> %arg6 = 0 to 5) {
  // CHECK:       [[LOAD_0:%.+]] = affine.load %arg0[%arg4, %arg6] : memref<10x5xf32>
  // CHECK:       [[LOAD_1:%.+]] = affine.load %arg1[%arg2, %arg3, %arg6, %arg5] : memref<2x3x5x10xf32>
  // CHECK:       [[LOAD_RES:%.+]] = affine.load [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<2x3x10x10xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_0]], [[LOAD_1]] : f32
  // CHECK:       [[ADD:%.+]] = addf [[LOAD_RES]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<2x3x10x10xf32>
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
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[LOOPS]]#0 -> %arg2 = 0 to 2, [[LOOPS]]#1 -> %arg3 = 0 to 3) {
  // CHECK:   krnl.iterate([[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) with ([[LOOPS]]#2 -> %arg4 = 0 to 10, [[LOOPS]]#3 -> %arg5 = 0 to 10) {
  // CHECK:     store [[CONSTANT]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<2x3x10x10xf32>
  // CHECK:     [[LOOPS_REDUCE:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[DEF_LOOPS_REDUCE]]) with ([[LOOPS_REDUCE]] -> %arg6 = 0 to 5) {
  // CHECK:       [[LOAD_0:%.+]] = affine.load %arg0[%arg2, %arg3, %arg4, %arg6] : memref<2x3x10x5xf32>
  // CHECK:       [[LOAD_1:%.+]] = affine.load %arg1[%arg2, %arg3, %arg6, %arg5] : memref<2x3x5x10xf32>
  // CHECK:       [[LOAD_RES:%.+]] = affine.load [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<2x3x10x10xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_0]], [[LOAD_1]] : f32
  // CHECK:       [[ADD:%.+]] = addf [[LOAD_RES]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<2x3x10x10xf32>
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
  // CHECK: krnl.iterate([[DEF_LOOPS]]) with ([[LOOPS]] -> %arg2 = 0 to 10) {
  // CHECK:   affine.store [[CONSTANT]], [[RES]][%arg2] : memref<10xf32>
  // CHECK:   [[LOOPS_REDUCE:%.+]] = krnl.define_loops 1
  // CHECK:   krnl.iterate([[DEF_LOOPS_REDUCE]]) with ([[LOOPS_REDUCE]] -> %arg3 = 0 to 5) {
  // CHECK:     [[LOAD_0:%.+]] = affine.load %arg0[%arg3] : memref<5xf32>
  // CHECK:     [[LOAD_1:%.+]] = affine.load %arg1[%arg3, %arg2] : memref<5x10xf32>
  // CHECK:     [[LOAD_RES:%.+]] = affine.load [[RES]][%arg2] : memref<10xf32>
  // CHECK:     [[MUL:%.+]] = mulf [[LOAD_0]], [[LOAD_1]] : f32
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_RES]], [[MUL]] : f32
  // CHECK:     affine.store [[ADD]], [[RES]][%arg2] : memref<10xf32>
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
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg1, [[C0]] : memref<?x5x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_1:%.+]] = dim [[RES]], [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to [[DIM_1]]) {
  // CHECK:   krnl.iterate([[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK:     affine.store [[CONSTANT]], [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK:     [[DEF_LOOPS_REDUCE:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[DEF_LOOPS_REDUCE]]) with ([[DEF_LOOPS_REDUCE]] -> %arg4 = 0 to 5) {
  // CHECK:       [[LOAD_0:%.+]] = affine.load %arg0[%arg4] : memref<5xf32>
  // CHECK:       [[LOAD_1:%.+]] = affine.load %arg1[%arg2, %arg4, %arg3] : memref<?x5x10xf32>
  // CHECK:       [[LOAD_RES:%.+]] = affine.load [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_0]], [[LOAD_1]] : f32
  // CHECK:       [[ADD:%.+]] = addf [[LOAD_RES]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[RES]][%arg2, %arg3] : memref<?x10xf32>
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
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10x5xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_1:%.+]] = dim [[RES]], [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[LOOPS]]#0) with ([[LOOPS]]#0 -> %arg2 = 0 to [[DIM_1]]) {
  // CHECK:   krnl.iterate([[LOOPS]]#1) with ([[LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK:     affine.store [[CONSTANT]], [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK:     [[LOOPS_REDUCE:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[LOOPS_REDUCE]]) with ([[LOOPS_REDUCE]] -> %arg4 = 0 to 5) {
  // CHECK:       [[LOAD_0:%.+]] = affine.load %arg0[%arg2, %arg3, %arg4] : memref<?x10x5xf32>
  // CHECK:       [[LOAD_1:%.+]] = affine.load %arg1[%arg4] : memref<5xf32>
  // CHECK:       [[LOAD_RES:%.+]] = affine.load [[RES]][%arg2, %arg3] : memref<?x10xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_0]], [[LOAD_1]] : f32
  // CHECK:       [[ADD:%.+]] = addf [[LOAD_RES]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[RES]][%arg2, %arg3] : memref<?x10xf32>
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
  // CHECK: affine.store [[CONSTANT]], [[RES]][%[[CONSTANT_INDEX]]] : memref<1xf32>
  // CHECK: [[LOOPS_REDUCE:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[LOOPS_REDUCE]]) with ([[LOOPS_REDUCE]] -> %arg2 = 0 to 5) {
  // CHECK:   [[LOAD_0:%.+]] = affine.load %arg0[%arg2] : memref<5xf32>
  // CHECK:   [[LOAD_1:%.+]] = affine.load %arg1[%arg2] : memref<5xf32>
  // CHECK:   [[LOAD_RES:%.+]] = affine.load [[RES]][%[[CONSTANT_INDEX]]] : memref<1xf32>
  // CHECK:   [[MUL:%.+]] = mulf [[LOAD_0]], [[LOAD_1]] : f32
  // CHECK:   [[ADD:%.+]] = addf [[LOAD_RES]], [[MUL]] : f32
  // CHECK:   affine.store [[ADD]], [[RES]][%[[CONSTANT_INDEX]]] : memref<1xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<1xf32>
}

// -----

func @test_conv_no_bias_no_pad(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_no_pad
  // CHECK: [[RES:%.+]] = alloc() : memref<1x5x27x58xf32>
  // CHECK: [[CONST0:%.+]] = constant 5 : index
  // CHECK: [[CONST1:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[CONST2:%.+]] = constant 2 : index
  // CHECK: [[OUTER_LOOPS:%.+]]:2 = krnl.define_loops 2

  // CHECK: krnl.iterate([[OUTER_LOOPS]]#0, [[OUTER_LOOPS]]#1) with ([[OUTER_LOOPS]]#0 -> %arg2 = 0 to 1, [[OUTER_LOOPS]]#1 -> %arg3 = 0 to 5) {
  // CHECK: [[SPATIAL_LOOPS:%.+]]:2 = krnl.define_loops 2

  // CHECK: krnl.iterate([[SPATIAL_LOOPS]]#0, [[SPATIAL_LOOPS]]#1) with ([[SPATIAL_LOOPS]]#0 -> %arg4 = 0 to 27, [[SPATIAL_LOOPS]]#1 -> %arg5 = 0 to 58) {
  // CHECK: affine.store [[CONST1]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<1x5x27x58xf32>
  // CHECK: [[INNER_LOOPS:%.+]]:3 = krnl.define_loops 3

  // CHECK: krnl.iterate([[INNER_LOOPS]]#0, [[INNER_LOOPS]]#1, [[INNER_LOOPS]]#2) with ([[INNER_LOOPS]]#0 -> %arg6 = 0 to 2, [[INNER_LOOPS]]#1 -> %arg7 = 0 to 6, [[INNER_LOOPS]]#2 -> %arg8 = 0 to 7) {
  // CHECK: [[R1PLUSK1:%.+]] = affine.apply #{{.*}}(%arg4, %arg7)
  // CHECK: [[R2PLUSK2:%.+]] = affine.apply #{{.*}}(%arg5, %arg8)
  // CHECK: [[DATA:%.+]] = affine.load %arg0[%arg2, %arg6, [[R1PLUSK1]], [[R2PLUSK2]]] : memref<1x2x32x64xf32>
  // CHECK: [[KERNEL:%.+]] = affine.load %arg1[%arg3, %arg6, %arg7, %arg8] : memref<5x2x6x7xf32>
  // CHECK: [[ACC_RES:%.+]] = affine.load %0[%arg2, %arg3, %arg4, %arg5] : memref<1x5x27x58xf32>
  // CHECK: [[MUL:%.+]] = mulf [[DATA]], [[KERNEL]] : f32
  // CHECK: [[ADD:%.+]] = addf [[ACC_RES]], [[MUL]] : f32
  // CHECK: affine.store [[ADD]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<1x5x27x58xf32>
  // CHECK: }
  // CHECK: }
  // CHECK: }

  // CHECK: return [[RES]] : memref<1x5x27x58xf32>
}

// -----

func @test_conv_bias_no_pad(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, tensor<5xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_bias_no_pad
  // CHECK: [[RES:%.+]] = alloc() : memref<1x5x27x58xf32>
  // CHECK: [[CONST0:%.+]] = constant 5 : index
  // CHECK: [[CONST1:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[CONST2:%.+]] = constant 2 : index
  // CHECK: [[OUTER_LOOPS:%.+]]:2 = krnl.define_loops 2

  // CHECK: krnl.iterate([[OUTER_LOOPS]]#0, [[OUTER_LOOPS]]#1) with ([[OUTER_LOOPS]]#0 -> %arg3 = 0 to 1, [[OUTER_LOOPS]]#1 -> %arg4 = 0 to 5) {
  // CHECK: [[SPATIAL_LOOPS:%.+]]:2 = krnl.define_loops 2

  // CHECK: krnl.iterate([[SPATIAL_LOOPS]]#0, [[SPATIAL_LOOPS]]#1) with ([[SPATIAL_LOOPS]]#0 -> %arg5 = 0 to 27, [[SPATIAL_LOOPS]]#1 -> %arg6 = 0 to 58) {
  // CHECK: affine.store [[CONST1]], [[RES]][%arg3, %arg4, %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: [[INNER_LOOPS:%.+]]:3 = krnl.define_loops 3

  // CHECK: krnl.iterate([[INNER_LOOPS]]#0, [[INNER_LOOPS]]#1, [[INNER_LOOPS]]#2) with ([[INNER_LOOPS]]#0 -> %arg7 = 0 to 2, [[INNER_LOOPS]]#1 -> %arg8 = 0 to 6, [[INNER_LOOPS]]#2 -> %arg9 = 0 to 7) {
  // CHECK: [[R1PLUSK1:%.+]] = affine.apply #{{.*}}(%arg5, %arg8)
  // CHECK: [[R2PLUSK2:%.+]] = affine.apply #{{.*}}(%arg6, %arg9)
  // CHECK: [[DATA:%.+]] = affine.load %arg0[%arg3, %arg7, [[R1PLUSK1]], [[R2PLUSK2]]] : memref<1x2x32x64xf32>
  // CHECK: [[KERNEL:%.+]] = affine.load %arg1[%arg4, %arg7, %arg8, %arg9] : memref<5x2x6x7xf32>
  // CHECK: [[ACC_RES:%.+]] = affine.load %0[%arg3, %arg4, %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: [[MUL:%.+]] = mulf [[DATA]], [[KERNEL]] : f32
  // CHECK: [[ADD:%.+]] = addf [[ACC_RES]], [[MUL]] : f32
  // CHECK: affine.store [[ADD]], [[RES]][%arg3, %arg4, %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: }
  // CHECK: [[BIAS1:%.+]] = affine.load [[RES]][%arg3, %arg4, %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: [[BIAS2:%.+]] = affine.load %arg2[%arg4] : memref<5xf32>
  // CHECK: [[BIAS3:%.+]] = addf [[BIAS1]], [[BIAS2]] : f32
  // CHECK: affine.store [[BIAS3]], [[RES]][%arg3, %arg4, %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: }
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x5x27x58xf32>
}

// -----

func @test_conv_no_bias_no_pad_w_group(%arg0 : tensor<1x9x32x64xf32>, %arg1 : tensor<5x3x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 3 : si64} : (tensor<1x9x32x64xf32>, tensor<5x3x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_no_pad_w_group
  // CHECK: [[RES:%.+]] = alloc() : memref<1x5x27x58xf32>
  // CHECK: %[[CONST0:.+]] = constant 1 : index
  // CHECK: [[CONST1:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[CONST2:%.+]] = constant 3 : index
  // CHECK: [[OUTER_LOOPS:%.+]]:3 = krnl.define_loops 3

  // CHECK: krnl.iterate([[OUTER_LOOPS]]#0, [[OUTER_LOOPS]]#1, [[OUTER_LOOPS]]#2) with ([[OUTER_LOOPS]]#0 -> %arg2 = 0 to 1, [[OUTER_LOOPS]]#1 -> %arg3 = 0 to 3, [[OUTER_LOOPS]]#2 -> %arg4 = 0 to 1) {
  // CHECK: %[[ADD1:.+]] = affine.apply #{{.*}}(%arg3, %arg4)[%[[CONST0]]]
  // CHECK: [[SPATIAL_LOOPS:%.+]]:2 = krnl.define_loops 2

  // CHECK: krnl.iterate([[SPATIAL_LOOPS]]#0, [[SPATIAL_LOOPS]]#1) with ([[SPATIAL_LOOPS]]#0 -> %arg5 = 0 to 27, [[SPATIAL_LOOPS]]#1 -> %arg6 = 0 to 58) {
  // CHECK: affine.store [[CONST1]], [[RES]][%arg2, %[[ADD1]], %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: [[INNER_LOOPS:%.+]]:3 = krnl.define_loops 3

  // CHECK: krnl.iterate([[INNER_LOOPS]]#0, [[INNER_LOOPS]]#1, [[INNER_LOOPS]]#2) with ([[INNER_LOOPS]]#0 -> %arg7 = 0 to 3, [[INNER_LOOPS]]#1 -> %arg8 = 0 to 6, [[INNER_LOOPS]]#2 -> %arg9 = 0 to 7) {
  // CHECK: [[ADD2:%.+]] = affine.apply #{{.*}}(%arg3, %arg7)[%c3]
  // CHECK: [[R1PLUSK1:%.+]] = affine.apply #{{.*}}(%arg5, %arg8) 
  // CHECK: [[R2PLUSK2:%.+]] = affine.apply #{{.*}}(%arg6, %arg9) 
  // CHECK: [[DATA:%.+]] = affine.load %arg0[%arg2, [[ADD2]], [[R1PLUSK1]], [[R2PLUSK2]]] : memref<1x9x32x64xf32>
  // CHECK: [[KERNEL:%.+]] = affine.load %arg1[%[[ADD1]], %arg7, %arg8, %arg9] : memref<5x3x6x7xf32>
  // CHECK: [[ACC_RES:%.+]] = affine.load %0[%arg2, %[[ADD1]], %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: [[MUL:%.+]] = mulf [[DATA]], [[KERNEL]] : f32
  // CHECK: [[ADD:%.+]] = addf [[ACC_RES]], [[MUL]] : f32
  // CHECK: affine.store [[ADD]], [[RES]][%arg2, %[[ADD1]], %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: }
  // CHECK: }
  // CHECK: }

  // CHECK: return [[RES]] : memref<1x5x27x58xf32>
}

// -----

func @test_conv_no_bias_no_pad_w_strides(%arg0 : tensor<1x9x32x64xf32>, %arg1 : tensor<5x9x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64, strides = [2, 2]} : (tensor<1x9x32x64xf32>, tensor<5x9x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_no_bias_no_pad_w_strides
  // CHECK: [[RES:%.+]] = alloc() : memref<1x5x14x29xf32>
  // CHECK: [[CONST0:%.+]] = constant 5 : index
  // CHECK: [[CONST1:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[CONST2:%.+]] = constant 9 : index
  // CHECK: [[OUTER_LOOPS:%.+]]:2 = krnl.define_loops 2

  // CHECK: krnl.iterate([[OUTER_LOOPS]]#0, [[OUTER_LOOPS]]#1) with ([[OUTER_LOOPS]]#0 -> %arg2 = 0 to 1, [[OUTER_LOOPS]]#1 -> %arg3 = 0 to 5) {
  // CHECK: [[SPATIAL_LOOPS:%.+]]:2 = krnl.define_loops 2

  // CHECK: krnl.iterate([[SPATIAL_LOOPS]]#0, [[SPATIAL_LOOPS]]#1) with ([[SPATIAL_LOOPS]]#0 -> %arg4 = 0 to 14, [[SPATIAL_LOOPS]]#1 -> %arg5 = 0 to 29) {
  // CHECK: affine.store [[CONST1]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<1x5x14x29xf32>
  // CHECK: [[INNER_LOOPS:%.+]]:3 = krnl.define_loops 3

  // CHECK: krnl.iterate([[INNER_LOOPS]]#0, [[INNER_LOOPS]]#1, [[INNER_LOOPS]]#2) with ([[INNER_LOOPS]]#0 -> %arg6 = 0 to 9, [[INNER_LOOPS]]#1 -> %arg7 = 0 to 6, [[INNER_LOOPS]]#2 -> %arg8 = 0 to 7) {
  // CHECK: [[R1PLUSK1:%.+]] = affine.apply #{{.*}}(%arg4, %arg7)
  // CHECK: [[R2PLUSK2:%.+]] = affine.apply #{{.*}}(%arg5, %arg8)
  // CHECK: [[DATA:%.+]] = affine.load %arg0[%arg2, %arg6, [[R1PLUSK1]], [[R2PLUSK2]]] : memref<1x9x32x64xf32>
  // CHECK: [[KERNEL:%.+]] = affine.load %arg1[%arg3, %arg6, %arg7, %arg8] : memref<5x9x6x7xf32>
  // CHECK: [[ACC_RES:%.+]] = affine.load %0[%arg2, %arg3, %arg4, %arg5] : memref<1x5x14x29xf32>
  // CHECK: [[MUL:%.+]] = mulf [[DATA]], [[KERNEL]] : f32
  // CHECK: [[ADD:%.+]] = addf [[ACC_RES]], [[MUL]] : f32
  // CHECK: affine.store [[ADD]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<1x5x14x29xf32>
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
  // CHECK: krnl.iterate([[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg5 = 0 to 2) {
  // CHECK:   [[SCALE:%.+]] = affine.load %arg1[%arg5] : memref<2xf32>
  // CHECK:   [[BIAS:%.+]] = affine.load %arg2[%arg5] : memref<2xf32>
  // CHECK:   [[MEAN:%.+]] = affine.load %arg3[%arg5] : memref<2xf32>
  // CHECK:   [[VARIANCE:%.+]] = affine.load %arg4[%arg5] : memref<2xf32>
  // CHECK:   krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) with ([[DEF_LOOPS]]#0 -> %arg6 = 0 to 1, [[DEF_LOOPS]]#2 -> %arg7 = 0 to 1, [[DEF_LOOPS]]#3 -> %arg8 = 0 to 3) {
  // CHECK:     [[LOADED_VAL:%.+]] = affine.load %arg0[%arg6, %arg5, %arg7, %arg8] : memref<1x2x1x3xf32>
  // CHECK:     [[DIVIDEND:%.+]] = subf [[LOADED_VAL]], [[MEAN]] : f32
  // CHECK:     [[ADJUSTED_VARIANCE:%.+]] = addf [[VARIANCE]], [[EPSILON]] : f32
  // CHECK:     [[DIVISOR:%.+]] = sqrt [[ADJUSTED_VARIANCE]] : f32
  // CHECK:     [[NORM:%.+]] = divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK:     [[SCALE_NORM:%.+]] = mulf [[SCALE]], [[NORM]] : f32
  // CHECK:     [[SHIFT_SCALE_NORM:%.+]] = addf [[SCALE_NORM]], [[BIAS]] : f32
  // CHECK:     affine.store [[SHIFT_SCALE_NORM]], [[RES]][%arg6, %arg5, %arg7, %arg8] : memref<1x2x1x3xf32>
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
  // CHECK: %[[ZERO_INDEX:.+]] = constant 0 : index
  // CHECK: [[SCALE:%.+]] = affine.load %arg1[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: [[BIAS:%.+]] = affine.load %arg2[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: [[MEAN:%.+]] = affine.load %arg3[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: [[VARIANCE:%.+]] = affine.load %arg4[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]) with ([[DEF_LOOPS]] -> %arg5 = 0 to 10) {
  // CHECK:   [[LOADED_VAL:%.+]] = affine.load %arg0[%arg5] : memref<10xf32>
  // CHECK:   [[DIVIDEND:%.+]] = subf [[LOADED_VAL]], [[MEAN]] : f32
  // CHECK:   [[ADJUSTED_VARIANCE:%.+]] = addf [[VARIANCE]], [[EPSILON]] : f32
  // CHECK:   [[DIVISOR:%.+]] = sqrt [[ADJUSTED_VARIANCE]] : f32
  // CHECK:   [[NORM:%.+]] = divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK:   [[SCALE_NORM:%.+]] = mulf [[SCALE]], [[NORM]] : f32
  // CHECK:   [[SHIFT_SCALE_NORM:%.+]] = addf [[SCALE_NORM]], [[BIAS]] : f32
  // CHECK:   affine.store [[SHIFT_SCALE_NORM]], [[RES]][%arg5] : memref<10xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<10xf32>
}

// -----

func @test_batchnorm_testmode_2d(%arg0: tensor<10x3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>) -> tensor<10x3xf32> {
  %0 = "onnx.BatchNormalizationTestMode"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<10x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<10x3xf32>
  return %0 : tensor<10x3xf32>

  // CHECK-LABEL: test_batchnorm_testmode_2d
  // CHECK: [[RES:%.+]] = alloc() : memref<10x3xf32>
  // CHECK: [[EPSILON:%.+]] = constant 9.99999974E-6 : f32
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:   [[SCALE:%.+]] = affine.load %arg1[%arg5] : memref<3xf32>
  // CHECK:   [[BIAS:%.+]] = affine.load %arg2[%arg5] : memref<3xf32>
  // CHECK:   [[MEAN:%.+]] = affine.load %arg3[%arg5] : memref<3xf32>
  // CHECK:   [[VARIANCE:%.+]] = affine.load %arg4[%arg5] : memref<3xf32>
  // CHECK:   krnl.iterate([[DEF_LOOPS]]#0) with ([[DEF_LOOPS]]#0 -> %arg6 = 0 to 10) {
  // CHECK:     [[LOADED_VAL:%.+]] = affine.load %arg0[%arg6, %arg5] : memref<10x3xf32>
  // CHECK:     [[DIVIDEND:%.+]] = subf [[LOADED_VAL]], [[MEAN]] : f32
  // CHECK:     [[ADJUSTED_VARIANCE:%.+]] = addf [[VARIANCE]], [[EPSILON]] : f32
  // CHECK:     [[DIVISOR:%.+]] = sqrt [[ADJUSTED_VARIANCE]] : f32
  // CHECK:     [[NORM:%.+]] = divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK:     [[SCALE_NORM:%.+]] = mulf [[SCALE]], [[NORM]] : f32
  // CHECK:     [[SHIFT_SCALE_NORM:%.+]] = addf [[SCALE_NORM]], [[BIAS]] : f32
  // CHECK:     affine.store [[SHIFT_SCALE_NORM]], [[RES]][%arg6, %arg5] : memref<10x3xf32>
  // CHECK:   }
  // CHECK: }
  // CHECK: return [[RES]] : memref<10x3xf32>
}

// -----

func @test_abs_float(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Abs"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_abs_float
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ABS:%.+]] = absf [[LOAD]] : f32
  // CHECK: affine.store [[ABS]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func @test_abs_int(%arg0 : tensor<?x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.Abs"(%arg0) : (tensor<?x10xi32>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_abs_int
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xi32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xi32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<?x10xi32>
  // CHECK: [[ZERO:%.+]] = constant 0 : i32
  // CHECK: [[LESS_THAN_ZERO:%.+]] = cmpi "slt", [[LOAD]], [[ZERO]] : i32
  // CHECK: [[NEGATIVE_LOAD:%.+]] = subi [[ZERO]], [[LOAD]] : i32
  // CHECK: [[SELECT:%.+]] = select [[LESS_THAN_ZERO]], [[NEGATIVE_LOAD]], [[LOAD]] : i32
  // CHECK: affine.store [[SELECT]], [[RES]][%arg1, %arg2] : memref<?x10xi32>
  // CHECK: return [[RES]] : memref<?x10xi32>
}

// -----

func @test_constant_pad1(%arg0: tensor<16x16xf32>) -> tensor<18x20xf32> {
  %0 = "onnx.PadConstantValuePad"(%arg0) {constant_value = 0.000000e+00 : f32, mode = "constant", pads = [0, 3, 2, 1]} : (tensor<16x16xf32>) -> tensor<18x20xf32>
  return %0 : tensor<18x20xf32>
  // CHECK-LABEL: test_constant_pad1
  // CHECK: [[RES:%.+]] = alloc() : memref<18x20xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 18, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 20) {
  // CHECK: [[CST:%.+]] = constant 0.000000e+00 : f32
  // CHECK: affine.store [[CST]], [[RES]][%arg1, %arg2] : memref<18x20xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOPS2:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 16, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 16) {
  // CHECK: [[ADD:%.+]] = affine.apply #{{.*}}(%arg2)
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<16x16xf32>
  // CHECK: affine.store [[LOAD]], [[RES]][%arg1, [[ADD]]] : memref<18x20xf32>
  // CHECK: }
}

func @test_pad1(%arg0: tensor<16x16xf32>) -> tensor<18x20xf32> {
  %cst = constant unit
  %0 = "onnx.Pad"(%arg0, %cst, %cst) {constant_value = dense<0.000000e+00> : tensor<1xf32>, mode = "constant", pads = dense<[0, 3, 2, 1]> : tensor<4xi32>} : (tensor<16x16xf32>, none, none) -> tensor<18x20xf32>
  return %0 : tensor<18x20xf32>
  // CHECK-LABEL: test_pad1
  // CHECK: [[RES:%.+]] = alloc() : memref<18x20xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 18, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 20) {
  // CHECK: [[CST:%.+]] = constant 0.000000e+00 : f32
  // CHECK: affine.store [[CST]], [[RES]][%arg1, %arg2] : memref<18x20xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOPS2:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 16, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 16) {
  // CHECK: [[ADD:%.+]] = affine.apply #{{.*}}(%arg2)
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[%arg1, %arg2] : memref<16x16xf32>
  // CHECK: affine.store [[LOAD]], [[RES]][%arg1, [[ADD]]] : memref<18x20xf32>
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
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = 2 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>)  -> tensor<5x5x9x32xf32>
  "std.return"(%1) : (tensor<5x5x9x32xf32>) -> ()

  // CHECK-LABEL: test_concat_1
  // CHECK: [[RES:%.+]] = alloc() : memref<5x5x9x32xf32>
  // CHECK: [[DEF_LOOPS0:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS0]]#0, [[DEF_LOOPS0]]#1, [[DEF_LOOPS0]]#2, [[DEF_LOOPS0]]#3) with ([[DEF_LOOPS0]]#0 -> %arg3 = 0 to 5, [[DEF_LOOPS0]]#1 -> %arg4 = 0 to 5, [[DEF_LOOPS0]]#2 -> %arg5 = 0 to 1, [[DEF_LOOPS0]]#3 -> %arg6 = 0 to 32) {
  // CHECK: [[LOAD0:%.+]] = affine.load %arg0[%arg3, %arg4, %arg5, %arg6] :  memref<5x5x1x32xf32>
  // CHECK: affine.store [[LOAD0]], [[RES]][%arg3, %arg4, %arg5, %arg6] : memref<5x5x9x32xf32>

  // CHECK: [[DEF_LOOPS1:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1, [[DEF_LOOPS1]]#2, [[DEF_LOOPS1]]#3) with ([[DEF_LOOPS1]]#0 -> %arg3 = 0 to 5, [[DEF_LOOPS1]]#1 -> %arg4 = 0 to 5, [[DEF_LOOPS1]]#2 -> %arg5 = 0 to 3, [[DEF_LOOPS1]]#3 -> %arg6 = 0 to 32) {
  // CHECK: [[AFFINE_APPLY1:%.+]] = affine.apply #{{.*}}(%arg5)
  // CHECK: [[LOAD1:%.+]] = affine.load %arg1[%arg3, %arg4, %arg5, %arg6] :  memref<5x5x3x32xf32>
  // CHECK: affine.store [[LOAD1]], [[RES]][%arg3, %arg4, [[AFFINE_APPLY1]], %arg6] : memref<5x5x9x32xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2, [[DEF_LOOPS2]]#3) with ([[DEF_LOOPS2]]#0 -> %arg3 = 0 to 5, [[DEF_LOOPS2]]#1 -> %arg4 = 0 to 5, [[DEF_LOOPS2]]#2 -> %arg5 = 0 to 5, [[DEF_LOOPS2]]#3 -> %arg6 = 0 to 32) {
  // CHECK: [[AFFINE_APPLY2:%.+]] = affine.apply #{{.*}}(%arg5)
  // CHECK: [[LOAD2:%.+]] = affine.load %arg2[%arg3, %arg4, %arg5, %arg6] :  memref<5x5x5x32xf32>
  // CHECK: affine.store [[LOAD2]], [[RES]][%arg3, %arg4, [[AFFINE_APPLY2]], %arg6] : memref<5x5x9x32xf32>

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
  // CHECK: krnl.iterate([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31) {

  // CHECK:   affine.store [[IDENTITY]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #{{.*}}(%arg3)[%c32, %c2, %c0, %c1, %c1_0], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #{{.*}}(%arg4)[%c32_1, %c2_2, %c0_3, %c1_4, %c1_5]) {
  // CHECK:     {{.*}} = load %arg0[%arg1, %arg2, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     {{.*}} = affine.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:     affine.store {{.*}}, [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   }

  // CHECK:   {{.*}} = affine.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   affine.store {{.*}}, [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK: }
}

// -----

func @test_pool_unknown_dimensions(%arg0 : tensor<1x3x?x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x?x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-DAG: #[[AFFINE_MAP:.+]] = affine_map<(d0)[s0, s1, s2, s3] -> ((d0 + s1 - (s0 - 1) * s3 - 1) floordiv s2 + 1)>
  // CHECK-LABEL: test_pool_unknown_dimensions
  // CHECK: [[C0:%.+]] = constant 2 : index
  // CHECK: [[DIM:%.+]] = dim %arg0, [[C0]] : memref<1x3x?x32xf32>
  // CHECK: [[KERNEL:%.+]] = constant 2 : index
  // CHECK: [[PAD:%.+]] = constant 0 : index
  // CHECK: [[STRIDE:%.+]] = constant 1 : index
  // CHECK: [[DILATION:%.+]] = constant 1 : index
  // CHECK: [[AFFINE_APPLY:%.+]] = affine.apply #[[AFFINE_MAP]]([[DIM]]){{.*}}[[KERNEL]], [[PAD]], [[STRIDE]], [[DILATION]]{{.*}}
  // CHECK: [[RES:%.+]] = alloc([[AFFINE_APPLY]]) : memref<1x3x?x31xf32>
}

// -----

func @test_averagepool_identity_value(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_averagepool_identity_value
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = constant 0.000000e+00 : f32
  // CHECK: affine.store [[IDENTITY]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
}

// -----

func @test_maxpool_identity_value(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_maxpool_identity_value
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = constant 0xFF800000 : f32
  // CHECK: affine.store [[IDENTITY]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
}

// -----

func @test_averagepool_pooling_operation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_averagepool_pooling_operation
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>

  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31) {

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #{{.*}}(%arg3)[%c32, %c2, %c0, %c1, %c1_0], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #{{.*}}(%arg4)[%c32_1, %c2_2, %c0_3, %c1_4, %c1_5]) {

  // CHECK:     [[INPUT_LOAD:%.+]] = load %arg0[%arg1, %arg2, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     [[OUTPUT_LOAD:%.+]] = affine.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:     [[SUM:%.+]] = addf [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     affine.store [[SUM]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   }

  // CHECK:   [[NUMERATOR:%.+]] = affine.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   [[AVERAGE:%.+]] = divf [[NUMERATOR]], {{.*}} : f32
  // CHECK:   affine.store [[AVERAGE]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK: }
}

// -----

func @test_maxpool_pooling_operation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_maxpool_pooling_operation
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>

  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31) {

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #{{.*}}(%arg3)[%c32, %c2, %c0, %c1, %c1_0], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #{{.*}}(%arg4)[%c32_1, %c2_2, %c0_3, %c1_4, %c1_5]) {

  // CHECK:     [[INPUT_LOAD:%.+]] = load %arg0[%arg1, %arg2, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     [[OUTPUT_LOAD:%.+]] = affine.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:     [[GREATER:%.+]] = cmpf "ogt", [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     [[SELECT:%.+]] = select [[GREATER]], [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     affine.store [[SELECT]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   }

  // CHECK-NOT:   {{.*}} = affine.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK-NOT:   affine.store {{.*}}, [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK: }
}

// -----

/// Check GRU with three required inputs (X, W, R). The optional inputs are default.
/// Also check the equation for 'ht' when linear_before_reset = 0 (default)
func @test_gru_general_computation(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_general_computation
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x3xf32>

  /// Check initialize loop.
  // CHECK: [[INITIAL_VAL:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[DEF_LOOPS_INIT:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS_INIT]]#0, [[DEF_LOOPS_INIT]]#1, [[DEF_LOOPS_INIT]]#2) with ([[DEF_LOOPS_INIT]]#0 -> %arg3 = 0 to 1, [[DEF_LOOPS_INIT]]#1 -> %arg4 = 0 to 3, [[DEF_LOOPS_INIT]]#2 -> %arg5 = 0 to 3) {
  // CHECK:   affine.store [[INITIAL_VAL]], [[RES]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK: }

  /// Check main loop.
  // CHECK: [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:   [[ztMemRef:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:   [[htMemRef:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:   [[ZERO_INDEX:%.+]] = constant 0 : index
  // CHECK:   [[INDEX_3:%.+]] = constant 3 : index
  // CHECK:   [[INDEX_0:%.+]] = constant 0 : index
  // CHECK:   [[INDEX_1:%.+]] = constant 1 : index
  // CHECK:   [[INDEX_2:%.+]] = constant 2 : index
  // CHECK:   [[DATA_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[DATA_LOOPS]]#0, [[DATA_LOOPS]]#1) with ([[DATA_LOOPS]]#0 -> %arg4 = 0 to 3, [[DATA_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:     [[ht:%.+]] = alloc() : memref<f32>
  // CHECK:     [[rt:%.+]] = alloc() : memref<f32>
  // CHECK:     [[zt:%.+]] = alloc() : memref<f32>

  // CHECK:     [[INITIAL_VAL_0:%.+]] = constant 0.000000e+00 : f32
  // CHECK:     [[XWZt:%.+]] = alloc() : memref<f32>
  // CHECK:     affine.store [[INITIAL_VAL_0]], [[XWZt]][] : memref<f32>
  // CHECK:     [[HRZt:%.+]] = alloc() : memref<f32>
  // CHECK:     affine.store [[INITIAL_VAL_0]], [[HRZt]][] : memref<f32>
  // CHECK:     [[XWRt:%.+]] = alloc() : memref<f32>
  // CHECK:     affine.store [[INITIAL_VAL_0]], [[XWRt]][] : memref<f32>
  // CHECK:     [[HRRt:%.+]] = alloc() : memref<f32>
  // CHECK:     affine.store [[INITIAL_VAL_0]], [[HRRt]][] : memref<f32>
  // CHECK:     [[XWHt:%.+]] = alloc() : memref<f32>
  // CHECK:     affine.store [[INITIAL_VAL_0]], [[XWHt]][] : memref<f32>
  // CHECK:     [[HRHt:%.+]] = alloc() : memref<f32>
  // CHECK:     affine.store [[INITIAL_VAL_0]], [[HRHt]][] : memref<f32>

  // CHECK:     [[REDUCTION_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[REDUCTION_LOOPS]]) with ([[REDUCTION_LOOPS]] -> %arg6 = 0 to 2) {
  // CHECK:       [[BIAS_MAP_FOR_Z:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_0]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_R:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_1]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_H:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_2]], [[INDEX_3]]]
  // CHECK:       [[Xt:%.+]] = affine.load %arg0[%arg3, %arg4, %arg6] : memref<4x3x2xf32>

  /// compute Xt*(Wz^T)
  // CHECK:       [[WZt:%.+]] = affine.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_Z]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WZt]] : f32
  // CHECK:       [[LOAD:%.+]] = affine.load [[XWZt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[XWZt]][] : memref<f32>

  /// compute Xt*(Wr^T)
  // CHECK:       [[WRt:%.+]] = affine.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_R]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WRt]] : f32
  // CHECK:       [[LOAD:%.+]] = affine.load [[XWRt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[XWRt]][] : memref<f32>

  /// compute Xt*(Wh^T)
  // CHECK:       [[WHt:%.+]] = affine.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_H]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WHt]] : f32
  // CHECK:       [[LOAD:%.+]] = affine.load [[XWHt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[XWHt]][] : memref<f32>
  // CHECK:     }

  // CHECK:     [[REDUCTION_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[REDUCTION_LOOPS]]) with ([[REDUCTION_LOOPS]] -> %arg6 = 0 to 3) {
  // CHECK:       [[BIAS_MAP_FOR_Z:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_0]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_R:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_1]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_H:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_2]], [[INDEX_3]]]
  // CHECK:       [[PREVIOUS_Ht:%.+]] = affine.load [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg6] : memref<1x3x3xf32>
  /// compute Ht-1*(Rz^T)
  // CHECK:       [[RZt:%.+]] = affine.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_Z]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[PREVIOUS_Ht]], [[RZt]] : f32
  // CHECK:       [[LOAD:%.+]] = affine.load [[HRZt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[HRZt]][] : memref<f32>

  /// compute Ht-1*(Rr^T)
  // CHECK:       [[RRt:%.+]] = affine.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_R]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[PREVIOUS_Ht]], [[RRt]] : f32
  // CHECK:       [[LOAD:%.+]] = affine.load [[HRRt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[HRRt]][] : memref<f32>
  // CHECK:     }

  /// compute zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
  // CHECK:     [[LOAD_XWZt:%.+]] = affine.load [[XWZt]][] : memref<f32>
  // CHECK:     [[LOAD_HRZt:%.+]] = affine.load [[HRZt]][] : memref<f32>
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWZt]], [[LOAD_HRZt]] : f32
  /// apply activation f = sigmoid
  // CHECK:     {{.*}} = alloc() : memref<f32>
  // CHECK:     affine.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = affine.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     affine.store {{.*}}, [[zt]][] : memref<f32>
  // CHECK:     affine.store {{.*}}, [[ztMemRef]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  /// compute rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
  // CHECK:     [[LOAD_XWRt:%.+]] = affine.load [[XWRt]][] : memref<f32>
  // CHECK:     [[LOAD_HRRt:%.+]] = affine.load [[HRRt]][] : memref<f32>
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWRt]], [[LOAD_HRRt]] : f32
  /// apply activation f = sigmoid
  // CHECK:     {{.*}} = alloc() : memref<f32>
  // CHECK:     affine.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = affine.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     affine.store {{.*}}, [[rt]][] : memref<f32>
  // CHECK:     [[LOAD_rt:%.+]] = affine.load [[rt]][] : memref<f32>

  /// compute  ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) since linear_before_reset = 0 (default)
  /// 'rt (.) Ht-1'
  // CHECK:     [[LOAD_XWHt:%.+]] = affine.load [[XWHt]][] : memref<f32>
  // CHECK:     [[PREVIOUS_Ht:%.+]] = affine.load [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[MUL_rt_PREVIOUS_Ht:%.+]] = mulf [[LOAD_rt]], [[PREVIOUS_Ht]] : f32
  /// '(rt (.) Ht-1)*(Rh^T)'
  // CHECK:     [[REDUCTION_LOOPS_1:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[REDUCTION_LOOPS_1]]) with ([[REDUCTION_LOOPS_1]] -> %arg6 = 0 to 2) {
  // CHECK:       [[BIAS_MAP_FOR_H:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_2]], [[INDEX_3]]]
  // CHECK:       [[LOAD_RHt:%.+]] = affine.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_H]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[MUL_rt_PREVIOUS_Ht]], [[LOAD_RHt]] : f32
  // CHECK:       [[LOAD:%.+]] = affine.load [[HRHt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[HRHt]][] : memref<f32>
  // CHECK:     }
  // CHECK:     [[LOAD_HRHt:%.+]] = affine.load [[HRHt]][] : memref<f32>
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWHt]], [[LOAD_HRHt]] : f32
  /// apply activation g = tanh
  // CHECK:     {{.*}} = alloc() : memref<f32>
  // CHECK:     affine.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = affine.load {{.*}}[] : memref<f32>
  // CHECK:     %cst_8 = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = subf %cst_8, {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     affine.store {{.*}}, [[ht]][] : memref<f32>
  // CHECK:     [[LOAD_ht:%.+]] = affine.load [[ht]][] : memref<f32>
  // CHECK:     affine.store [[LOAD_ht]], [[htMemRef]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:     dealloc [[XWZt]] : memref<f32>
  // CHECK:     dealloc [[XWRt]] : memref<f32>
  // CHECK:     dealloc [[XWHt]] : memref<f32>
  // CHECK:     dealloc [[HRZt]] : memref<f32>
  // CHECK:     dealloc [[HRRt]] : memref<f32>
  // CHECK:     dealloc [[HRHt]] : memref<f32>
  // CHECK:     dealloc [[zt]] : memref<f32>
  // CHECK:     dealloc [[rt]] : memref<f32>
  // CHECK:     dealloc [[ht]] : memref<f32>
  // CHECK:   }

  // CHECK:   [[GATE_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[GATE_LOOPS]]#0, [[GATE_LOOPS]]#1) with ([[GATE_LOOPS]]#0 -> %arg4 = 0 to 3, [[GATE_LOOPS]]#1 -> %arg5 = 0 to 3) {
  /// compute  Ht = (1 - zt) (.) ht + zt (.) Ht-1
  // CHECK:     [[LOAD_zt:%.+]] = affine.load [[ztMemRef]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[LOAD_ht:%.+]] = affine.load [[htMemRef]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[PREVIOUS_Ht:%.+]] = affine.load [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[ONE:%.+]] = constant 1.000000e+00 : f32
  // CHECK:     [[SUB:%.+]] = subf [[ONE]], [[LOAD_zt]] : f32
  // CHECK:     [[MUL:%.+]] = mulf [[SUB]], [[LOAD_ht]] : f32
  // CHECK:     [[MUL_1:%.+]] = mulf [[LOAD_zt]], [[PREVIOUS_Ht]] : f32
  // CHECK:     [[ADD:%.+]] = addf [[MUL]], [[MUL_1]] : f32
  // CHECK:     affine.store [[ADD]], [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:   }
  // CHECK:    dealloc [[htMemRef]] : memref<1x3x3xf32>
  // CHECK:    dealloc [[ztMemRef]] : memref<1x3x3xf32>

  // CHECK: }
  // CHECK: return [[RES]] : memref<1x3x3xf32>
}

// -----

/// GRU with three required inputs (X, W, R). The optional inputs are default.
/// Check the equation for 'ht' when linear_before_reset !=0.
func @test_gru_linear_before_reset(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64, linear_before_reset = 1 : si64} : (tensor<4x3x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_linear_before_reset
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x3xf32>

  /// Check initialize loop.
  // CHECK: [[INITIAL_VAL:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[DEF_LOOPS_INIT:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS_INIT]]#0, [[DEF_LOOPS_INIT]]#1, [[DEF_LOOPS_INIT]]#2) with ([[DEF_LOOPS_INIT]]#0 -> %arg3 = 0 to 1, [[DEF_LOOPS_INIT]]#1 -> %arg4 = 0 to 3, [[DEF_LOOPS_INIT]]#2 -> %arg5 = 0 to 3) {
  // CHECK:   affine.store [[INITIAL_VAL]], [[RES]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK: }

  /// Check main loop.
  // CHECK: [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:   [[ztMemRef:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:   [[htMemRef:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:   [[ZERO_INDEX:%.+]] = constant 0 : index
  // CHECK:   [[INDEX_3:%.+]] = constant 3 : index
  // CHECK:   [[INDEX_0:%.+]] = constant 0 : index
  // CHECK:   [[INDEX_1:%.+]] = constant 1 : index
  // CHECK:   [[INDEX_2:%.+]] = constant 2 : index
  // CHECK:   [[DATA_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[DATA_LOOPS]]#0, [[DATA_LOOPS]]#1) with ([[DATA_LOOPS]]#0 -> %arg4 = 0 to 3, [[DATA_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:     [[ht:%.+]] = alloc() : memref<f32>
  // CHECK:     [[rt:%.+]] = alloc() : memref<f32>
  // CHECK:     [[zt:%.+]] = alloc() : memref<f32>

  // CHECK:     [[INITIAL_VAL_0:%.+]] = constant 0.000000e+00 : f32
  // CHECK:     [[XWZt:%.+]] = alloc() : memref<f32>
  // CHECK:     affine.store [[INITIAL_VAL_0]], [[XWZt]][] : memref<f32>
  // CHECK:     [[HRZt:%.+]] = alloc() : memref<f32>
  // CHECK:     affine.store [[INITIAL_VAL_0]], [[HRZt]][] : memref<f32>
  // CHECK:     [[XWRt:%.+]] = alloc() : memref<f32>
  // CHECK:     affine.store [[INITIAL_VAL_0]], [[XWRt]][] : memref<f32>
  // CHECK:     [[HRRt:%.+]] = alloc() : memref<f32>
  // CHECK:     affine.store [[INITIAL_VAL_0]], [[HRRt]][] : memref<f32>
  // CHECK:     [[XWHt:%.+]] = alloc() : memref<f32>
  // CHECK:     affine.store [[INITIAL_VAL_0]], [[XWHt]][] : memref<f32>
  // CHECK:     [[HRHt:%.+]] = alloc() : memref<f32>
  // CHECK:     affine.store [[INITIAL_VAL_0]], [[HRHt]][] : memref<f32>

  // CHECK:     [[REDUCTION_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[REDUCTION_LOOPS]]) with ([[REDUCTION_LOOPS]] -> %arg6 = 0 to 2) {
  // CHECK:       [[BIAS_MAP_FOR_Z:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_0]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_R:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_1]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_H:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_2]], [[INDEX_3]]]
  // CHECK:       [[Xt:%.+]] = affine.load %arg0[%arg3, %arg4, %arg6] : memref<4x3x2xf32>

  /// compute Xt*(Wz^T)
  // CHECK:       [[WZt:%.+]] = affine.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_Z]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WZt]] : f32
  // CHECK:       [[LOAD:%.+]] = affine.load [[XWZt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[XWZt]][] : memref<f32>

  /// compute Xt*(Wr^T)
  // CHECK:       [[WRt:%.+]] = affine.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_R]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WRt]] : f32
  // CHECK:       [[LOAD:%.+]] = affine.load [[XWRt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[XWRt]][] : memref<f32>

  /// compute Xt*(Wh^T)
  // CHECK:       [[WHt:%.+]] = affine.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_H]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WHt]] : f32
  // CHECK:       [[LOAD:%.+]] = affine.load [[XWHt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[XWHt]][] : memref<f32>
  // CHECK:     }

  // CHECK:     [[REDUCTION_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[REDUCTION_LOOPS]]) with ([[REDUCTION_LOOPS]] -> %arg6 = 0 to 3) {
  // CHECK:       [[BIAS_MAP_FOR_Z:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_0]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_R:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_1]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_H:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_2]], [[INDEX_3]]]
  // CHECK:       [[PREVIOUS_Ht:%.+]] = affine.load [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg6] : memref<1x3x3xf32>
  /// compute Ht-1*(Rz^T)
  // CHECK:       [[RZt:%.+]] = affine.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_Z]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[PREVIOUS_Ht]], [[RZt]] : f32
  // CHECK:       [[LOAD:%.+]] = affine.load [[HRZt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[HRZt]][] : memref<f32>

  /// compute Ht-1*(Rr^T)
  // CHECK:       [[RRt:%.+]] = affine.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_R]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[PREVIOUS_Ht]], [[RRt]] : f32
  // CHECK:       [[LOAD:%.+]] = affine.load [[HRRt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[HRRt]][] : memref<f32>

  /// compute Ht-1*(Rh^T)
  // CHECK:       [[RHt:%.+]] = affine.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_H]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[PREVIOUS_Ht]], [[RHt]] : f32
  // CHECK:       [[LOAD:%.+]] = affine.load [[HRHt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       affine.store [[ADD]], [[HRHt]][] : memref<f32>
  // CHECK:     }

  /// compute zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
  // CHECK:     [[LOAD_XWZt:%.+]] = affine.load [[XWZt]][] : memref<f32>
  // CHECK:     [[LOAD_HRZt:%.+]] = affine.load [[HRZt]][] : memref<f32>
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWZt]], [[LOAD_HRZt]] : f32
  /// apply activation f = sigmoid
  // CHECK:     {{.*}} = alloc() : memref<f32>
  // CHECK:     affine.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = affine.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     affine.store {{.*}}, [[zt]][] : memref<f32>
  // CHECK:     affine.store {{.*}}, [[ztMemRef]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  /// compute rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
  // CHECK:     [[LOAD_XWRt:%.+]] = affine.load [[XWRt]][] : memref<f32>
  // CHECK:     [[LOAD_HRRt:%.+]] = affine.load [[HRRt]][] : memref<f32>
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWRt]], [[LOAD_HRRt]] : f32
  /// apply activation f = sigmoid
  // CHECK:     {{.*}} = alloc() : memref<f32>
  // CHECK:     affine.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = affine.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     affine.store {{.*}}, [[rt]][] : memref<f32>
  // CHECK:     [[LOAD_rt:%.+]] = affine.load [[rt]][] : memref<f32>

  /// compute ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) since linear_before_reset != 0
  // CHECK:     [[LOAD_XWHt:%.+]] = affine.load [[XWHt]][] : memref<f32>
  // CHECK:     [[LOAD_HRHt:%.+]] = affine.load [[HRHt]][] : memref<f32>
  // CHECK:     [[MUL_rt_HRHt:%.+]] = mulf [[LOAD_rt]], [[LOAD_HRHt]] : f32
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWHt]], [[MUL_rt_HRHt]] : f32
  /// apply activation g = tanh
  // CHECK:     {{.*}} = alloc() : memref<f32>
  // CHECK:     affine.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = affine.load {{.*}}[] : memref<f32>
  // CHECK:     %cst_8 = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = subf %cst_8, {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     affine.store {{.*}}, [[ht]][] : memref<f32>
  // CHECK:     [[LOAD_ht:%.+]] = affine.load [[ht]][] : memref<f32>
  // CHECK:     affine.store [[LOAD_ht]], [[htMemRef]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:     dealloc [[XWZt]] : memref<f32>
  // CHECK:     dealloc [[XWRt]] : memref<f32>
  // CHECK:     dealloc [[XWHt]] : memref<f32>
  // CHECK:     dealloc [[HRZt]] : memref<f32>
  // CHECK:     dealloc [[HRRt]] : memref<f32>
  // CHECK:     dealloc [[HRHt]] : memref<f32>
  // CHECK:     dealloc [[zt]] : memref<f32>
  // CHECK:     dealloc [[rt]] : memref<f32>
  // CHECK:     dealloc [[ht]] : memref<f32>
  // CHECK:   }

  // CHECK:   [[GATE_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[GATE_LOOPS]]#0, [[GATE_LOOPS]]#1) with ([[GATE_LOOPS]]#0 -> %arg4 = 0 to 3, [[GATE_LOOPS]]#1 -> %arg5 = 0 to 3) {
  /// compute  Ht = (1 - zt) (.) ht + zt (.) Ht-1
  // CHECK:     [[LOAD_zt:%.+]] = affine.load [[ztMemRef]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[LOAD_ht:%.+]] = affine.load [[htMemRef]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[PREVIOUS_Ht:%.+]] = affine.load [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[ONE:%.+]] = constant 1.000000e+00 : f32
  // CHECK:     [[SUB:%.+]] = subf [[ONE]], [[LOAD_zt]] : f32
  // CHECK:     [[MUL:%.+]] = mulf [[SUB]], [[LOAD_ht]] : f32
  // CHECK:     [[MUL_1:%.+]] = mulf [[LOAD_zt]], [[PREVIOUS_Ht]] : f32
  // CHECK:     [[ADD:%.+]] = addf [[MUL]], [[MUL_1]] : f32
  // CHECK:     affine.store [[ADD]], [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:   }
  // CHECK:    dealloc [[htMemRef]] : memref<1x3x3xf32>
  // CHECK:    dealloc [[ztMemRef]] : memref<1x3x3xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x3x3xf32>
}

// -----

/// Check GRU with three required inputs (X, W, R), and bias input.
func @test_gru_with_bias(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>, %arg3: tensor<1x18xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, tensor<1x18xf32>, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_with_bias

  // CHECK: [[LOAD_WZ_BIAS:%.+]] = affine.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_WZ_BIAS]] : f32
  // CHECK: [[LOAD_RZ_BIAS:%.+]] = affine.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_RZ_BIAS]] : f32

  // CHECK: [[LOAD_WR_BIAS:%.+]] = affine.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_WR_BIAS]] : f32
  // CHECK: [[LOAD_RR_BIAS:%.+]] = affine.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_RR_BIAS]] : f32

  // CHECK: [[LOAD_WH_BIAS:%.+]] = affine.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_WH_BIAS]] : f32
  // CHECK: [[LOAD_RH_BIAS:%.+]] = affine.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_RH_BIAS]] : f32
}

// -----

// Check handling unknown dimensions for GRU by checking the 
// correctness of allocating and deallocating memory.
func @test_gru_unkown_dims_allocation(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x9x?xf32>, %arg2: tensor<1x9x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<?x?x?xf32>, tensor<1x9x?xf32>, tensor<1x9x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: @test_gru_unkown_dims_allocation

  // allocate memory for Hidden (Y_h).
  // CHECK: [[C1_0:%.+]] = constant 1 : index
  // CHECK: [[BATCH_SIZE:%.+]] = dim %arg0, [[C1_0]] : memref<?x?x?xf32>
  // CHECK: [[Y_h:%.+]] = alloc([[BATCH_SIZE]]) : memref<1x?x3xf32>

  // CHECK: return [[Y_h]] : memref<1x?x3xf32>
}

// -----

func @test_lstm_general_computation(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

  // CHECK-DAG: [[ACCESS_BY_OFFSET_MAP:#.+]] = affine_map<(d0)[s0, s1] -> (d0 + s0 * s1)>
  // CHECK-LABEL: @test_lstm_general_computation

  // CHECK:  [[CELL_STATE:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:  [[HIDDEN_STATE:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:  {{.*}} = constant unit

  // CHECK:  [[INITIAL_VALUE:%.+]] = constant 0.000000e+00 : f32
  // CHECK:  [[INITIALIZE_LOOPS:%.+]]:3 = krnl.define_loops 3
  // CHECK:  krnl.iterate([[INITIALIZE_LOOPS]]#0, [[INITIALIZE_LOOPS]]#1, [[INITIALIZE_LOOPS]]#2) with ([[INITIALIZE_LOOPS]]#0 -> %arg3 = 0 to 1, [[INITIALIZE_LOOPS]]#1 -> %arg4 = 0 to 3, [[INITIALIZE_LOOPS]]#2 -> %arg5 = 0 to 3) {
  // CHECK:    affine.store [[INITIAL_VALUE]], [[HIDDEN_STATE]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:    affine.store [[INITIAL_VALUE]], [[CELL_STATE]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:  }

  // CHECK:  [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {

  // CHECK:    [[HtRc_GEMM:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:    [[XtWc_GEMM:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:    [[HtRf_GEMM:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:    [[XtWf_GEMM:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:    [[HtRo_GEMM:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:    [[XtWo_GEMM:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:    [[HtRi_GEMM:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:    [[XtWi_GEMM:%.+]] = alloc() : memref<1x3x3xf32>

  // CHECK:    [[C0_INDEX:%.+]] = constant 0 : index
  // CHECK:    {{.*}} = constant 3 : index
  // CHECK:    {{.*}} = constant 0 : index
  // CHECK:    {{.*}} = constant 1 : index
  // CHECK:    {{.*}} = constant 2 : index
  // CHECK:    {{.*}} = constant 3 : index
  // CHECK:    {{.*}} = constant 4 : index
  // CHECK:    {{.*}} = constant 5 : index
  // CHECK:    {{.*}} = constant 6 : index
  // CHECK:    {{.*}} = constant 7 : index
  // CHECK:    [[MATRIX_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:    krnl.iterate([[MATRIX_LOOPS]]#0, [[MATRIX_LOOPS]]#1) with ([[MATRIX_LOOPS]]#0 -> %arg4 = 0 to 3, [[MATRIX_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:      [[CST0:%.+]] = constant 0.000000e+00 : f32
  // CHECK:      affine.store [[CST0]], [[XtWi_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      affine.store [[CST0]], [[HtRi_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      affine.store [[CST0]], [[XtWo_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      affine.store [[CST0]], [[HtRo_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      affine.store [[CST0]], [[XtWf_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      affine.store [[CST0]], [[HtRf_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      affine.store [[CST0]], [[XtWc_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      affine.store [[CST0]], [[HtRc_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      [[XW_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:      krnl.iterate([[XW_LOOPS]]) with ([[XW_LOOPS]] -> %arg6 = 0 to 2) {
  // CHECK:        [[INPUT_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[OUTPUT_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[FORGET_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[CELL_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]

  // CHECK:        [[Xt_LOAD:%.+]] = affine.load %arg0[%arg3, %arg4, %arg6] : memref<4x3x2xf32>
  // CHECK:        [[Wi_LOAD:%.+]] = affine.load %arg1{{\[}}[[C0_INDEX]], [[INPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wi_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[XtWi_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        affine.store {{.*}}, [[XtWi_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:        [[Wo_LOAD:%.+]] = affine.load %arg1{{\[}}[[C0_INDEX]], [[OUTPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wo_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[XtWo_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, %26 : f32
  // CHECK:        affine.store {{.*}}, [[XtWo_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:        [[Wf_LOAD:%.+]] = affine.load %arg1{{\[}}[[C0_INDEX]], [[FORGET_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wf_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[XtWf_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, %30 : f32
  // CHECK:        affine.store {{.*}}, [[XtWf_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:        [[Wc_LOAD:%.+]] = affine.load %arg1{{\[}}[[C0_INDEX]], [[CELL_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wc_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[XtWc_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, %34 : f32
  // CHECK:        affine.store {{.*}}, [[XtWc_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      }
  // CHECK:      [[HR_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:      krnl.iterate([[HR_LOOPS]]) with ([[HR_LOOPS]] -> %arg6 = 0 to 3) {
  // CHECK:        [[INPUT_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[OUTPUT_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[FORGET_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[CELL_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]

  // CHECK:        [[Ht_LOAD:%.+]] = affine.load %1{{\[}}[[C0_INDEX]], %arg4, %arg6] : memref<1x3x3xf32>

  // CHECK:        [[Ri_LOAD:%.+]] = affine.load %arg2{{\[}}[[C0_INDEX]], [[INPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht_LOAD]], [[Ri_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[HtRi_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        affine.store {{.*}}, [[HtRi_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:        [[Ro_LOAD:%.+]] = affine.load %arg2{{\[}}[[C0_INDEX]], [[OUTPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht_LOAD]], [[Ro_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[HtRo_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        affine.store {{.*}}, [[HtRo_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:        [[Rf_LOAD:%.+]] = affine.load %arg2{{\[}}[[C0_INDEX]], [[FORGET_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht_LOAD]], [[Rf_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[HtRf_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        affine.store {{.*}}, [[HtRf_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:        [[Rc_LOAD:%.+]] = affine.load %arg2{{\[}}[[C0_INDEX]], [[CELL_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht_LOAD]], [[Rc_LOAD]] : f32
  // CHECK:        {{.*}} = affine.load [[HtRc_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        affine.store {{.*}}, [[HtRc_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      } 
  // CHECK:    }

  // CHECK:    [[DATA_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:    krnl.iterate([[DATA_LOOPS]]#0, [[DATA_LOOPS]]#1) with ([[DATA_LOOPS]]#0 -> %arg4 = 0 to 3, [[DATA_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:      [[hCt:%.+]] = alloc() : memref<f32>
  // CHECK:      [[Ot:%.+]] = alloc() : memref<f32>
  // CHECK:      [[ct:%.+]] = alloc() : memref<f32>
  // CHECK:      [[Ft:%.+]] = alloc() : memref<f32>
  // CHECK:      [[It:%.+]] = alloc() : memref<f32>

  // CHECK:      [[Ct1_LOAD:%.+]] = affine.load [[CELL_STATE]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      [[XtWi_LOAD:%.+]] = affine.load [[XtWi_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      [[HtRi_LOAD:%.+]] = affine.load [[HtRi_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      [[It_OUTPUT:%.+]] = addf [[XtWi_LOAD]], [[HtRi_LOAD]] : f32

  // CHECK:      [[SIGMOID_INPUT:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[It_OUTPUT]], [[SIGMOID_INPUT]][] : memref<f32>
  // CHECK:      {{.*}} = affine.load [[SIGMOID_INPUT]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}}: f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      affine.store {{.*}}, [[It]][] : memref<f32>
  // CHECK:      [[It_LOAD:%.+]] = affine.load [[It]][] : memref<f32>

  // CHECK:      [[XtWf_LOAD:%.+]] = affine.load [[XtWf_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      [[HtRf_LOAD:%.+]] = affine.load [[HtRf_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      [[Ft_OUTPUT:%.+]] = addf [[XtWf_LOAD]], [[HtRf_LOAD]] : f32

  // CHECK:      [[SIGMOID_FORGET:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[Ft_OUTPUT]], [[SIGMOID_FORGET]][] : memref<f32>
  // CHECK:      {{.*}} = affine.load [[SIGMOID_FORGET]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}}: f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      affine.store {{.*}}, [[Ft]][] : memref<f32>
  // CHECK:      [[Ft_LOAD:%.+]] = affine.load [[Ft]][] : memref<f32>

  // CHECK:      [[XtWc_LOAD:%.+]] = affine.load [[XtWc_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      [[HtRc_LOAD:%.+]] = affine.load [[HtRc_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      [[ct_OUTPUT:%.+]] = addf [[XtWc_LOAD]], [[HtRc_LOAD]] : f32

  // CHECK:      [[TANH_CELL:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[ct_OUTPUT]], [[TANH_CELL]][] : memref<f32>
  // CHECK:      {{.*}} = affine.load [[TANH_CELL]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      affine.store {{.*}}, [[ct]][] : memref<f32>
  // CHECK:      [[ct_LOAD:%.+]] = affine.load [[ct]][] : memref<f32>

  // CHECK:      [[FtCt1:%.+]] = mulf [[Ft_LOAD]], [[Ct1_LOAD]] : f32
  // CHECK:      [[Itct:%.+]] = mulf [[It_LOAD]], [[ct_LOAD]] : f32
  // CHECK:      [[Ct:%.+]] = addf [[FtCt1]], [[Itct]] : f32
  // CHECK:      affine.store [[Ct]], [[CELL_STATE]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:      [[XtWo_LOAD:%.+]] = affine.load [[XtWo_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      [[HtRo_LOAD:%.+]] = affine.load [[HtRo_GEMM]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      [[Ot_OUTPUT:%.+]] = addf [[XtWo_LOAD]], [[HtRo_LOAD]] : f32

  // CHECK:      [[SIGMOID_OUTPUT:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[Ot_OUTPUT]], [[SIGMOID_OUTPUT]][] : memref<f32>
  // CHECK:      {{.*}} = affine.load [[SIGMOID_OUTPUT]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}}: f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      affine.store {{.*}}, [[Ot]][] : memref<f32>
  // CHECK:      [[Ot_LOAD:%.+]] = affine.load [[Ot]][] : memref<f32>

  // CHECK:      [[TANH_HIDDEN:%.+]] = alloc() : memref<f32>
  // CHECK:      affine.store [[Ct]], [[TANH_HIDDEN]][] : memref<f32>
  // CHECK:      {{.*}} = affine.load [[TANH_HIDDEN]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      affine.store {{.*}}, [[hCt]][] : memref<f32>
  // CHECK:      [[hCt_LOAD:%.+]] = affine.load [[hCt]][] : memref<f32>

  // CHECK:      [[Ht:%.+]] = mulf [[Ot_LOAD]], [[hCt_LOAD]] : f32
  // CHECK:      affine.store [[Ht]], [[HIDDEN_STATE]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:      dealloc [[It]] : memref<f32>
  // CHECK:      dealloc [[Ft]] : memref<f32>
  // CHECK:      dealloc [[ct]] : memref<f32>
  // CHECK:      dealloc [[Ot]] : memref<f32>
  // CHECK:      dealloc [[hCt]] : memref<f32>
  // CHECK:    }
  // CHECK:    dealloc [[XtWi_GEMM]] : memref<1x3x3xf32>
  // CHECK:    dealloc [[XtWo_GEMM]] : memref<1x3x3xf32>
  // CHECK:    dealloc [[XtWf_GEMM]] : memref<1x3x3xf32>
  // CHECK:    dealloc [[XtWc_GEMM]] : memref<1x3x3xf32>
  // CHECK:    dealloc [[HtRi_GEMM]] : memref<1x3x3xf32>
  // CHECK:    dealloc [[HtRo_GEMM]] : memref<1x3x3xf32>
  // CHECK:    dealloc [[HtRf_GEMM]] : memref<1x3x3xf32>
  // CHECK:    dealloc [[HtRc_GEMM]] : memref<1x3x3xf32>
 
  // CHECK:  }
  // CHECK:  dealloc [[CELL_STATE]] : memref<1x3x3xf32>
  // CHECK:  return [[HIDDEN_STATE]] : memref<1x3x3xf32>
}

// -----

func @test_lstm_reverse_mode(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64, direction = "reverse"} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

  // CHECK: [[REVERSE_IV_MAP:#.+]] = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
  // CHECK-LABEL: @test_lstm_reverse_mode

  // CHECK:  [[REVERSE_SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[REVERSE_SEQUENCE_LOOPS]]) with ([[REVERSE_SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:  %[[SEQUENCE_LEN:.+]] = constant 4 : index
  // CHECK:  %[[REVERSE_SEQUENCE_IV:.+]] = affine.apply [[REVERSE_IV_MAP]](%arg3)[%[[SEQUENCE_LEN]]{{]}}
  // CHECK:  [[Xt_LOAD:%.+]] = affine.load %arg0[%[[REVERSE_SEQUENCE_IV]], {{.*}}, {{.*}}] : memref<4x3x2xf32>
}

// -----

func @test_lstm_bidirectional_mode(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64, direction = "bidirectional"} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

  // CHECK: [[REVERSE_IV_MAP:#.+]] = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
  // CHECK-LABEL: @test_lstm_bidirectional_mode

  // CHECK:  [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:  {{.*}} = krnl.define_loops 2
  // CHECK:  {{.*}} = krnl.define_loops 1
  // CHECK:  [[Xt_LOAD:%.+]] = affine.load %arg0[%arg3, {{.*}}, {{.*}}] : memref<4x3x2xf32>
  // CHECK:  {{.*}} = krnl.define_loops 1
  // CHECK:  {{.*}} = krnl.define_loops 2

  // CHECK:  [[REVERSE_SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[REVERSE_SEQUENCE_LOOPS]]) with ([[REVERSE_SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:  %[[SEQUENCE_LEN:.+]] = constant 4 : index
  // CHECK:  %[[REVERSE_SEQUENCE_IV:.+]] = affine.apply [[REVERSE_IV_MAP]](%arg3)[%[[SEQUENCE_LEN]]{{]}}
  // CHECK:  {{.*}} = krnl.define_loops 2
  // CHECK:  {{.*}} = krnl.define_loops 1
  // CHECK:  [[Xt_LOAD:%.+]] = affine.load %arg0[%[[REVERSE_SEQUENCE_IV]], {{.*}}, {{.*}}] : memref<4x3x2xf32>
  // CHECK:  {{.*}} = krnl.define_loops 1
  // CHECK:  {{.*}} = krnl.define_loops 2
}

// -----

// Check handling unknown dimensions for LSTM by checking the 
// correctness of allocating and deallocating memory.
func @test_lstm_unkown_dims_allocation(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x12x?xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<?x?x?xf32>, tensor<1x12x?xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (tensor<*xf32>, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: @test_lstm_unkown_dims_allocation

  // allocate memory for all Hidden (Y).
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[SEQUENCE_LENGTH:%.+]] = dim %arg0, [[C0]] : memref<?x?x?xf32>
  // CHECK: [[C1:%.+]] = constant 1 : index
  // CHECK: [[BATCH_SIZE:%.+]] = dim %arg0, [[C1]] : memref<?x?x?xf32>
  // CHECK: [[Y:%.+]] = alloc([[SEQUENCE_LENGTH]], [[BATCH_SIZE]]) : memref<?x1x?x3xf32>

  // allocate memory for Hidden (Y_h).
  // CHECK: [[C1_0:%.+]] = constant 1 : index
  // CHECK: [[BATCH_SIZE:%.+]] = dim %arg0, [[C1_0]] : memref<?x?x?xf32>
  // CHECK: [[Y_h:%.+]] = alloc([[BATCH_SIZE]]) : memref<1x?x3xf32>

  // allocate memory for Cell (Y_c).
  // CHECK: [[C1_1:%.+]] = constant 1 : index
  // CHECK: [[BATCH_SIZE:%.+]] = dim %arg0, [[C1_1]] : memref<?x?x?xf32>
  // CHECK: [[Y_c:%.+]] = alloc([[BATCH_SIZE]]) : memref<1x?x3xf32>

  // deallocate Y since there is no operation consuming it. 
  // CHECK: dealloc [[Y]] : memref<?x1x?x3xf32>
  // deallocate Y_c since it is not a return value.
  // CHECK: dealloc [[Y_c]] : memref<1x?x3xf32>
  // CHECK: return [[Y_h]] : memref<1x?x3xf32>
}

// -----

/// Check RNN with three required inputs (X, W, R). The optional inputs are default.
func @test_rnn_general_computation(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x3x2xf32>, %arg2: tensor<1x3x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x3x2xf32>, tensor<1x3x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_rnn_general_computation
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x3xf32>

  /// Check initialize loop.
  // CHECK: [[INITIAL_VAL:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[DEF_LOOPS_INIT:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS_INIT]]#0, [[DEF_LOOPS_INIT]]#1, [[DEF_LOOPS_INIT]]#2) with ([[DEF_LOOPS_INIT]]#0 -> %arg3 = 0 to 1, [[DEF_LOOPS_INIT]]#1 -> %arg4 = 0 to 3, [[DEF_LOOPS_INIT]]#2 -> %arg5 = 0 to 3) {
  // CHECK:   affine.store [[INITIAL_VAL]], [[RES]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK: }

  /// Check main loop.
  // CHECK: [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:   [[HtRi_GEMM:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:   [[XtWi_GEMM:%.+]] = alloc() : memref<1x3x3xf32>
  // CHECK:   [[ZERO_INDEX:%.+]] = constant 0 : index
  // CHECK:   {{.*}} = constant 3 : index
  // CHECK:   {{.*}} = constant 0 : index
  // CHECK:   {{.*}} = constant 1 : index

  /// Check reduction loop to compute matrix multiplication for 'Xt*(Wi^T)' and 'Ht-1*(Ri^T)'
  // CHECK:   [[MATRIX_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[MATRIX_LOOPS]]#0, [[MATRIX_LOOPS]]#1) with ([[MATRIX_LOOPS]]#0 -> %arg4 = 0 to 3, [[MATRIX_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:     [[CST0:%.+]] = constant 0.000000e+00 : f32
  // CHECK:     affine.store [[CST0]], [[XtWi_GEMM]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     affine.store [[CST0]], [[HtRi_GEMM]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[XW_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[XW_LOOPS]]) with ([[XW_LOOPS]] -> %arg6 = 0 to 2) {
  // CHECK:       [[Xt_LOAD:%.+]] = affine.load %arg0[%arg3, %arg4, %arg6] : memref<4x3x2xf32>
  // CHECK:       [[Wi_LOAD:%.+]] = affine.load %arg1{{\[}}[[ZERO_INDEX]], %arg5, %arg6] : memref<1x3x2xf32>
  // CHECK:       {{.*}} = mulf [[Xt_LOAD]], [[Wi_LOAD]] : f32
  // CHECK:       {{.*}} = affine.load [[XtWi_GEMM]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:       {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:       affine.store {{.*}}, [[XtWi_GEMM]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     }
  // CHECK:     [[HR_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[HR_LOOPS]]) with ([[HR_LOOPS]] -> %arg6 = 0 to 3) {
  // CHECK:       [[Ht_LOAD:%.+]] = affine.load %0{{\[}}[[ZERO_INDEX]], %arg4, %arg6] : memref<1x3x3xf32>
  // CHECK:       [[Ri_LOAD:%.+]] = affine.load %arg2{{\[}}[[ZERO_INDEX]], %arg5, %arg6] : memref<1x3x3xf32>
  // CHECK:       {{.*}} = mulf [[Ht_LOAD]], [[Ri_LOAD]] : f32
  // CHECK:       {{.*}} = affine.load [[HtRi_GEMM]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:       {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:       affine.store {{.*}}, [[HtRi_GEMM]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     }
  // CHECK:   }
 
  // CHECK:   [[DATA_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[DATA_LOOPS]]#0, [[DATA_LOOPS]]#1) with ([[DATA_LOOPS]]#0 -> %arg4 = 0 to 3, [[DATA_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:     [[Ht:%.+]] = alloc() : memref<f32>

  /// Check 'Xt*(Wi^T) + Ht-1*(Ri^T)'
  // CHECK:     [[LOAD_XWi:%.+]] = affine.load [[XtWi_GEMM]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[LOAD_HRi:%.+]] = affine.load [[HtRi_GEMM]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[XWi_PLUS_HRi:%.+]] = addf [[LOAD_XWi]], [[LOAD_HRi]] : f32

  /// Check calling 'Tanh'
  // CHECK:     {{.*}} = alloc() : memref<f32>
  // CHECK:     affine.store [[XWi_PLUS_HRi]], {{.*}} : memref<f32>
  // CHECK:     {{.*}} = affine.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*.}} = exp {{.*.}} : f32
  // CHECK:     {{.*.}} = subf {{.*.}}, {{.*.}} : f32
  // CHECK:     {{.*.}} = addf {{.*.}}, {{.*.}} : f32
  // CHECK:     {{.*}} = divf {{.*.}}, {{.*.}} : f32
  // CHECK:     affine.store {{.*}}, [[Ht]][] : memref<f32>

  /// Check storing the result.
  // CHECK:     [[NEW_Ht_LOAD:%.+]] = affine.load [[Ht]][] : memref<f32>
  // CHECK:     affine.store [[NEW_Ht_LOAD]], [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     dealloc [[Ht]] : memref<f32>
  // CHECK:   }
  // CHECK:   dealloc [[XtWi_GEMM]] : memref<1x3x3xf32>
  // CHECK:   dealloc [[HtRi_GEMM]] : memref<1x3x3xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x3x3xf32>
}

// -----

/// Check RNN with three required inputs (X, W, R), and bias input.
func @test_rnn_with_bias(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x3x2xf32>, %arg2: tensor<1x3x3xf32>, %arg3: tensor<1x6xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x3x2xf32>, tensor<1x3x3xf32>, tensor<1x6xf32>, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_rnn_with_bias
  // CHECK: [[LOAD_W_BIAS:%.+]] = affine.load %arg3[{{.*}}, {{.*}}] : memref<1x6xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_W_BIAS]] : f32
  // CHECK: [[LOAD_R_BIAS:%.+]] = affine.load %arg3[{{.*}}, {{.*}}] : memref<1x6xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_R_BIAS]] : f32
}

// -----

// Check handling unknown dimensions for RNN by checking the 
// correctness of allocating and deallocating memory.
func @test_rnn_unkown_dims_allocation(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x3x?xf32>, %arg2: tensor<1x3x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<?x?x?xf32>, tensor<1x3x?xf32>, tensor<1x3x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: @test_rnn_unkown_dims_allocation

  // allocate memory for Hidden (Y_h).
  // CHECK: [[C1_0:%.+]] = constant 1 : index
  // CHECK: [[BATCH_SIZE:%.+]] = dim %arg0, [[C1_0]] : memref<?x?x?xf32>
  // CHECK: [[Y_h:%.+]] = alloc([[BATCH_SIZE]]) : memref<1x?x3xf32>

  // CHECK: return [[Y_h]] : memref<1x?x3xf32>
}


// -----

func @test_squeeze(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Squeeze"(%arg0) { axes = [1, -2]} : (tensor<16x1x32x1x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_squeeze
  // CHECK: [[RES:%.+]] = alloc() : memref<16x32x64xf32>
  // CHECK: [[TENSOR_SIZE:%.+]] = constant 131072 : i64
  // CHECK: "krnl.memcpy"([[RES]], %arg0, [[TENSOR_SIZE]]) : (memref<16x32x64xf32>, memref<16x1x32x1x64xf32>, i64) -> ()
  // CHECK: return [[RES]] : memref<16x32x64xf32>
}

// -----

func @test_squeeze_unknown_dimensions(%arg0 : tensor<?x1x32x?x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Squeeze"(%arg0) { axes = [1,-2]} : (tensor<?x1x32x?x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_squeeze_unknown_dimensions
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x1x32x?x64xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x32x64xf32>
  // CHECK: [[TENSOR_SIZE_0:%.+]] = constant 8192 : i64
  // CHECK: [[DIM_0_i64:%.+]] = index_cast [[DIM_0]] : index to i64
  // CHECK: [[TENSOR_SIZE_1:%.+]] = muli [[TENSOR_SIZE_0]], [[DIM_0_i64]] : i64
  // CHECK: "krnl.memcpy"([[RES]], %arg0, [[TENSOR_SIZE_1]]) : (memref<?x32x64xf32>, memref<?x1x32x?x64xf32>, i64) -> ()
  // CHECK: return [[RES]] : memref<?x32x64xf32>
}

// -----

func @test_split_equal(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.Split"(%arg0) { axis = 0 : si64} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 8)>
  // CHECK-LABEL: @test_split_equal

  // CHECK: [[RES_1:%.+]] = alloc() : memref<8x32x64xf32>
  // CHECK: [[RES_0:%.+]] = alloc() : memref<8x32x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 8, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   [[LOAD_0:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<16x32x64xf32>
  // CHECK:   affine.store [[LOAD_0]], [[RES_0]][%arg1, %arg2, %arg3] : memref<8x32x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 8, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   %[[INDEX:.+]] = affine.apply [[INDEX_MAP]](%arg1)
  // CHECK:   [[LOAD_1:%.+]] = affine.load %arg0[%[[INDEX]], %arg2, %arg3] : memref<16x32x64xf32>
  // CHECK:   affine.store [[LOAD_1]], [[RES_1]][%arg1, %arg2, %arg3] : memref<8x32x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<8x32x64xf32>, memref<8x32x64xf32>
}

// -----

func @test_split_variable(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.Split"(%arg0) { axis = 1 : si64, split = [2, 30]} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 2)>
  // CHECK-LABEL: @test_split_variable

  // CHECK: [[RES_1:%.+]] = alloc() : memref<16x30x64xf32>
  // CHECK: [[RES_0:%.+]] = alloc() : memref<16x2x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 2, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   [[LOAD_0:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<16x32x64xf32>
  // CHECK:   affine.store [[LOAD_0]], [[RES_0]][%arg1, %arg2, %arg3] : memref<16x2x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 30, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   %[[INDEX:.+]] = affine.apply [[INDEX_MAP]](%arg2)
  // CHECK:   [[LOAD_1:%.+]] = affine.load %arg0[%arg1, %[[INDEX]], %arg3] : memref<16x32x64xf32>
  // CHECK:   affine.store [[LOAD_1]], [[RES_1]][%arg1, %arg2, %arg3] : memref<16x30x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<16x2x64xf32>, memref<16x30x64xf32>
}

// -----

func @test_split_unknown_dimension(%arg0 : tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.Split"(%arg0) { axis = 1 : si64, split = [2, 30]} : (tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 2)>
  // CHECK-LABEL: @test_split_unknown_dimension

  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x?x64xf32>
  // CHECK: [[RES_0:%.+]] = alloc([[DIM_0]]) : memref<?x2x64xf32>
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_1:%.+]] = dim %arg0, [[C0_0]] : memref<?x?x64xf32>
  // CHECK: [[RES_1:%.+]] = alloc([[DIM_1]]) : memref<?x30x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[C0_2:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim [[RES_0]], [[C0_2]] : memref<?x2x64xf32>
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to [[DIM_0]], [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 2, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   [[LOAD_0:%.+]] = affine.load %arg0[%arg1, %arg2, %arg3] : memref<?x?x64xf32>
  // CHECK:   affine.store [[LOAD_0]], [[RES_0]][%arg1, %arg2, %arg3] : memref<?x2x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[C0_3:%.+]] = constant 0 : index
  // CHECK: [[DIM_1:%.+]] = dim [[RES_1]], [[C0_3]] : memref<?x30x64xf32>
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to [[DIM_1]], [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 30, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   %[[INDEX:.+]] = affine.apply [[INDEX_MAP]](%arg2)
  // CHECK:   [[LOAD_1:%.+]] = affine.load %arg0[%arg1, %[[INDEX]], %arg3] : memref<?x?x64xf32>
  // CHECK:   affine.store [[LOAD_1]], [[RES_1]][%arg1, %arg2, %arg3] : memref<?x30x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<?x2x64xf32>, memref<?x30x64xf32>
}

// -----

func @cast_lowering_sametype(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "onnx.Cast"(%arg0) {to = 1 : si64} : (tensor<f32>) -> tensor<f32>
  "std.return"(%0) : (tensor<f32>) -> ()

  // CHECK-LABEL: cast_lowering_sametype
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[] : memref<f32>
  // CHECK: affine.store [[LOAD]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func @cast_lowering_intfloat(%arg0: tensor<i64>) -> tensor<f32> {
  %0 = "onnx.Cast"(%arg0) {to = 1 : si64} : (tensor<i64>) -> tensor<f32>
  "std.return"(%0) : (tensor<f32>) -> ()

  // CHECK-LABEL: cast_lowering_intfloat
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[] : memref<i64>
  // CHECK: [[VAL:%.+]] = sitofp [[LOAD]] : i64 to f32
  // CHECK: affine.store [[VAL]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func @cast_lowering_floatint(%arg0: tensor<f32>) -> tensor<i64> {
  %0 = "onnx.Cast"(%arg0) {to = 7 : si64} : (tensor<f32>) -> tensor<i64>
  "std.return"(%0) : (tensor<i64>) -> ()

  // CHECK-LABEL: cast_lowering_floatint
  // CHECK: [[RES:%.+]] = alloc() : memref<i64>
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[] : memref<f32>
  // CHECK: [[VAL:%.+]] = fptosi [[LOAD]] : f32 to i64
  // CHECK: affine.store [[VAL]], [[RES]][] : memref<i64>
  // CHECK: return [[RES]] : memref<i64>
}

// -----

func @cast_lowering_f16f32(%arg0: tensor<f16>) -> tensor<f32> {
  %0 = "onnx.Cast"(%arg0) {to = 1 : si64} : (tensor<f16>) -> tensor<f32>
  "std.return"(%0) : (tensor<f32>) -> ()

  // CHECK-LABEL: cast_lowering_f16f32
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[] : memref<f16>
  // CHECK: [[VAL:%.+]] = fpext [[LOAD]] : f16 to f32
  // CHECK: affine.store [[VAL]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func @cast_lowering_f64f32(%arg0: tensor<f64>) -> tensor<f32> {
  %0 = "onnx.Cast"(%arg0) {to = 1 : si64} : (tensor<f64>) -> tensor<f32>
  "std.return"(%0) : (tensor<f32>) -> ()

  // CHECK-LABEL: cast_lowering_f64f32
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[LOAD:%.+]] = affine.load %arg0[] : memref<f64>
  // CHECK: [[VAL:%.+]] = fptrunc [[LOAD]] : f64 to f32
  // CHECK: affine.store [[VAL]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func @cast_lowering_f64f32_10(%arg0: tensor<10xf64>) -> tensor<*xf32> {
  %0 = "onnx.Cast"(%arg0) {to = 1 : si64} : (tensor<10xf64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: cast_lowering_f64f32_10
  // CHECK: [[RES:%.+]] = alloc() : memref<10xf32>
  // CHECK: [[DEF_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[DEF_LOOPS]]) with ([[DEF_LOOPS]] -> %arg1 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = affine.load %arg0[%arg1] : memref<10xf64>
  // CHECK: [[FPTRUNC:%.+]] = fptrunc [[LOAD1]] : f64 to f32
  // CHECK: affine.store [[FPTRUNC]], [[RES]][%arg1] : memref<10xf32>
  // CHECK: return [[RES]] : memref<10xf32>
}

// -----

func @test_size_known(%arg0: tensor<2x2xf32>) -> tensor<i64> {
  %1 = "onnx.Size"(%arg0) : (tensor<2x2xf32>) -> tensor<i64>
  "std.return"(%1) : (tensor<i64>) -> ()

  // CHECK-LABEL: test_size_known
  // CHECK:      [[RES:%.+]] = alloc() : memref<i64>
  // CHECK-NEXT  [[SIZE:%.+]] = constant 4 : i64
  // CHECK-NEXT  affine.store [[SIZE]], [[RES]][] : memref<i64>
  // CHECK-NEXT  return [[RES]] : memref<i64>

}

// -----

func @test_size_unknown(%arg0 : tensor<?x2x?xf32>) -> tensor<i64> {

  // CHECK-LABEL: test_size_unknown
  // CHECK:       [[RES:%.+]] = alloc() : memref<i64>
  // CHECK-NEXT:  [[INIT:%.+]] = constant 2 : i64
  // CHECK-NEXT:  [[IND1:%.+]] = constant 0 : index
  // CHECK-NEXT:  [[DIM1:%.+]] = dim %arg0, [[IND1]] : memref<?x2x?xf32>
  // CHECK-NEXT:  [[CAST1:%.+]] = index_cast [[DIM1]] : index to i64
  // CHECK-NEXT:  [[TMP1:%.+]] = muli [[INIT]], [[CAST1]] : i64
  // CHECK-NEXT:  [[IND2:%.+]] = constant 2 : index
  // CHECK-NEXT:  [[DIM2:%.+]] = dim %arg0, [[IND2]] : memref<?x2x?xf32>
  // CHECK-NEXT:  [[IND3:%.+]] = index_cast [[DIM2]] : index to i64
  // CHECK-NEXT:  [[SIZE:%.+]] = muli [[TMP1]], [[IND3]] : i64
  // CHECK-NEXT:  affine.store [[SIZE]], [[RES]][] : memref<i64>
  // CHECK-NEXT:  return [[RES]] : memref<i64>

  %1 = "onnx.Size"(%arg0)  : (tensor<?x2x?xf32>) -> tensor<i64>
  "std.return"(%1) : (tensor<i64>) -> ()
}

// -----

// Test gather along axis 0, first example in ONNX for Gather.
func @test_gather_axis0(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>} : () -> tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "std.return"(%0) : (tensor<2x2x2xf32>) -> ()

  // CHECK-LABEL: test_gather_axis0
  // CHECK: [[ALLOC:%.+]] = alloc() : memref<2x2x2xf32>
  // CHECK: [[GLOBAL:%.+]] = "krnl.global"() {name = "{{.*}}", shape = [2, 2], value = dense<{{\[+}}0, 1], [1, 2{{\]+}}> : tensor<2x2xi64>} : () -> memref<2x2xi64>
  // CHECK: [[LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[ZERO:%.+]] = constant 0 : index
  // CHECK: [[DIM_INDEX:%.+]] = constant 0 : index
  // CHECK: [[DIM:%.+]] = "krnl.dim"(%arg0, [[DIM_INDEX]]) : (memref<3x2xf32>, index) -> index
  // CHECK: krnl.iterate([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2) with ([[LOOP]]#0 -> [[ARG1:%.+]] = 0 to 2, [[LOOP]]#1 -> [[ARG2:%.+]] = 0 to 2, [[LOOP]]#2 -> [[ARG3:%.+]] = 0 to 2) {
  // CHECK: [[AFFINE1:%.+]] = affine.load [[GLOBAL]]{{.}}[[ARG1]], [[ARG2]]{{.}} : memref<2x2xi64>
  // CHECK: [[AFFINE2:%.+]] = index_cast [[AFFINE1]] : i64 to index
  // CHECK: [[AFFINE3:%.+]] = addi [[AFFINE2]], [[DIM]] : index
  // CHECK: [[CMP:%.+]] = cmpi "slt", [[AFFINE2]], [[ZERO]] : index
  // CHECK: [[AFFINE4:%.+]] = select [[CMP]], [[AFFINE3]], [[AFFINE2]] : index
  // CHECK: [[DATA:%.+]] = load %arg0{{.}}[[AFFINE4]], [[ARG3]]{{.}} : memref<3x2xf32>
  // CHECK: affine.store [[DATA]], [[ALLOC]]{{.}}[[ARG1]], [[ARG2]], [[ARG3]]{{.}} : memref<2x2x2xf32>
}

// -----

// Test gather along axis 1, second example in ONNX for Gather.
func @test_gather_axis1(%arg0 : tensor<3x3xf32>) -> tensor<3x1x2xf32> {
  %indices = "onnx.Constant"() {value = dense<[[0, 2]]> : tensor<1x2xi64>} : () -> tensor<1x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  "std.return"(%0) : (tensor<3x1x2xf32>) -> ()

  // CHECK-LABEL: test_gather_axis1
  // CHECK: [[ALLOC:%.+]] = alloc() : memref<3x1x2xf32>
  // CHECK: [[GLOBAL:%.+]] = "krnl.global"() {name = "constant_0", shape = [1, 2], value = dense<{{\[+}}0, 2{{\]+}}> : tensor<1x2xi64>} : () -> memref<1x2xi64>
  // CHECK: [[LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[ZERO:%.+]] = constant 0 : index
  // CHECK: [[DIM_INDEX:%.+]] = constant 1 : index
  // CHECK: [[DIM:%.+]] = "krnl.dim"(%arg0, [[DIM_INDEX]]) : (memref<3x3xf32>, index) -> index
  // CHECK: krnl.iterate([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2) with ([[LOOP]]#0 -> [[ARG1:%.+]] = 0 to 3, [[LOOP]]#1 -> [[ARG2:%.+]] = 0 to 1, [[LOOP]]#2 -> [[ARG3:%.+]] = 0 to 2) {
  // CHECK: [[AFFINE1:%.+]] = affine.load [[GLOBAL]]{{.}}[[ARG2]], [[ARG3]]{{.}} : memref<1x2xi64>
  // CHECK: [[AFFINE2:%.+]] = index_cast [[AFFINE1]] : i64 to index
  // CHECK: [[AFFINE3:%.+]] = addi [[AFFINE2]], [[DIM]] : index
  // CHECK: [[CMP:%.+]] = cmpi "slt", [[AFFINE2]], [[ZERO]] : index
  // CHECK: [[AFFINE4:%.+]] = select [[CMP]], [[AFFINE3]], [[AFFINE2]] : index
  // CHECK: [[DATA:%.+]] = load %arg0{{.}}[[ARG1]], [[AFFINE4]]{{.}} : memref<3x3xf32>
  // CHECK: affine.store [[DATA]], [[ALLOC]]{{.}}[[ARG1]], [[ARG2]], [[ARG3]]{{.}} : memref<3x1x2xf32>
}

// -----

// Check the lowering of ConstantOfShape when:
//   - No value attribute.
//   - The input is an empty tensor.
// Expected emitted code:
//   - No need a Krnl iterate.
//   - The output is a scalar tensor.
func @test_constant_of_shape_empty_tensor(%arg0 : tensor<0xi64>) -> tensor<*xf32> {
  %0 = "onnx.ConstantOfShape"(%arg0) : (tensor<0xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_of_shape_empty_tensor
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[CST_VALUE:%.+]] = constant 0.000000e+00 : f32
  // CHECK: affine.store [[CST_VALUE]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

// Check the lowering of ConstantOfShape when:
//   - The input is not a constant tensor.
// Expected emitted code:
//   - Emit code to compute output dimensions from the input's dimensions.
//   - Krnl iterates are used to set values to the output.
func @test_constant_of_shape_dynamic_dims(%arg0 : tensor<3xi64>) -> tensor<*xf32> {
  %0 = "onnx.ConstantOfShape"(%arg0) {value = dense<[1.0]> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_of_shape_dynamic_dims
  // CHECK: [[CST0:%.+]] = constant 0 : index
  // CHECK: [[LOAD_DIM_0:%.+]] = affine.load %arg0{{\[}}[[CST0]]{{\]}} : memref<3xi64>
  // CHECK: [[DIM_0:%.+]] = index_cast [[LOAD_DIM_0]] : i64 to index
  // CHECK: [[CST1:%.+]] = constant 1 : index
  // CHECK: [[LOAD_DIM_1:%.+]] = affine.load %arg0{{\[}}[[CST1]]{{\]}} : memref<3xi64>
  // CHECK: [[DIM_1:%.+]] = index_cast [[LOAD_DIM_1]] : i64 to index
  // CHECK: [[CST2:%.+]] = constant 2 : index
  // CHECK: [[LOAD_DIM_2:%.+]] = affine.load %arg0{{\[}}[[CST2]]{{\]}} : memref<3xi64>
  // CHECK: [[DIM_2:%.+]] = index_cast [[LOAD_DIM_2]] : i64 to index
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]], [[DIM_1]], [[DIM_2]]) : memref<?x?x?xf32>

  // CHECK: [[CST_VALUE:%.+]] = constant 1.000000e+00 : f32
  // CHECK: [[LOOP_DEF:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[CST00:%.+]] = constant 0 : index
  // CHECK: [[RES_DIM_0:%.+]] = dim [[RES]], [[CST00]] : memref<?x?x?xf32>
  // CHECK: [[CST11:%.+]] = constant 1 : index
  // CHECK: [[RES_DIM_1:%.+]] = dim [[RES]], [[CST11]] : memref<?x?x?xf32>
  // CHECK: [[CST22:%.+]] = constant 2 : index
  // CHECK: [[RES_DIM_2:%.+]] = dim [[RES]], [[CST22]] : memref<?x?x?xf32>
  // CHECK: krnl.iterate([[LOOP_DEF]]#0, [[LOOP_DEF]]#1, [[LOOP_DEF]]#2) with ([[LOOP_DEF]]#0 -> %arg1 = 0 to [[RES_DIM_0]], [[LOOP_DEF]]#1 -> %arg2 = 0 to [[RES_DIM_1]], [[LOOP_DEF]]#2 -> %arg3 = 0 to [[RES_DIM_2]]) {
  // CHECK:   affine.store [[CST_VALUE]], [[RES]][%arg1, %arg2, %arg3] : memref<?x?x?xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<?x?x?xf32>
}

// -----

// Check the lowering of ConstantOfShape when:
//   - The input is a constant tensor.
// Expected emitted code:
//   - Output dimensions are computed during compilation time.
//   - Krnl iterates are used to set values to the output.
func @test_constant_of_shape_static_dims() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[3, 4, 5]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.ConstantOfShape"(%0) {value = dense<[1.0]> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_of_shape_static_dims
  // CHECK: [[RES:%.+]] = alloc() : memref<3x4x5xf32>
  // CHECK: [[GLOBAL_CST:%.+]] = "krnl.global"() {name = "constant_0", shape = [3], value = dense<[3, 4, 5]> : tensor<3xi64>} : () -> memref<3xi64>
  // CHECK: [[CST_VALUE:%.+]] = constant 1.000000e+00 : f32
  // CHECK: [[LOOP_DEF:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[LOOP_DEF]]#0, [[LOOP_DEF]]#1, [[LOOP_DEF]]#2) with ([[LOOP_DEF]]#0 -> %arg0 = 0 to 3, [[LOOP_DEF]]#1 -> %arg1 = 0 to 4, [[LOOP_DEF]]#2 -> %arg2 = 0 to 5) {
  // CHECK:   affine.store [[CST_VALUE]], [[RES]][%arg0, %arg1, %arg2] : memref<3x4x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xf32>
}

// -----

// Test Tile with 2D input and constant repeats
func @test_tile1(%arg0 : tensor<4x8xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() { value = dense<[3, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Tile"(%arg0, %0) : (tensor<4x8xf32>, tensor<2xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0)[s0] -> (d0 mod s0)>
  // CHECK-LABEL: test_tile1
  // CHECK:  [[R0:%.+]] = alloc() : memref<12x16xf32>
  // CHECK:  [[R1:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[3, 2]> : tensor<2xi64>} : () -> memref<2xi64>
  // CHECK:  [[R2:%.+]]:2 = krnl.define_loops 2
  // CHECK:  krnl.iterate([[R2]]#0, [[R2]]#1) with ([[R2]]#0 -> [[ARG1:%.+]] = 0 to 12, [[R2]]#1 -> [[ARG2:%.+]] = 0 to 16) {
  // CHECK:    [[C0:%.+]] = constant 0 : index
  // CHECK:    [[R3:%.+]] = dim %arg0, [[C0]] : memref<4x8xf32>
  // CHECK:    [[R4:%.+]] = affine.apply [[INDEX_MAP]]([[ARG1]]){{\[}}[[R3]]{{\]}}
  // CHECK:    [[C1:%.+]] = constant 1 : index
  // CHECK:    [[R5:%.+]] = dim %arg0, [[C1]] : memref<4x8xf32>
  // CHECK:    [[R6:%.+]] = affine.apply [[INDEX_MAP]]([[ARG2]]){{\[}}[[R5]]{{\]}}
  // CHECK:    [[R7:%.+]] = affine.load %arg0{{\[}}[[R4]], [[R6]]{{\]}} : memref<4x8xf32>
  // CHECK:    affine.store [[R7]], %0{{\[}}[[ARG1]], [[ARG2]]{{\]}} : memref<12x16xf32>
}

// -----

// Test Tile with 1D input and unknown repeats
func @test_tile2(%arg0 : tensor<8xf32>, %arg1 : tensor<1xi64>) -> tensor<*xf32> {
  %1 = "onnx.Tile"(%arg0, %arg1) : (tensor<8xf32>, tensor<1xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0)[s0] -> (d0 mod s0)>
  // CHECK-LABEL test_tile2
  // CHECK:  [[C0:%.+]] = constant 0 : index
  // CHECK:  [[R0:%.+]] = affine.load %arg1{{\[}}[[C0]]{{\]}} : memref<1xi64>
  // CHECK:  [[R1:%.+]] = index_cast [[R0]] : i64 to index
  // CHECK:  [[C0_0:%.+]] = constant 0 : index
  // CHECK:  [[R2:%.+]] = dim %arg0, [[C0_0]] : memref<8xf32>
  // CHECK:  [[R3:%.+]] = muli [[R2]], [[R1]] : index
  // CHECK:  [[R4:%.+]] = alloc([[R3]]) : memref<?xf32>
  // CHECK:  [[R5:%.+]] = krnl.define_loops 1
  // CHECK:  [[C0_1:%.+]] = constant 0 : index
  // CHECK:  [[R6:%.+]] = dim [[R4]], [[C0_1]] : memref<?xf32>
  // CHECK:  krnl.iterate([[R5]]) with ([[R5]] -> [[ARG2:%.+]] = 0 to [[R6]]) {
  // CHECK:    [[C0_2:%.+]] = constant 0 : index
  // CHECK:    [[R7:%.+]] = dim %arg0, [[C0_2]] : memref<8xf32>
  // CHECK:    [[R8:%.+]] = affine.apply [[INDEX_MAP]]([[ARG2]]){{\[}}[[R7]]{{\]}}
  // CHECK:    [[R9:%.+]] = affine.load %arg0{{\[}}[[R8]]{{\]}} : memref<8xf32>
  // CHECK:    affine.store [[R9]], [[R4]]{{\[}}[[ARG2]]{{\]}} : memref<?xf32>
}

// -----

func @test_flatten0(%arg0 : tensor<2x3x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Flatten"(%arg0) {axis = 0 : si64} : (tensor<2x3x4xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
  // CHECK: [[MAP_FIRST:#.+]] = affine_map<() -> (0)>
  // CHECK: [[MAP_SECOND:#.+]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2 + d1 * s2 + d0 * (s1 * s2))>
  // CHECK-LABEL test_flatten0
  // CHECK:  [[ALLOC:%.+]] = alloc() : memref<1x24xf32>
  // CHECK:  [[LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK:  krnl.iterate([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2) with ([[LOOP]]#0 -> [[LOOPARG1:%.+]] = 0 to 2, [[LOOP]]#1 -> [[LOOPARG2:%.+]] = 0 to 3, [[LOOP]]#2 -> [[LOOPARG3:%.+]] = 0 to 4) {
  // CHECK:    [[LOAD:%.+]] = affine.load %arg0{{\[}}[[LOOPARG1]], [[LOOPARG2]], [[LOOPARG3]]{{\]}} : memref<2x3x4xf32>
  // CHECK:    [[FIRSTDIM:%.+]] = affine.apply [[MAP_FIRST]]()
  // CHECK:    [[C0:%.+]] = constant 0 : index
  // CHECK:    [[R4:%.+]] = dim %arg0, [[C0]] : memref<2x3x4xf32>
  // CHECK:    [[C1:%.+]] = constant 1 : index
  // CHECK:    [[R5:%.+]] = dim %arg0, [[C1]] : memref<2x3x4xf32>
  // CHECK:    [[C2:%.+]] = constant 2 : index
  // CHECK:    [[R6:%.+]] = dim %arg0, [[C2]] : memref<2x3x4xf32>
  // CHECK:    [[SECONDDIM:%.+]] = affine.apply [[MAP_SECOND]]([[LOOPARG1]], [[LOOPARG2]], [[LOOPARG3]]){{\[}}[[R4]], [[R5]], [[R6]]{{\]}}
  // CHECK:    affine.store [[LOAD]], [[ALLOC]]{{\[}}[[FIRSTDIM]], [[SECONDDIM]]{{\]}} : memref<1x24xf32>
}

// -----

// test partially known input shape
func @test_flatten1(%arg0 : tensor<2x?x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<2x?x4xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
 
  // CHECK:  [[MAP1:#.+]] = affine_map<(d0, d1)[s0, s1] -> (d1 + d0 * s1)>
  // CHECK:  [[MAP2:#.+]] = affine_map<(d0)[s0] -> (d0)>
  // CHECK-LABEL test_flatten1
  // CHECK:  [[C1:%.+]] = constant 1 : index
  // CHECK:  [[C0:%.+]] = constant 0 : index
  // CHECK:  [[R0:%.+]] = dim %arg0, [[C0]] : memref<2x?x4xf32>
  // CHECK:  [[R1:%.+]] = muli [[C1]], [[R0]] : index
  // CHECK:  [[C1_0:%.+]] = constant 1 : index
  // CHECK:  [[R2:%.+]] = dim %arg0, [[C1_0]] : memref<2x?x4xf32>
  // CHECK:  [[R3:%.+]] = muli [[R1]], [[R2]] : index
  // CHECK:  [[R4:%.+]] = alloc([[R3]]) : memref<?x4xf32>
  // CHECK:  [[R5:%.+]]:3 = krnl.define_loops 3
  // CHECK:  [[C1_1:%.+]] = constant 1 : index
  // CHECK:  [[R6:%.+]] = dim %arg0, [[C1_1]] : memref<2x?x4xf32>
  // CHECK:  krnl.iterate([[R5]]#0, [[R5]]#1, [[R5]]#2) with ([[R5]]#0 -> [[ARG1:%.+]] = 0 to 2, [[R5]]#1 -> [[ARG2:%.+]] = 0 to [[R6]], [[R5]]#2 -> [[ARG3:%.+]] = 0 to 4) {
  // CHECK:    [[R7:%.+]] = affine.load %arg0{{\[}}[[ARG1]], [[ARG2]], [[ARG3]]{{\]}} : memref<2x?x4xf32>
  // CHECK:    [[C0_2:%.+]] = constant 0 : index
  // CHECK:    [[R8:%.+]] = dim %arg0, [[C0_2]] : memref<2x?x4xf32>
  // CHECK:    [[C1_3:%.+]] = constant 1 : index
  // CHECK:    [[R9:%.+]] = dim %arg0, [[C1_3]] : memref<2x?x4xf32>
  // CHECK:    [[R10:%.+]] = affine.apply [[MAP1]]([[ARG1]], [[ARG2]]){{\[}}[[R8]], [[R9]]{{\]}}
  // CHECK:    [[C2:%.+]] = constant 2 : index
  // CHECK:    [[R11:%.+]] = dim %arg0, [[C2]] : memref<2x?x4xf32>
  // CHECK:    [[R12:%.+]] = affine.apply [[MAP2]]([[ARG3]]){{\[}}[[R11]]{{\]}}
  // CHECK:    store [[R7]], [[R4]]{{\[}}[[R10]], [[R12]]{{\]}} : memref<?x4xf32>

}

// -----

// Test Tile with 1D unknown input 
func @test_tile3(%arg0 : tensor<?xf32>, %arg1 : tensor<1xi64>) -> tensor<*xf32> {
  %1 = "onnx.Tile"(%arg0, %arg1) : (tensor<?xf32>, tensor<1xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0)[s0] -> (d0 mod s0)>
  // CHECK-LABEL test_tile3
  // CHECK:  [[C0:%.+]] = constant 0 : index
  // CHECK:  [[R0:%.+]] = affine.load %arg1{{\[}}[[C0]]{{\]}} : memref<1xi64>
  // CHECK:  [[R1:%.+]] = index_cast [[R0]] : i64 to index
  // CHECK:  [[C0_0:%.+]] = constant 0 : index
  // CHECK:  [[R2:%.+]] = dim %arg0, [[C0_0]] : memref<?xf32>
  // CHECK:  [[R3:%.+]] = muli [[R2]], [[R1]] : index
  // CHECK:  [[R4:%.+]] = alloc([[R3]]) : memref<?xf32>
  // CHECK:  [[R5:%.+]] = krnl.define_loops 1
  // CHECK:  [[C0_1:%.+]] = constant 0 : index
  // CHECK:  [[R6:%.+]] = dim %4, [[C0_1]] : memref<?xf32>
  // CHECK:  krnl.iterate([[R5]]) with ([[R5]] -> [[ARG2:%.+]] = 0 to [[R6]]) {
  // CHECK:    [[C0_2:%.+]] = constant 0 : index
  // CHECK:    [[R7:%.+]] = dim %arg0, [[C0_2]] : memref<?xf32>
  // CHECK:    [[R8:%.+]] = affine.apply [[INDEX_MAP]]([[ARG2]]){{\[}}[[R7]]{{\]}}
  // CHECK:    [[R9:%.+]] = load %arg0{{\[}}[[R8]]{{\]}} : memref<?xf32>
  // CHECK:    affine.store [[R9]], [[R4]]{{\[}}[[ARG2]]{{\]}} : memref<?xf32>
}

// -----

func @test_less(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xi1>
  return %0 : tensor<3x4x5xi1>

  // CHECK-LABEL: test_less
  // CHECK: [[RES:%.+]] = alloc() : memref<3x4x5xi1>
  // CHECK: [[DEF_LOOPS]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 3, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 4, [[DEF_LOOPS]]#2 -> %arg4 = 0 to 5) {
  // CHECK:   [[LHS:%.+]] = affine.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:   [[RHS:%.+]] = affine.load %arg1[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:   [[LESS:%.+]] = cmpf "olt", [[LHS]], [[RHS]] : f32
  // CHECK:   affine.store [[LESS]], [[RES]][%arg2, %arg3, %arg4] : memref<3x4x5xi1>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xi1>
}

// -----

func @test_less_broadcast(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5xf32>) -> tensor<3x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<5xf32>) -> tensor<3x4x5xi1>
  return %0 : tensor<3x4x5xi1>

  // CHECK-LABEL: test_less_broadcast
  // CHECK: [[RES:%.+]] = alloc() : memref<3x4x5xi1>
  // CHECK: [[DEF_LOOPS]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 3, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 4, [[DEF_LOOPS]]#2 -> %arg4 = 0 to 5) {
  // CHECK:   [[LHS:%.+]] = affine.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:   [[RHS:%.+]] = affine.load %arg1[%arg4] : memref<5xf32>
  // CHECK:   [[LESS:%.+]] = cmpf "olt", [[LHS]], [[RHS]] : f32
  // CHECK:   affine.store [[LESS]], [[RES]][%arg2, %arg3, %arg4] : memref<3x4x5xi1>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xi1>
}

// -----

func @test_less_unknown_dims(%arg0: tensor<3x4x5xf32>, %arg1: tensor<?x4x5xf32>) -> tensor<3x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<?x4x5xf32>) -> tensor<3x4x5xi1>
  return %0 : tensor<3x4x5xi1>

  // CHECK-LABEL: test_less_unknown_dims
  // CHECK: [[RES:%.+]] = alloc() : memref<3x4x5xi1>
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM0:%.+]] = dim %arg1, [[C0]] : memref<?x4x5xf32>
  // CHECK: [[C1:%.+]] = constant 1 : index
  // CHECK: [[LOAD_DIM0:%.+]] = cmpi "eq", [[DIM0]], [[C1]] : index
  // CHECK: [[DEF_LOOPS:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 3, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 4, [[DEF_LOOPS]]#2 -> %arg4 = 0 to 5) {
  // CHECK:   [[LHS:%.+]] = affine.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:   [[C0_INDEX:%.+]] = constant 0 : index
  // CHECK:   [[RHS_DIM0:%.+]] = select [[LOAD_DIM0]], [[C0_INDEX]], %arg2 : index
  // CHECK:   [[RHS:%.+]] = affine.load %arg1{{\[}}[[RHS_DIM0]], %arg3, %arg4] : memref<?x4x5xf32>
  // CHECK:   [[LESS:%.+]] = cmpf "olt", [[LHS]], [[RHS]] : f32
  // CHECK:   affine.store [[LESS]], [[RES]][%arg2, %arg3, %arg4] : memref<3x4x5xi1>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xi1>
}

