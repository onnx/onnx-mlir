// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// ----

// TODO: Remove test_no_argument_1 from the test - empty function body is no longer
// supported in mlir: https://reviews.llvm.org/D91886
func private @test_no_argument_2() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value =  dense<[[1.000000e+0, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>} : () -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

}

// CHECK-LABEL: test_no_argument_2
// CHECK: [[GLOBAL:%.+]] = "{{.*}}"({{.*}}) {{.*}} : ({{.*}}) -> memref<2x2xf32>
// CHECK: [[ALLOC:%.+]] = alloc() : memref<2x2xf32>
// CHECK: [[CONST_4:%.+]] = constant 4 : i64
// CHECK: [[CONST_4_0:%.+]] = constant 4 : i64
// CHECK: [[SIZE:%.+]] = muli [[CONST_4]], [[CONST_4_0]] : i64
// CHECK: "krnl.memcpy"([[ALLOC]], [[GLOBAL]], [[SIZE]]) : (memref<2x2xf32>, memref<2x2xf32>, i64) -> ()
// CHECK: return [[ALLOC]] : memref<2x2xf32>

// -----

func private @test_elementwise_op_with_scalar_values_1(%arg0 : tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Exp"(%arg0) : (tensor<f32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_elementwise_op_with_scalar_values_1
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<f32>
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: krnl.store [[EXP]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func private @test_elementwise_op_with_scalar_values_2(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_elementwise_op_with_scalar_values_2
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[] : memref<f32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[] : memref<f32>
  // CHECK: [[ADD:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[ADD]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func private @test_elementwise_op_with_scalar_values_3(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_elementwise_op_with_scalar_values_3
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[] : memref<f32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[] : memref<f32>
  // CHECK: [[ADD1:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[LOAD3:%.+]] = krnl.load %arg2[] : memref<f32>
  // CHECK: [[ADD2:%.+]] = addf [[ADD1]], [[LOAD3]] : f32
  // CHECK: krnl.store [[ADD2]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func private @test_add(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_add
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[ADDF:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[ADDF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func private @test_mul(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_mul
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MULF:%.+]] = mulf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[MULF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func private @test_div(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_div
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[DIVF:%.+]] = divf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[DIVF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func private @test_sub(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sub
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[SUBF:%.+]] = subf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[SUBF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func private @test_and(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<*xi1> {
  %0 = "onnx.And"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  "std.return"(%0) : (tensor<*xi1>) -> ()

  // CHECK-LABEL: test_and
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xi1>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[AND:%.+]] = and [[LOAD1]], [[LOAD2]] : i1
  // CHECK: krnl.store [[AND]], [[RES]][%arg2, %arg3] : memref<10x10xi1>
  // CHECK: return [[RES]] : memref<10x10xi1>
}

// -----

func private @test_or(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<*xi1> {
  %0 = "onnx.Or"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  "std.return"(%0) : (tensor<*xi1>) -> ()

  // CHECK-LABEL: test_or
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xi1>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[OR:%.+]] = or [[LOAD1]], [[LOAD2]] : i1
  // CHECK: krnl.store [[OR]], [[RES]][%arg2, %arg3] : memref<10x10xi1>
  // CHECK: return [[RES]] : memref<10x10xi1>
}

// -----

func private @test_xor(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<*xi1> {
  %0 = "onnx.Xor"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  "std.return"(%0) : (tensor<*xi1>) -> ()

  // CHECK-LABEL: test_xor
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xi1>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[XOR:%.+]] = xor [[LOAD1]], [[LOAD2]] : i1
  // CHECK: krnl.store [[XOR]], [[RES]][%arg2, %arg3] : memref<10x10xi1>
  // CHECK: return [[RES]] : memref<10x10xi1>
}

// -----

func private @test_exp(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: krnl.store [[EXP]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_tanh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[X:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ONE:%.+]] = constant 1.000000e+00 : f32
  // CHECK: [[TWO:%.+]] = constant 2.000000e+00 : f32
  // CHECK: [[X_MUL_2:%.+]] = mulf [[X]], [[TWO]] : f32
  // CHECK: [[NEG_X_MUL_2:%.+]] = negf [[X_MUL_2]] : f32
  // CHECK: [[EXP_1:%.+]] = exp [[NEG_X_MUL_2]] : f32
  // CHECK: [[SUB_1:%.+]] = subf %cst, [[EXP_1]] : f32
  // CHECK: [[ADD_1:%.+]] = addf %cst, [[EXP_1]] : f32
  // CHECK: [[DIV_1:%.+]] = divf [[SUB_1]], [[ADD_1]] : f32
  // CHECK: [[EXP_2:%.+]] = exp [[X_MUL_2]] : f32
  // CHECK: [[SUB_2:%.+]] = subf [[EXP_2]], %cst : f32
  // CHECK: [[ADD_2:%.+]] = addf [[EXP_2]], %cst : f32
  // CHECK: [[DIV_2:%.+]] = divf [[SUB_2]], [[ADD_2]] : f32
  // CHECK: [[ZERO:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[CMP:%.+]] = cmpf oge, [[X]], [[ZERO]] : f32
  // CHECK: [[TANH:%.+]] = select [[CMP]], [[DIV_1]], [[DIV_2]] : f32
  // CHECK: krnl.store [[TANH]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_sinh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[TWO:%.+]] = constant {{2.+}} : f32
  // CHECK: [[NLOAD:%.+]] = subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = exp [[NLOAD]] : f32
  // CHECK: [[DIVIDEND:%.+]] = subf [[EXP]], [[NEXP]] : f32
  // CHECK: [[SINH_RES:%.+]] = divf [[DIVIDEND]], [[TWO]] : f32
  // CHECK: krnl.store [[SINH_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_cosh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[TWO:%.+]] = constant {{2.+}} : f32
  // CHECK: [[NLOAD:%.+]] = subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = exp [[NLOAD]] : f32
  // CHECK: [[DIVIDEND:%.+]] = addf [[EXP]], [[NEXP]] : f32
  // CHECK: [[COSH_RES:%.+]] = divf [[DIVIDEND]], [[TWO]] : f32
  // CHECK: krnl.store [[COSH_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_cos(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[COS:%.+]] = cos [[LOAD]] : f32
  // CHECK: krnl.store [[COS]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_sin(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sin"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sin
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[SIN:%.+]] = sin [[LOAD]] : f32
  // CHECK: krnl.store [[SIN]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_log(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[LOG:%.+]] = log [[LOAD]] : f32
  // CHECK: krnl.store [[LOG]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_sigmoid(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[NLOAD:%.+]] = subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = exp [[NLOAD]] : f32
  // CHECK: [[DIVISOR:%.+]] = addf [[ONE]], [[NEXP]] : f32
  // CHECK: [[SIGMOID_RES:%.+]] = divf [[ONE]], [[DIVISOR]] : f32
  // CHECK: krnl.store [[SIGMOID_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_relu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[LTZERO:%.+]] = cmpf olt, [[LOAD]], [[ZERO]] : f32
  // CHECK: [[RELU_RES:%.+]] = select [[LTZERO]], [[ZERO]], [[LOAD]] : f32
  // CHECK: krnl.store [[RELU_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_reshape(%arg0 : tensor<?x10xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD_0:%.+]] = krnl.load %arg1[%[[CONSTANT_1]]] : memref<4xi64>
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_1:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: [[DIM_1_CAST:%.+]] = index_cast [[DIM_1]] : index to i64
  // CHECK: [[CONSTANT_2:%.+]] = constant 0 : i64
  // CHECK: [[CMP_0:%.+]] = cmpi eq, [[LOAD_0]], [[CONSTANT_2]] : i64
  // CHECK: [[SELECT_0:%.+]] = select [[CMP_0]], [[DIM_1_CAST]], [[LOAD_0]] : i64
  // CHECK: [[MUL_1:%.+]] = muli [[TYPE_IN_BYTES_1]], [[SELECT_0]] : i64

  // CHECK: %[[CONSTANT_3:.+]] = constant 1 : index
  // CHECK: [[LOAD_1:%.+]] = krnl.load %arg1[%[[CONSTANT_3]]] : memref<4xi64>
  // CHECK: [[CONSTANT_3:%.+]] = constant 10 : i64
  // CHECK: [[CONSTANT_4:%.+]] = constant 0 : i64
  // CHECK: [[CMP_1:%.+]] = cmpi eq, [[LOAD_1]], [[CONSTANT_4]] : i64
  // CHECK: [[SELECT_1:%.+]] = select [[CMP_1]], [[CONSTANT_3]], [[LOAD_1]] : i64
  // CHECK: [[MUL_2:%.+]] = muli [[MUL_1]], [[SELECT_1]] : i64

  // CHECK: %[[CONSTANT_5:.+]] = constant 2 : index
  // CHECK: [[LOAD_2:%.+]] = krnl.load %arg1[%[[CONSTANT_5]]] : memref<4xi64>
  // CHECK: [[MUL_3:%.+]] = muli [[MUL_2]], [[LOAD_2]] : i64

  // CHECK: %[[CONSTANT_6:.+]] = constant 3 : index
  // CHECK: [[LOAD_3:%.+]] = krnl.load %arg1[%[[CONSTANT_6]]] : memref<4xi64>
  // CHECK: [[MUL_4:%.+]] = muli [[MUL_3]], [[LOAD_3]] : i64

  // CHECK: [[CONSTANT_8:%.+]] = constant -1 : i64
  // CHECK: [[TENSOR_SIZE_FROM_SHAPE:%.+]] = muli [[MUL_4]], [[CONSTANT_8]] : i64

  // CHECK: [[CMP_2:%.+]] = cmpi eq, [[SELECT_0]], [[CONSTANT_8]] : i64
  // CHECK: [[DIVISIGNED_0:%.+]] = divi_signed [[TENSOR_SIZE]], [[TENSOR_SIZE_FROM_SHAPE]] : i64
  // CHECK: [[SELECT_2:%.+]] = select [[CMP_2]], [[DIVISIGNED_0]], [[SELECT_0]] : i64
  // CHECK: [[CAST_0:%.+]] = index_cast [[SELECT_2]] : i64 to index

  // CHECK: [[CMP_3:%.+]] = cmpi eq, [[SELECT_1]], [[CONSTANT_8]] : i64
  // CHECK: [[DIVISIGNED_1:%.+]] = divi_signed [[TENSOR_SIZE]], [[TENSOR_SIZE_FROM_SHAPE]] : i64
  // CHECK: [[SELECT_3:%.+]] = select [[CMP_3]], [[DIVISIGNED_1]], [[SELECT_1]] : i64
  // CHECK: [[CAST_1:%.+]] = index_cast [[SELECT_3]] : i64 to index

  // CHECK: [[CMP_4:%.+]] = cmpi eq, [[LOAD_2]], [[CONSTANT_8]] : i64
  // CHECK: [[DIVISIGNED_2:%.+]] = divi_signed [[TENSOR_SIZE]], [[TENSOR_SIZE_FROM_SHAPE]] : i64
  // CHECK: [[SELECT_4:%.+]] = select [[CMP_4]], [[DIVISIGNED_2]], [[LOAD_2]] : i64
  // CHECK: [[CAST_2:%.+]] = index_cast [[SELECT_4]] : i64 to index

  // CHECK: [[CMP_5:%.+]] = cmpi eq, [[LOAD_3]], [[CONSTANT_8]] : i64
  // CHECK: [[DIVISIGNED_3:%.+]] = divi_signed [[TENSOR_SIZE]], [[TENSOR_SIZE_FROM_SHAPE]] : i64
  // CHECK: [[SELECT_5:%.+]] = select [[CMP_5]], [[DIVISIGNED_3]], [[LOAD_3]] : i64
  // CHECK: [[CAST_3:%.+]] = index_cast [[SELECT_5]] : i64 to index

  // CHECK: [[ALLOC:%.+]] = alloc([[CAST_0]], [[CAST_1]], [[CAST_2]], [[CAST_3]]) : memref<?x?x?x?xf32>
  // CHECK: "krnl.memcpy"([[ALLOC]], %arg0, [[TENSOR_SIZE]]) : (memref<?x?x?x?xf32>, memref<?x10xf32>, i64) -> ()
  // CHECK: return [[ALLOC]] : memref<?x?x?x?xf32>
}

// ----
func private @test_reshape_constant(%arg0 : tensor<?x10xf32>) -> tensor<?x5xf32> {
  %0 = "onnx.Constant"() {value = dense<[-1, 5]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<?x10xf32>, tensor<2xi64>) -> tensor<?x5xf32>
  "std.return"(%1) : (tensor<?x5xf32>) -> ()
// CHECK-LABEL:     test_reshape_constant
// CHECK-SAME:     ([[VAR_arg0:%.+]]: memref<?x10xf32>) -> memref<?x5xf32> {
// CHECK:           [[VAR_0:%.+]] = "krnl.global"() {name = "constant_0", shape = [2], value = dense<[-1, 5]> : tensor<2xi64>} : () -> memref<2xi64>
// CHECK:           [[VAR_c4_i64:%.+]] = constant 4 : i64
// CHECK:           [[VAR_c0:%.+]] = constant 0 : index
// CHECK:           [[VAR_1:%.+]] = dim [[VAR_arg0]], [[VAR_c0]] : memref<?x10xf32>
// CHECK:           [[VAR_2:%.+]] = index_cast [[VAR_1]] : index to i64
// CHECK:           [[VAR_3:%.+]] = muli [[VAR_c4_i64]], [[VAR_2]] : i64
// CHECK:           [[VAR_c10_i64:%.+]] = constant 10 : i64
// CHECK:           [[VAR_4:%.+]] = muli [[VAR_3]], [[VAR_c10_i64]] : i64
// CHECK:           [[VAR_c4_i64_0:%.+]] = constant 4 : i64
// CHECK:           [[VAR_c_min_1_i64:%.+]] = constant -1 : i64
// CHECK:           [[VAR_5:%.+]] = muli [[VAR_c4_i64_0]], [[VAR_c_min_1_i64]] : i64
// CHECK:           [[VAR_c5_i64:%.+]] = constant 5 : i64
// CHECK:           [[VAR_6:%.+]] = muli [[VAR_5]], [[VAR_c5_i64]] : i64
// CHECK:           [[VAR_c_min_1_i64_1:%.+]] = constant -1 : i64
// CHECK:           [[VAR_7:%.+]] = muli [[VAR_6]], [[VAR_c_min_1_i64_1]] : i64
// CHECK:           [[VAR_8:%.+]] = cmpi eq, [[VAR_c_min_1_i64]], [[VAR_c_min_1_i64_1]] : i64
// CHECK:           [[VAR_9:%.+]] = divi_signed [[VAR_4]], [[VAR_7]] : i64
// CHECK:           [[VAR_10:%.+]] = select [[VAR_8]], [[VAR_9]], [[VAR_c_min_1_i64]] : i64
// CHECK:           [[VAR_11:%.+]] = index_cast [[VAR_10]] : i64 to index
// CHECK:           [[VAR_12:%.+]] = alloc([[VAR_11]]) : memref<?x5xf32>
// CHECK:           "krnl.memcpy"([[VAR_12]], [[VAR_arg0]], [[VAR_4]]) : (memref<?x5xf32>, memref<?x10xf32>, i64) -> ()
// CHECK:           return [[VAR_12]] : memref<?x5xf32>
// CHECK:         }
}

// -----

func private @test_sum(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sum
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[ADD:%.+]] = addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[ADD]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func private @test_max(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Max"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_max
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MAX:%.+]] = cmpf ogt, [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[RELU_RES:%.+]] = select [[MAX]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[RELU_RES]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func private @test_min(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Min"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_min
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MIN:%.+]] = cmpf olt, [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[RELU_RES:%.+]] = select [[MIN]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[RELU_RES]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func private @test_elu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[ALPHA:%.+]] = constant {{2.+}} : f32
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[CMP:%.+]] = cmpf olt, [[LOAD]], [[ZERO]] : f32
  // CHECK: [[SUB:%.+]] = subf [[EXP]], [[ONE]] : f32
  // CHECK: [[MUL:%.+]] = mulf [[ALPHA]], [[SUB]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[MUL]], [[LOAD]] : f32
  // CHECK: krnl.store [[SELECT]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_leakyrelu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[ALPHA:%.+]] = constant {{1.+}} : f32
  // CHECK: [[CMP:%.+]] = cmpf olt, [[LOAD]], [[ZERO]] : f32
  // CHECK: [[MUL:%.+]] = mulf [[ALPHA]], [[LOAD]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[MUL]], [[LOAD]] : f32
  // CHECK: krnl.store [[SELECT]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_selu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[ALPHA:%.+]] = constant {{1.+}} : f32
  // CHECK: [[GAMMA:%.+]] = constant {{2.+}} : f32
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[CMP:%.+]] = cmpf ogt, [[LOAD]], [[ZERO]] : f32
  // CHECK: [[MUL:%.+]] = mulf [[ALPHA]], [[EXP]] : f32
  // CHECK: [[SUB:%.+]] = subf [[MUL]], [[ALPHA]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[LOAD]], [[SUB]] : f32
  // CHECK: [[SELU_RES:%.+]] = mulf [[GAMMA]], [[SELECT]] : f32
  // CHECK: krnl.store [[SELU_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_hardsigmoid(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[ALPHA:%.+]] = constant {{1.+}} : f32
  // CHECK: [[BETA:%.+]] = constant {{2.+}} : f32
  // CHECK: [[MUL:%.+]] = mulf [[ALPHA]], [[LOAD]] : f32
  // CHECK: [[ADD:%.+]] = addf [[MUL]], [[BETA]] : f32
  // CHECK: [[CMP1:%.+]] = cmpf ogt, [[ADD]], [[ZERO]] : f32
  // CHECK: [[SELECT1:%.+]] = select [[CMP1]], [[ADD]], [[ZERO]] : f32
  // CHECK: [[CMP2:%.+]] = cmpf olt, [[SELECT1]], [[ONE]] : f32
  // CHECK: [[SELECT2:%.+]] = select [[CMP2]], [[SELECT1]], [[ONE]] : f32
  // CHECK: krnl.store [[SELECT2]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_reciprocal(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[RECIPROCAL_RES:%.+]] = divf [[ONE]], [[LOAD]] : f32
  // CHECK: krnl.store [[RECIPROCAL_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_softplus(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[EXP:%.+]] = exp [[LOAD]] : f32
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[ADD:%.+]] = addf [[EXP]], [[ONE]] : f32
  // CHECK: [[SOFTPLUS_RES:%.+]] = log [[ADD]] : f32
  // CHECK: krnl.store [[SOFTPLUS_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_softsign(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ABS:%.+]] = absf [[LOAD]] : f32
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[ADD:%.+]] = addf [[ABS]], [[ONE]] : f32
  // CHECK: [[SOFTSIGN_RES:%.+]] = divf [[LOAD]], [[ADD]] : f32
  // CHECK: krnl.store [[SOFTSIGN_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_reducemax(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMax"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducemax
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 0xFF800000 : f32
  // CHECK: krnl.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[CMP:%.+]] = cmpf ogt, [[LOAD2]], [[LOAD1]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[LOAD2]], [[LOAD1]] : f32
  // CHECK: krnl.store [[SELECT]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func private @test_reducemin(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMin"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducemin
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 0x7F800000 : f32
  // CHECK: krnl.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[CMP:%.+]] = cmpf olt, [[LOAD2]], [[LOAD1]] : f32
  // CHECK: [[SELECT:%.+]] = select [[CMP]], [[LOAD2]], [[LOAD1]] : f32
  // CHECK: krnl.store [[SELECT]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func private @test_reduceprod(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceProd"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduceprod
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 1.000000e+00 : f32
  // CHECK: krnl.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[REDUCE:%.+]] = mulf [[LOAD2]], [[LOAD1]] : f32
  // CHECK: krnl.store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func private @test_reducesum(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceSum"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducesum
  // CHECK: [[RES:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2) {
  // CHECK: [[IDENTITY:%.+]] = constant 0.000000e+00 : f32
  // CHECK: krnl.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[REDUCE:%.+]] = addf [[LOAD2]], [[LOAD1]] : f32
  // CHECK: krnl.store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func private @test_softmax(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis=1: si64} : (tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_softmax
  // CHECK: [[MAX:%.+]] = alloc() : memref<f32>
  // CHECK: [[SUM:%.+]] = alloc() : memref<f32>
  // CHECK: [[RES:%.+]] = alloc() : memref<10x10xf32>
  // CHECK: [[CST:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[CST_0:%.+]] = constant 0xFF800000 : f32
  // CHECK: [[OUTER_LOOP:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[OUTER_LOOP]]) with ([[OUTER_LOOP]] -> %arg1 = 0 to 10) {
  // CHECK: krnl.store [[CST]], [[SUM]][] : memref<f32>
  // CHECK: krnl.store [[CST_0]], [[MAX]][] : memref<f32>
  // CHECK: [[INNER_MAX_LOOP:%.+]] = krnl.define_loops 1
  // CHECK: [[INNER_SUM_LOOP:%.+]] = krnl.define_loops 1
  // CHECK: [[INNER_SOFTMAX_LOOP:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[INNER_MAX_LOOP]]) with ([[INNER_MAX_LOOP]] -> %arg2 = 0 to 10) {
  // CHECK:   [[LOAD1:%.+]] = krnl.load [[MAX]][] : memref<f32>
  // CHECK:   [[LOAD2:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
  // CHECK:   [[COND:%.+]] = cmpf ogt, [[LOAD1]], [[LOAD2]] : f32
  // CHECK:   [[SELECT:%.+]] = select [[COND]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK:   krnl.store [[SELECT]], [[MAX]][] : memref<f32>
  // CHECK: }
  // CHECK: [[LOAD_MAX:%.+]] = krnl.load [[MAX]][] : memref<f32>
  // CHECK: krnl.iterate([[INNER_SUM_LOOP]]) with ([[INNER_SUM_LOOP]] -> %arg2 = 0 to 10) {
  // CHECK:   [[LOAD1]] = krnl.load [[SUM]][] : memref<f32>
  // CHECK:   [[LOAD2]] = krnl.load %arg0[%arg1, %arg2] : memref<10x10xf32>
  // CHECK:   [[SUB:%.+]] = subf [[LOAD2]], [[LOAD_MAX]] : f32
  // CHECK:   [[EXP:%.+]] = exp [[SUB]] : f32
  // CHECK:   [[ADD:%.+]] = addf [[LOAD1]], [[EXP]] : f32
  // CHECK:   krnl.store [[ADD]], [[SUM]][] : memref<f32>
  // CHECK:   krnl.store [[EXP]], [[RES]][%arg1, %arg2] : memref<10x10xf32>
  // CHECK: }
  // CHECK: [[LOAD_SUM:%.+]] = krnl.load [[SUM]][] : memref<f32>

  // CHECK: krnl.iterate([[INNER_SOFTMAX_LOOP]]) with ([[INNER_SOFTMAX_LOOP]] -> %arg2 = 0 to 10) {
  // CHECK:   [[LOAD1]] = krnl.load [[RES]][%arg1, %arg2] : memref<10x10xf32>
  // CHECK:   [[DIV:%.+]] = divf [[LOAD1]], [[LOAD_SUM]] : f32
  // CHECK:   krnl.store [[DIV]], [[RES]][%arg1, %arg2] : memref<10x10xf32>
  // CHECK: }
  // CHECK: }
  // CHECK: dealloc [[SUM]] : memref<f32>
  // CHECK: dealloc [[MAX]] : memref<f32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func private @test_sqrt(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[SQRT:%.+]] = sqrt [[LOAD]] : f32
  // CHECK: krnl.store [[SQRT]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_unsqueeze(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
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

func private @test_transpose(%arg0 : tensor<10x20x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<10x20x30x40xf32>) -> tensor<*xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_transpose
  // CHECK: [[RES0:%.+]] = alloc() : memref<40x10x30x20xf32>
  // CHECK: [[RES1:%.+]] = alloc() : memref<40x30x20x10xf32>

  // CHECK: [[DEF_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg2 = 0 to 20, [[DEF_LOOPS]]#2 -> %arg3 = 0 to 30, [[DEF_LOOPS]]#3 -> %arg4 = 0 to 40) {
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3, %arg4] : memref<10x20x30x40xf32>
  // CHECK: krnl.store [[LOAD]], [[RES1]][%arg4, %arg3, %arg2, %arg1] : memref<40x30x20x10xf32>

  // CHECK: [[DEF_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to 40, [[DEF_LOOPS]]#1 -> %arg2 = 0 to 30, [[DEF_LOOPS]]#2 -> %arg3 = 0 to 20, [[DEF_LOOPS]]#3 -> %arg4 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = krnl.load [[RES1]][%arg1, %arg2, %arg3, %arg4] : memref<40x30x20x10xf32>
  // CHECK: krnl.store [[LOAD]], [[RES0]][%arg1, %arg4, %arg2, %arg3] : memref<40x10x30x20xf32>

  // CHECK: dealloc [[RES1]] : memref<40x30x20x10xf32>
  // CHECK: return [[RES0]] : memref<40x10x30x20xf32>
}

// -----

// COM: Test whether the lowering is correct in the presence of dynamic dimensions.
func private @test_transpose_dynamic_dims(%arg0 : tensor<10x?x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<10x?x30x40xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL:  func private @test_transpose_dynamic_dims
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x?x30x40xf32>) -> memref<10x40x?x30xf32> {
  // CHECK:           [[CST_1_:%.+]] = constant 1 : index
  // CHECK:           [[DIM_0_:%.+]] = dim [[PARAM_0_]], [[CST_1_]] : memref<10x?x30x40xf32>
  // CHECK-DAG:       [[RES_:%.+]] = alloc([[DIM_0_]]) : memref<10x40x?x30xf32>
  // CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
  // CHECK-DAG:       [[CST_1_1_:%.+]] = constant 1 : index
  // CHECK:           [[DIM_1_:%.+]] = dim [[PARAM_0_]], [[CST_1_1_]] : memref<10x?x30x40xf32>
  // CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[DIM_1_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 30, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 40) {
  // CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]], [[I_3_]]{{.}} : memref<10x?x30x40xf32>
  // CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_3_]], [[I_1_]], [[I_2_]]{{.}} : memref<10x40x?x30xf32>
  // CHECK:           }
  // CHECK:           return [[RES_]] : memref<10x40x?x30xf32>
  // CHECK:         }
}

// -----

func private @test_identity(%arg0 : tensor<10x20x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Identity"(%arg0) : (tensor<10x20x30x40xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_identity
  // CHECK: return %arg0 : memref<10x20x30x40xf32>
}

// -----

func private @test_sign_f(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = constant {{0.+}} : f32
  // CHECK: [[ONE:%.+]] = constant {{1.+}} : f32
  // CHECK: [[MINUS_ONE:%.+]] = constant {{-1.+}} : f32
  // CHECK: [[GTZERO:%.+]] = cmpf ogt, [[LOAD]], [[ZERO]] : f32
  // CHECK: [[SELECT_PLUS:%.+]] = select [[GTZERO]], [[ONE]], [[MINUS_ONE]] : f32
  // CHECK: [[EQZERO:%.+]] = cmpf oeq, [[LOAD]], [[ZERO]] : f32
  // CHECK: [[SIGN_RES:%.+]] = select [[EQZERO]], [[ZERO]], [[SELECT_PLUS]] : f32
  // CHECK: krnl.store [[SIGN_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_sign_i(%arg0 : tensor<?x10xi32>) -> tensor<*xi32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xi32>
  // CHECK: [[ZERO:%.+]] = constant 0 : i32
  // CHECK: [[ONE:%.+]] = constant 1 : i32
  // CHECK: [[MINUS_ONE:%.+]] = constant -1 : i32
  // CHECK: [[GTZERO:%.+]] = cmpi sgt, [[LOAD]], [[ZERO]] : i32
  // CHECK: [[SELECT_PLUS:%.+]] = select [[GTZERO]], [[ONE]], [[MINUS_ONE]] : i32
  // CHECK: [[EQZERO:%.+]] = cmpi eq, [[LOAD]], [[ZERO]] : i32
  // CHECK: [[SIGN_RES:%.+]] = select [[EQZERO]], [[ZERO]], [[SELECT_PLUS]] : i32
  // CHECK: krnl.store [[SIGN_RES]], [[RES]][%arg1, %arg2] : memref<?x10xi32>
  // CHECK: return [[RES]] : memref<?x10xi32>
}

// -----

// 2-D x 2-D
func private @test_matmul1(%arg0 : tensor<10x5xf32>, %arg1 : tensor<5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<10x5xf32>, tensor<5x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

//CHECK-LABEL:  func private @test_matmul1
//CHECK-SAME:   ([[A_:%.+]]: memref<10x5xf32>, [[B_:%.+]]: memref<5x10xf32>) -> memref<10x10xf32> {
//CHECK:           [[RES_:%.+]] = alloc() : memref<10x10xf32>
//CHECK:           [[VAR_cst_:%.+]] = constant 0.000000e+00 : f32
//CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
//CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10) {
//CHECK:             [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
//CHECK:             krnl.store [[VAR_cst_]], [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
//CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 5) {
//CHECK:               [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[I_0_]], [[I_2_]]{{.}} : memref<10x5xf32>
//CHECK:               [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[I_2_]], [[I_1_]]{{.}} : memref<5x10xf32>
//CHECK:               [[LOAD_RES_MEM_:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
//CHECK:               [[VAR_6_:%.+]] = mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
//CHECK:               [[VAR_7_:%.+]] = addf [[LOAD_RES_MEM_]], [[VAR_6_]] : f32
//CHECK:               krnl.store [[VAR_7_]], [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             }
//CHECK:             [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             krnl.store [[LOAD_REDUCTION]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
//CHECK:           }
//CHECK:           return [[RES_]] : memref<10x10xf32>
//CHECK:         }
}

// -----

// 2-D x N-D
func private @test_matmul2(%arg0 : tensor<10x5xf32>, %arg1 : tensor<2x3x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<10x5xf32>, tensor<2x3x5x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

//CHECK-LABEL:  func private @test_matmul2
//CHECK-SAME:   ([[A_:%.+]]: memref<10x5xf32>, [[B_:%.+]]: memref<2x3x5x10xf32>) -> memref<2x3x10x10xf32> {
//CHECK:           [[RES_:%.+]] = alloc() : memref<2x3x10x10xf32>
//CHECK:           [[VAR_cst_:%.+]] = constant 0.000000e+00 : f32
//CHECK:           [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
//CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 10, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 10) {
//CHECK:             [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
//CHECK:             krnl.store [[VAR_cst_]], [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
//CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_4_:%.+]] = 0 to 5) {
//CHECK:               [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[I_2_]], [[I_4_]]{{.}} : memref<10x5xf32>
//CHECK:               [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[I_0_]], [[I_1_]], [[I_4_]], [[I_3_]]{{.}} : memref<2x3x5x10xf32>
//CHECK:               [[LOAD_RES_MEM_:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
//CHECK:               [[VAR_6_:%.+]] = mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
//CHECK:               [[VAR_7_:%.+]] = addf [[LOAD_RES_MEM_]], [[VAR_6_]] : f32
//CHECK:               krnl.store [[VAR_7_]], [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             }
//CHECK:             [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             krnl.store [[LOAD_REDUCTION]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]], [[I_3_]]{{.}} : memref<2x3x10x10xf32>
//CHECK:           }
//CHECK:           return [[RES_]] : memref<2x3x10x10xf32>
//CHECK:         }
}

// -----

// N-D x N-D
func private @test_matmul3(%arg0 : tensor<2x3x10x5xf32>, %arg1 : tensor<2x3x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<2x3x10x5xf32>, tensor<2x3x5x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

//CHECK-LABEL:  func private @test_matmul3
//CHECK-SAME:   ([[A_:%.+]]: memref<2x3x10x5xf32>, [[B_:%.+]]: memref<2x3x5x10xf32>) -> memref<2x3x10x10xf32> {
//CHECK:           [[RES_:%.+]] = alloc() : memref<2x3x10x10xf32>
//CHECK:           [[VAR_cst_:%.+]] = constant 0.000000e+00 : f32
//CHECK:           [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
//CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 10, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 10) {
//CHECK:             [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
//CHECK:             krnl.store [[VAR_cst_]], [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
//CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_4_:%.+]] = 0 to 5) {
//CHECK:               [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]], [[I_4_]]{{.}} : memref<2x3x10x5xf32>
//CHECK:               [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[I_0_]], [[I_1_]], [[I_4_]], [[I_3_]]{{.}} : memref<2x3x5x10xf32>
//CHECK:               [[LOAD_RES_MEM_:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
//CHECK:               [[VAR_6_:%.+]] = mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
//CHECK:               [[VAR_7_:%.+]] = addf [[LOAD_RES_MEM_]], [[VAR_6_]] : f32
//CHECK:               krnl.store [[VAR_7_]], [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             }
//CHECK:             [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             krnl.store [[LOAD_REDUCTION]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]], [[I_3_]]{{.}} : memref<2x3x10x10xf32>
//CHECK:           }
//CHECK:           return [[RES_]] : memref<2x3x10x10xf32>
//CHECK:         }
}

// -----

// 1-D x 2-D
func private @test_matmul4(%arg0 : tensor<5xf32>, %arg1 : tensor<5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<5x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

//CHECK-LABEL:  func private @test_matmul4
//CHECK-SAME:   ([[A_:%.+]]: memref<5xf32>, [[B_:%.+]]: memref<5x10xf32>) -> memref<10xf32> {
//CHECK:           [[RES_:%.+]] = alloc() : memref<10xf32>
//CHECK:           [[VAR_cst_:%.+]] = constant 0.000000e+00 : f32
//CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
//CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 10) {
//CHECK:             [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
//CHECK:             krnl.store [[VAR_cst_]], [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
//CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 5) {
//CHECK:               [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[I_1_]]{{.}} : memref<5xf32>
//CHECK:               [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[I_1_]], [[I_0_]]{{.}} : memref<5x10xf32>
//CHECK:               [[LOAD_RES_MEM_:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
//CHECK:               [[VAR_6_:%.+]] = mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
//CHECK:               [[VAR_7_:%.+]] = addf [[LOAD_RES_MEM_]], [[VAR_6_]] : f32
//CHECK:               krnl.store [[VAR_7_]], [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             }
//CHECK:             [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             krnl.store [[LOAD_REDUCTION]], [[RES_]]{{.}}[[I_0_]]{{.}} : memref<10xf32>
//CHECK:           }
//CHECK:           return [[RES_]] : memref<10xf32>
//CHECK:         }
}

// -----

// 1-D x N-D
func private @test_matmul5(%arg0 : tensor<5xf32>, %arg1 : tensor<?x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<?x5x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

//CHECK-LABEL:  func private @test_matmul5
//CHECK-SAME:   ([[A_:%.+]]: memref<5xf32>, [[B_:%.+]]: memref<?x5x10xf32>) -> memref<?x10xf32> {
//CHECK:           [[VAR_c0_:%.+]] = constant 0 : index
//CHECK:           [[VAR_0_:%.+]] = dim [[B_]], [[VAR_c0_]] : memref<?x5x10xf32>
//CHECK:           [[RES_:%.+]] = alloc([[VAR_0_]]) : memref<?x10xf32>
//CHECK:           [[VAR_cst_:%.+]] = constant 0.000000e+00 : f32
//CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
//CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_0_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10) {
//CHECK:             [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
//CHECK:             krnl.store [[VAR_cst_]], [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
//CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 5) {
//CHECK:               [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[I_2_]]{{.}} : memref<5xf32>
//CHECK:               [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[I_0_]], [[I_2_]], [[I_1_]]{{.}} : memref<?x5x10xf32>
//CHECK:               [[LOAD_RES_MEM_:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
//CHECK:               [[VAR_7_:%.+]] = mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
//CHECK:               [[VAR_8_:%.+]] = addf [[LOAD_RES_MEM_]], [[VAR_7_]] : f32
//CHECK:               krnl.store [[VAR_8_]], [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             }
//CHECK:             [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             krnl.store [[LOAD_REDUCTION]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x10xf32>
//CHECK:           }
//CHECK:           return [[RES_]] : memref<?x10xf32>
//CHECK:         }
}

// -----

// N-D x 1-D
func private @test_matmul6(%arg0 : tensor<?x10x5xf32>, %arg1 : tensor<5xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<?x10x5xf32>, tensor<5xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

//CHECK-LABEL:  func private @test_matmul6
//CHECK-SAME:   ([[A_:%.+]]: memref<?x10x5xf32>, [[B_:%.+]]: memref<5xf32>) -> memref<?x10xf32> {
//CHECK:           [[VAR_c0_:%.+]] = constant 0 : index
//CHECK:           [[VAR_0_:%.+]] = dim [[A_]], [[VAR_c0_]] : memref<?x10x5xf32>
//CHECK:           [[RES_:%.+]] = alloc([[VAR_0_]]) : memref<?x10xf32>
//CHECK:           [[VAR_cst_:%.+]] = constant 0.000000e+00 : f32
//CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
//CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_0_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10) {
//CHECK:             [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
//CHECK:             krnl.store [[VAR_cst_]], [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
//CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 5) {
//CHECK:               [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<?x10x5xf32>
//CHECK:               [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[I_2_]]{{.}} : memref<5xf32>
//CHECK:               [[LOAD_RES_MEM_:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
//CHECK:               [[VAR_7_:%.+]] = mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
//CHECK:               [[VAR_8_:%.+]] = addf [[LOAD_RES_MEM_]], [[VAR_7_]] : f32
//CHECK:               krnl.store [[VAR_8_]], [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             }
//CHECK:             [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             krnl.store [[LOAD_REDUCTION]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x10xf32>
//CHECK:           }
//CHECK:           return [[RES_]] : memref<?x10xf32>
//CHECK:         }
}

// -----

// 1-D x 1-D
func private @test_matmul7(%arg0 : tensor<5xf32>, %arg1 : tensor<5xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<5xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

//CHECK-LABEL:  func private @test_matmul7
//CHECK-SAME:   ([[A_:%.+]]: memref<5xf32>, [[B_:%.+]]: memref<5xf32>) -> memref<1xf32> {
//CHECK:           [[RES_:%.+]] = alloc() : memref<1xf32>
//CHECK:           [[VAR_cst_:%.+]] = constant 0.000000e+00 : f32
//CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
//CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 1) {
//CHECK:             [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
//CHECK:             krnl.store [[VAR_cst_]], [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
//CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 5) {
//CHECK:               [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[I_1_]]{{.}} : memref<5xf32>
//CHECK:               [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[I_1_]]{{.}} : memref<5xf32>
//CHECK:               [[LOAD_RES_MEM_:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
//CHECK:               [[VAR_6_:%.+]] = mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
//CHECK:               [[VAR_7_:%.+]] = addf [[LOAD_RES_MEM_]], [[VAR_6_]] : f32
//CHECK:               krnl.store [[VAR_7_]], [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             }
//CHECK:             [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
//CHECK:             krnl.store [[LOAD_REDUCTION]], [[RES_]]{{.}}[[I_0_]]{{.}} : memref<1xf32>
//CHECK:           }
//CHECK:           return [[RES_]] : memref<1xf32>
//CHECK:         }
}

// -----

func private @test_conv_no_bias_no_pad(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-DAG: #[[ZERO_MAP2:.+]] = affine_map<(d0) -> (0, d0)>
  // CHECK-DAG: #{{.*}} = affine_map<(d0) -> (32, d0 + 6)>
  // CHECK-DAG: #[[ZERO_MAP4:.+]] = affine_map<(d0, d1) -> (0, d1)>
  // CHECK-DAG: #{{.*}} = affine_map<(d0, d1) -> (64, d1 + 7)>
  // CHECK-DAG: #[[BOUND:.+]] = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>

  // CHECK-LABEL: test_conv_no_bias_no_pad
  // CHECK: [[RES:%.+]] = alloc() : memref<1x5x27x58xf32>
  // CHECK: [[CONST1:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[OUTER_LOOPS:%.+]]:2 = krnl.define_loops 2

  // CHECK: krnl.iterate([[OUTER_LOOPS]]#0, [[OUTER_LOOPS]]#1) with ([[OUTER_LOOPS]]#0 -> %arg2 = 0 to 1, [[OUTER_LOOPS]]#1 -> %arg3 = 0 to 5) {
  // CHECK: [[SPATIAL_LOOPS:%.+]]:2 = krnl.define_loops 2

  // CHECK: krnl.iterate([[SPATIAL_LOOPS]]#0, [[SPATIAL_LOOPS]]#1) with ([[SPATIAL_LOOPS]]#0 -> %arg4 = 0 to 27, [[SPATIAL_LOOPS]]#1 -> %arg5 = 0 to 58) {
  // CHECK: [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
  // CHECK: krnl.store [[CONST1]], [[REDUCTION_VAL]][] : memref<f32>
  // CHECK: [[START1:%.+]] = affine.max #[[ZERO_MAP2]](%arg4)
  // CHECK: {{.*}} = affine.min {{.*}}
  // CHECK: [[KERNEL_OFFSET1:%.+]] = affine.min #[[ZERO_MAP2]](%arg4)
  // CHECK: [[START2:%.+]] = affine.max #[[ZERO_MAP4]](%arg4, %arg5)
  // CHECK: {{.*}} = affine.min {{.*}}
  // CHECK: [[KERNEL_OFFSET2:%.+]] = affine.min #[[ZERO_MAP4]](%arg4, %arg5)

  // CHECK: [[INNER_LOOPS:%.+]]:3 = krnl.define_loops 3

  // CHECK: krnl.iterate([[INNER_LOOPS]]#0, [[INNER_LOOPS]]#1, [[INNER_LOOPS]]#2) with ([[INNER_LOOPS]]#0 -> %arg6 = 0 to 2, [[INNER_LOOPS]]#1 -> %arg7 = 0 to min #[[BOUND]](%arg4)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}], [[INNER_LOOPS]]#2 -> %arg8 = 0 to min #[[BOUND]](%arg5)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]) {
  // CHECK: [[R1:%.+]] = addi %arg7, [[START1]] : index
  // CHECK: [[R2:%.+]] = addi %arg8, [[START2]] : index
  // CHECK: [[K1:%.+]] = subi %arg7, [[KERNEL_OFFSET1]] : index
  // CHECK: [[K2:%.+]] = subi %arg8, [[KERNEL_OFFSET2]] : index
  // CHECK: [[DATA:%.+]] = krnl.load %arg0[%arg2, %arg6, [[R1]], [[R2]]{{\]}} : memref<1x2x32x64xf32>
  // CHECK: [[KERNEL:%.+]] = krnl.load %arg1[%arg3, %arg6, [[K1]], [[K2]]{{\]}} : memref<5x2x6x7xf32>
  // CHECK: [[ACC_RES:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK: [[MUL:%.+]] = mulf [[DATA]], [[KERNEL]] : f32
  // CHECK: [[ADD:%.+]] = addf [[ACC_RES]], [[MUL]] : f32
  // CHECK: krnl.store [[ADD]], [[REDUCTION_VAL]][] : memref<f32>
  // CHECK: }
  // CHECK: [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK: krnl.store [[LOAD_REDUCTION]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<1x5x27x58xf32>
  // CHECK: }
  // CHECK: }

  // CHECK: return [[RES]] : memref<1x5x27x58xf32>
}

// -----

func private @test_conv_bias_no_pad(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, tensor<5xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_conv_bias_no_pad
  // CHECK: [[RES:%.+]] = alloc() : memref<1x5x27x58xf32>
  // CHECK: [[CONST1:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[OUTER_LOOPS:%.+]]:2 = krnl.define_loops 2

  // CHECK: krnl.iterate([[OUTER_LOOPS]]#0, [[OUTER_LOOPS]]#1) with ([[OUTER_LOOPS]]#0 -> %arg3 = 0 to 1, [[OUTER_LOOPS]]#1 -> %arg4 = 0 to 5) {
  // CHECK: [[SPATIAL_LOOPS:%.+]]:2 = krnl.define_loops 2

  // CHECK: krnl.iterate([[SPATIAL_LOOPS]]#0, [[SPATIAL_LOOPS]]#1) with ([[SPATIAL_LOOPS]]#0 -> %arg5 = 0 to 27, [[SPATIAL_LOOPS]]#1 -> %arg6 = 0 to 58) {
  // CHECK: [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
  // CHECK: krnl.store [[CONST1]], [[REDUCTION_VAL]][] : memref<f32>
  // CHECK: [[INNER_LOOPS:%.+]]:3 = krnl.define_loops 3

  // CHECK: krnl.iterate([[INNER_LOOPS]]#0, [[INNER_LOOPS]]#1, [[INNER_LOOPS]]#2) with ([[INNER_LOOPS]]#0 -> %arg7 = 0 to 2, [[INNER_LOOPS]]#1 -> %arg8 = 0 to min #{{.*}}(%arg5)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}], [[INNER_LOOPS]]#2 -> %arg9 = 0 to min #{{.*}}(%arg6)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]) {
  // CHECK: }
  // CHECK: [[BIAS1:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK: [[BIAS2:%.+]] = krnl.load %arg2[%arg4] : memref<5xf32>
  // CHECK: [[BIAS3:%.+]] = addf [[BIAS1]], [[BIAS2]] : f32
  // CHECK: krnl.store [[BIAS3]], [[RES]][%arg3, %arg4, %arg5, %arg6] : memref<1x5x27x58xf32>
  // CHECK: }
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x5x27x58xf32>
}

// -----

func private @test_conv_no_bias_no_pad_w_group(%arg0 : tensor<1x9x32x64xf32>, %arg1 : tensor<5x3x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 3 : si64} : (tensor<1x9x32x64xf32>, tensor<5x3x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()


  // CHECK-LABEL: test_conv_no_bias_no_pad_w_group
  // CHECK: [[RES:%.+]] = alloc() : memref<1x5x27x58xf32>
  // CHECK: [[OUTER_LOOPS:%.+]]:3 = krnl.define_loops 3

  // CHECK: krnl.iterate([[OUTER_LOOPS]]#0, [[OUTER_LOOPS]]#1, [[OUTER_LOOPS]]#2) with ([[OUTER_LOOPS]]#0 -> %arg2 = 0 to 1, [[OUTER_LOOPS]]#1 -> %arg3 = 0 to 3, [[OUTER_LOOPS]]#2 -> %arg4 = 0 to 1) {
  // CHECK: [[SPATIAL_LOOPS:%.+]]:2 = krnl.define_loops 2

  // CHECK: krnl.iterate([[SPATIAL_LOOPS]]#0, [[SPATIAL_LOOPS]]#1) with ([[SPATIAL_LOOPS]]#0 -> %arg5 = 0 to 27, [[SPATIAL_LOOPS]]#1 -> %arg6 = 0 to 58) {
  // CHECK: krnl.store {{.*}}
  // CHECK: [[INNER_LOOPS:%.+]]:3 = krnl.define_loops 3

  // CHECK: krnl.iterate([[INNER_LOOPS]]#0, [[INNER_LOOPS]]#1, [[INNER_LOOPS]]#2) with ([[INNER_LOOPS]]#0 -> %arg7 = 0 to 3, [[INNER_LOOPS]]#1 -> %arg8 = 0 to min #{{.*}}(%arg5)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}], [[INNER_LOOPS]]#2 -> %arg9 = 0 to min #{{.*}}(%arg6)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]) {
  // CHECK: [[GROUP:%.+]] = affine.apply #{{.*}}(%arg3, %arg4, %arg5, %arg6, %arg3, %arg7)
  // CHECK: [[DATA:%.+]] = krnl.load %arg0[%arg2, [[GROUP]], {{.*}}, {{.*}}] : memref<1x9x32x64xf32>
  // CHECK: }
  // CHECK: }
  // CHECK: }

  // CHECK: return [[RES]] : memref<1x5x27x58xf32>
}

// -----

func private @test_conv_no_bias_no_pad_w_strides(%arg0 : tensor<1x9x32x64xf32>, %arg1 : tensor<5x9x6x7xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64, strides = [2, 2]} : (tensor<1x9x32x64xf32>, tensor<5x9x6x7xf32>, none) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-DAG: #[[BOUND:.+]] = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>

  // CHECK-LABEL: test_conv_no_bias_no_pad_w_strides
  // CHECK: [[RES:%.+]] = alloc() : memref<1x5x14x29xf32>
  // CHECK: [[CONST1:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[OUTER_LOOPS:%.+]]:2 = krnl.define_loops 2

  // CHECK: krnl.iterate([[OUTER_LOOPS]]#0, [[OUTER_LOOPS]]#1) with ([[OUTER_LOOPS]]#0 -> %arg2 = 0 to 1, [[OUTER_LOOPS]]#1 -> %arg3 = 0 to 5) {
  // CHECK: [[SPATIAL_LOOPS:%.+]]:2 = krnl.define_loops 2

  // CHECK: krnl.iterate([[SPATIAL_LOOPS]]#0, [[SPATIAL_LOOPS]]#1) with ([[SPATIAL_LOOPS]]#0 -> %arg4 = 0 to 14, [[SPATIAL_LOOPS]]#1 -> %arg5 = 0 to 29) {
  // CHECK: [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
  // CHECK: krnl.store [[CONST1]], [[REDUCTION_VAL]][] : memref<f32>
  // CHECK: [[INNER_LOOPS:%.+]]:3 = krnl.define_loops 3

  // CHECK: krnl.iterate([[INNER_LOOPS]]#0, [[INNER_LOOPS]]#1, [[INNER_LOOPS]]#2) with ([[INNER_LOOPS]]#0 -> %arg6 = 0 to 9, [[INNER_LOOPS]]#1 -> %arg7 = 0 to min #[[BOUND]](%arg4)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}], [[INNER_LOOPS]]#2 -> %arg8 = 0 to min #[[BOUND]](%arg5)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]) {
  // CHECK: [[R1:%.+]] = addi %arg7, [[START1]] : index
  // CHECK: [[R2:%.+]] = addi %arg8, [[START2]] : index
  // CHECK: [[K1:%.+]] = subi %arg7, [[KERNEL_OFFSET1]] : index
  // CHECK: [[K2:%.+]] = subi %arg8, [[KERNEL_OFFSET2]] : index
  // CHECK: [[DATA:%.+]] = krnl.load %arg0[%arg2, %arg6, [[R1]], [[R2]]{{\]}} : memref<1x9x32x64xf32>
  // CHECK: [[KERNEL:%.+]] = krnl.load %arg1[%arg3, %arg6, [[K1]], [[K2]]{{\]}} : memref<5x9x6x7xf32>
  // CHECK: [[ACC_RES:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK: [[MUL:%.+]] = mulf [[DATA]], [[KERNEL]] : f32
  // CHECK: [[ADD:%.+]] = addf [[ACC_RES]], [[MUL]] : f32
  // CHECK: krnl.store [[ADD]], [[REDUCTION_VAL]][] : memref<f32>
  // CHECK: }
  // CHECK: [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK: krnl.store [[LOAD_REDUCTION]], [[RES]][%arg2, %arg3, %arg4, %arg5] : memref<1x5x14x29xf32>
  // CHECK: }
  // CHECK: }

  // CHECK: return [[RES]] : memref<1x5x14x29xf32>
}


// -----

func private @test_batchnorm_testmode_Nd(%arg0: tensor<1x2x1x3xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> tensor<1x2x1x3xf32> {
  %0 = "onnx.BatchNormalizationTestMode"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x2x1x3xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x1x3xf32>
  return %0 : tensor<1x2x1x3xf32>

  // CHECK-LABEL: test_batchnorm_testmode_Nd
  // CHECK-DAG: [[RES:%.+]] = alloc() : memref<1x2x1x3xf32>
  // CHECK-DAG: [[EPSILON:%.+]] = constant 9.99999974E-6 : f32
  // CHECK: [[DEF_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg5 = 0 to 2) {
  // CHECK:   [[SCALE:%.+]] = krnl.load %arg1[%arg5] : memref<2xf32>
  // CHECK:   [[BIAS:%.+]] = krnl.load %arg2[%arg5] : memref<2xf32>
  // CHECK:   [[MEAN:%.+]] = krnl.load %arg3[%arg5] : memref<2xf32>
  // CHECK:   [[VARIANCE:%.+]] = krnl.load %arg4[%arg5] : memref<2xf32>
  // CHECK:   krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) with ([[DEF_LOOPS]]#0 -> %arg6 = 0 to 1, [[DEF_LOOPS]]#2 -> %arg7 = 0 to 1, [[DEF_LOOPS]]#3 -> %arg8 = 0 to 3) {
  // CHECK:     [[LOADED_VAL:%.+]] = krnl.load %arg0[%arg6, %arg5, %arg7, %arg8] : memref<1x2x1x3xf32>
  // CHECK:     [[DIVIDEND:%.+]] = subf [[LOADED_VAL]], [[MEAN]] : f32
  // CHECK:     [[ADJUSTED_VARIANCE:%.+]] = addf [[VARIANCE]], [[EPSILON]] : f32
  // CHECK:     [[DIVISOR:%.+]] = sqrt [[ADJUSTED_VARIANCE]] : f32
  // CHECK:     [[NORM:%.+]] = divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK:     [[SCALE_NORM:%.+]] = mulf [[SCALE]], [[NORM]] : f32
  // CHECK:     [[SHIFT_SCALE_NORM:%.+]] = addf [[SCALE_NORM]], [[BIAS]] : f32
  // CHECK:     krnl.store [[SHIFT_SCALE_NORM]], [[RES]][%arg6, %arg5, %arg7, %arg8] : memref<1x2x1x3xf32>
  // CHECK:   }
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x2x1x3xf32>
}

// -----

func private @test_batchnorm_testmode_1d(%arg0: tensor<10xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<10xf32> {
  %0 = "onnx.BatchNormalizationTestMode"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<10xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>

  // CHECK-LABEL: test_batchnorm_testmode_1d
  // CHECK: [[RES:%.+]] = alloc() : memref<10xf32>
  // CHECK: [[EPSILON:%.+]] = constant 9.99999974E-6 : f32
  // CHECK: [[DEF_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: %[[ZERO_INDEX:.+]] = constant 0 : index
  // CHECK: [[SCALE:%.+]] = krnl.load %arg1[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: [[BIAS:%.+]] = krnl.load %arg2[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: [[MEAN:%.+]] = krnl.load %arg3[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: [[VARIANCE:%.+]] = krnl.load %arg4[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]) with ([[DEF_LOOPS]] -> %arg5 = 0 to 10) {
  // CHECK:   [[LOADED_VAL:%.+]] = krnl.load %arg0[%arg5] : memref<10xf32>
  // CHECK:   [[DIVIDEND:%.+]] = subf [[LOADED_VAL]], [[MEAN]] : f32
  // CHECK:   [[ADJUSTED_VARIANCE:%.+]] = addf [[VARIANCE]], [[EPSILON]] : f32
  // CHECK:   [[DIVISOR:%.+]] = sqrt [[ADJUSTED_VARIANCE]] : f32
  // CHECK:   [[NORM:%.+]] = divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK:   [[SCALE_NORM:%.+]] = mulf [[SCALE]], [[NORM]] : f32
  // CHECK:   [[SHIFT_SCALE_NORM:%.+]] = addf [[SCALE_NORM]], [[BIAS]] : f32
  // CHECK:   krnl.store [[SHIFT_SCALE_NORM]], [[RES]][%arg5] : memref<10xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<10xf32>
}

// -----

func private @test_batchnorm_testmode_2d(%arg0: tensor<10x3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>) -> tensor<10x3xf32> {
  %0 = "onnx.BatchNormalizationTestMode"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<10x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<10x3xf32>
  return %0 : tensor<10x3xf32>

  // CHECK-LABEL: test_batchnorm_testmode_2d
  // CHECK: [[RES:%.+]] = alloc() : memref<10x3xf32>
  // CHECK: [[EPSILON:%.+]] = constant 9.99999974E-6 : f32
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:   [[SCALE:%.+]] = krnl.load %arg1[%arg5] : memref<3xf32>
  // CHECK:   [[BIAS:%.+]] = krnl.load %arg2[%arg5] : memref<3xf32>
  // CHECK:   [[MEAN:%.+]] = krnl.load %arg3[%arg5] : memref<3xf32>
  // CHECK:   [[VARIANCE:%.+]] = krnl.load %arg4[%arg5] : memref<3xf32>
  // CHECK:   krnl.iterate([[DEF_LOOPS]]#0) with ([[DEF_LOOPS]]#0 -> %arg6 = 0 to 10) {
  // CHECK:     [[LOADED_VAL:%.+]] = krnl.load %arg0[%arg6, %arg5] : memref<10x3xf32>
  // CHECK:     [[DIVIDEND:%.+]] = subf [[LOADED_VAL]], [[MEAN]] : f32
  // CHECK:     [[ADJUSTED_VARIANCE:%.+]] = addf [[VARIANCE]], [[EPSILON]] : f32
  // CHECK:     [[DIVISOR:%.+]] = sqrt [[ADJUSTED_VARIANCE]] : f32
  // CHECK:     [[NORM:%.+]] = divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK:     [[SCALE_NORM:%.+]] = mulf [[SCALE]], [[NORM]] : f32
  // CHECK:     [[SHIFT_SCALE_NORM:%.+]] = addf [[SCALE_NORM]], [[BIAS]] : f32
  // CHECK:     krnl.store [[SHIFT_SCALE_NORM]], [[RES]][%arg6, %arg5] : memref<10x3xf32>
  // CHECK:   }
  // CHECK: }
  // CHECK: return [[RES]] : memref<10x3xf32>
}

// -----

func private @test_abs_float(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ABS:%.+]] = absf [[LOAD]] : f32
  // CHECK: krnl.store [[ABS]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_abs_int(%arg0 : tensor<?x10xi32>) -> tensor<*xi32> {
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
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xi32>
  // CHECK: [[ZERO:%.+]] = constant 0 : i32
  // CHECK: [[LESS_THAN_ZERO:%.+]] = cmpi slt, [[LOAD]], [[ZERO]] : i32
  // CHECK: [[NEGATIVE_LOAD:%.+]] = subi [[ZERO]], [[LOAD]] : i32
  // CHECK: [[SELECT:%.+]] = select [[LESS_THAN_ZERO]], [[NEGATIVE_LOAD]], [[LOAD]] : i32
  // CHECK: krnl.store [[SELECT]], [[RES]][%arg1, %arg2] : memref<?x10xi32>
  // CHECK: return [[RES]] : memref<?x10xi32>
}

func private @test_pad1(%arg0: tensor<16x16xf32>) -> tensor<18x20xf32> {
  %0 = "onnx.Constant"() {value = dense<[0, 3, 2, 1]> : tensor<4xi64> } : () -> tensor<4xi64>
  %1 = "onnx.Constant"() {value = dense<0.000000e+00> : tensor<1xf32> } : () -> tensor<1xf32>
  %2 = "onnx.Pad"(%arg0, %0, %1) {mode = "constant"} : (tensor<16x16xf32>, tensor<4xi64>, tensor<1xf32>) -> tensor<18x20xf32>
  return %2 : tensor<18x20xf32>
  // CHECK-LABEL: test_pad1
  // CHECK: [[RES:%.+]] = alloc() : memref<18x20xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 18, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 20) {
  // CHECK: [[CST:%.+]] = constant 0.000000e+00 : f32
  // CHECK: krnl.store [[CST]], [[RES]][%arg1, %arg2] : memref<18x20xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOPS2:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 16, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 16) {
  // CHECK: [[ADD:%.+]] = affine.apply #{{.*}}(%arg2)
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<16x16xf32>
  // CHECK: krnl.store [[LOAD]], [[RES]][%arg1, [[ADD]]] : memref<18x20xf32>
  // CHECK: }
}

// -----

func private @test_constant_dense_2d_value(%arg0: tensor<1xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[[0.0, 0.0], [1.0, 1.1], [2.0, 2.1]]> : tensor<3x2xf32>} : () -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_constant_dense_2d_value
  // CHECK: [[GLOBAL:%.+]] = "krnl.global"() {name = "constant_0", shape = [3, 2], value = dense<{{.*}}[0.000000e+00, 0.000000e+00], [1.000000e+00, 1.100000e+00], [2.000000e+00, 2.100000e+00]{{.*}}> : tensor<3x2xf32>} : () -> memref<3x2xf32>
  // CHECK: [[ALLOC:%.+]] = alloc() : memref<3x2xf32>
  // CHECK: [[CONST_4:%.+]] = constant 4 : i64
  // CHECK: [[CONST_6:%.+]] = constant 6 : i64
  // CHECK: [[SIZE:%.+]] = muli [[CONST_4]], [[CONST_6]] : i64
  // CHECK: "krnl.memcpy"([[ALLOC]], [[GLOBAL]], [[SIZE]]) : (memref<3x2xf32>, memref<3x2xf32>, i64) -> ()
  // CHECK: return [[ALLOC]] : memref<3x2xf32>
}

// -----

func private @test_pool_general_computation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-DAG: #{{.*}} = affine_map<(d0) -> (0, d0)>
  // CHECK-DAG: #{{.*}} = affine_map<(d0) -> (32, d0 + 2)>
  // CHECK-DAG: #{{.*}} = affine_map<(d0, d1) -> (0, d1)>
  // CHECK-DAG: #{{.*}} = affine_map<(d0, d1) -> (32, d1 + 2)>
  // CHECK-DAG: #[[BOUND:.+]] = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>

  // CHECK-LABEL: @test_pool_general_computation

  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = constant 0.000000e+00 : f32

  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31) {

  // CHECK:   [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
  // CHECK:   krnl.store [[IDENTITY]], [[REDUCTION_VAL]][] : memref<f32>

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #[[BOUND]](%arg3)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #[[BOUND]](%arg4)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]) {
  // CHECK:     {{.*}} = krnl.load %arg0[%arg1, %arg2, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     {{.*}} = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:     krnl.store {{.*}}, [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   }

  // CHECK:   [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   krnl.store [[LOAD_REDUCTION]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK: }
}

// -----

func private @test_pool_unknown_dimensions(%arg0 : tensor<1x3x?x32xf32>) -> tensor<*xf32> {
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

func private @test_averagepool_identity_value(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_averagepool_identity_value
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
  // CHECK: krnl.store [[IDENTITY]], [[REDUCTION_VAL]][] : memref<f32>
}

// -----

func private @test_maxpool_identity_value(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_maxpool_identity_value
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = constant 0xFF800000 : f32
  // CHECK: [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
  // CHECK: krnl.store [[IDENTITY]], [[REDUCTION_VAL]][] : memref<f32>
}

// -----

func private @test_averagepool_pooling_operation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_averagepool_pooling_operation
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>

  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31) {

  // CHECK:   [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
  // CHECK:   krnl.store {{.*}}, [[REDUCTION_VAL]][] : memref<f32>

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #{{.*}}(%arg3)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #{{.*}}(%arg4)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]) {

  // CHECK:     [[INPUT_LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     [[OUTPUT_LOAD:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:     [[SUM:%.+]] = addf [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     krnl.store [[SUM]], [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   }
  // CHECK:   [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   krnl.store [[LOAD_REDUCTION]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>

  // CHECK:   [[NUMERATOR:%.+]] = krnl.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   [[AVERAGE:%.+]] = divf [[NUMERATOR]], {{.*}} : f32
  // CHECK:   krnl.store [[AVERAGE]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK: }
}

// -----

func private @test_maxpool_pooling_operation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_maxpool_pooling_operation
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x31x31xf32>

  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31) {

  // CHECK:   [[REDUCTION_VAL:%.+]] = alloca() : memref<f32>
  // CHECK:   krnl.store {{.*}}, [[REDUCTION_VAL]][] : memref<f32>

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #{{.*}}(%arg3)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #{{.*}}(%arg4)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]) {

  // CHECK:     [[INPUT_LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     [[OUTPUT_LOAD:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:     [[GREATER:%.+]] = cmpf ogt, [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     [[SELECT:%.+]] = select [[GREATER]], [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     krnl.store [[SELECT]], [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   }
  // CHECK:   [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   krnl.store [[LOAD_REDUCTION]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>

  // CHECK-NOT:   {{.*}} = krnl.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK-NOT:   krnl.store {{.*}}, [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK: }
}

// -----

/// Check GRU with three required inputs (X, W, R). The optional inputs are default.
/// Also check the equation for 'ht' when linear_before_reset = 0 (default)
func private @test_gru_general_computation(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_general_computation
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x3xf32>

  /// Check initialize loop.
  // CHECK: [[INITIAL_VAL:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[DEF_LOOPS_INIT:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS_INIT]]#0, [[DEF_LOOPS_INIT]]#1, [[DEF_LOOPS_INIT]]#2) with ([[DEF_LOOPS_INIT]]#0 -> %arg3 = 0 to 1, [[DEF_LOOPS_INIT]]#1 -> %arg4 = 0 to 3, [[DEF_LOOPS_INIT]]#2 -> %arg5 = 0 to 3) {
  // CHECK:   krnl.store [[INITIAL_VAL]], [[RES]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK: }

  /// Check main loop.
  // CHECK: [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:   [[rhrHMemRef:%.+]] = alloc() : memref<3x3xf32>
  // CHECK:   [[rhMemRef:%.+]] = alloc() : memref<3x3xf32>
  // CHECK:   [[xwHMemRef:%.+]] = alloc() : memref<3x3xf32>
  // CHECK:   [[ztMemRef:%.+]] = alloc() : memref<3x3xf32>
  // CHECK:   [[htMemRef:%.+]] = alloc() : memref<3x3xf32>
  // CHECK:   [[ZERO_INDEX:%.+]] = constant 0 : index
  // CHECK:   [[INDEX_3:%.+]] = constant 3 : index
  // CHECK:   [[INDEX_0:%.+]] = constant 0 : index
  // CHECK:   [[INDEX_1:%.+]] = constant 1 : index
  // CHECK:   [[INDEX_2:%.+]] = constant 2 : index
  // CHECK:   [[DATA_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[DATA_LOOPS]]#0, [[DATA_LOOPS]]#1) with ([[DATA_LOOPS]]#0 -> %arg4 = 0 to 3, [[DATA_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:     [[rt:%.+]] = alloc() : memref<f32>
  // CHECK:     [[zt:%.+]] = alloc() : memref<f32>

  // CHECK:     [[INITIAL_VAL_0:%.+]] = constant 0.000000e+00 : f32
  // CHECK:     [[XWZt:%.+]] = alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[XWZt]][] : memref<f32>
  // CHECK:     [[HRZt:%.+]] = alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[HRZt]][] : memref<f32>
  // CHECK:     [[XWRt:%.+]] = alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[XWRt]][] : memref<f32>
  // CHECK:     [[HRRt:%.+]] = alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[HRRt]][] : memref<f32>

  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[xwHMemRef]][%arg4, %arg5] : memref<3x3xf32>

  // CHECK:     [[REDUCTION_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[REDUCTION_LOOPS]]) with ([[REDUCTION_LOOPS]] -> %arg6 = 0 to 2) {
  // CHECK:       [[BIAS_MAP_FOR_Z:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_0]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_R:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_1]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_H:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_2]], [[INDEX_3]]]
  // CHECK:       [[Xt:%.+]] = krnl.load %arg0[%arg3, %arg4, %arg6] : memref<4x3x2xf32>

  /// compute Xt*(Wz^T)
  // CHECK:       [[WZt:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_Z]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WZt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[XWZt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[XWZt]][] : memref<f32>

  /// compute Xt*(Wr^T)
  // CHECK:       [[WRt:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_R]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WRt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[XWRt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[XWRt]][] : memref<f32>

  /// compute Xt*(Wh^T)
  // CHECK:       [[WHt:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_H]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WHt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[xwHMemRef]][%arg4, %arg5] : memref<3x3xf32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[xwHMemRef]][%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     }

  // CHECK:     [[REDUCTION_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[REDUCTION_LOOPS]]) with ([[REDUCTION_LOOPS]] -> %arg6 = 0 to 3) {
  // CHECK:       [[BIAS_MAP_FOR_Z:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_0]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_R:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_1]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_H:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_2]], [[INDEX_3]]]
  // CHECK:       [[PREVIOUS_Ht:%.+]] = krnl.load [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg6] : memref<1x3x3xf32>
  /// compute Ht-1*(Rz^T)
  // CHECK:       [[RZt:%.+]] = krnl.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_Z]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[PREVIOUS_Ht]], [[RZt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[HRZt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[HRZt]][] : memref<f32>

  /// compute Ht-1*(Rr^T)
  // CHECK:       [[RRt:%.+]] = krnl.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_R]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[PREVIOUS_Ht]], [[RRt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[HRRt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[HRRt]][] : memref<f32>
  // CHECK:     }

  /// compute zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
  // CHECK:     [[LOAD_XWZt:%.+]] = krnl.load [[XWZt]][] : memref<f32>
  // CHECK:     [[LOAD_HRZt:%.+]] = krnl.load [[HRZt]][] : memref<f32>
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWZt]], [[LOAD_HRZt]] : f32
  /// apply activation f = sigmoid
  // CHECK:     {{.*}} = alloc() : memref<f32>
  // CHECK:     krnl.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = krnl.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     krnl.store {{.*}}, [[zt]][] : memref<f32>
  // CHECK:     krnl.store {{.*}}, [[ztMemRef]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  /// compute rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
  // CHECK:     [[LOAD_XWRt:%.+]] = krnl.load [[XWRt]][] : memref<f32>
  // CHECK:     [[LOAD_HRRt:%.+]] = krnl.load [[HRRt]][] : memref<f32>
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWRt]], [[LOAD_HRRt]] : f32
  /// apply activation f = sigmoid
  // CHECK:     {{.*}} = alloc() : memref<f32>
  // CHECK:     krnl.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = krnl.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     krnl.store {{.*}}, [[rt]][] : memref<f32>
  // CHECK:     [[LOAD_rt:%.+]] = krnl.load [[rt]][] : memref<f32>

  // COM: 'rt (.) Ht-1'
  // CHECK:     [[LOAD_ht:%.+]] = krnl.load [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[RtHt:%.+]] = mulf [[LOAD_rt]], [[LOAD_ht]] : f32
  // CHECK:     krnl.store [[RtHt]], [[rhMemRef]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  // CHECK:     dealloc [[XWZt]] : memref<f32>
  // CHECK:     dealloc [[XWRt]] : memref<f32>
  // CHECK:     dealloc [[HRZt]] : memref<f32>
  // CHECK:     dealloc [[HRRt]] : memref<f32>
  // CHECK:   }

  // COM: compute '(rt (.) Ht-1)*(Rh^T)'
  // CHECK:   [[HT_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[HT_LOOPS]]#0, [[HT_LOOPS]]#1) with ([[HT_LOOPS]]#0 -> %arg4 = 0 to 3, [[HT_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:     [[BIAS_MAP_FOR_H:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_2]], [[INDEX_3]]]
  // CHECK:     [[INITIAL_VAL:%.+]] = constant 0.000000e+00 : f32
  // CHECK:     krnl.store [[INITIAL_VAL]], [[rhrHMemRef]][%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[REDUCTION_LOOPS_1:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[REDUCTION_LOOPS_1]]) with ([[REDUCTION_LOOPS_1]] -> %arg6 = 0 to 3) {
  // CHECK:       [[LOAD_RtHt:%.+]] = krnl.load [[rhMemRef]][%arg4, %arg6] : memref<3x3xf32>
  // CHECK:       [[LOAD_RHt:%.+]] = krnl.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_H]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_RtHt]], [[LOAD_RHt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[rhrHMemRef]][%arg4, %arg5] : memref<3x3xf32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[rhrHMemRef]][%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     }
  // CHECK:   }

  // CHECK:   [[GATE_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[GATE_LOOPS]]#0, [[GATE_LOOPS]]#1) with ([[GATE_LOOPS]]#0 -> %arg4 = 0 to 3, [[GATE_LOOPS]]#1 -> %arg5 = 0 to 3) {

  // COM: compute  ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh) since linear_before_reset = 0 (default)
  // CHECK:     [[ht:%.+]] = alloc() : memref<f32>
  // CHECK:     [[LOAD_XWHt:%.+]] = krnl.load [[xwHMemRef]][%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[LOAD_HRHt:%.+]] = krnl.load [[rhrHMemRef]][%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWHt]], [[LOAD_HRHt]] : f32
  /// apply activation g = tanh
  // CHECK:     krnl.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = krnl.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = constant 2.000000e+00 : f32
  // CHECK:     {{.*}} = mulf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = negf {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = cmpf oge, {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = select {{.*}}, {{.*}}, {{.*}} : f32
  // CHECK:     krnl.store {{.*}}, [[ht]][] : memref<f32>
  // CHECK:     [[LOAD_ht:%.+]] = krnl.load [[ht]][] : memref<f32>

  // COM: compute  Ht = (1 - zt) (.) ht + zt (.) Ht-1
  // CHECK:     [[LOAD_zt:%.+]] = krnl.load [[ztMemRef]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[PREVIOUS_Ht:%.+]] = krnl.load [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[ONE:%.+]] = constant 1.000000e+00 : f32
  // CHECK:     [[SUB:%.+]] = subf [[ONE]], [[LOAD_zt]] : f32
  // CHECK:     [[MUL:%.+]] = mulf [[SUB]], [[LOAD_ht]] : f32
  // CHECK:     [[MUL_1:%.+]] = mulf [[LOAD_zt]], [[PREVIOUS_Ht]] : f32
  // CHECK:     [[ADD:%.+]] = addf [[MUL]], [[MUL_1]] : f32
  // CHECK:     krnl.store [[ADD]], [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:   }
  // CHECK:   dealloc [[htMemRef]] : memref<3x3xf32>
  // CHECK:   dealloc [[ztMemRef]] : memref<3x3xf32>
  // CHECK:   dealloc [[xwHMemRef]] : memref<3x3xf32>
  // CHECK:   dealloc [[rhMemRef]] : memref<3x3xf32>
  // CHECK:   dealloc [[rhrHMemRef]] : memref<3x3xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x3x3xf32>
}

// -----

/// GRU with three required inputs (X, W, R). The optional inputs are default.
/// Check the equation for 'ht' when linear_before_reset !=0.
func private @test_gru_linear_before_reset(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64, linear_before_reset = 1 : si64} : (tensor<4x3x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_linear_before_reset
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x3xf32>

  /// Check initialize loop.
  // CHECK: [[INITIAL_VAL:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[DEF_LOOPS_INIT:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS_INIT]]#0, [[DEF_LOOPS_INIT]]#1, [[DEF_LOOPS_INIT]]#2) with ([[DEF_LOOPS_INIT]]#0 -> %arg3 = 0 to 1, [[DEF_LOOPS_INIT]]#1 -> %arg4 = 0 to 3, [[DEF_LOOPS_INIT]]#2 -> %arg5 = 0 to 3) {
  // CHECK:   krnl.store [[INITIAL_VAL]], [[RES]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK: }

  /// Check main loop.
  // CHECK: [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:   [[ztMemRef:%.+]] = alloc() : memref<3x3xf32>
  // CHECK:   [[htMemRef:%.+]] = alloc() : memref<3x3xf32>
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
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[XWZt]][] : memref<f32>
  // CHECK:     [[HRZt:%.+]] = alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[HRZt]][] : memref<f32>
  // CHECK:     [[XWRt:%.+]] = alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[XWRt]][] : memref<f32>
  // CHECK:     [[HRRt:%.+]] = alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[HRRt]][] : memref<f32>
  // CHECK:     [[XWHt:%.+]] = alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[XWHt]][] : memref<f32>
  // CHECK:     [[HRHt:%.+]] = alloc() : memref<f32>
  // CHECK:     krnl.store [[INITIAL_VAL_0]], [[HRHt]][] : memref<f32>

  // CHECK:     [[REDUCTION_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[REDUCTION_LOOPS]]) with ([[REDUCTION_LOOPS]] -> %arg6 = 0 to 2) {
  // CHECK:       [[BIAS_MAP_FOR_Z:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_0]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_R:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_1]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_H:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_2]], [[INDEX_3]]]
  // CHECK:       [[Xt:%.+]] = krnl.load %arg0[%arg3, %arg4, %arg6] : memref<4x3x2xf32>

  /// compute Xt*(Wz^T)
  // CHECK:       [[WZt:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_Z]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WZt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[XWZt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[XWZt]][] : memref<f32>

  /// compute Xt*(Wr^T)
  // CHECK:       [[WRt:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_R]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WRt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[XWRt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[XWRt]][] : memref<f32>

  /// compute Xt*(Wh^T)
  // CHECK:       [[WHt:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_H]], %arg6] : memref<1x9x2xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[Xt]], [[WHt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[XWHt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[XWHt]][] : memref<f32>
  // CHECK:     }

  // CHECK:     [[REDUCTION_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[REDUCTION_LOOPS]]) with ([[REDUCTION_LOOPS]] -> %arg6 = 0 to 3) {
  // CHECK:       [[BIAS_MAP_FOR_Z:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_0]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_R:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_1]], [[INDEX_3]]]
  // CHECK:       [[BIAS_MAP_FOR_H:%.+]] = affine.apply {{.*}}(%arg5){{\[}}[[INDEX_2]], [[INDEX_3]]]
  // CHECK:       [[PREVIOUS_Ht:%.+]] = krnl.load [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg6] : memref<1x3x3xf32>
  /// compute Ht-1*(Rz^T)
  // CHECK:       [[RZt:%.+]] = krnl.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_Z]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[PREVIOUS_Ht]], [[RZt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[HRZt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[HRZt]][] : memref<f32>

  /// compute Ht-1*(Rr^T)
  // CHECK:       [[RRt:%.+]] = krnl.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_R]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[PREVIOUS_Ht]], [[RRt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[HRRt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[HRRt]][] : memref<f32>

  /// compute Ht-1*(Rh^T)
  // CHECK:       [[RHt:%.+]] = krnl.load %arg2{{\[}}[[ZERO_INDEX]], [[BIAS_MAP_FOR_H]], %arg6] : memref<1x9x3xf32>
  // CHECK:       [[MUL:%.+]] = mulf [[PREVIOUS_Ht]], [[RHt]] : f32
  // CHECK:       [[LOAD:%.+]] = krnl.load [[HRHt]][] : memref<f32>
  // CHECK:       [[ADD:%.+]] = addf [[LOAD]], [[MUL]] : f32
  // CHECK:       krnl.store [[ADD]], [[HRHt]][] : memref<f32>
  // CHECK:     }

  /// compute zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
  // CHECK:     [[LOAD_XWZt:%.+]] = krnl.load [[XWZt]][] : memref<f32>
  // CHECK:     [[LOAD_HRZt:%.+]] = krnl.load [[HRZt]][] : memref<f32>
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWZt]], [[LOAD_HRZt]] : f32
  /// apply activation f = sigmoid
  // CHECK:     {{.*}} = alloc() : memref<f32>
  // CHECK:     krnl.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = krnl.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     krnl.store {{.*}}, [[zt]][] : memref<f32>
  // CHECK:     krnl.store {{.*}}, [[ztMemRef]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  /// compute rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
  // CHECK:     [[LOAD_XWRt:%.+]] = krnl.load [[XWRt]][] : memref<f32>
  // CHECK:     [[LOAD_HRRt:%.+]] = krnl.load [[HRRt]][] : memref<f32>
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWRt]], [[LOAD_HRRt]] : f32
  /// apply activation f = sigmoid
  // CHECK:     {{.*}} = alloc() : memref<f32>
  // CHECK:     krnl.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = krnl.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     krnl.store {{.*}}, [[rt]][] : memref<f32>
  // CHECK:     [[LOAD_rt:%.+]] = krnl.load [[rt]][] : memref<f32>

  /// compute ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh) since linear_before_reset != 0
  // CHECK:     [[LOAD_XWHt:%.+]] = krnl.load [[XWHt]][] : memref<f32>
  // CHECK:     [[LOAD_HRHt:%.+]] = krnl.load [[HRHt]][] : memref<f32>
  // CHECK:     [[MUL_rt_HRHt:%.+]] = mulf [[LOAD_rt]], [[LOAD_HRHt]] : f32
  // CHECK:     [[ADD:%.+]] = addf [[LOAD_XWHt]], [[MUL_rt_HRHt]] : f32
  /// apply activation g = tanh
  // CHECK:     {{.*}} = alloc() : memref<f32>
  // CHECK:     krnl.store [[ADD]], {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = krnl.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = constant 2.000000e+00 : f32
  // CHECK:     {{.*}} = mulf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = negf {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = cmpf oge, {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = select {{.*}}, {{.*}}, {{.*}} : f32
  // CHECK:     krnl.store {{.*}}, [[ht]][] : memref<f32>
  // CHECK:     [[LOAD_ht:%.+]] = krnl.load [[ht]][] : memref<f32>
  // CHECK:     krnl.store [[LOAD_ht]], [[htMemRef]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

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
  // CHECK:     [[LOAD_ht:%.+]] = krnl.load [[htMemRef]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[LOAD_zt:%.+]] = krnl.load [[ztMemRef]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[PREVIOUS_Ht:%.+]] = krnl.load [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     [[ONE:%.+]] = constant 1.000000e+00 : f32
  // CHECK:     [[SUB:%.+]] = subf [[ONE]], [[LOAD_zt]] : f32
  // CHECK:     [[MUL:%.+]] = mulf [[SUB]], [[LOAD_ht]] : f32
  // CHECK:     [[MUL_1:%.+]] = mulf [[LOAD_zt]], [[PREVIOUS_Ht]] : f32
  // CHECK:     [[ADD:%.+]] = addf [[MUL]], [[MUL_1]] : f32
  // CHECK:     krnl.store [[ADD]], [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:   }
  // CHECK:    dealloc [[htMemRef]] : memref<3x3xf32>
  // CHECK:    dealloc [[ztMemRef]] : memref<3x3xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x3x3xf32>
}

// -----

/// Check GRU with three required inputs (X, W, R), and bias input.
func private @test_gru_with_bias(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x9x2xf32>, %arg2: tensor<1x9x3xf32>, %arg3: tensor<1x18xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.GRU"(%arg0, %arg1, %arg2, %arg3, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x9x2xf32>, tensor<1x9x3xf32>, tensor<1x18xf32>, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_gru_with_bias

  // CHECK: [[LOAD_WZ_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_WZ_BIAS]] : f32
  // CHECK: [[LOAD_RZ_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_RZ_BIAS]] : f32

  // CHECK: [[LOAD_WR_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_WR_BIAS]] : f32
  // CHECK: [[LOAD_RR_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_RR_BIAS]] : f32

  // CHECK: [[LOAD_WH_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_WH_BIAS]] : f32
  // CHECK: [[LOAD_RH_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x18xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_RH_BIAS]] : f32
}

// -----

// Check handling unknown dimensions for GRU by checking the
// correctness of allocating and deallocating memory.
func private @test_gru_unkown_dims_allocation(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x9x?xf32>, %arg2: tensor<1x9x3xf32>) -> tensor<*xf32> {
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

func private @test_lstm_general_computation(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
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
  // CHECK:    krnl.store [[INITIAL_VALUE]], [[HIDDEN_STATE]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:    krnl.store [[INITIAL_VALUE]], [[CELL_STATE]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:  }

  // CHECK:  [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {

  // CHECK:    [[HtRc_GEMM:%.+]] = alloc() : memref<3x3xf32>
  // CHECK:    [[XtWc_GEMM:%.+]] = alloc() : memref<3x3xf32>
  // CHECK:    [[HtRf_GEMM:%.+]] = alloc() : memref<3x3xf32>
  // CHECK:    [[XtWf_GEMM:%.+]] = alloc() : memref<3x3xf32>
  // CHECK:    [[HtRo_GEMM:%.+]] = alloc() : memref<3x3xf32>
  // CHECK:    [[XtWo_GEMM:%.+]] = alloc() : memref<3x3xf32>
  // CHECK:    [[HtRi_GEMM:%.+]] = alloc() : memref<3x3xf32>
  // CHECK:    [[XtWi_GEMM:%.+]] = alloc() : memref<3x3xf32>

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
  // CHECK:      krnl.store [[CST0]], [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      krnl.store [[CST0]], [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      krnl.store [[CST0]], [[XtWo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      krnl.store [[CST0]], [[HtRo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      krnl.store [[CST0]], [[XtWf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      krnl.store [[CST0]], [[HtRf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      krnl.store [[CST0]], [[XtWc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      krnl.store [[CST0]], [[HtRc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[XW_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:      krnl.iterate([[XW_LOOPS]]) with ([[XW_LOOPS]] -> %arg6 = 0 to 2) {
  // CHECK:        [[INPUT_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[OUTPUT_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[FORGET_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[CELL_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]

  // CHECK:        [[Xt_LOAD:%.+]] = krnl.load %arg0[%arg3, %arg4, %arg6] : memref<4x3x2xf32>
  // CHECK:        [[Wi_LOAD:%.+]] = krnl.load %arg1{{\[}}[[C0_INDEX]], [[INPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wi_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        krnl.store {{.*}}, [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  // CHECK:        [[Wo_LOAD:%.+]] = krnl.load %arg1{{\[}}[[C0_INDEX]], [[OUTPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wo_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[XtWo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, %26 : f32
  // CHECK:        krnl.store {{.*}}, [[XtWo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  // CHECK:        [[Wf_LOAD:%.+]] = krnl.load %arg1{{\[}}[[C0_INDEX]], [[FORGET_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wf_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[XtWf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, %30 : f32
  // CHECK:        krnl.store {{.*}}, [[XtWf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  // CHECK:        [[Wc_LOAD:%.+]] = krnl.load %arg1{{\[}}[[C0_INDEX]], [[CELL_HIDDEN_INDEX]], %arg6] : memref<1x12x2xf32>
  // CHECK:        {{.*}} = mulf [[Xt_LOAD]], [[Wc_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[XtWc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, %34 : f32
  // CHECK:        krnl.store {{.*}}, [[XtWc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      }
  // CHECK:      [[HR_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:      krnl.iterate([[HR_LOOPS]]) with ([[HR_LOOPS]] -> %arg6 = 0 to 3) {
  // CHECK:        [[INPUT_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[OUTPUT_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[FORGET_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]
  // CHECK:        [[CELL_HIDDEN_INDEX:%.+]] = affine.apply #{{.*}}(%arg5)[{{.*}}, {{.*}}]

  // CHECK:        [[Ht_LOAD:%.+]] = krnl.load %1{{\[}}[[C0_INDEX]], %arg4, %arg6] : memref<1x3x3xf32>

  // CHECK:        [[Ri_LOAD:%.+]] = krnl.load %arg2{{\[}}[[C0_INDEX]], [[INPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht_LOAD]], [[Ri_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        krnl.store {{.*}}, [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  // CHECK:        [[Ro_LOAD:%.+]] = krnl.load %arg2{{\[}}[[C0_INDEX]], [[OUTPUT_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht_LOAD]], [[Ro_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[HtRo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        krnl.store {{.*}}, [[HtRo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  // CHECK:        [[Rf_LOAD:%.+]] = krnl.load %arg2{{\[}}[[C0_INDEX]], [[FORGET_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht_LOAD]], [[Rf_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[HtRf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        krnl.store {{.*}}, [[HtRf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>

  // CHECK:        [[Rc_LOAD:%.+]] = krnl.load %arg2{{\[}}[[C0_INDEX]], [[CELL_HIDDEN_INDEX]], %arg6] : memref<1x12x3xf32>
  // CHECK:        {{.*}} = mulf [[Ht_LOAD]], [[Rc_LOAD]] : f32
  // CHECK:        {{.*}} = krnl.load [[HtRc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:        {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:        krnl.store {{.*}}, [[HtRc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      } 
  // CHECK:    }

  // CHECK:    [[DATA_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:    krnl.iterate([[DATA_LOOPS]]#0, [[DATA_LOOPS]]#1) with ([[DATA_LOOPS]]#0 -> %arg4 = 0 to 3, [[DATA_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:      [[hCt:%.+]] = alloc() : memref<f32>
  // CHECK:      [[Ot:%.+]] = alloc() : memref<f32>
  // CHECK:      [[ct:%.+]] = alloc() : memref<f32>
  // CHECK:      [[Ft:%.+]] = alloc() : memref<f32>
  // CHECK:      [[It:%.+]] = alloc() : memref<f32>

  // CHECK:      [[Ct1_LOAD:%.+]] = krnl.load [[CELL_STATE]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:      [[XtWi_LOAD:%.+]] = krnl.load [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[HtRi_LOAD:%.+]] = krnl.load [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[It_OUTPUT:%.+]] = addf [[XtWi_LOAD]], [[HtRi_LOAD]] : f32

  // CHECK:      [[SIGMOID_INPUT:%.+]] = alloc() : memref<f32>
  // CHECK:      krnl.store [[It_OUTPUT]], [[SIGMOID_INPUT]][] : memref<f32>
  // CHECK:      {{.*}} = krnl.load [[SIGMOID_INPUT]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}}: f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      krnl.store {{.*}}, [[It]][] : memref<f32>
  // CHECK:      [[It_LOAD:%.+]] = krnl.load [[It]][] : memref<f32>

  // CHECK:      [[XtWf_LOAD:%.+]] = krnl.load [[XtWf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[HtRf_LOAD:%.+]] = krnl.load [[HtRf_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[Ft_OUTPUT:%.+]] = addf [[XtWf_LOAD]], [[HtRf_LOAD]] : f32

  // CHECK:      [[SIGMOID_FORGET:%.+]] = alloc() : memref<f32>
  // CHECK:      krnl.store [[Ft_OUTPUT]], [[SIGMOID_FORGET]][] : memref<f32>
  // CHECK:      {{.*}} = krnl.load [[SIGMOID_FORGET]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}}: f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      krnl.store {{.*}}, [[Ft]][] : memref<f32>
  // CHECK:      [[Ft_LOAD:%.+]] = krnl.load [[Ft]][] : memref<f32>

  // CHECK:      [[XtWc_LOAD:%.+]] = krnl.load [[XtWc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[HtRc_LOAD:%.+]] = krnl.load [[HtRc_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[ct_OUTPUT:%.+]] = addf [[XtWc_LOAD]], [[HtRc_LOAD]] : f32

  // CHECK:      [[TANH_CELL:%.+]] = alloc() : memref<f32>
  // CHECK:      krnl.store [[ct_OUTPUT]], [[TANH_CELL]][] : memref<f32>
  // CHECK:      {{.*}} = krnl.load [[TANH_CELL]][] : memref<f32>
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = constant 2.000000e+00 : f32
  // CHECK:      {{.*}} = mulf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = negf {{.*}} : f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = cmpf oge, {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = select {{.*}}, {{.*}}, {{.*}} : f32
  // CHECK:      krnl.store {{.*}}, [[ct]][] : memref<f32>
  // CHECK:      [[ct_LOAD:%.+]] = krnl.load [[ct]][] : memref<f32>

  // CHECK:      [[FtCt1:%.+]] = mulf [[Ft_LOAD]], [[Ct1_LOAD]] : f32
  // CHECK:      [[Itct:%.+]] = mulf [[It_LOAD]], [[ct_LOAD]] : f32
  // CHECK:      [[Ct:%.+]] = addf [[FtCt1]], [[Itct]] : f32
  // CHECK:      krnl.store [[Ct]], [[CELL_STATE]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:      [[XtWo_LOAD:%.+]] = krnl.load [[XtWo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[HtRo_LOAD:%.+]] = krnl.load [[HtRo_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:      [[Ot_OUTPUT:%.+]] = addf [[XtWo_LOAD]], [[HtRo_LOAD]] : f32

  // CHECK:      [[SIGMOID_OUTPUT:%.+]] = alloc() : memref<f32>
  // CHECK:      krnl.store [[Ot_OUTPUT]], [[SIGMOID_OUTPUT]][] : memref<f32>
  // CHECK:      {{.*}} = krnl.load [[SIGMOID_OUTPUT]][] : memref<f32>
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}}: f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      krnl.store {{.*}}, [[Ot]][] : memref<f32>
  // CHECK:      [[Ot_LOAD:%.+]] = krnl.load [[Ot]][] : memref<f32>

  // CHECK:      [[TANH_HIDDEN:%.+]] = alloc() : memref<f32>
  // CHECK:      krnl.store [[Ct]], [[TANH_HIDDEN]][] : memref<f32>
  // CHECK:      {{.*}} = krnl.load [[TANH_HIDDEN]][] : memref<f32>
  // CHECK:      {{.*}} = constant 1.000000e+00 : f32
  // CHECK:      {{.*}} = constant 2.000000e+00 : f32
  // CHECK:      {{.*}} = mulf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = negf {{.*}} : f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = exp {{.*}} : f32
  // CHECK:      {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = constant 0.000000e+00 : f32
  // CHECK:      {{.*}} = cmpf oge, {{.*}}, {{.*}} : f32
  // CHECK:      {{.*}} = select {{.*}}, {{.*}}, {{.*}} : f32
  // CHECK:      krnl.store {{.*}}, [[hCt]][] : memref<f32>
  // CHECK:      [[hCt_LOAD:%.+]] = krnl.load [[hCt]][] : memref<f32>

  // CHECK:      [[Ht:%.+]] = mulf [[Ot_LOAD]], [[hCt_LOAD]] : f32
  // CHECK:      krnl.store [[Ht]], [[HIDDEN_STATE]]{{\[}}[[C0_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>

  // CHECK:      dealloc [[It]] : memref<f32>
  // CHECK:      dealloc [[Ft]] : memref<f32>
  // CHECK:      dealloc [[ct]] : memref<f32>
  // CHECK:      dealloc [[Ot]] : memref<f32>
  // CHECK:      dealloc [[hCt]] : memref<f32>
  // CHECK:    }
  // CHECK:    dealloc [[XtWi_GEMM]] : memref<3x3xf32>
  // CHECK:    dealloc [[XtWo_GEMM]] : memref<3x3xf32>
  // CHECK:    dealloc [[XtWf_GEMM]] : memref<3x3xf32>
  // CHECK:    dealloc [[XtWc_GEMM]] : memref<3x3xf32>
  // CHECK:    dealloc [[HtRi_GEMM]] : memref<3x3xf32>
  // CHECK:    dealloc [[HtRo_GEMM]] : memref<3x3xf32>
  // CHECK:    dealloc [[HtRf_GEMM]] : memref<3x3xf32>
  // CHECK:    dealloc [[HtRc_GEMM]] : memref<3x3xf32>
 
  // CHECK:  }
  // CHECK:  dealloc [[CELL_STATE]] : memref<1x3x3xf32>
  // CHECK:  return [[HIDDEN_STATE]] : memref<1x3x3xf32>
}

// -----

func private @test_lstm_reverse_mode(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64, direction = "reverse"} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

  // CHECK: [[REVERSE_IV_MAP:#.+]] = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
  // CHECK-LABEL: @test_lstm_reverse_mode

  // CHECK:  [[REVERSE_SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[REVERSE_SEQUENCE_LOOPS]]) with ([[REVERSE_SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:  %[[SEQUENCE_LEN:.+]] = constant 4 : index
  // CHECK:  %[[REVERSE_SEQUENCE_IV:.+]] = affine.apply [[REVERSE_IV_MAP]](%arg3)[%[[SEQUENCE_LEN]]{{]}}
  // CHECK:  [[Xt_LOAD:%.+]] = krnl.load %arg0[%[[REVERSE_SEQUENCE_IV]], {{.*}}, {{.*}}] : memref<4x3x2xf32>
}

// -----

func private @test_lstm_bidirectional_mode(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x12x2xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h, %Y_c = "onnx.LSTM"(%arg0, %arg1, %arg2, %cst, %cst, %cst, %cst, %cst) {hidden_size = 3 : si64, direction = "bidirectional"} : (tensor<4x3x2xf32>, tensor<1x12x2xf32>, tensor<1x12x3xf32>, none, none, none, none, none) -> (none, tensor<*xf32>, none)
  return %Y_h : tensor<*xf32>

  // CHECK: [[REVERSE_IV_MAP:#.+]] = affine_map<(d0)[s0] -> (-d0 + s0 - 1)>
  // CHECK-LABEL: @test_lstm_bidirectional_mode

  // CHECK:  [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:  {{.*}} = krnl.define_loops 2
  // CHECK:  {{.*}} = krnl.define_loops 1
  // CHECK:  [[Xt_LOAD:%.+]] = krnl.load %arg0[%arg3, {{.*}}, {{.*}}] : memref<4x3x2xf32>
  // CHECK:  {{.*}} = krnl.define_loops 1
  // CHECK:  {{.*}} = krnl.define_loops 2

  // CHECK:  [[REVERSE_SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:  krnl.iterate([[REVERSE_SEQUENCE_LOOPS]]) with ([[REVERSE_SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:  %[[SEQUENCE_LEN:.+]] = constant 4 : index
  // CHECK:  %[[REVERSE_SEQUENCE_IV:.+]] = affine.apply [[REVERSE_IV_MAP]](%arg3)[%[[SEQUENCE_LEN]]{{]}}
  // CHECK:  {{.*}} = krnl.define_loops 2
  // CHECK:  {{.*}} = krnl.define_loops 1
  // CHECK:  [[Xt_LOAD:%.+]] = krnl.load %arg0[%[[REVERSE_SEQUENCE_IV]], {{.*}}, {{.*}}] : memref<4x3x2xf32>
  // CHECK:  {{.*}} = krnl.define_loops 1
  // CHECK:  {{.*}} = krnl.define_loops 2
}

// -----

// Check handling unknown dimensions for LSTM by checking the
// correctness of allocating and deallocating memory.
func private @test_lstm_unkown_dims_allocation(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x12x?xf32>, %arg2: tensor<1x12x3xf32>) -> tensor<*xf32> {
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
func private @test_rnn_general_computation(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x3x2xf32>, %arg2: tensor<1x3x3xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %cst, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x3x2xf32>, tensor<1x3x3xf32>, none, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_rnn_general_computation
  // CHECK: [[RES:%.+]] = alloc() : memref<1x3x3xf32>

  /// Check initialize loop.
  // CHECK: [[INITIAL_VAL:%.+]] = constant 0.000000e+00 : f32
  // CHECK: [[DEF_LOOPS_INIT:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS_INIT]]#0, [[DEF_LOOPS_INIT]]#1, [[DEF_LOOPS_INIT]]#2) with ([[DEF_LOOPS_INIT]]#0 -> %arg3 = 0 to 1, [[DEF_LOOPS_INIT]]#1 -> %arg4 = 0 to 3, [[DEF_LOOPS_INIT]]#2 -> %arg5 = 0 to 3) {
  // CHECK:   krnl.store [[INITIAL_VAL]], [[RES]][%arg3, %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK: }

  /// Check main loop.
  // CHECK: [[SEQUENCE_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[SEQUENCE_LOOPS]]) with ([[SEQUENCE_LOOPS]] -> %arg3 = 0 to 4) {
  // CHECK:   [[HtRi_GEMM:%.+]] = alloc() : memref<3x3xf32>
  // CHECK:   [[XtWi_GEMM:%.+]] = alloc() : memref<3x3xf32>
  // CHECK:   [[ZERO_INDEX:%.+]] = constant 0 : index
  // CHECK:   {{.*}} = constant 3 : index
  // CHECK:   {{.*}} = constant 0 : index
  // CHECK:   {{.*}} = constant 1 : index

  /// Check reduction loop to compute matrix multiplication for 'Xt*(Wi^T)' and 'Ht-1*(Ri^T)'
  // CHECK:   [[MATRIX_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[MATRIX_LOOPS]]#0, [[MATRIX_LOOPS]]#1) with ([[MATRIX_LOOPS]]#0 -> %arg4 = 0 to 3, [[MATRIX_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:     [[CST0:%.+]] = constant 0.000000e+00 : f32
  // CHECK:     krnl.store [[CST0]], [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     krnl.store [[CST0]], [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[XW_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[XW_LOOPS]]) with ([[XW_LOOPS]] -> %arg6 = 0 to 2) {
  // CHECK:       [[Xt_LOAD:%.+]] = krnl.load %arg0[%arg3, %arg4, %arg6] : memref<4x3x2xf32>
  // CHECK:       [[Wi_LOAD:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], %arg5, %arg6] : memref<1x3x2xf32>
  // CHECK:       {{.*}} = mulf [[Xt_LOAD]], [[Wi_LOAD]] : f32
  // CHECK:       {{.*}} = krnl.load [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:       {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:       krnl.store {{.*}}, [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     }
  // CHECK:     [[HR_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK:     krnl.iterate([[HR_LOOPS]]) with ([[HR_LOOPS]] -> %arg6 = 0 to 3) {
  // CHECK:       [[Ht_LOAD:%.+]] = krnl.load %0{{\[}}[[ZERO_INDEX]], %arg4, %arg6] : memref<1x3x3xf32>
  // CHECK:       [[Ri_LOAD:%.+]] = krnl.load %arg2{{\[}}[[ZERO_INDEX]], %arg5, %arg6] : memref<1x3x3xf32>
  // CHECK:       {{.*}} = mulf [[Ht_LOAD]], [[Ri_LOAD]] : f32
  // CHECK:       {{.*}} = krnl.load [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:       {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:       krnl.store {{.*}}, [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     }
  // CHECK:   }
 
  // CHECK:   [[DATA_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[DATA_LOOPS]]#0, [[DATA_LOOPS]]#1) with ([[DATA_LOOPS]]#0 -> %arg4 = 0 to 3, [[DATA_LOOPS]]#1 -> %arg5 = 0 to 3) {
  // CHECK:     [[Ht:%.+]] = alloc() : memref<f32>

  /// Check 'Xt*(Wi^T) + Ht-1*(Ri^T)'
  // CHECK:     [[LOAD_XWi:%.+]] = krnl.load [[XtWi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[LOAD_HRi:%.+]] = krnl.load [[HtRi_GEMM]]{{\[}}%arg4, %arg5] : memref<3x3xf32>
  // CHECK:     [[XWi_PLUS_HRi:%.+]] = addf [[LOAD_XWi]], [[LOAD_HRi]] : f32

  /// Check calling 'Tanh'
  // CHECK:     {{.*}} = alloc() : memref<f32>
  // CHECK:     krnl.store [[XWi_PLUS_HRi]], {{.*}} : memref<f32>
  // CHECK:     {{.*}} = krnl.load {{.*}}[] : memref<f32>
  // CHECK:     {{.*}} = constant 1.000000e+00 : f32
  // CHECK:     {{.*}} = constant 2.000000e+00 : f32
  // CHECK:     {{.*}} = mulf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = negf {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = exp {{.*}} : f32
  // CHECK:     {{.*}} = subf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = addf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = divf {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = constant 0.000000e+00 : f32
  // CHECK:     {{.*}} = cmpf oge, {{.*}}, {{.*}} : f32
  // CHECK:     {{.*}} = select {{.*}}, {{.*}}, {{.*}} : f32
  // CHECK:     krnl.store {{.*}}, [[Ht]][] : memref<f32>

  /// Check storing the result.
  // CHECK:     [[NEW_Ht_LOAD:%.+]] = krnl.load [[Ht]][] : memref<f32>
  // CHECK:     krnl.store [[NEW_Ht_LOAD]], [[RES]]{{\[}}[[ZERO_INDEX]], %arg4, %arg5] : memref<1x3x3xf32>
  // CHECK:     dealloc [[Ht]] : memref<f32>
  // CHECK:   }
  // CHECK:   dealloc [[XtWi_GEMM]] : memref<3x3xf32>
  // CHECK:   dealloc [[HtRi_GEMM]] : memref<3x3xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x3x3xf32>
}

// -----

/// Check RNN with three required inputs (X, W, R), and bias input.
func private @test_rnn_with_bias(%arg0: tensor<4x3x2xf32>, %arg1: tensor<1x3x2xf32>, %arg2: tensor<1x3x3xf32>, %arg3: tensor<1x6xf32>) -> tensor<*xf32> {
  %cst = constant unit
  %Y, %Y_h = "onnx.RNN"(%arg0, %arg1, %arg2, %arg3, %cst, %cst) {hidden_size = 3 : si64} : (tensor<4x3x2xf32>, tensor<1x3x2xf32>, tensor<1x3x3xf32>, tensor<1x6xf32>, none, none) -> (none, tensor<*xf32>)
  return %Y_h : tensor<*xf32>

  // CHECK-LABEL: test_rnn_with_bias
  // CHECK: [[LOAD_W_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x6xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_W_BIAS]] : f32
  // CHECK: [[LOAD_R_BIAS:%.+]] = krnl.load %arg3[{{.*}}, {{.*}}] : memref<1x6xf32>
  // CHECK: {{.*}} = addf {{.*}}, [[LOAD_R_BIAS]] : f32
}

// -----

// Check handling unknown dimensions for RNN by checking the
// correctness of allocating and deallocating memory.
func private @test_rnn_unkown_dims_allocation(%arg0: tensor<?x?x?xf32>, %arg1: tensor<1x3x?xf32>, %arg2: tensor<1x3x3xf32>) -> tensor<*xf32> {
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

func private @test_squeeze(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Squeeze"(%arg0) { axes = [1, -2]} : (tensor<16x1x32x1x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_squeeze
  // CHECK: [[RES:%.+]] = alloc() : memref<16x32x64xf32>
  // CHECK: [[TENSOR_SIZE:%.+]] = constant 131072 : i64
  // CHECK: "krnl.memcpy"([[RES]], %arg0, [[TENSOR_SIZE]]) : (memref<16x32x64xf32>, memref<16x1x32x1x64xf32>, i64) -> ()
  // CHECK: return [[RES]] : memref<16x32x64xf32>
}

// -----

func private @test_squeeze_unknown_dimensions(%arg0 : tensor<?x1x32x?x64xf32>) -> tensor<*xf32> {
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

func private @test_split_equal(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.Split"(%arg0) { axis = 0 : si64} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 8)>
  // CHECK-LABEL: @test_split_equal

  // CHECK: [[RES_1:%.+]] = alloc() : memref<8x32x64xf32>
  // CHECK: [[RES_0:%.+]] = alloc() : memref<8x32x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 8, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   [[LOAD_0:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_0]], [[RES_0]][%arg1, %arg2, %arg3] : memref<8x32x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 8, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   %[[INDEX:.+]] = affine.apply [[INDEX_MAP]](%arg1)
  // CHECK:   [[LOAD_1:%.+]] = krnl.load %arg0[%[[INDEX]], %arg2, %arg3] : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_1]], [[RES_1]][%arg1, %arg2, %arg3] : memref<8x32x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<8x32x64xf32>, memref<8x32x64xf32>
}

// -----

func private @test_split_variable(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.Split"(%arg0) { axis = 1 : si64, split = [2, 30]} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 2)>
  // CHECK-LABEL: @test_split_variable

  // CHECK: [[RES_1:%.+]] = alloc() : memref<16x30x64xf32>
  // CHECK: [[RES_0:%.+]] = alloc() : memref<16x2x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 2, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   [[LOAD_0:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_0]], [[RES_0]][%arg1, %arg2, %arg3] : memref<16x2x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 30, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64) {
  // CHECK:   %[[INDEX:.+]] = affine.apply [[INDEX_MAP]](%arg2)
  // CHECK:   [[LOAD_1:%.+]] = krnl.load %arg0[%arg1, %[[INDEX]], %arg3] : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_1]], [[RES_1]][%arg1, %arg2, %arg3] : memref<16x30x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<16x2x64xf32>, memref<16x30x64xf32>
}

// -----

func private @cast_lowering_sametype(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<f32>) -> tensor<f32>
  "std.return"(%0) : (tensor<f32>) -> ()

  // CHECK-LABEL: cast_lowering_sametype
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<f32>
  // CHECK: krnl.store [[LOAD]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func private @cast_lowering_intfloat(%arg0: tensor<i64>) -> tensor<f32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<i64>) -> tensor<f32>
  "std.return"(%0) : (tensor<f32>) -> ()

  // CHECK-LABEL: cast_lowering_intfloat
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<i64>
  // CHECK: [[VAL:%.+]] = sitofp [[LOAD]] : i64 to f32
  // CHECK: krnl.store [[VAL]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func private @cast_lowering_floatint(%arg0: tensor<f32>) -> tensor<i64> {
  %0 = "onnx.Cast"(%arg0) {to = i64} : (tensor<f32>) -> tensor<i64>
  "std.return"(%0) : (tensor<i64>) -> ()

  // CHECK-LABEL: cast_lowering_floatint
  // CHECK: [[RES:%.+]] = alloc() : memref<i64>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<f32>
  // CHECK: [[VAL:%.+]] = fptosi [[LOAD]] : f32 to i64
  // CHECK: krnl.store [[VAL]], [[RES]][] : memref<i64>
  // CHECK: return [[RES]] : memref<i64>
}

// -----

func private @cast_lowering_f16f32(%arg0: tensor<f16>) -> tensor<f32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<f16>) -> tensor<f32>
  "std.return"(%0) : (tensor<f32>) -> ()

  // CHECK-LABEL: cast_lowering_f16f32
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<f16>
  // CHECK: [[VAL:%.+]] = fpext [[LOAD]] : f16 to f32
  // CHECK: krnl.store [[VAL]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func private @cast_lowering_f64f32(%arg0: tensor<f64>) -> tensor<f32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<f64>) -> tensor<f32>
  "std.return"(%0) : (tensor<f32>) -> ()

  // CHECK-LABEL: cast_lowering_f64f32
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<f64>
  // CHECK: [[VAL:%.+]] = fptrunc [[LOAD]] : f64 to f32
  // CHECK: krnl.store [[VAL]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func private @cast_lowering_f64f32_10(%arg0: tensor<10xf64>) -> tensor<*xf32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<10xf64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: cast_lowering_f64f32_10
  // CHECK: [[RES:%.+]] = alloc() : memref<10xf32>
  // CHECK: [[DEF_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[DEF_LOOPS]]) with ([[DEF_LOOPS]] -> %arg1 = 0 to 10) {
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg1] : memref<10xf64>
  // CHECK: [[FPTRUNC:%.+]] = fptrunc [[LOAD1]] : f64 to f32
  // CHECK: krnl.store [[FPTRUNC]], [[RES]][%arg1] : memref<10xf32>
  // CHECK: return [[RES]] : memref<10xf32>
}

// -----

func private @cast_lowering_int_wider_int(%arg0: tensor<i32>) -> tensor<i64> {
  %0 = "onnx.Cast"(%arg0) {to = i64} : (tensor<i32>) -> tensor<i64>
  "std.return"(%0) : (tensor<i64>) -> ()

  // CHECK-LABEL: cast_lowering_int_wider_int
  // CHECK: [[RES:%.+]] = alloc() : memref<i64>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<i32>
  // CHECK: [[CAST:%.+]] = sexti [[LOAD]] : i32 to i64
  // CHECK: krnl.store [[CAST]], [[RES]][] : memref<i64>
  // CHECK: return [[RES]] : memref<i64>
}

// -----

func private @cast_lowering_int_narrow_int(%arg0: tensor<i64>) -> tensor<i32> {
  %0 = "onnx.Cast"(%arg0) {to = i32 } : (tensor<i64>) -> tensor<i32>
  "std.return"(%0) : (tensor<i32>) -> ()

  // CHECK-LABEL: cast_lowering_int_narrow_int
  // CHECK: [[RES:%.+]] = alloc() : memref<i32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<i64>
  // CHECK: [[CAST:%.+]] = trunci [[LOAD]] : i64 to i32
  // CHECK: krnl.store [[CAST]], [[RES]][] : memref<i32>
  // CHECK: return [[RES]] : memref<i32>
}

// -----

func private @test_size_known(%arg0: tensor<2x2xf32>) -> tensor<i64> {
  %1 = "onnx.Size"(%arg0) : (tensor<2x2xf32>) -> tensor<i64>
  "std.return"(%1) : (tensor<i64>) -> ()

  // CHECK-LABEL: test_size_known
  // CHECK:      [[RES:%.+]] = alloc() : memref<i64>
  // CHECK-NEXT  [[SIZE:%.+]] = constant 4 : i64
  // CHECK-NEXT  krnl.store [[SIZE]], [[RES]][] : memref<i64>
  // CHECK-NEXT  return [[RES]] : memref<i64>

}

// -----

func private @test_size_unknown(%arg0 : tensor<?x2x?xf32>) -> tensor<i64> {

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
  // CHECK-NEXT:  krnl.store [[SIZE]], [[RES]][] : memref<i64>
  // CHECK-NEXT:  return [[RES]] : memref<i64>

  %1 = "onnx.Size"(%arg0)  : (tensor<?x2x?xf32>) -> tensor<i64>
  "std.return"(%1) : (tensor<i64>) -> ()
}

// -----

// Check the lowering of ConstantOfShape when:
//   - No value attribute.
//   - The input is an empty tensor.
// Expected emitted code:
//   - No need a Krnl iterate.
//   - The output is a scalar tensor.
func private @test_constant_of_shape_empty_tensor(%arg0 : tensor<0xi64>) -> tensor<*xf32> {
  %0 = "onnx.ConstantOfShape"(%arg0) : (tensor<0xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_of_shape_empty_tensor
  // CHECK: [[RES:%.+]] = alloc() : memref<f32>
  // CHECK: [[CST_VALUE:%.+]] = constant 0.000000e+00 : f32
  // CHECK: krnl.store [[CST_VALUE]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

// Check the lowering of ConstantOfShape when:
//   - The input is not a constant tensor.
// Expected emitted code:
//   - Emit code to compute output dimensions from the input's dimensions.
//   - Krnl iterates are used to set values to the output.
func private @test_constant_of_shape_dynamic_dims(%arg0 : tensor<3xi64>) -> tensor<*xf32> {
  %0 = "onnx.ConstantOfShape"(%arg0) {value = dense<[1.0]> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_of_shape_dynamic_dims
  // CHECK: [[CST0:%.+]] = constant 0 : index
  // CHECK: [[LOAD_DIM_0:%.+]] = krnl.load %arg0{{\[}}[[CST0]]{{\]}} : memref<3xi64>
  // CHECK: [[DIM_0:%.+]] = index_cast [[LOAD_DIM_0]] : i64 to index
  // CHECK: [[CST1:%.+]] = constant 1 : index
  // CHECK: [[LOAD_DIM_1:%.+]] = krnl.load %arg0{{\[}}[[CST1]]{{\]}} : memref<3xi64>
  // CHECK: [[DIM_1:%.+]] = index_cast [[LOAD_DIM_1]] : i64 to index
  // CHECK: [[CST2:%.+]] = constant 2 : index
  // CHECK: [[LOAD_DIM_2:%.+]] = krnl.load %arg0{{\[}}[[CST2]]{{\]}} : memref<3xi64>
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
  // CHECK:   krnl.store [[CST_VALUE]], [[RES]][%arg1, %arg2, %arg3] : memref<?x?x?xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<?x?x?xf32>
}

// -----

// Check the lowering of ConstantOfShape when:
//   - The input is a constant tensor.
// Expected emitted code:
//   - Output dimensions are computed during compilation time.
//   - Krnl iterates are used to set values to the output.
func private @test_constant_of_shape_static_dims() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[3, 4, 5]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.ConstantOfShape"(%0) {value = dense<[1.0]> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_of_shape_static_dims
  // CHECK: [[RES:%.+]] = alloc() : memref<3x4x5xf32>
  // CHECK: [[GLOBAL_CST:%.+]] = "krnl.global"() {name = "constant_0", shape = [3], value = dense<[3, 4, 5]> : tensor<3xi64>} : () -> memref<3xi64>
  // CHECK: [[CST_VALUE:%.+]] = constant 1.000000e+00 : f32
  // CHECK: [[LOOP_DEF:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[LOOP_DEF]]#0, [[LOOP_DEF]]#1, [[LOOP_DEF]]#2) with ([[LOOP_DEF]]#0 -> %arg0 = 0 to 3, [[LOOP_DEF]]#1 -> %arg1 = 0 to 4, [[LOOP_DEF]]#2 -> %arg2 = 0 to 5) {
  // CHECK:   krnl.store [[CST_VALUE]], [[RES]][%arg0, %arg1, %arg2] : memref<3x4x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xf32>
}

// -----

func private @test_flatten0(%arg0 : tensor<2x3x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Flatten"(%arg0) {axis = 0 : si64} : (tensor<2x3x4xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
  // CHECK: [[MAP_FIRST:#.+]] = affine_map<() -> (0)>
  // CHECK: [[MAP_SECOND:#.+]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2 + d1 * s2 + d0 * (s1 * s2))>
  // CHECK-LABEL test_flatten0
  // CHECK:  [[ALLOC:%.+]] = alloc() : memref<1x24xf32>
  // CHECK:  [[LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK:  krnl.iterate([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2) with ([[LOOP]]#0 -> [[LOOPARG1:%.+]] = 0 to 2, [[LOOP]]#1 -> [[LOOPARG2:%.+]] = 0 to 3, [[LOOP]]#2 -> [[LOOPARG3:%.+]] = 0 to 4) {
  // CHECK:    [[LOAD:%.+]] = krnl.load %arg0{{\[}}[[LOOPARG1]], [[LOOPARG2]], [[LOOPARG3]]{{\]}} : memref<2x3x4xf32>
  // CHECK:    [[FIRSTDIM:%.+]] = affine.apply [[MAP_FIRST]]()
  // CHECK:    [[C0:%.+]] = constant 0 : index
  // CHECK:    [[R4:%.+]] = dim %arg0, [[C0]] : memref<2x3x4xf32>
  // CHECK:    [[C1:%.+]] = constant 1 : index
  // CHECK:    [[R5:%.+]] = dim %arg0, [[C1]] : memref<2x3x4xf32>
  // CHECK:    [[C2:%.+]] = constant 2 : index
  // CHECK:    [[R6:%.+]] = dim %arg0, [[C2]] : memref<2x3x4xf32>
  // CHECK:    [[SECONDDIM:%.+]] = affine.apply [[MAP_SECOND]]([[LOOPARG1]], [[LOOPARG2]], [[LOOPARG3]]){{\[}}[[R4]], [[R5]], [[R6]]{{\]}}
  // CHECK:    krnl.store [[LOAD]], [[ALLOC]]{{\[}}[[FIRSTDIM]], [[SECONDDIM]]{{\]}} : memref<1x24xf32>
}

// -----

// test partially known input shape
func private @test_flatten1(%arg0 : tensor<2x?x4xf32>) -> tensor<*xf32> {
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
  // CHECK:    [[R7:%.+]] = krnl.load %arg0{{\[}}[[ARG1]], [[ARG2]], [[ARG3]]{{\]}} : memref<2x?x4xf32>
  // CHECK:    [[C0_2:%.+]] = constant 0 : index
  // CHECK:    [[R8:%.+]] = dim %arg0, [[C0_2]] : memref<2x?x4xf32>
  // CHECK:    [[C1_3:%.+]] = constant 1 : index
  // CHECK:    [[R9:%.+]] = dim %arg0, [[C1_3]] : memref<2x?x4xf32>
  // CHECK:    [[R10:%.+]] = affine.apply [[MAP1]]([[ARG1]], [[ARG2]]){{\[}}[[R8]], [[R9]]{{\]}}
  // CHECK:    [[C2:%.+]] = constant 2 : index
  // CHECK:    [[R11:%.+]] = dim %arg0, [[C2]] : memref<2x?x4xf32>
  // CHECK:    [[R12:%.+]] = affine.apply [[MAP2]]([[ARG3]]){{\[}}[[R11]]{{\]}}
  // CHECK:    krnl.store [[R7]], [[R4]]{{\[}}[[R10]], [[R12]]{{\]}} : memref<?x4xf32>

}

// -----

func private @test_less(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xi1>
  return %0 : tensor<3x4x5xi1>

  // CHECK-LABEL: test_less
  // CHECK: [[RES:%.+]] = alloc() : memref<3x4x5xi1>
  // CHECK: [[DEF_LOOPS]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 3, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 4, [[DEF_LOOPS]]#2 -> %arg4 = 0 to 5) {
  // CHECK:   [[LHS:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:   [[RHS:%.+]] = krnl.load %arg1[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:   [[LESS:%.+]] = cmpf olt, [[LHS]], [[RHS]] : f32
  // CHECK:   krnl.store [[LESS]], [[RES]][%arg2, %arg3, %arg4] : memref<3x4x5xi1>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xi1>
}

// -----

func private @test_less_broadcast(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5xf32>) -> tensor<3x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<5xf32>) -> tensor<3x4x5xi1>
  return %0 : tensor<3x4x5xi1>

  // CHECK-LABEL: test_less_broadcast
  // CHECK: [[RES:%.+]] = alloc() : memref<3x4x5xi1>
  // CHECK: [[DEF_LOOPS]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 3, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 4, [[DEF_LOOPS]]#2 -> %arg4 = 0 to 5) {
  // CHECK:   [[LHS:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:   [[RHS:%.+]] = krnl.load %arg1[%arg4] : memref<5xf32>
  // CHECK:   [[LESS:%.+]] = cmpf olt, [[LHS]], [[RHS]] : f32
  // CHECK:   krnl.store [[LESS]], [[RES]][%arg2, %arg3, %arg4] : memref<3x4x5xi1>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xi1>
}

// -----

func private @test_floor(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Floor"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_floor
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[FLOOR:%.+]] = floorf [[LOAD]] : f32
  // CHECK: krnl.store [[FLOOR]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_ceil(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Ceil"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_ceil
  // CHECK: [[C0:%.+]] = constant 0 : index
  // CHECK: [[DIM_0:%.+]] = dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = alloc([[DIM_0]]) : memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = constant 0 : index
  // CHECK: [[DIM_2:%.+]] = dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10) {
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[CEIL:%.+]] = ceilf [[LOAD]] : f32
  // CHECK: krnl.store [[CEIL]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_clip(%arg0: tensor<3xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<3xf32> attributes {input_names = ["x", "min", "max"], output_names = ["y"]} {
  %0 = "onnx.Clip"(%arg0, %arg1, %arg2) : (tensor<3xf32>, tensor<f32>, tensor<f32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

// CHECK-LABEL: test_clip
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3xf32>, [[MIN_:%.+]]: memref<f32>, [[MAX_:%.+]]: memref<f32>) -> memref<3xf32> attributes {input_names = ["x", "min", "max"], output_names = ["y"]} {
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3) {
// CHECK-DAG:         [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[I_0_]]{{.}} : memref<3xf32>
// CHECK-DAG:         [[LOAD_MIN_MEM_:%.+]] = krnl.load [[MIN_]][] : memref<f32>
// CHECK:             [[VAR_4_:%.+]] = cmpf olt, [[LOAD_INPUT_MEM_]], [[LOAD_MIN_MEM_]] : f32
// CHECK-DAG:         [[VAR_5_:%.+]] = select [[VAR_4_]], [[LOAD_MIN_MEM_]], [[LOAD_INPUT_MEM_]] : f32
// CHECK-DAG:         [[LOAD_MAX_MEM_:%.+]] = krnl.load [[MAX_]][] : memref<f32>
// CHECK:             [[VAR_7_:%.+]] = cmpf olt, [[VAR_5_]], [[LOAD_MAX_MEM_]] : f32
// CHECK:             [[VAR_8_:%.+]] = select [[VAR_7_]], [[VAR_5_]], [[LOAD_MAX_MEM_]] : f32
// CHECK:             krnl.store [[VAR_8_]], [[RES_]]{{.}}[[I_0_]]{{.}} : memref<3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3xf32>
// CHECK:         }
}

// -----

func private @test_clip_default_min(%arg0: tensor<3xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<3xf32> attributes {input_names = ["x", "min", "max"], output_names = ["y"]} {
  %cst = constant unit
  %0 = "onnx.Clip"(%arg0, %cst, %arg2) : (tensor<3xf32>, none, tensor<f32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

// CHECK-LABEL: test_clip_default_min
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3xf32>, [[MIN_:%.+]]: memref<f32>, [[MAX_:%.+]]: memref<f32>) -> memref<3xf32> attributes {input_names = ["x", "min", "max"], output_names = ["y"]} {
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3) {
// CHECK-DAG:         [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[I_0_]]{{.}} : memref<3xf32>
// CHECK-DAG:         [[LOAD_MAX_MEM_:%.+]] = krnl.load [[MAX_]][] : memref<f32>
// CHECK:             [[VAR_7_:%.+]] = cmpf olt, [[LOAD_INPUT_MEM_]], [[LOAD_MAX_MEM_]] : f32
// CHECK:             [[VAR_8_:%.+]] = select [[VAR_7_]], [[LOAD_INPUT_MEM_]], [[LOAD_MAX_MEM_]] : f32
// CHECK:             krnl.store [[VAR_8_]], [[RES_]]{{.}}[[I_0_]]{{.}} : memref<3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3xf32>
// CHECK:         }
}

// -----

func private @test_pown(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> attributes {input_names = ["x", "y"], output_names = ["z"]} {
    %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
    return %0 : tensor<3x4x5xf32>
// CHECK-LABEL: test_pow
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x4x5xf32>, [[POWER_:%.+]]: memref<3x4x5xf32>) -> memref<3x4x5xf32> attributes {input_names = ["x", "y"], output_names = ["z"]} {
// CHECK-DAG:       [[RES_:%.+]] = alloc() : memref<3x4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5) {
// CHECK-DAG:         [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<3x4x5xf32>
// CHECK-DAG:         [[LOAD_POWER_MEM_:%.+]] = krnl.load [[POWER_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<3x4x5xf32>
// CHECK:             [[VAR_4_:%.+]] = powf [[LOAD_INPUT_MEM_]], [[LOAD_POWER_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<3x4x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x4x5xf32>
// CHECK:         }
}

// -----

// COM: Check float PRelu without broadcasting.
func @test_prelu_float(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_prelu_float
  // CHECK: [[RES:%.+]] = alloc() : memref<3x4x5xf32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5) {
  // CHECK:   [[LOAD_X:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:   [[LOAD_SLOPE:%.+]] = krnl.load %arg1[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:   [[CST_0:%.+]] = constant 0.000000e+00 : f32
  // CHECK:   [[LESS_THAN_ZERO:%.+]] = cmpf olt, [[LOAD_X]], [[CST_0]] : f32
  // CHECK:   [[MUL:%.+]] = mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:   [[SELECT:%.+]] = select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:   krnl.store [[SELECT]], [[RES]][%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xf32>
}

// -----

// COM: Check int PRelu without broadcasting.
func @test_prelu_int(%arg0: tensor<3x4x5xi32>, %arg1: tensor<3x4x5xi32>) -> tensor<*xi32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xi32>, tensor<3x4x5xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>

  // CHECK-LABEL: func @test_prelu_int
  // CHECK: [[RES:%.+]] = alloc() : memref<3x4x5xi32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5) {
  // CHECK:   [[LOAD_X:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xi32>
  // CHECK:   [[LOAD_SLOPE:%.+]] = krnl.load %arg1[%arg2, %arg3, %arg4] : memref<3x4x5xi32>
  // CHECK:   [[CST_0:%.+]] = constant 0 : i32
  // CHECK:   [[LESS_THAN_ZERO:%.+]] = cmpi slt, [[LOAD_X]], [[CST_0]] : i32
  // CHECK:   [[MUL:%.+]] = muli [[LOAD_SLOPE]], [[LOAD_X]] : i32
  // CHECK:   [[SELECT:%.+]] = select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : i32
  // CHECK:   krnl.store [[SELECT]], [[RES]][%arg2, %arg3, %arg4] : memref<3x4x5xi32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xi32>
}


// -----

// COM: Check PRelu with unidirectional broadcasting.
func @test_prelu_broadcast1(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_prelu_broadcast1
  // CHECK: [[RES:%.+]] = alloc() : memref<3x4x5xf32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5) {
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1[%arg4] : memref<5xf32>
  // CHECK-DAG:   [[CST_0:%.+]] = constant 0.000000e+00 : f32
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = cmpf olt, [[LOAD_X]], [[CST_0]] : f32
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT:%.+]] = select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:       krnl.store [[SELECT]], [[RES]][%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xf32>
}

// -----

// COM: Check PRelu with unidirectional broadcasting.
// COM: Tensor slope should be unidirectional broadcastable to input tensor X
func @test_prelu_broadcast2(%arg0: tensor<3x4x5xf32>, %arg1: tensor<1x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<1x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_prelu_broadcast2
  // CHECK: [[RES:%.+]] = alloc() : memref<3x4x5xf32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5) {
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK-DAG:   [[ZERO_INDEX:%.+]] = constant 0 : index
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], %arg4] : memref<1x5xf32>
  // CHECK-DAG:   [[CST_0:%.+]] = constant 0.000000e+00 : f32
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = cmpf olt, [[LOAD_X]], [[CST_0]] : f32
  // CHECK:       [[MUL:%.+]] = mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT:%.+]] = select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:       krnl.store [[SELECT]], [[RES]][%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xf32>
}

// -----
// COM: Check simple loop lowering.
func private @test_loop_simple_main_graph(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) -> tensor<1xi64> {
  %0 = "onnx.Loop"(%arg0, %arg1, %arg2) ({
    ^bb0(%body_arg0: tensor<i64>, %body_arg1: tensor<i1>, %body_arg2: tensor<1xi64>):
    %0 = "onnx.Identity"(%body_arg1) : (tensor<i1>) -> tensor<i1>
    %1 = "onnx.Add"(%body_arg2, %body_arg0) : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
    onnx.Return %0, %1 : tensor<i1>, tensor<1xi64>
  }) : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> tensor<1xi64>
  return %0 : tensor<1xi64>
  // CHECK:       module  {
  // CHECK-LABEL:       func private @test_loop_simple_main_graph
  // CHECK-SAME:     ([[TRIP_COUNT:%.+]]: memref<i64>, [[COND:%.+]]: memref<i1>, [[Y_INIT:%.+]]: memref<1xi64>) -> memref<1xi64> {
  // CHECK:           [[COND_GLOBAL:%.+]] = alloc() : memref<i1>
  // CHECK:           [[Y:%.+]] = alloc() : memref<1xi64>
  // CHECK:           [[Y_COPY_LOOP:%.+]] = krnl.define_loops 1
  // CHECK:           krnl.iterate([[Y_COPY_LOOP]]) with ([[Y_COPY_LOOP]] -> [[YCOPY_IV:%.+]] = 0 to 1) {
  // CHECK:             [[Y_VAL:%.+]] = krnl.load [[Y_INIT]]{{.}}[[YCOPY_IV]]{{.}} : memref<1xi64>
  // CHECK:             krnl.store [[Y_VAL]], [[Y]]{{.}}[[YCOPY_IV]]{{.}} : memref<1xi64>
  // CHECK:           }
  // CHECK:           [[COND_VAL:%.+]] = krnl.load [[COND]][] : memref<i1>
  // CHECK:           krnl.store [[COND_VAL]], [[COND_GLOBAL]][] : memref<i1>
  // CHECK:           [[LOOP:%.+]] = krnl.define_loops 1
  // CHECK:           [[TRIP_COUNT_VAL:%.+]] = krnl.load [[TRIP_COUNT]][] : memref<i64>
  // CHECK:           [[TRIP_COUNT_IDX:%.+]] = index_cast [[TRIP_COUNT_VAL]] : i64 to index
  // CHECK:           krnl.iterate([[LOOP]]) with ([[LOOP]] -> [[LOOP_IV:%.+]] = 0 to [[TRIP_COUNT_IDX]]) {
  // CHECK:             [[COND_VAL:%.+]] = krnl.load [[COND_GLOBAL]][] : memref<i1>
  // CHECK:             scf.if [[COND_VAL]] {
  // CHECK:               [[Y_CURR:%.+]] = alloc() : memref<1xi64>
  // CHECK:               [[LOOP_IV_VAL:%.+]] = index_cast [[LOOP_IV]] : index to i64
  // CHECK:               [[CURR_LOOP_IV:%.+]] = alloc() : memref<i64>
  // CHECK:               krnl.store [[LOOP_IV_VAL]], [[CURR_LOOP_IV]][] : memref<i64>
  // CHECK:               [[Y_COMPUTE_LOOP:%.+]] = krnl.define_loops 1
  // CHECK:               krnl.iterate([[Y_COMPUTE_LOOP]]) with ([[Y_COMPUTE_LOOP]] -> [[Y_COMPUTE_IV:%.+]] = 0 to 1) {
  // CHECK:                 [[Y_VAL:%.+]] = krnl.load [[Y]]{{.}}[[Y_COMPUTE_IV]]{{.}} : memref<1xi64>
  // CHECK:                 [[LOO_IV_VAL:%.+]] = krnl.load [[CURR_LOOP_IV]][] : memref<i64>
  // CHECK:                 [[NEW_Y_VAL:%.+]] = addi [[Y_VAL]], [[LOO_IV_VAL]] : i64
  // CHECK:                 krnl.store [[NEW_Y_VAL]], [[Y_CURR]]{{.}}[[Y_COMPUTE_IV]]{{.}} : memref<1xi64>
  // CHECK:               }
  // CHECK:               [[COND_CAST:%.+]] = krnl.dummy_cast [[COND]] : (memref<i1>) -> memref<i1>
  // CHECK:               [[Y_CURR_CAST:%.+]] = krnl.dummy_cast [[Y_CURR]] : (memref<1xi64>) -> memref<1xi64>
  // CHECK:               [[COND_CAST_VAL:%.+]] = krnl.load [[COND_CAST]][] : memref<i1>
  // CHECK:               krnl.store [[COND_CAST_VAL]], [[COND_GLOBAL]][] : memref<i1>
  // CHECK:               [[Y_COPY_LOOP:%.+]] = krnl.define_loops 1
  // CHECK:               krnl.iterate([[Y_COPY_LOOP]]) with ([[Y_COPY_LOOP]] -> [[Y_COPY_IV:%.+]] = 0 to 1) {
  // CHECK:                 [[Y_SCAN_VAL:%.+]] = krnl.load [[Y_CURR_CAST]]{{.}}[[Y_COPY_IV]]{{.}} : memref<1xi64>
  // CHECK:                 krnl.store [[Y_SCAN_VAL]], [[Y]]{{.}}[[Y_COPY_IV]]{{.}} : memref<1xi64>
  // CHECK:               }
  // CHECK:               dealloc [[Y_CURR]] : memref<1xi64>
  // CHECK:             }
  // CHECK:           }
  // CHECK:           dealloc [[COND_GLOBAL]] : memref<i1>
  // CHECK:           return [[Y]] : memref<1xi64>
  // CHECK:         }
  // CHECK:       }
}

