// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// TODO: Remove test_no_argument_1 from the test - empty function body is no longer
// supported in mlir: https://reviews.llvm.org/D91886
func private @test_no_argument_2() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value =  dense<[[1.000000e+0, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>} : () -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_no_argument_2
// CHECK: [[GLOBAL:%.+]] = "{{.*}}"({{.*}}) {{.*}} : ({{.*}}) -> memref<2x2xf32>
// CHECK: return [[GLOBAL]] : memref<2x2xf32>

// -----

func private @test_elementwise_op_with_scalar_values_1(%arg0 : tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Exp"(%arg0) : (tensor<f32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_elementwise_op_with_scalar_values_1
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<f32>
  // CHECK: [[EXP:%.+]] = math.exp [[LOAD]] : f32
  // CHECK: krnl.store [[EXP]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func private @test_elementwise_op_with_scalar_values_2(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_elementwise_op_with_scalar_values_2
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<f32>
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[] : memref<f32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[] : memref<f32>
  // CHECK: [[ADD:%.+]] = arith.addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[ADD]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func private @test_elementwise_op_with_scalar_values_3(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_elementwise_op_with_scalar_values_3
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<f32>
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[] : memref<f32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[] : memref<f32>
  // CHECK: [[ADD1:%.+]] = arith.addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[LOAD3:%.+]] = krnl.load %arg2[] : memref<f32>
  // CHECK: [[ADD2:%.+]] = arith.addf [[ADD1]], [[LOAD3]] : f32
  // CHECK: krnl.store [[ADD2]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func private @test_add(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_add
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[ADDF:%.+]] = arith.addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[ADDF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func private @test_mul(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_mul
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MULF:%.+]] = arith.mulf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[MULF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func private @test_div(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_div
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[DIVF:%.+]] = arith.divf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[DIVF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func private @test_sub(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sub
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[SUBF:%.+]] = arith.subf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[SUBF]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func private @test_and(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<*xi1> {
  %0 = "onnx.And"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  "std.return"(%0) : (tensor<*xi1>) -> ()

  // CHECK-LABEL: test_and
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xi1>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[AND:%.+]] = arith.andi [[LOAD1]], [[LOAD2]] : i1
  // CHECK: krnl.store [[AND]], [[RES]][%arg2, %arg3] : memref<10x10xi1>
  // CHECK: return [[RES]] : memref<10x10xi1>
}

// -----

func private @test_or(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<*xi1> {
  %0 = "onnx.Or"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  "std.return"(%0) : (tensor<*xi1>) -> ()

  // CHECK-LABEL: test_or
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xi1>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[OR:%.+]] = arith.ori [[LOAD1]], [[LOAD2]] : i1
  // CHECK: krnl.store [[OR]], [[RES]][%arg2, %arg3] : memref<10x10xi1>
  // CHECK: return [[RES]] : memref<10x10xi1>
}

// -----

func private @test_xor(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<*xi1> {
  %0 = "onnx.Xor"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  "std.return"(%0) : (tensor<*xi1>) -> ()

  // CHECK-LABEL: test_xor
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xi1>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xi1>
  // CHECK: [[XOR:%.+]] = arith.xori [[LOAD1]], [[LOAD2]] : i1
  // CHECK: krnl.store [[XOR]], [[RES]][%arg2, %arg3] : memref<10x10xi1>
  // CHECK: return [[RES]] : memref<10x10xi1>
}

// -----

func private @test_exp(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Exp"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_exp
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[EXP:%.+]] = math.exp [[LOAD]] : f32
  // CHECK: krnl.store [[EXP]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_tanh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Tanh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func private @test_tanh
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
  // CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
  // CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x10xf32>
  // CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x10xf32>
  // CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
  // CHECK-DAG:       [[VAR_c0_0_:%.+]] = arith.constant 0 : index
  // CHECK:           [[VAR_3_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_0_]] : memref<?x10xf32>
  // CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_3_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
  // CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x10xf32>
  // CHECK:             [[VAR_5_:%.+]] = math.tanh [[LOAD_PARAM_0_MEM_]] : f32
  // CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<?x10xf32>
  // CHECK:           }
  // CHECK:           return [[RES_]] : memref<?x10xf32>
  // CHECK:         }
}

// -----

func private @test_sinh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sinh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sinh
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = arith.constant {{0.+}} : f32
  // CHECK: [[TWO:%.+]] = arith.constant {{2.+}} : f32
  // CHECK: [[NLOAD:%.+]] = arith.subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[EXP:%.+]] = math.exp [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = math.exp [[NLOAD]] : f32
  // CHECK: [[DIVIDEND:%.+]] = arith.subf [[EXP]], [[NEXP]] : f32
  // CHECK: [[SINH_RES:%.+]] = arith.divf [[DIVIDEND]], [[TWO]] : f32
  // CHECK: krnl.store [[SINH_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_cosh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Cosh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_cosh
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = arith.constant {{0.+}} : f32
  // CHECK: [[TWO:%.+]] = arith.constant {{2.+}} : f32
  // CHECK: [[NLOAD:%.+]] = arith.subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[EXP:%.+]] = math.exp [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = math.exp [[NLOAD]] : f32
  // CHECK: [[DIVIDEND:%.+]] = arith.addf [[EXP]], [[NEXP]] : f32
  // CHECK: [[COSH_RES:%.+]] = arith.divf [[DIVIDEND]], [[TWO]] : f32
  // CHECK: krnl.store [[COSH_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_cos(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Cos"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_cos
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[COS:%.+]] = math.cos [[LOAD]] : f32
  // CHECK: krnl.store [[COS]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_sin(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sin"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sin
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[SIN:%.+]] = math.sin [[LOAD]] : f32
  // CHECK: krnl.store [[SIN]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_log(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Log"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_log
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[LOG:%.+]] = math.log [[LOAD]] : f32
  // CHECK: krnl.store [[LOG]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_sigmoid(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sigmoid
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = arith.constant {{0.+}} : f32
  // CHECK: [[ONE:%.+]] = arith.constant {{1.+}} : f32
  // CHECK: [[NLOAD:%.+]] = arith.subf [[ZERO]], [[LOAD]] : f32
  // CHECK: [[NEXP:%.+]] = math.exp [[NLOAD]] : f32
  // CHECK: [[DIVISOR:%.+]] = arith.addf [[ONE]], [[NEXP]] : f32
  // CHECK: [[SIGMOID_RES:%.+]] = arith.divf [[ONE]], [[DIVISOR]] : f32
  // CHECK: krnl.store [[SIGMOID_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_relu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_relu
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = arith.constant {{0.+}} : f32
  // CHECK: [[LTZERO:%.+]] = arith.cmpf olt, [[LOAD]], [[ZERO]] : f32
  // CHECK: [[RELU_RES:%.+]] = arith.select [[LTZERO]], [[ZERO]], [[LOAD]] : f32
  // CHECK: krnl.store [[RELU_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_reshape_constant(%arg0 : tensor<1x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[2, 5]> : tensor<2xi64> } : () -> tensor<2xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<1x10xf32>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()
// CHECK-LABEL:     test_reshape_constant
// CHECK: krnl.global
// CHECK: [[RES:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [2, 5], strides: [5, 1] : memref<1x10xf32> to memref<2x5xf32>
// CHECK: return [[RES]] : memref<2x5xf32>
}

// -----

// `Reshape` ops are lowerd to `reinterpret_cast` op. `reinterpret_cast` ops just change
// the view of input memref. So, input memref should not be deallocated if it is retuned.
// This test confirms the deallocation is not generated.

func private @test_reshape_constant_dealloc(%arg0 : tensor<10x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<10x1xf32>) -> tensor<*xf32>
  %1 = "onnx.Constant"() {value = dense<[2, 5]> : tensor<2xi64> } : () -> tensor<2xi64>
  %2 = "onnx.Reshape"(%0, %1) : (tensor<*xf32>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%2) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: func private @test_reshape_constant_dealloc
  // CHECK:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<1x10xf32>
  // CHECK:       [[VAR_3_:%.+]] = memref.reinterpret_cast [[VAR_0_]] to offset: [0], sizes: [2, 5], strides: [5, 1] : memref<1x10xf32> to memref<2x5xf32>
  // CHECK-NOT:   memref.dealloc [[VAR_0_]] : memref<1x10xf32>
  // CHECK:       return [[VAR_3_]] : memref<2x5xf32>
}

// -----

func private @test_sum(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sum
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[ADD:%.+]] = arith.addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[ADD]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func private @test_max(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Max"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_max
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MAX:%.+]] = arith.cmpf ogt, [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[RELU_RES:%.+]] = arith.select [[MAX]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[RELU_RES]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func private @test_min(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Min"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_min
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[%arg2, %arg3] : memref<10x10xf32>
  // CHECK: [[MIN:%.+]] = arith.cmpf olt, [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[RELU_RES:%.+]] = arith.select [[MIN]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[RELU_RES]], [[RES]][%arg2, %arg3] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func private @test_elu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Elu"(%arg0) {alpha=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_elu
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = arith.constant {{0.+}} : f32
  // CHECK: [[ONE:%.+]] = arith.constant {{1.+}} : f32
  // CHECK: [[ALPHA:%.+]] = arith.constant {{2.+}} : f32
  // CHECK: [[EXP:%.+]] = math.exp [[LOAD]] : f32
  // CHECK: [[CMP:%.+]] = arith.cmpf olt, [[LOAD]], [[ZERO]] : f32
  // CHECK: [[SUB:%.+]] = arith.subf [[EXP]], [[ONE]] : f32
  // CHECK: [[MUL:%.+]] = arith.mulf [[ALPHA]], [[SUB]] : f32
  // CHECK: [[SELECT:%.+]] = arith.select [[CMP]], [[MUL]], [[LOAD]] : f32
  // CHECK: krnl.store [[SELECT]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_leakyrelu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha=1.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_leakyrelu
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = arith.constant {{0.+}} : f32
  // CHECK: [[ALPHA:%.+]] = arith.constant {{1.+}} : f32
  // CHECK: [[CMP:%.+]] = arith.cmpf olt, [[LOAD]], [[ZERO]] : f32
  // CHECK: [[MUL:%.+]] = arith.mulf [[ALPHA]], [[LOAD]] : f32
  // CHECK: [[SELECT:%.+]] = arith.select [[CMP]], [[MUL]], [[LOAD]] : f32
  // CHECK: krnl.store [[SELECT]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_selu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Selu"(%arg0) {alpha=1.0:f32, gamma=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_selu
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = arith.constant {{0.+}} : f32
  // CHECK: [[ALPHA:%.+]] = arith.constant {{1.+}} : f32
  // CHECK: [[GAMMA:%.+]] = arith.constant {{2.+}} : f32
  // CHECK: [[EXP:%.+]] = math.exp [[LOAD]] : f32
  // CHECK: [[CMP:%.+]] = arith.cmpf ogt, [[LOAD]], [[ZERO]] : f32
  // CHECK: [[MUL:%.+]] = arith.mulf [[ALPHA]], [[EXP]] : f32
  // CHECK: [[SUB:%.+]] = arith.subf [[MUL]], [[ALPHA]] : f32
  // CHECK: [[SELECT:%.+]] = arith.select [[CMP]], [[LOAD]], [[SUB]] : f32
  // CHECK: [[SELU_RES:%.+]] = arith.mulf [[GAMMA]], [[SELECT]] : f32
  // CHECK: krnl.store [[SELU_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_hardsigmoid(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha=1.0:f32, beta=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_hardsigmoid
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = arith.constant {{0.+}} : f32
  // CHECK: [[ONE:%.+]] = arith.constant {{1.+}} : f32
  // CHECK: [[ALPHA:%.+]] = arith.constant {{1.+}} : f32
  // CHECK: [[BETA:%.+]] = arith.constant {{2.+}} : f32
  // CHECK: [[MUL:%.+]] = arith.mulf [[ALPHA]], [[LOAD]] : f32
  // CHECK: [[ADD:%.+]] = arith.addf [[MUL]], [[BETA]] : f32
  // CHECK: [[CMP1:%.+]] = arith.cmpf ogt, [[ADD]], [[ZERO]] : f32
  // CHECK: [[SELECT1:%.+]] = arith.select [[CMP1]], [[ADD]], [[ZERO]] : f32
  // CHECK: [[CMP2:%.+]] = arith.cmpf olt, [[SELECT1]], [[ONE]] : f32
  // CHECK: [[SELECT2:%.+]] = arith.select [[CMP2]], [[SELECT1]], [[ONE]] : f32
  // CHECK: krnl.store [[SELECT2]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_reciprocal(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Reciprocal"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reciprocal
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ONE:%.+]] = arith.constant {{1.+}} : f32
  // CHECK: [[RECIPROCAL_RES:%.+]] = arith.divf [[ONE]], [[LOAD]] : f32
  // CHECK: krnl.store [[RECIPROCAL_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_softplus(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softplus"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_softplus
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[EXP:%.+]] = math.exp [[LOAD]] : f32
  // CHECK: [[ONE:%.+]] = arith.constant {{1.+}} : f32
  // CHECK: [[ADD:%.+]] = arith.addf [[EXP]], [[ONE]] : f32
  // CHECK: [[SOFTPLUS_RES:%.+]] = math.log [[ADD]] : f32
  // CHECK: krnl.store [[SOFTPLUS_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_softsign(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softsign"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_softsign
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ABS:%.+]] = math.abs [[LOAD]] : f32
  // CHECK: [[ONE:%.+]] = arith.constant {{1.+}} : f32
  // CHECK: [[ADD:%.+]] = arith.addf [[ABS]], [[ONE]] : f32
  // CHECK: [[SOFTSIGN_RES:%.+]] = arith.divf [[LOAD]], [[ADD]] : f32
  // CHECK: krnl.store [[SOFTSIGN_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_reducemax(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMax"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducemax
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2){
  // CHECK: [[IDENTITY:%.+]] = arith.constant 0xFF800000 : f32
  // CHECK: krnl.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[CMP:%.+]] = arith.cmpf ogt, [[LOAD2]], [[LOAD1]] : f32
  // CHECK: [[SELECT:%.+]] = arith.select [[CMP]], [[LOAD2]], [[LOAD1]] : f32
  // CHECK: krnl.store [[SELECT]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func private @test_reducemin(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMin"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducemin
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2){
  // CHECK: [[IDENTITY:%.+]] = arith.constant 0x7F800000 : f32
  // CHECK: krnl.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[CMP:%.+]] = arith.cmpf olt, [[LOAD2]], [[LOAD1]] : f32
  // CHECK: [[SELECT:%.+]] = arith.select [[CMP]], [[LOAD2]], [[LOAD1]] : f32
  // CHECK: krnl.store [[SELECT]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func private @test_reduceprod(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceProd"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reduceprod
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2){
  // CHECK: [[IDENTITY:%.+]] = arith.constant 1.000000e+00 : f32
  // CHECK: krnl.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[REDUCE:%.+]] = arith.mulf [[LOAD2]], [[LOAD1]] : f32
  // CHECK: krnl.store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func private @test_reducesum(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %cst = "onnx.Constant"() {value = dense<[1]> : tensor<1xi64> } : () -> tensor<1xi64>
  %0 ="onnx.ReduceSum"(%arg0, %cst) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<1xi64>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducesum
  // CHECK: [[GLOBAL:%.+]] = "krnl.global"() {name = {{.*}}, shape = [1], value = dense<1> : tensor<1xi64>} : () -> memref<1xi64>
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2){
  // CHECK: [[IDENTITY:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: krnl.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[REDUCE:%.+]] = arith.addf [[LOAD2]], [[LOAD1]] : f32
  // CHECK: krnl.store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// -----

func private @test_reducesumV11(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceSumV11"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_reducesumV11
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
  // CHECK: [[DEF_LOOPS1:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 2){
  // CHECK: [[IDENTITY:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: krnl.store [[IDENTITY]], [[RES]][%arg1, %arg2] : memref<3x2xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2) with ([[DEF_LOOPS2]]#0 -> %arg1 = 0 to 3, [[DEF_LOOPS2]]#1 -> %arg2 = 0 to 2, [[DEF_LOOPS2]]#2 -> %arg3 = 0 to 2){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<3x2x2xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: [[REDUCE:%.+]] = arith.addf [[LOAD2]], [[LOAD1]] : f32
  // CHECK: krnl.store [[REDUCE]], [[RES]][%arg1, %arg3] : memref<3x2xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x2xf32>
}

// --
  func private @test_reducesum1(%arg0: tensor<3x2x2xf32>, %arg1: tensor<?xi64>) -> tensor<3x1x2xf32> {
    %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<3x2x2xf32>, tensor<?xi64>) -> tensor<3x1x2xf32>
    return %0 : tensor<3x1x2xf32>
// CHECK-LABEL:       @test_reducesum1
// CHECK-SAME:     ([[VAR_arg0:%.+]]: memref<3x2x2xf32>, [[VAR_arg1:%.+]]: memref<?xi64>) -> memref<3x1x2xf32> {
// CHECK:           [[VAR_1:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3xi1>
// CHECK:           [[VAR_false:%.+]] = arith.constant false
// CHECK:           [[VAR_true:%.+]] = arith.constant true
// CHECK:           [[VAR_c1:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_c0:%.+]] = arith.constant 0 : index
// CHECK:           krnl.store [[VAR_false]], [[VAR_1]]{{.}}[[VAR_c0]]{{.}} : memref<3xi1>
// CHECK:           [[VAR_c1_0:%.+]] = arith.constant 1 : index
// CHECK:           krnl.store [[VAR_false]], [[VAR_1]]{{.}}[[VAR_c1_0]]{{.}} : memref<3xi1>
// CHECK:           [[VAR_c2:%.+]] = arith.constant 2 : index
// CHECK:           krnl.store [[VAR_false]], [[VAR_1]]{{.}}[[VAR_c2]]{{.}} : memref<3xi1>
// CHECK:           [[VAR_c3_i64:%.+]] = arith.constant 3 : i64
// CHECK:           [[VAR_c0_i64:%.+]] = arith.constant 0 : i64
// CHECK:           [[VAR_2:%.+]] = krnl.define_loops 1
// CHECK:           [[VAR_c0_1:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_3:%.+]] = memref.dim [[VAR_arg1]], [[VAR_c0_1]] : memref<?xi64>
// CHECK:           krnl.iterate([[VAR_2]]) with ([[VAR_2]] -> [[VAR_arg2:%.+]] = 0 to [[VAR_3]]){
// CHECK:             [[VAR_6:%.+]] = krnl.load [[VAR_arg1]]{{.}}[[VAR_arg2]]{{.}} : memref<?xi64>
// CHECK:             [[VAR_7:%.+]] = arith.cmpi slt, [[VAR_6]], [[VAR_c0_i64]] : i64
// CHECK:             [[VAR_8:%.+]] = arith.addi [[VAR_6]], [[VAR_c3_i64]] : i64
// CHECK:             [[VAR_9:%.+]] = arith.select [[VAR_7]], [[VAR_8]], [[VAR_6]] : i64
// CHECK:             [[VAR_10:%.+]] = arith.index_cast [[VAR_9]] : i64 to index
// CHECK:             krnl.store [[VAR_true]], [[VAR_1]]{{.}}[[VAR_10]]{{.}} : memref<3xi1>
// CHECK:           }
// CHECK:           [[VAR_0:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x1x2xf32>
// CHECK:           [[VAR_4:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[VAR_4]]#0, [[VAR_4]]#1, [[VAR_4]]#2) with ([[VAR_4]]#0 -> [[VAR_arg2:%.+]] = 0 to 3, [[VAR_4]]#1 -> [[VAR_arg3:%.+]] = 0 to 1, [[VAR_4]]#2 -> [[VAR_arg4:%.+]] = 0 to 2){
// CHECK:             [[VAR_cst:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:             krnl.store [[VAR_cst]], [[VAR_0]]{{.}}[[VAR_arg2]], [[VAR_arg3]], [[VAR_arg4]]{{.}} : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           [[VAR_5:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[VAR_5]]#0, [[VAR_5]]#1, [[VAR_5]]#2) with ([[VAR_5]]#0 -> [[VAR_arg2:%.+]] = 0 to 3, [[VAR_5]]#1 -> [[VAR_arg3:%.+]] = 0 to 2, [[VAR_5]]#2 -> [[VAR_arg4:%.+]] = 0 to 2){
// CHECK:             [[VAR_c0_2:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_c0_3:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_6:%.+]] = krnl.load [[VAR_1]]{{.}}[[VAR_c0_3]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7:%.+]] = arith.cmpi eq, [[VAR_6]], [[VAR_true]] : i1
// CHECK:             [[VAR_8:%.+]] = arith.select [[VAR_7]], [[VAR_c0_2]], [[VAR_arg2]] : index
// CHECK:             [[VAR_c1_4:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_9:%.+]] = krnl.load [[VAR_1]]{{.}}[[VAR_c1_4]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_10:%.+]] = arith.cmpi eq, [[VAR_9]], [[VAR_true]] : i1
// CHECK:             [[VAR_11:%.+]] = arith.select [[VAR_10]], [[VAR_c0_2]], [[VAR_arg3]] : index
// CHECK:             [[VAR_c2_5:%.+]] = arith.constant 2 : index
// CHECK:             [[VAR_12:%.+]] = krnl.load [[VAR_1]]{{.}}[[VAR_c2_5]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_13:%.+]] = arith.cmpi eq, [[VAR_12]], [[VAR_true]] : i1
// CHECK:             [[VAR_14:%.+]] = arith.select [[VAR_13]], [[VAR_c0_2]], [[VAR_arg4]] : index
// CHECK:             [[VAR_15:%.+]] = krnl.load [[VAR_arg0]]{{.}}[[VAR_arg2]], [[VAR_arg3]], [[VAR_arg4]]{{.}} : memref<3x2x2xf32>
// CHECK:             [[VAR_16:%.+]] = krnl.load [[VAR_0]]{{.}}[[VAR_8]], [[VAR_11]], [[VAR_14]]{{.}} : memref<3x1x2xf32>
// CHECK:             [[VAR_17:%.+]] = arith.addf [[VAR_16]], [[VAR_15]] : f32
// CHECK:             krnl.store [[VAR_17]], [[VAR_0]]{{.}}[[VAR_8]], [[VAR_11]], [[VAR_14]]{{.}} : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           return [[VAR_0]] : memref<3x1x2xf32>
// CHECK:         }
}

// -----

  func @test_reducesum2(%arg0: tensor<3x2x2xf32>, %arg1: tensor<?xi64>) -> tensor<3x1x2xf32> {
    %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<?xi64>) -> tensor<3x1x2xf32>
    return %0 : tensor<3x1x2xf32>
// CHECK-LABEL:     @test_reducesum2
// CHECK-SAME:     ([[VAR_arg0:%.+]]: memref<3x2x2xf32>, [[VAR_arg1:%.+]]: memref<?xi64>) -> memref<3x1x2xf32> {
// CHECK:           [[VAR_1:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3xi1>
// CHECK:           [[VAR_false:%.+]] = arith.constant false
// CHECK:           [[VAR_true:%.+]] = arith.constant true
// CHECK:           [[VAR_c1:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_c0:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_c0_0:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_2:%.+]] = memref.dim [[VAR_arg1]], [[VAR_c0_0]] : memref<?xi64>
// CHECK:           [[VAR_3:%.+]] = arith.cmpi eq, [[VAR_2]], [[VAR_c0]] : index
// CHECK:           [[VAR_4:%.+]] = arith.select [[VAR_3]], [[VAR_true]], [[VAR_false]] : i1
// CHECK:           [[VAR_c0_1:%.+]] = arith.constant 0 : index
// CHECK:           krnl.store [[VAR_4]], [[VAR_1]]{{.}}[[VAR_c0_1]]{{.}} : memref<3xi1>
// CHECK:           [[VAR_c1_2:%.+]] = arith.constant 1 : index
// CHECK:           krnl.store [[VAR_4]], [[VAR_1]]{{.}}[[VAR_c1_2]]{{.}} : memref<3xi1>
// CHECK:           [[VAR_c2:%.+]] = arith.constant 2 : index
// CHECK:           krnl.store [[VAR_4]], [[VAR_1]]{{.}}[[VAR_c2]]{{.}} : memref<3xi1>
// CHECK:           [[VAR_c3_i64:%.+]] = arith.constant 3 : i64
// CHECK:           [[VAR_c0_i64:%.+]] = arith.constant 0 : i64
// CHECK:           [[VAR_5:%.+]] = krnl.define_loops 1
// CHECK:           [[VAR_c0_3:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_6:%.+]] = memref.dim [[VAR_arg1]], [[VAR_c0_3]] : memref<?xi64>
// CHECK:           krnl.iterate([[VAR_5]]) with ([[VAR_5]] -> [[VAR_arg2:%.+]] = 0 to [[VAR_6]]){
// CHECK:             [[VAR_9:%.+]] = krnl.load [[VAR_arg1]]{{.}}[[VAR_arg2]]{{.}} : memref<?xi64>
// CHECK:             [[VAR_10:%.+]] = arith.cmpi slt, [[VAR_9]], [[VAR_c0_i64]] : i64
// CHECK:             [[VAR_11:%.+]] = arith.addi [[VAR_9]], [[VAR_c3_i64]] : i64
// CHECK:             [[VAR_12:%.+]] = arith.select [[VAR_10]], [[VAR_11]], [[VAR_9]] : i64
// CHECK:             [[VAR_13:%.+]] = arith.index_cast [[VAR_12]] : i64 to index
// CHECK:             krnl.store [[VAR_true]], [[VAR_1]]{{.}}[[VAR_13]]{{.}} : memref<3xi1>
// CHECK:           }
// CHECK:           [[VAR_0:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x1x2xf32>
// CHECK:           [[VAR_7:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[VAR_7]]#0, [[VAR_7]]#1, [[VAR_7]]#2) with ([[VAR_7]]#0 -> [[VAR_arg2:%.+]] = 0 to 3, [[VAR_7]]#1 -> [[VAR_arg3:%.+]] = 0 to 1, [[VAR_7]]#2 -> [[VAR_arg4:%.+]] = 0 to 2){
// CHECK:             [[VAR_cst:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:             krnl.store [[VAR_cst]], [[VAR_0]]{{.}}[[VAR_arg2]], [[VAR_arg3]], [[VAR_arg4]]{{.}} : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           [[VAR_8:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[VAR_8]]#0, [[VAR_8]]#1, [[VAR_8]]#2) with ([[VAR_8]]#0 -> [[VAR_arg2:%.+]] = 0 to 3, [[VAR_8]]#1 -> [[VAR_arg3:%.+]] = 0 to 2, [[VAR_8]]#2 -> [[VAR_arg4:%.+]] = 0 to 2){
// CHECK:             [[VAR_c0_4:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_c0_5:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_9:%.+]] = krnl.load [[VAR_1]]{{.}}[[VAR_c0_5]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_10:%.+]] = arith.cmpi eq, [[VAR_9]], [[VAR_true]] : i1
// CHECK:             [[VAR_11:%.+]] = arith.select [[VAR_10]], [[VAR_c0_4]], [[VAR_arg2]] : index
// CHECK:             [[VAR_c1_6:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_12:%.+]] = krnl.load [[VAR_1]]{{.}}[[VAR_c1_6]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_13:%.+]] = arith.cmpi eq, [[VAR_12]], [[VAR_true]] : i1
// CHECK:             [[VAR_14:%.+]] = arith.select [[VAR_13]], [[VAR_c0_4]], [[VAR_arg3]] : index
// CHECK:             [[VAR_c2_7:%.+]] = arith.constant 2 : index
// CHECK:             [[VAR_15:%.+]] = krnl.load [[VAR_1]]{{.}}[[VAR_c2_7]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_16:%.+]] = arith.cmpi eq, [[VAR_15]], [[VAR_true]] : i1
// CHECK:             [[VAR_17:%.+]] = arith.select [[VAR_16]], [[VAR_c0_4]], [[VAR_arg4]] : index
// CHECK:             [[VAR_18:%.+]] = krnl.load [[VAR_arg0]]{{.}}[[VAR_arg2]], [[VAR_arg3]], [[VAR_arg4]]{{.}} : memref<3x2x2xf32>
// CHECK:             [[VAR_19:%.+]] = krnl.load [[VAR_0]]{{.}}[[VAR_11]], [[VAR_14]], [[VAR_17]]{{.}} : memref<3x1x2xf32>
// CHECK:             [[VAR_20:%.+]] = arith.addf [[VAR_19]], [[VAR_18]] : f32
// CHECK:             krnl.store [[VAR_20]], [[VAR_0]]{{.}}[[VAR_11]], [[VAR_14]], [[VAR_17]]{{.}} : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           return [[VAR_0]] : memref<3x1x2xf32>

}

// -----

func private @test_sqrt(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sqrt
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[SQRT:%.+]] = math.sqrt [[LOAD]] : f32
  // CHECK: krnl.store [[SQRT]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_unsqueeze(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[0, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_unsqueeze
  // CHECK: [[RES:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 10, 10, 1], strides: [100, 10, 1, 1] : memref<10x10xf32> to memref<1x10x10x1xf32>
  // CHECK: return [[RES]] : memref<1x10x10x1xf32>
}

// -----

func private @test_unsqueezev11(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.UnsqueezeV11"(%arg0) {axes=[0,3]} : (tensor<10x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_unsqueezev11
  // CHECK: [[RES:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 10, 10, 1], strides: [100, 10, 1, 1] : memref<10x10xf32> to memref<1x10x10x1xf32>
  // CHECK: return [[RES]] : memref<1x10x10x1xf32>
}

// -----

// `UnsqueezeV11` ops are lowerd to `reinterpret_cast` op. `reinterpret_cast` ops just
// change the view of input memref. So, input memref should not be deallocated if it
// is retuned. This test confirms the deallocation is not generated.

func private @test_unsqueeze_dealloc(%arg0 : tensor<10x20xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[0, 3]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Transpose"(%arg0) : (tensor<10x20xf32>) -> tensor<*xf32>
  %2 = "onnx.Unsqueeze"(%1, %0) : (tensor<*xf32>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%2) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: func private @test_unsqueeze_dealloc
  // CHECK:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<20x10xf32>
  // CHECK:       [[VAR_2_:%.+]] = memref.reinterpret_cast [[VAR_0_]] to offset: [0], sizes: [1, 20, 10, 1], strides: [200, 10, 1, 1] : memref<20x10xf32> to memref<1x20x10x1xf32>
  // CHECK-NOT:   memref.dealloc [[VAR_0_]] : memref<20x10xf32>
  // CHECK:	  return [[VAR_2_]] : memref<1x20x10x1xf32>
}

// -----

func private @test_unsqueezev11_dealloc(%arg0 : tensor<10x20xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<10x20xf32>) -> tensor<*xf32>
  %1 = "onnx.UnsqueezeV11"(%0) {axes=[0,3]} : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: func private @test_unsqueezev11_dealloc
  // CHECK:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<20x10xf32>
  // CHECK:       [[VAR_2_:%.+]] = memref.reinterpret_cast [[VAR_0_]] to offset: [0], sizes: [1, 20, 10, 1], strides: [200, 10, 1, 1] : memref<20x10xf32> to memref<1x20x10x1xf32>
  // CHECK-NOT:   memref.dealloc [[VAR_0_]] : memref<20x10xf32>
  // CHECK:	  return [[VAR_2_]] : memref<1x20x10x1xf32>
}

// -----

// Test for multiple `reinterpret_cast` in a function. Only returned memrefs should not be deallocated.
func private @test_unsqueeze_squeeze_dealloc(%arg0 : tensor<10x20xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -1]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Constant"() {value = dense<[1, 2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %2 = "onnx.Transpose"(%arg0) : (tensor<10x20xf32>) -> tensor<*xf32>
  %3 = "onnx.Unsqueeze"(%2, %0) : (tensor<*xf32>, tensor<2xi64>) -> tensor<*xf32>
  %4 = "onnx.Transpose"(%3) {perm = [0, 3, 1, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  %5 = "onnx.Squeeze"(%4, %1) : (tensor<*xf32>, tensor<2xi64>) -> tensor<*xf32>
  "std.return"(%5) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: func private @test_unsqueeze_squeeze_dealloc
  // CHECK:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<20x10xf32>
  // CHECK:       [[VAR_3_:%.+]] = memref.reinterpret_cast [[VAR_1_]] to offset: [0], sizes: [20, 1, 10, 1], strides: [10, 10, 1, 1] : memref<20x10xf32> to memref<20x1x10x1xf32>
  // CHECK:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<20x1x1x10xf32>
  // CHECK:       [[VAR_5_:%.+]] = memref.reinterpret_cast [[VAR_0_]] to offset: [0], sizes: [20, 10], strides: [10, 1] : memref<20x1x1x10xf32> to memref<20x10xf32>
  // CHECK:       return [[VAR_5_]] : memref<20x10xf32>
}

// -----

func private @test_unsqueezev11_squeezev11_dealloc(%arg0 : tensor<10x20xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<10x20xf32>) -> tensor<*xf32>
  %1 = "onnx.UnsqueezeV11"(%0) { axes = [1, -1]} : (tensor<*xf32>) -> (tensor<*xf32>)
  %2 = "onnx.Transpose"(%1) {perm = [0, 3, 1, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "onnx.SqueezeV11"(%2) {axes=[1, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%3) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: func private @test_unsqueezev11_squeezev11_dealloc
  // CHECK:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<20x10xf32>
  // CHECK:       [[VAR_3_:%.+]] = memref.reinterpret_cast [[VAR_1_]] to offset: [0], sizes: [20, 1, 10, 1], strides: [10, 10, 1, 1] : memref<20x10xf32> to memref<20x1x10x1xf32>
  // CHECK:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<20x1x1x10xf32>
  // CHECK:       [[VAR_5_:%.+]] = memref.reinterpret_cast [[VAR_0_]] to offset: [0], sizes: [20, 10], strides: [10, 1] : memref<20x1x1x10xf32> to memref<20x10xf32>
  // CHECK:       return [[VAR_5_]] : memref<20x10xf32>
}

// -----

func private @test_transpose(%arg0 : tensor<10x20x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<10x20x30x40xf32>) -> tensor<*xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_transpose
  // CHECK: [[RES1:%.+]] = memref.alloc() {{.*}}: memref<40x30x20x10xf32>

  // CHECK: [[DEF_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg2 = 0 to 20, [[DEF_LOOPS]]#2 -> %arg3 = 0 to 30, [[DEF_LOOPS]]#3 -> %arg4 = 0 to 40){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3, %arg4] : memref<10x20x30x40xf32>
  // CHECK: krnl.store [[LOAD]], [[RES1]][%arg4, %arg3, %arg2, %arg1] : memref<40x30x20x10xf32>

  // CHECK: [[RES0:%.+]] = memref.alloc() {{.*}}: memref<40x10x30x20xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to 40, [[DEF_LOOPS]]#1 -> %arg2 = 0 to 30, [[DEF_LOOPS]]#2 -> %arg3 = 0 to 20, [[DEF_LOOPS]]#3 -> %arg4 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load [[RES1]][%arg1, %arg2, %arg3, %arg4] : memref<40x30x20x10xf32>
  // CHECK: krnl.store [[LOAD]], [[RES0]][%arg1, %arg4, %arg2, %arg3] : memref<40x10x30x20xf32>

  // CHECK: return [[RES0]] : memref<40x10x30x20xf32>
}

// -----

// COM: Test whether the lowering is correct in the presence of dynamic dimensions.
func private @test_transpose_dynamic_dims(%arg0 : tensor<10x?x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<10x?x30x40xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL:  func private @test_transpose_dynamic_dims
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x?x30x40xf32>) -> memref<10x40x?x30xf32> {
  // CHECK:           [[CST_1_:%.+]] = arith.constant 1 : index
  // CHECK:           [[DIM_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<10x?x30x40xf32>
  // CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[DIM_0_]]) {{.*}}: memref<10x40x?x30xf32>
  // CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
  // CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
  // CHECK:           [[DIM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_1_]] : memref<10x?x30x40xf32>
  // CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[DIM_1_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 30, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 40){
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
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ZERO:%.+]] = arith.constant {{0.+}} : f32
  // CHECK: [[ONE:%.+]] = arith.constant {{1.+}} : f32
  // CHECK: [[MINUS_ONE:%.+]] = arith.constant {{-1.+}} : f32
  // CHECK: [[GTZERO:%.+]] = arith.cmpf ogt, [[LOAD]], [[ZERO]] : f32
  // CHECK: [[SELECT_PLUS:%.+]] = arith.select [[GTZERO]], [[ONE]], [[MINUS_ONE]] : f32
  // CHECK: [[EQZERO:%.+]] = arith.cmpf oeq, [[LOAD]], [[ZERO]] : f32
  // CHECK: [[SIGN_RES:%.+]] = arith.select [[EQZERO]], [[ZERO]], [[SELECT_PLUS]] : f32
  // CHECK: krnl.store [[SIGN_RES]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_sign_i(%arg0 : tensor<?x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.Sign"(%arg0) : (tensor<?x10xi32>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_sign_i
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xi32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xi32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xi32>
  // CHECK: [[ZERO:%.+]] = arith.constant 0 : i32
  // CHECK: [[ONE:%.+]] = arith.constant 1 : i32
  // CHECK: [[MINUS_ONE:%.+]] = arith.constant -1 : i32
  // CHECK: [[GTZERO:%.+]] = arith.cmpi sgt, [[LOAD]], [[ZERO]] : i32
  // CHECK: [[SELECT_PLUS:%.+]] = arith.select [[GTZERO]], [[ONE]], [[MINUS_ONE]] : i32
  // CHECK: [[EQZERO:%.+]] = arith.cmpi eq, [[LOAD]], [[ZERO]] : i32
  // CHECK: [[SIGN_RES:%.+]] = arith.select [[EQZERO]], [[ZERO]], [[SELECT_PLUS]] : i32
  // CHECK: krnl.store [[SIGN_RES]], [[RES]][%arg1, %arg2] : memref<?x10xi32>
  // CHECK: return [[RES]] : memref<?x10xi32>
}

// -----

func private @test_batchnorm_testmode_Nd(%arg0: tensor<1x2x1x3xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> tensor<1x2x1x3xf32> {
  %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<1x2x1x3xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>, tensor<2xf32>) -> tensor<1x2x1x3xf32>
  return %0 : tensor<1x2x1x3xf32>

  // CHECK-LABEL: test_batchnorm_testmode_Nd
  // CHECK-DAG: [[RES:%.+]] = memref.alloc() {{.*}}: memref<1x2x1x3xf32>
  // CHECK-DAG: [[EPSILON:%.+]] = arith.constant 9.99999974E-6 : f32
  // CHECK: [[DEF_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg5 = 0 to 2){
  // CHECK:   [[SCALE:%.+]] = krnl.load %arg1[%arg5] : memref<2xf32>
  // CHECK:   [[BIAS:%.+]] = krnl.load %arg2[%arg5] : memref<2xf32>
  // CHECK:   [[MEAN:%.+]] = krnl.load %arg3[%arg5] : memref<2xf32>
  // CHECK:   [[VARIANCE:%.+]] = krnl.load %arg4[%arg5] : memref<2xf32>
  // CHECK:   krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) with ([[DEF_LOOPS]]#0 -> %arg6 = 0 to 1, [[DEF_LOOPS]]#2 -> %arg7 = 0 to 1, [[DEF_LOOPS]]#3 -> %arg8 = 0 to 3){
  // CHECK:     [[LOADED_VAL:%.+]] = krnl.load %arg0[%arg6, %arg5, %arg7, %arg8] : memref<1x2x1x3xf32>
  // CHECK:     [[DIVIDEND:%.+]] = arith.subf [[LOADED_VAL]], [[MEAN]] : f32
  // CHECK:     [[ADJUSTED_VARIANCE:%.+]] = arith.addf [[VARIANCE]], [[EPSILON]] : f32
  // CHECK:     [[DIVISOR:%.+]] = math.sqrt [[ADJUSTED_VARIANCE]] : f32
  // CHECK:     [[NORM:%.+]] = arith.divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK:     [[SCALE_NORM:%.+]] = arith.mulf [[SCALE]], [[NORM]] : f32
  // CHECK:     [[SHIFT_SCALE_NORM:%.+]] = arith.addf [[SCALE_NORM]], [[BIAS]] : f32
  // CHECK:     krnl.store [[SHIFT_SCALE_NORM]], [[RES]][%arg6, %arg5, %arg7, %arg8] : memref<1x2x1x3xf32>
  // CHECK:   }
  // CHECK: }
  // CHECK: return [[RES]] : memref<1x2x1x3xf32>
}

// -----

func private @test_batchnorm_testmode_1d(%arg0: tensor<10xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<10xf32> {
  %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<10xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>

  // CHECK-LABEL: test_batchnorm_testmode_1d
  // CHECK: [[EPSILON:%.+]] = arith.constant 9.99999974E-6 : f32
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10xf32>
  // CHECK: [[DEF_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: %[[ZERO_INDEX:.+]] = arith.constant 0 : index
  // CHECK: [[SCALE:%.+]] = krnl.load %arg1[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: [[BIAS:%.+]] = krnl.load %arg2[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: [[MEAN:%.+]] = krnl.load %arg3[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: [[VARIANCE:%.+]] = krnl.load %arg4[%[[ZERO_INDEX]]] : memref<1xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]) with ([[DEF_LOOPS]] -> %arg5 = 0 to 10){
  // CHECK:   [[LOADED_VAL:%.+]] = krnl.load %arg0[%arg5] : memref<10xf32>
  // CHECK:   [[DIVIDEND:%.+]] = arith.subf [[LOADED_VAL]], [[MEAN]] : f32
  // CHECK:   [[ADJUSTED_VARIANCE:%.+]] = arith.addf [[VARIANCE]], [[EPSILON]] : f32
  // CHECK:   [[DIVISOR:%.+]] = math.sqrt [[ADJUSTED_VARIANCE]] : f32
  // CHECK:   [[NORM:%.+]] = arith.divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK:   [[SCALE_NORM:%.+]] = arith.mulf [[SCALE]], [[NORM]] : f32
  // CHECK:   [[SHIFT_SCALE_NORM:%.+]] = arith.addf [[SCALE_NORM]], [[BIAS]] : f32
  // CHECK:   krnl.store [[SHIFT_SCALE_NORM]], [[RES]][%arg5] : memref<10xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<10xf32>
}

// -----

func private @test_batchnorm_testmode_2d(%arg0: tensor<10x3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>) -> tensor<10x3xf32> {
  %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<10x3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<10x3xf32>
  return %0 : tensor<10x3xf32>

  // CHECK-LABEL: test_batchnorm_testmode_2d
  // CHECK: [[EPSILON:%.+]] = arith.constant 9.99999974E-6 : f32
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x3xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#1 -> %arg5 = 0 to 3){
  // CHECK:   [[SCALE:%.+]] = krnl.load %arg1[%arg5] : memref<3xf32>
  // CHECK:   [[BIAS:%.+]] = krnl.load %arg2[%arg5] : memref<3xf32>
  // CHECK:   [[MEAN:%.+]] = krnl.load %arg3[%arg5] : memref<3xf32>
  // CHECK:   [[VARIANCE:%.+]] = krnl.load %arg4[%arg5] : memref<3xf32>
  // CHECK:   krnl.iterate([[DEF_LOOPS]]#0) with ([[DEF_LOOPS]]#0 -> %arg6 = 0 to 10){
  // CHECK:     [[LOADED_VAL:%.+]] = krnl.load %arg0[%arg6, %arg5] : memref<10x3xf32>
  // CHECK:     [[DIVIDEND:%.+]] = arith.subf [[LOADED_VAL]], [[MEAN]] : f32
  // CHECK:     [[ADJUSTED_VARIANCE:%.+]] = arith.addf [[VARIANCE]], [[EPSILON]] : f32
  // CHECK:     [[DIVISOR:%.+]] = math.sqrt [[ADJUSTED_VARIANCE]] : f32
  // CHECK:     [[NORM:%.+]] = arith.divf [[DIVIDEND]], [[DIVISOR]] : f32
  // CHECK:     [[SCALE_NORM:%.+]] = arith.mulf [[SCALE]], [[NORM]] : f32
  // CHECK:     [[SHIFT_SCALE_NORM:%.+]] = arith.addf [[SCALE_NORM]], [[BIAS]] : f32
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
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[ABS:%.+]] = math.abs [[LOAD]] : f32
  // CHECK: krnl.store [[ABS]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_abs_int(%arg0 : tensor<?x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.Abs"(%arg0) : (tensor<?x10xi32>) -> tensor<*xi32>
  "std.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_abs_int
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xi32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xi32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xi32>
  // CHECK: [[ZERO:%.+]] = arith.constant 0 : i32
  // CHECK: [[LESS_THAN_ZERO:%.+]] = arith.cmpi slt, [[LOAD]], [[ZERO]] : i32
  // CHECK: [[NEGATIVE_LOAD:%.+]] = arith.subi [[ZERO]], [[LOAD]] : i32
  // CHECK: [[SELECT:%.+]] = arith.select [[LESS_THAN_ZERO]], [[NEGATIVE_LOAD]], [[LOAD]] : i32
  // CHECK: krnl.store [[SELECT]], [[RES]][%arg1, %arg2] : memref<?x10xi32>
  // CHECK: return [[RES]] : memref<?x10xi32>
}

// -----

func private @test_constant_dense_2d_value(%arg0: tensor<1xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[[0.0, 0.0], [1.0, 1.1], [2.0, 2.1]]> : tensor<3x2xf32>} : () -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_constant_dense_2d_value
  // CHECK: [[GLOBAL:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3, 2], value = dense<{{.*}}[0.000000e+00, 0.000000e+00], [1.000000e+00, 1.100000e+00], [2.000000e+00, 2.100000e+00]{{.*}}> : tensor<3x2xf32>} : () -> memref<3x2xf32>
  // CHECK: return [[GLOBAL]] : memref<3x2xf32>
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

  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = arith.constant 0.000000e+00 : f32

  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31){

  // CHECK:   [[REDUCTION_VAL:%.+]] = memref.alloca() : memref<f32>
  // CHECK:   krnl.store [[IDENTITY]], [[REDUCTION_VAL]][] : memref<f32>

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #[[BOUND]](%arg3)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #[[BOUND]](%arg4)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]){
  // CHECK:     {{.*}} = krnl.load %arg0[%arg1, %arg2, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     {{.*}} = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:     krnl.store {{.*}}, [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   }

  // CHECK:   [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   krnl.store [[LOAD_REDUCTION]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK: }
}

// -----

func private @test_averagepool_identity_value(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_averagepool_identity_value
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: [[REDUCTION_VAL:%.+]] = memref.alloca() : memref<f32>
  // CHECK: krnl.store [[IDENTITY]], [[REDUCTION_VAL]][] : memref<f32>
}

// -----

func private @test_maxpool_identity_value(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_maxpool_identity_value
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = arith.constant 0xFF800000 : f32
  // CHECK: [[REDUCTION_VAL:%.+]] = memref.alloca() : memref<f32>
  // CHECK: krnl.store [[IDENTITY]], [[REDUCTION_VAL]][] : memref<f32>
}

// -----

func private @test_averagepool_pooling_operation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_averagepool_pooling_operation
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<1x3x31x31xf32>

  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31){

  // CHECK:   [[REDUCTION_VAL:%.+]] = memref.alloca() : memref<f32>
  // CHECK:   krnl.store {{.*}}, [[REDUCTION_VAL]][] : memref<f32>

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #{{.*}}(%arg3)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #{{.*}}(%arg4)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]){

  // CHECK:     [[INPUT_LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     [[OUTPUT_LOAD:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:     [[SUM:%.+]] = arith.addf [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     krnl.store [[SUM]], [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   }
  // CHECK:   [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   krnl.store [[LOAD_REDUCTION]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>

  // CHECK:   [[NUMERATOR:%.+]] = krnl.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK:   [[AVERAGE:%.+]] = arith.divf [[NUMERATOR]], {{.*}} : f32
  // CHECK:   krnl.store [[AVERAGE]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK: }
}

// -----

func private @test_maxpool_pooling_operation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_maxpool_pooling_operation
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<1x3x31x31xf32>

  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31){

  // CHECK:   [[REDUCTION_VAL:%.+]] = memref.alloca() : memref<f32>
  // CHECK:   krnl.store {{.*}}, [[REDUCTION_VAL]][] : memref<f32>

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #{{.*}}(%arg3)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #{{.*}}(%arg4)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]){

  // CHECK:     [[INPUT_LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     [[OUTPUT_LOAD:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:     [[GREATER:%.+]] = arith.cmpf ogt, [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     [[SELECT:%.+]] = arith.select [[GREATER]], [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     krnl.store [[SELECT]], [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   }
  // CHECK:   [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   krnl.store [[LOAD_REDUCTION]], [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>

  // CHECK-NOT:   {{.*}} = krnl.load [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK-NOT:   krnl.store {{.*}}, [[RES]][%arg1, %arg2, %arg3, %arg4] : memref<1x3x31x31xf32>
  // CHECK: }
}

// -----

func private @test_squeeze(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<16x1x32x1x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_squeeze
  // CHECK: [[RES:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 32, 64], strides: [2048, 64, 1] : memref<16x1x32x1x64xf32> to memref<16x32x64xf32>
  // CHECK: return [[RES]] : memref<16x32x64xf32>
}

// -----

func private @test_squeezev11(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) { axes = [1, -2]} : (tensor<16x1x32x1x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_squeezev11
  // CHECK: [[RES:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 32, 64], strides: [2048, 64, 1] : memref<16x1x32x1x64xf32> to memref<16x32x64xf32>
  // CHECK: return [[RES]] : memref<16x32x64xf32>
}

// -----

// `SqueezeV11` ops are lowerd to `reinterpret_cast` op. `reinterpret_cast` ops just change the view of input memref. So, input memref should not be deallocated if it is retuned. This test confirms the deallocation is not generated.

func private @test_squeeze_dealloc(%arg0 : tensor<16x32x1x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2, 4]} : (tensor<16x32x1x1x64xf32>) -> tensor<*xf32>
  %2 = "onnx.Squeeze"(%1, %0) : (tensor<*xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "std.return"(%2) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func private @test_squeeze_dealloc
  // CHECK:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<16x1x32x1x64xf32>
// CHECK-DAG:     [[VAR_2_:%.+]] = memref.reinterpret_cast [[VAR_0_]] to offset: [0], sizes: [16, 32, 64], strides: [2048, 64, 1] : memref<16x1x32x1x64xf32> to memref<16x32x64xf32>
  // CHECK-NOT:   memref.dealloc [[VAR_0_]] : memref<20x10xf32>
  // CHECK:       return [[VAR_2_]] : memref<16x32x64xf32>
}

// -----

func private @test_squeezev11_dealloc(%arg0 : tensor<16x32x1x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2, 4]} : (tensor<16x32x1x1x64xf32>) -> tensor<*xf32>
  %1 = "onnx.SqueezeV11"(%0) { axes = [1, -2]} : (tensor<*xf32>) -> (tensor<*xf32>)
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func private @test_squeezev11_dealloc
  // CHECK:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<16x1x32x1x64xf32>
// CHECK-DAG:     [[VAR_2_:%.+]] = memref.reinterpret_cast [[VAR_0_]] to offset: [0], sizes: [16, 32, 64], strides: [2048, 64, 1] : memref<16x1x32x1x64xf32> to memref<16x32x64xf32>
  // CHECK-NOT:   memref.dealloc [[VAR_0_]] : memref<20x10xf32>
  // CHECK:       return [[VAR_2_]] : memref<16x32x64xf32>
}

// -----

func private @test_squeeze_unknown_dimensions(%arg0 : tensor<?x1x32x?x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[1, -2]> : tensor<2xi64>} : () -> tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<?x1x32x?x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func private @test_squeeze_unknown_dimensions
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x32x?x64xf32>) -> memref<?x32x64xf32> {
  // CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
  // CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x1x32x?x64xf32>
  // CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
  // CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
  // CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
  // CHECK-DAG:       [[CST_64_1_:%.+]] = arith.constant 64 : index
  // CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : index
  // CHECK:           [[VAR_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: {{.}}[[VAR_0_]], 32, 64], strides: [2048, 64, 1] : memref<?x1x32x?x64xf32> to memref<?x32x64xf32>
  // CHECK:           return [[VAR_1_]] : memref<?x32x64xf32>
  // CHECK:         }
}

// -----

func private @test_squeezev11_unknown_dimensions(%arg0 : tensor<?x1x32x?x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) { axes = [1,-2]} : (tensor<?x1x32x?x64xf32>) -> (tensor<*xf32>)
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL:  func private @test_squeezev11_unknown_dimensions
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x32x?x64xf32>) -> memref<?x32x64xf32> {
  // CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
  // CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x1x32x?x64xf32>
  // CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
  // CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : index
  // CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
  // CHECK-DAG:       [[CST_64_1_:%.+]] = arith.constant 64 : index
  // CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : index
  // CHECK:           [[VAR_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: {{.}}[[VAR_0_]], 32, 64], strides: [2048, 64, 1] : memref<?x1x32x?x64xf32> to memref<?x32x64xf32>
  // CHECK:           return [[VAR_1_]] : memref<?x32x64xf32>
  // CHECK:         }
}

// -----

func private @test_split_equal(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = 0 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 8)>
  // CHECK-LABEL: @test_split_equal

  // CHECK: [[RES_0:%.+]] = memref.alloc() {{.*}}: memref<8x32x64xf32>
  // CHECK: [[RES_1:%.+]] = memref.alloc() {{.*}}: memref<8x32x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 8, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[LOAD_0:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_0]], [[RES_0]][%arg1, %arg2, %arg3] : memref<8x32x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 8, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64){
  // CHECK:   %[[INDEX:.+]] = affine.apply [[INDEX_MAP]](%arg1)
  // CHECK:   [[LOAD_1:%.+]] = krnl.load %arg0[%[[INDEX]], %arg2, %arg3] : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_1]], [[RES_1]][%arg1, %arg2, %arg3] : memref<8x32x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<8x32x64xf32>, memref<8x32x64xf32>
}

// -----

func private @test_split_variable(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %split = "onnx.Constant"() {value = dense<[2, 30]> : tensor<2xi64>} : () -> tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) { axis = 1 : si64} : (tensor<16x32x64xf32>, tensor<2xi64>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 2)>
  // CHECK-LABEL: @test_split_variable

  // CHECK: [[RES_0:%.+]] = memref.alloc() {{.*}}: memref<16x2x64xf32>
  // CHECK: [[RES_1:%.+]] = memref.alloc() {{.*}}: memref<16x30x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 2, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[LOAD_0:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_0]], [[RES_0]][%arg1, %arg2, %arg3] : memref<16x2x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 30, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64){
  // CHECK:   %[[INDEX:.+]] = affine.apply [[INDEX_MAP]](%arg2)
  // CHECK:   [[LOAD_1:%.+]] = krnl.load %arg0[%arg1, %[[INDEX]], %arg3] : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_1]], [[RES_1]][%arg1, %arg2, %arg3] : memref<16x30x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<16x2x64xf32>, memref<16x30x64xf32>
}

// -----

func private @test_splitv11_equal(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.SplitV11"(%arg0) { axis = 0 : si64} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 8)>
  // CHECK-LABEL: @test_splitv11_equal

  // CHECK: [[RES_0:%.+]] = memref.alloc() {{.*}}: memref<8x32x64xf32>
  // CHECK: [[RES_1:%.+]] = memref.alloc() {{.*}}: memref<8x32x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 8, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[LOAD_0:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_0]], [[RES_0]][%arg1, %arg2, %arg3] : memref<8x32x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 8, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64){
  // CHECK:   %[[INDEX:.+]] = affine.apply [[INDEX_MAP]](%arg1)
  // CHECK:   [[LOAD_1:%.+]] = krnl.load %arg0[%[[INDEX]], %arg2, %arg3] : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_1]], [[RES_1]][%arg1, %arg2, %arg3] : memref<8x32x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<8x32x64xf32>, memref<8x32x64xf32>
}

// -----

func private @test_splitv11_variable(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.SplitV11"(%arg0) { axis = 1 : si64, split = [2, 30]} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "std.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 2)>
  // CHECK-LABEL: @test_splitv11_variable

  // CHECK: [[RES_0:%.+]] = memref.alloc() {{.*}}: memref<16x2x64xf32>
  // CHECK: [[RES_1:%.+]] = memref.alloc() {{.*}}: memref<16x30x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 2, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[LOAD_0:%.+]] = krnl.load %arg0[%arg1, %arg2, %arg3] : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_0]], [[RES_0]][%arg1, %arg2, %arg3] : memref<16x2x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 30, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64){
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
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<f32>
  // CHECK: krnl.store [[LOAD]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func private @cast_lowering_intfloat(%arg0: tensor<i64>) -> tensor<f32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<i64>) -> tensor<f32>
  "std.return"(%0) : (tensor<f32>) -> ()

  // CHECK-LABEL: cast_lowering_intfloat
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<i64>
  // CHECK: [[VAL:%.+]] = arith.sitofp [[LOAD]] : i64 to f32
  // CHECK: krnl.store [[VAL]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func private @cast_lowering_floatint(%arg0: tensor<f32>) -> tensor<i64> {
  %0 = "onnx.Cast"(%arg0) {to = i64} : (tensor<f32>) -> tensor<i64>
  "std.return"(%0) : (tensor<i64>) -> ()

  // CHECK-LABEL: cast_lowering_floatint
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<i64>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<f32>
  // CHECK: [[VAL:%.+]] = arith.fptosi [[LOAD]] : f32 to i64
  // CHECK: krnl.store [[VAL]], [[RES]][] : memref<i64>
  // CHECK: return [[RES]] : memref<i64>
}

// -----

func private @cast_lowering_f16f32(%arg0: tensor<f16>) -> tensor<f32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<f16>) -> tensor<f32>
  "std.return"(%0) : (tensor<f32>) -> ()

  // CHECK-LABEL: cast_lowering_f16f32
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<f16>
  // CHECK: [[VAL:%.+]] = arith.extf [[LOAD]] : f16 to f32
  // CHECK: krnl.store [[VAL]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func private @cast_lowering_f64f32(%arg0: tensor<f64>) -> tensor<f32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<f64>) -> tensor<f32>
  "std.return"(%0) : (tensor<f32>) -> ()

  // CHECK-LABEL: cast_lowering_f64f32
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<f64>
  // CHECK: [[VAL:%.+]] = arith.truncf [[LOAD]] : f64 to f32
  // CHECK: krnl.store [[VAL]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func private @cast_lowering_f64f32_10(%arg0: tensor<10xf64>) -> tensor<*xf32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<10xf64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: cast_lowering_f64f32_10
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10xf32>
  // CHECK: [[DEF_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[DEF_LOOPS]]) with ([[DEF_LOOPS]] -> %arg1 = 0 to 10){
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[%arg1] : memref<10xf64>
  // CHECK: [[FPTRUNC:%.+]] = arith.truncf [[LOAD1]] : f64 to f32
  // CHECK: krnl.store [[FPTRUNC]], [[RES]][%arg1] : memref<10xf32>
  // CHECK: return [[RES]] : memref<10xf32>
}

// -----

func private @cast_lowering_int_wider_int(%arg0: tensor<i32>) -> tensor<i64> {
  %0 = "onnx.Cast"(%arg0) {to = i64} : (tensor<i32>) -> tensor<i64>
  "std.return"(%0) : (tensor<i64>) -> ()

  // CHECK-LABEL: cast_lowering_int_wider_int
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<i64>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<i32>
  // CHECK: [[CAST:%.+]] = arith.extsi [[LOAD]] : i32 to i64
  // CHECK: krnl.store [[CAST]], [[RES]][] : memref<i64>
  // CHECK: return [[RES]] : memref<i64>
}

// -----

func private @cast_lowering_int_narrow_int(%arg0: tensor<i64>) -> tensor<i32> {
  %0 = "onnx.Cast"(%arg0) {to = i32 } : (tensor<i64>) -> tensor<i32>
  "std.return"(%0) : (tensor<i32>) -> ()

  // CHECK-LABEL: cast_lowering_int_narrow_int
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<i32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<i64>
  // CHECK: [[CAST:%.+]] = arith.trunci [[LOAD]] : i64 to i32
  // CHECK: krnl.store [[CAST]], [[RES]][] : memref<i32>
  // CHECK: return [[RES]] : memref<i32>
}

// -----

func private @cast_lowering_int_to_bool(%arg0: tensor<i64>) -> tensor<i1> {
  %0 = "onnx.Cast"(%arg0) {to = i1 } : (tensor<i64>) -> tensor<i1>
  "std.return"(%0) : (tensor<i1>) -> ()

// CHECK-LABEL:  func private @cast_lowering_int_to_bool
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<i64>) -> memref<i1> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<i1>
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][] : memref<i64>
// CHECK-DAG:       [[VAR_c0_i64_:%.+]] = arith.constant 0 : i64
// CHECK:           [[VAR_2_:%.+]] = arith.cmpi ne, [[LOAD_PARAM_0_MEM_]], [[VAR_c0_i64_]] : i64
// CHECK:           krnl.store [[VAR_2_]], [[RES_]][] : memref<i1>
// CHECK:           return [[RES_]] : memref<i1>
// CHECK:         }
}

// -----

func private @cast_lowering_uint_to_bool(%arg0 : tensor<ui8>) -> tensor<i1> {
  %0 = "onnx.Cast"(%arg0) {to = i1} : (tensor<ui8>) -> tensor<i1>
  "std.return"(%0): (tensor<i1>) -> ()

// CHECK-LABEL:  func private @cast_lowering_uint_to_bool
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<ui8>) -> memref<i1> {  
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<i1>
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][] : memref<ui8>  
// CHECK-DAG:       [[LOAD_PARAM_UCC:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_:%.+]] : ui8 to i8
// CHECK-DAG:       [[VAR_c0_i8_:%.+]] = arith.constant 0 : i8
// CHECK:           [[VAR_2_:%.+]] = arith.cmpi ne, [[LOAD_PARAM_UCC]], [[VAR_c0_i8_]] : i8
// CHECK:           krnl.store [[VAR_2_]], [[RES_]][] : memref<i1>
// CHECK:           return [[RES_]] : memref<i1>
// CHECK:         }
}

// -----

func private @cast_lowering_float_to_bool(%arg0: tensor<f32>) -> tensor<i1> {
  %0 = "onnx.Cast"(%arg0) {to = i1 } : (tensor<f32>) -> tensor<i1>
  "std.return"(%0) : (tensor<i1>) -> ()

// CHECK-LABEL:  func private @cast_lowering_float_to_bool
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<f32>) -> memref<i1> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<i1>
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][] : memref<f32>
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK:           [[VAR_2_:%.+]] = arith.cmpf one, [[LOAD_PARAM_0_MEM_]], [[VAR_cst_]] : f32
// CHECK:           krnl.store [[VAR_2_]], [[RES_]][] : memref<i1>
// CHECK:           return [[RES_]] : memref<i1>
// CHECK:         }
}

// -----

func private @cast_lowering_uint_narrow_uint(%arg0: tensor<ui64>) -> tensor<ui32> {
  %0 = "onnx.Cast"(%arg0) {to = ui32 } : (tensor<ui64>) -> tensor<ui32>
  "std.return"(%0) : (tensor<ui32>) -> ()
// CHECK-LABEL:  func private @cast_lowering_uint_narrow_uint
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<ui64>) -> memref<ui32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<ui32>
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][] : memref<ui64>
// CHECK:           [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_]] : ui64 to i64
// CHECK:           [[VAR_3_:%.+]] = arith.trunci [[VAR_2_]] : i64 to i32
// CHECK:           [[VAR_4_:%.+]] = builtin.unrealized_conversion_cast [[VAR_3_]] : i32 to ui32
// CHECK:           krnl.store [[VAR_4_]], [[RES_]][] : memref<ui32>
// CHECK:           return [[RES_]] : memref<ui32>
// CHECK:         }
}

// -----

func private @cast_lowering_uint_wider_uint(%arg0: tensor<ui32>) -> tensor<ui64> {
  %0 = "onnx.Cast"(%arg0) {to = ui64 } : (tensor<ui32>) -> tensor<ui64>
  "std.return"(%0) : (tensor<ui64>) -> ()
// CHECK-LABEL:  func private @cast_lowering_uint_wider_uint
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<ui32>) -> memref<ui64> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<ui64>
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][] : memref<ui32>
// CHECK:           [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_]] : ui32 to i32
// CHECK:           [[VAR_3_:%.+]] = arith.extui [[VAR_2_]] : i32 to i64
// CHECK:           [[VAR_4_:%.+]] = builtin.unrealized_conversion_cast [[VAR_3_]] : i64 to ui64
// CHECK:           krnl.store [[VAR_4_]], [[RES_]][] : memref<ui64>
// CHECK:           return [[RES_]] : memref<ui64>
// CHECK:         }
}

// -----

func private @test_size_known(%arg0: tensor<2x2xf32>) -> tensor<i64> {
  %1 = "onnx.Size"(%arg0) : (tensor<2x2xf32>) -> tensor<i64>
  "std.return"(%1) : (tensor<i64>) -> ()

  // CHECK-LABEL: test_size_known
  // CHECK:      [[RES:%.+]] = memref.alloc() {{.*}}: memref<i64>
  // CHECK-NEXT  [[SIZE:%.+]] = arith.constant 4 : i64
  // CHECK-NEXT  krnl.store [[SIZE]], [[RES]][] : memref<i64>
  // CHECK-NEXT  return [[RES]] : memref<i64>

}

// -----

func private @test_size_unknown(%arg0 : tensor<?x2x?xf32>) -> tensor<i64> {

  // CHECK-LABEL: test_size_unknown
  // CHECK:       [[RES:%.+]] = memref.alloc() {{.*}}: memref<i64>
  // CHECK-NEXT:  [[INIT:%.+]] = arith.constant 2 : i64
  // CHECK-NEXT:  [[IND1:%.+]] = arith.constant 0 : index
  // CHECK-NEXT:  [[DIM1:%.+]] = memref.dim %arg0, [[IND1]] : memref<?x2x?xf32>
  // CHECK-NEXT:  [[CAST1:%.+]] = arith.index_cast [[DIM1]] : index to i64
  // CHECK-NEXT:  [[TMP1:%.+]] = arith.muli [[INIT]], [[CAST1]] : i64
  // CHECK-NEXT:  [[IND2:%.+]] = arith.constant 2 : index
  // CHECK-NEXT:  [[DIM2:%.+]] = memref.dim %arg0, [[IND2]] : memref<?x2x?xf32>
  // CHECK-NEXT:  [[IND3:%.+]] = arith.index_cast [[DIM2]] : index to i64
  // CHECK-NEXT:  [[SIZE:%.+]] = arith.muli [[TMP1]], [[IND3]] : i64
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
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<f32>
  // CHECK: [[CST_VALUE:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: krnl.store [[CST_VALUE]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

// Check the lowering of ConstantOfShape when:
//   - The input is not a arith.constant tensor.
// Expected emitted code:
//   - Emit code to compute output dimensions from the input's dimensions.
//   - Krnl iterates are used to set values to the output.
func private @test_constant_of_shape_dynamic_dims(%arg0 : tensor<3xi64>) -> tensor<*xf32> {
  %0 = "onnx.ConstantOfShape"(%arg0) {value = dense<[1.0]> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_of_shape_dynamic_dims
  // CHECK: [[CST0:%.+]] = arith.constant 0 : index
  // CHECK: [[LOAD_DIM_0:%.+]] = krnl.load %arg0{{\[}}[[CST0]]{{\]}} : memref<3xi64>
  // CHECK: [[DIM_0:%.+]] = arith.index_cast [[LOAD_DIM_0]] : i64 to index
  // CHECK: [[CST1:%.+]] = arith.constant 1 : index
  // CHECK: [[LOAD_DIM_1:%.+]] = krnl.load %arg0{{\[}}[[CST1]]{{\]}} : memref<3xi64>
  // CHECK: [[DIM_1:%.+]] = arith.index_cast [[LOAD_DIM_1]] : i64 to index
  // CHECK: [[CST2:%.+]] = arith.constant 2 : index
  // CHECK: [[LOAD_DIM_2:%.+]] = krnl.load %arg0{{\[}}[[CST2]]{{\]}} : memref<3xi64>
  // CHECK: [[DIM_2:%.+]] = arith.index_cast [[LOAD_DIM_2]] : i64 to index
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]], [[DIM_1]], [[DIM_2]]) {{.*}}: memref<?x?x?xf32>

  // CHECK: [[CST_VALUE:%.+]] = arith.constant 1.000000e+00 : f32
  // CHECK: [[LOOP_DEF:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[CST00:%.+]] = arith.constant 0 : index
  // CHECK: [[RES_DIM_0:%.+]] = memref.dim [[RES]], [[CST00]] : memref<?x?x?xf32>
  // CHECK: [[CST11:%.+]] = arith.constant 1 : index
  // CHECK: [[RES_DIM_1:%.+]] = memref.dim [[RES]], [[CST11]] : memref<?x?x?xf32>
  // CHECK: [[CST22:%.+]] = arith.constant 2 : index
  // CHECK: [[RES_DIM_2:%.+]] = memref.dim [[RES]], [[CST22]] : memref<?x?x?xf32>
  // CHECK: krnl.iterate([[LOOP_DEF]]#0, [[LOOP_DEF]]#1, [[LOOP_DEF]]#2) with ([[LOOP_DEF]]#0 -> %arg1 = 0 to [[RES_DIM_0]], [[LOOP_DEF]]#1 -> %arg2 = 0 to [[RES_DIM_1]], [[LOOP_DEF]]#2 -> %arg3 = 0 to [[RES_DIM_2]]){
  // CHECK:   krnl.store [[CST_VALUE]], [[RES]][%arg1, %arg2, %arg3] : memref<?x?x?xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<?x?x?xf32>
}

// -----

// Check the lowering of ConstantOfShape when:
//   - The input is a arith.constant tensor.
// Expected emitted code:
//   - Output dimensions are computed during compilation time.
//   - Krnl iterates are used to set values to the output.
func private @test_constant_of_shape_static_dims() -> tensor<*xf32> {
  %0 = "onnx.Constant"() {value = dense<[3, 4, 5]> : tensor<3xi64> } : () -> tensor<3xi64>
  %1 = "onnx.ConstantOfShape"(%0) {value = dense<[1.0]> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_of_shape_static_dims
  // CHECK: [[GLOBAL_CST:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<[3, 4, 5]> : tensor<3xi64>} : () -> memref<3xi64>
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
  // CHECK: [[CST_VALUE:%.+]] = arith.constant 1.000000e+00 : f32
  // CHECK: [[LOOP_DEF:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[LOOP_DEF]]#0, [[LOOP_DEF]]#1, [[LOOP_DEF]]#2) with ([[LOOP_DEF]]#0 -> %arg0 = 0 to 3, [[LOOP_DEF]]#1 -> %arg1 = 0 to 4, [[LOOP_DEF]]#2 -> %arg2 = 0 to 5){
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
  // CHECK:  [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x24xf32>
  // CHECK:  [[LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK:  krnl.iterate([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2) with ([[LOOP]]#0 -> [[LOOPARG1:%.+]] = 0 to 2, [[LOOP]]#1 -> [[LOOPARG2:%.+]] = 0 to 3, [[LOOP]]#2 -> [[LOOPARG3:%.+]] = 0 to 4){
  // CHECK:    [[LOAD:%.+]] = krnl.load %arg0{{\[}}[[LOOPARG1]], [[LOOPARG2]], [[LOOPARG3]]{{\]}} : memref<2x3x4xf32>
  // CHECK:    [[FIRSTDIM:%.+]] = affine.apply [[MAP_FIRST]]()
  // CHECK:    [[C0:%.+]] = arith.constant 0 : index
  // CHECK:    [[R4:%.+]] = arith.constant 2 : index
  // CHECK:    [[C1:%.+]] = arith.constant 1 : index
  // CHECK:    [[R5:%.+]] = arith.constant 3 : index
  // CHECK:    [[C2:%.+]] = arith.constant 2 : index
  // CHECK:    [[R6:%.+]] = arith.constant 4 : index
  // CHECK:    [[SECONDDIM:%.+]] = affine.apply [[MAP_SECOND]]([[LOOPARG1]], [[LOOPARG2]], [[LOOPARG3]]){{\[}}[[R4]], [[R5]], [[R6]]{{\]}}
  // CHECK:    krnl.store [[LOAD]], [[ALLOC]]{{\[}}[[FIRSTDIM]], [[SECONDDIM]]{{\]}} : memref<1x24xf32>
}

// -----

// test partially known input shape
func private @test_flatten1(%arg0 : tensor<2x?x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<2x?x4xf32>) -> tensor<*xf32>
  "std.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func private @test_flatten1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x?x4xf32>) -> memref<?x4xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.muli [[CST_1_]], [[CST_2_]] : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_1_]] : memref<2x?x4xf32>
// CHECK:           [[VAR_2_:%.+]] = arith.muli [[VAR_0_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_2_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_5_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_2_]] : memref<2x?x4xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[VAR_5_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<2x?x4xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:         [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_7_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_3_]] : memref<2x?x4xf32>
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.apply #map0([[I_0_]], [[I_1_]]){{.}}[[CST_2_1_]], [[VAR_7_]]{{.}}
// CHECK-DAG:         [[CST_2_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:         [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK:             [[VAR_9_:%.+]] = affine.apply #map1([[I_2_]]){{.}}[[CST_4_]]{{.}}
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_8_]], [[VAR_9_]]{{.}} : memref<?x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x4xf32>
// CHECK:         }
}

// -----

func private @test_less(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xi1>
  return %0 : tensor<3x4x5xi1>

  // CHECK-LABEL: test_less
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xi1>
  // CHECK: [[DEF_LOOPS]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 3, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 4, [[DEF_LOOPS]]#2 -> %arg4 = 0 to 5){
  // CHECK:   [[LHS:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:   [[RHS:%.+]] = krnl.load %arg1[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:   [[LESS:%.+]] = arith.cmpf olt, [[LHS]], [[RHS]] : f32
  // CHECK:   krnl.store [[LESS]], [[RES]][%arg2, %arg3, %arg4] : memref<3x4x5xi1>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xi1>
}

// -----

func private @test_less_broadcast(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5xf32>) -> tensor<3x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<5xf32>) -> tensor<3x4x5xi1>
  return %0 : tensor<3x4x5xi1>

  // CHECK-LABEL: test_less_broadcast
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xi1>
  // CHECK: [[DEF_LOOPS]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 3, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 4, [[DEF_LOOPS]]#2 -> %arg4 = 0 to 5){
  // CHECK:   [[LHS:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:   [[RHS:%.+]] = krnl.load %arg1[%arg4] : memref<5xf32>
  // CHECK:   [[LESS:%.+]] = arith.cmpf olt, [[LHS]], [[RHS]] : f32
  // CHECK:   krnl.store [[LESS]], [[RES]][%arg2, %arg3, %arg4] : memref<3x4x5xi1>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xi1>
}

// -----

func private @test_floor(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Floor"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_floor
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[FLOOR:%.+]] = math.floor [[LOAD]] : f32
  // CHECK: krnl.store [[FLOOR]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_ceil(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Ceil"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "std.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_ceil
  // CHECK: [[C0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_0:%.+]] = memref.dim %arg0, [[C0]] : memref<?x10xf32>
  // CHECK: [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<?x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[C0_0:%.+]] = arith.constant 0 : index
  // CHECK: [[DIM_2:%.+]] = memref.dim %arg0, [[C0_0]] : memref<?x10xf32>
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to [[DIM_2]], [[DEF_LOOPS]]#1 -> %arg2 = 0 to 10){
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[%arg1, %arg2] : memref<?x10xf32>
  // CHECK: [[CEIL:%.+]] = math.ceil [[LOAD]] : f32
  // CHECK: krnl.store [[CEIL]], [[RES]][%arg1, %arg2] : memref<?x10xf32>
  // CHECK: return [[RES]] : memref<?x10xf32>
}

// -----

func private @test_clip(%arg0: tensor<3xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<3xf32> attributes {input_names = ["x", "min", "max"], output_names = ["y"]} {
  %0 = "onnx.Clip"(%arg0, %arg1, %arg2) : (tensor<3xf32>, tensor<f32>, tensor<f32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

// CHECK-LABEL: test_clip
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3xf32>, [[MIN_:%.+]]: memref<f32>, [[MAX_:%.+]]: memref<f32>) -> memref<3xf32> attributes {input_names = ["x", "min", "max"], output_names = ["y"]} {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3){
// CHECK-DAG:         [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[I_0_]]{{.}} : memref<3xf32>
// CHECK-DAG:         [[LOAD_MIN_MEM_:%.+]] = krnl.load [[MIN_]][] : memref<f32>
// CHECK:             [[VAR_4_:%.+]] = arith.cmpf olt, [[LOAD_INPUT_MEM_]], [[LOAD_MIN_MEM_]] : f32
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.select [[VAR_4_]], [[LOAD_MIN_MEM_]], [[LOAD_INPUT_MEM_]] : f32
// CHECK-DAG:         [[LOAD_MAX_MEM_:%.+]] = krnl.load [[MAX_]][] : memref<f32>
// CHECK:             [[VAR_7_:%.+]] = arith.cmpf olt, [[VAR_5_]], [[LOAD_MAX_MEM_]] : f32
// CHECK:             [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_5_]], [[LOAD_MAX_MEM_]] : f32
// CHECK:             krnl.store [[VAR_8_]], [[RES_]]{{.}}[[I_0_]]{{.}} : memref<3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3xf32>
// CHECK:         }
}

// -----

func private @test_clip_default_min(%arg0: tensor<3xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<3xf32> attributes {input_names = ["x", "min", "max"], output_names = ["y"]} {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Clip"(%arg0, %cst, %arg2) : (tensor<3xf32>, none, tensor<f32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

// CHECK-LABEL: test_clip_default_min
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3xf32>, [[MIN_:%.+]]: memref<f32>, [[MAX_:%.+]]: memref<f32>) -> memref<3xf32> attributes {input_names = ["x", "min", "max"], output_names = ["y"]} {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3){
// CHECK-DAG:         [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[I_0_]]{{.}} : memref<3xf32>
// CHECK-DAG:         [[LOAD_MAX_MEM_:%.+]] = krnl.load [[MAX_]][] : memref<f32>
// CHECK:             [[VAR_7_:%.+]] = arith.cmpf olt, [[LOAD_INPUT_MEM_]], [[LOAD_MAX_MEM_]] : f32
// CHECK:             [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[LOAD_INPUT_MEM_]], [[LOAD_MAX_MEM_]] : f32
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
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:         [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<3x4x5xf32>
// CHECK-DAG:         [[LOAD_POWER_MEM_:%.+]] = krnl.load [[POWER_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<3x4x5xf32>
// CHECK:             [[VAR_4_:%.+]] = math.powf [[LOAD_INPUT_MEM_]], [[LOAD_POWER_MEM_]] : f32
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
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5){
  // CHECK:   [[LOAD_X:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:   [[LOAD_SLOPE:%.+]] = krnl.load %arg1[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:   [[CST_0:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK:   [[LESS_THAN_ZERO:%.+]] = arith.cmpf olt, [[LOAD_X]], [[CST_0]] : f32
  // CHECK:   [[MUL:%.+]] = arith.mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:   [[SELECT:%.+]] = arith.select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
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
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xi32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5){
  // CHECK:   [[LOAD_X:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xi32>
  // CHECK:   [[LOAD_SLOPE:%.+]] = krnl.load %arg1[%arg2, %arg3, %arg4] : memref<3x4x5xi32>
  // CHECK:   [[CST_0:%.+]] = arith.constant 0 : i32
  // CHECK:   [[LESS_THAN_ZERO:%.+]] = arith.cmpi slt, [[LOAD_X]], [[CST_0]] : i32
  // CHECK:   [[MUL:%.+]] = arith.muli [[LOAD_SLOPE]], [[LOAD_X]] : i32
  // CHECK:   [[SELECT:%.+]] = arith.select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : i32
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
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5){
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1[%arg4] : memref<5xf32>
  // CHECK-DAG:   [[CST_0:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = arith.cmpf olt, [[LOAD_X]], [[CST_0]] : f32
  // CHECK:       [[MUL:%.+]] = arith.mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT:%.+]] = arith.select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
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
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5){
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[%arg2, %arg3, %arg4] : memref<3x4x5xf32>
  // CHECK-DAG:   [[ZERO_INDEX:%.+]] = arith.constant 0 : index
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], %arg4] : memref<1x5xf32>
  // CHECK-DAG:   [[CST_0:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = arith.cmpf olt, [[LOAD_X]], [[CST_0]] : f32
  // CHECK:       [[MUL:%.+]] = arith.mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT:%.+]] = arith.select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
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
  // CHECK-LABEL:  func private @test_loop_simple_main_graph
  // CHECK-SAME:     ([[TRIP_COUNT:%.+]]: memref<i64>, [[COND:%.+]]: memref<i1>, [[Y_INIT:%.+]]: memref<1xi64>) -> memref<1xi64> {
  // CHECK:           [[Y:%.+]] = memref.alloc() {{.*}}: memref<1xi64>
  // CHECK:           [[Y_COPY_LOOP:%.+]] = krnl.define_loops 1
  // CHECK:           krnl.iterate([[Y_COPY_LOOP]]) with ([[Y_COPY_LOOP]] -> [[YCOPY_IV:%.+]] = 0 to 1){
  // CHECK:             [[Y_VAL:%.+]] = krnl.load [[Y_INIT]]{{.}}[[YCOPY_IV]]{{.}} : memref<1xi64>
  // CHECK:             krnl.store [[Y_VAL]], [[Y]]{{.}}[[YCOPY_IV]]{{.}} : memref<1xi64>
  // CHECK:           }
  // CHECK:           [[COND_GLOBAL:%.+]] = memref.alloc() {{.*}}: memref<i1>
  // CHECK:           [[COND_VAL:%.+]] = krnl.load [[COND]][] : memref<i1>
  // CHECK:           krnl.store [[COND_VAL]], [[COND_GLOBAL]][] : memref<i1>
  // CHECK:           [[LOOP:%.+]] = krnl.define_loops 1
  // CHECK:           [[TRIP_COUNT_VAL:%.+]] = krnl.load [[TRIP_COUNT]][] : memref<i64>
  // CHECK:           [[TRIP_COUNT_IDX:%.+]] = arith.index_cast [[TRIP_COUNT_VAL]] : i64 to index
  // CHECK:           krnl.iterate([[LOOP]]) with ([[LOOP]] -> [[LOOP_IV:%.+]] = 0 to [[TRIP_COUNT_IDX]]){
  // CHECK:             [[COND_VAL:%.+]] = krnl.load [[COND_GLOBAL]][] : memref<i1>
  // CHECK:             scf.if [[COND_VAL]] {
  // CHECK:               [[LOOP_IV_VAL:%.+]] = arith.index_cast [[LOOP_IV]] : index to i64
  // CHECK:               [[CURR_LOOP_IV:%.+]] = memref.alloc() {{.*}}: memref<i64>
  // CHECK:               krnl.store [[LOOP_IV_VAL]], [[CURR_LOOP_IV]][] : memref<i64>
  // CHECK:               [[UCC_COND:%.+]] = builtin.unrealized_conversion_cast [[COND]] : memref<i1> to tensor<i1>  
  // CHECK:               [[Y_CURR:%.+]] = memref.alloc() {{.*}}: memref<1xi64>
  // CHECK:               [[Y_COMPUTE_LOOP:%.+]] = krnl.define_loops 1
  // CHECK:               krnl.iterate([[Y_COMPUTE_LOOP]]) with ([[Y_COMPUTE_LOOP]] -> [[Y_COMPUTE_IV:%.+]] = 0 to 1){
  // CHECK:                 [[Y_VAL:%.+]] = krnl.load [[Y]]{{.}}[[Y_COMPUTE_IV]]{{.}} : memref<1xi64>
  // CHECK:                 [[LOO_IV_VAL:%.+]] = krnl.load [[CURR_LOOP_IV]][] : memref<i64>
  // CHECK:                 [[NEW_Y_VAL:%.+]] = arith.addi [[Y_VAL]], [[LOO_IV_VAL]] : i64
  // CHECK:                 krnl.store [[NEW_Y_VAL]], [[Y_CURR]]{{.}}[[Y_COMPUTE_IV]]{{.}} : memref<1xi64>
  // CHECK:               }
  // CHECK:               [[UCC_Y_CURR:%.+]] = builtin.unrealized_conversion_cast [[Y_CURR]] : memref<1xi64> to tensor<1xi64>
  // CHECK:               [[COND_CAST:%.+]] = builtin.unrealized_conversion_cast [[UCC_COND]] : tensor<i1> to memref<i1> 
  // CHECK:               [[Y_CURR_CAST:%.+]] = builtin.unrealized_conversion_cast [[UCC_Y_CURR]] : tensor<1xi64> to memref<1xi64>
  // CHECK:               [[COND_CAST_VAL:%.+]] = krnl.load [[COND_CAST]][] : memref<i1>
  // CHECK:               krnl.store [[COND_CAST_VAL]], [[COND_GLOBAL]][] : memref<i1>
  // CHECK:               [[Y_COPY_LOOP:%.+]] = krnl.define_loops 1
  // CHECK:               krnl.iterate([[Y_COPY_LOOP]]) with ([[Y_COPY_LOOP]] -> [[Y_COPY_IV:%.+]] = 0 to 1){
  // CHECK:                 [[Y_SCAN_VAL:%.+]] = krnl.load [[Y_CURR_CAST]]{{.}}[[Y_COPY_IV]]{{.}} : memref<1xi64>
  // CHECK:                 krnl.store [[Y_SCAN_VAL]], [[Y]]{{.}}[[Y_COPY_IV]]{{.}} : memref<1xi64>
  // CHECK:               }
  // CHECK:             }
  // CHECK:           }
  // CHECK:           return [[Y]] : memref<1xi64>
  // CHECK:        }
}

// -----

func @test_resize1(%arg0 : tensor<3x4xf32>) -> tensor<*xf32> {
    %cst = "onnx.NoValue"() {value} : () -> none
    %0 = "onnx.Constant"() {value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>} : () -> tensor<4xf32>
    %1 = "onnx.Constant"() {value = dense<[1.000000e+00,  3.000000e+00]> : tensor<2xf32>} : () -> tensor<2xf32>
    %2 = "onnx.Resize"(%arg0, %0, %1, %cst) {coordinate_transformation_mode = "asymmetric", mode = "nearest", nearest_mode = "floor"} : (tensor<3x4xf32>, tensor<4xf32>, tensor<2xf32>, none) -> tensor<*xf32>
    "std.return"(%2) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  func @test_resize1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4xf32>) -> memref<3x12xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [4], value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2], value = dense<[1.000000e+00, 3.000000e+00]> : tensor<2xf32>} : () -> memref<2xf32>
// CHECK-DAG:       [[VAR_cst_0_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[VAR_cst_1_:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x12xf32>
// CHECK-DAG:       [[VAR_c0_i64_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 12){
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[I_0_]] : index to i64
// CHECK:             [[VAR_5_:%.+]] = arith.sitofp [[VAR_4_]] : i64 to f32
// CHECK:             [[VAR_6_:%.+]] = arith.divf [[VAR_5_]], [[VAR_cst_0_]] : f32
// CHECK:             [[VAR_7_:%.+]] = math.floor [[VAR_6_]] : f32
// CHECK:             [[VAR_8_:%.+]] = arith.fptosi [[VAR_7_]] : f32 to i64
// CHECK:             [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[VAR_c0_i64_]] : i64
// CHECK:             [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[VAR_c0_i64_]], [[VAR_8_]] : i64
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : i64 to index
// CHECK-DAG:         [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpi slt, [[VAR_11_]], [[VAR_c3_]] : index
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.subi [[VAR_c3_]], [[VAR_c1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_11_]], [[VAR_13_]] : index
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.index_cast [[I_1_]] : index to i64
// CHECK:             [[VAR_16_:%.+]] = arith.sitofp [[VAR_15_]] : i64 to f32
// CHECK:             [[VAR_17_:%.+]] = arith.divf [[VAR_16_]], [[VAR_cst_1_]] : f32
// CHECK:             [[VAR_18_:%.+]] = math.floor [[VAR_17_]] : f32
// CHECK:             [[VAR_19_:%.+]] = arith.fptosi [[VAR_18_]] : f32 to i64
// CHECK:             [[VAR_20_:%.+]] = arith.cmpi slt, [[VAR_19_]], [[VAR_c0_i64_]] : i64
// CHECK:             [[VAR_21_:%.+]] = arith.select [[VAR_20_]], [[VAR_c0_i64_]], [[VAR_19_]] : i64
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.index_cast [[VAR_21_]] : i64 to index
// CHECK-DAG:         [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.cmpi slt, [[VAR_22_]], [[VAR_c4_]] : index
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.subi [[VAR_c4_]], [[VAR_c1_]] : index
// CHECK:             [[VAR_25_:%.+]] = arith.select [[VAR_23_]], [[VAR_22_]], [[VAR_24_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_14_]], [[VAR_25_]]{{.}} : memref<3x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<3x12xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x12xf32>
// CHECK:         }
}

//-----
  func @test_gather_scalar(%arg0: tensor<4xi64>, %arg1: tensor<i64>) -> tensor<i64> {
    %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64} : (tensor<4xi64>, tensor<i64>) -> tensor<i64>
    return %0 : tensor<i64>
// CHECK-LABEL:  @test_gather_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4xi64>, [[PARAM_1_:%.+]]: memref<i64>) -> memref<i64> {
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<i64>
// CHECK:           krnl.define_loops 0
// CHECK:           krnl.iterate() with (){
// CHECK-DAG:         [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<i64>
// CHECK:             [[VAR_2_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.cmpi slt, [[VAR_2_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addi [[VAR_2_]], [[CST_4_1_]] : index
// CHECK:             [[VAR_5_:%.+]] = arith.select [[VAR_3_]], [[VAR_4_]], [[VAR_2_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_5_]]{{.}} : memref<4xi64>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][] : memref<i64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<i64>
}

  func @test_reversesequence_1(%arg0: tensor<10x?xf32>, %arg1: tensor<10xi64>) -> tensor<*xf32> {
    %0 = "onnx.ReverseSequence"(%arg0, %arg1) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<10x?xf32>, tensor<10xi64>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
// CHECK-LABEL:  @test_reversesequence_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x?xf32>, [[PARAM_1_:%.+]]: memref<10xi64>) -> memref<10x?xf32> {
// CHECK-DAG:       [[VAR_c10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<10x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<10x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[VAR_0_]]){
// CHECK-DAG:         [[VAR_c1_0_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[I_1_]]{{.}} : memref<10xi64>
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi slt, [[I_0_]], [[VAR_4_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.subi [[VAR_4_]], [[I_0_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.subi [[VAR_6_]], [[VAR_c1_0_]] : index
// CHECK:             [[VAR_8_:%.+]] = arith.select [[VAR_5_]], [[VAR_7_]], [[I_0_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_8_]], [[I_1_]]{{.}} : memref<10x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x?xf32>
  }

//-----
  func @test_random_normal1() -> tensor<*xf32> {
    %0 = "onnx.RandomNormal"() {shape = [3, 4, 5], dtype = 1 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : () -> tensor<*xf32>
    "std.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal1
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[SEED:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf32>, index, f32, f32, f32) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf32>
}

//-----
  func @test_random_normal2() -> tensor<*xf32> {
    %0 = "onnx.RandomNormal"() {shape = [3, 4, 5], dtype = 2 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : () -> tensor<*xf32>
    "std.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal2
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf64>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       [[SEED:%.+]] = arith.constant 2.000000e+00 : f64
// CHECK-DAG:       "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf64>, index, f64, f64, f64) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf64>
}

//-----
  func @test_random_normal3() -> tensor<*xf32> {
    %0 = "onnx.RandomNormal"() {shape = [3, 4, 5], dtype = 2 : si64, mean = 0.0 :f32, scale = 1.0 : f32} : () -> tensor<*xf32>
    "std.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal3
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf64>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       [[SEED:%.+]] = arith.constant
// CHECK-DAG:       "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf64>, index, f64, f64, f64) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf64>
}

//-----
  func @test_random_normal_like1(%arg0: tensor<3x4x5xf32>) -> tensor<*xf32> {
    %0 = "onnx.RandomNormalLike"(%arg0) {dtype = 1 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<3x4x5xf32>) -> tensor<*xf32>
    "std.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal_like1
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[SEED:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf32>, index, f32, f32, f32) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf32>
}

//-----
  func @test_random_normal_like2(%arg0: tensor<3x4x5xf32>) -> tensor<*xf32> {
    %0 = "onnx.RandomNormalLike"(%arg0) {dtype = 2 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<3x4x5xf32>) -> tensor<*xf32>
    "std.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal_like2
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf64>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       [[SEED:%.+]] = arith.constant 2.000000e+00 : f64
// CHECK-DAG:       "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf64>, index, f64, f64, f64) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf64>
}

//-----
  func @test_random_normal_like3(%arg0: tensor<3x4x5xf32>) -> tensor<*xf32> {
    %0 = "onnx.RandomNormalLike"(%arg0) {dtype = 2 : si64, mean = 0.0 :f32, scale = 1.0 : f32} : (tensor<3x4x5xf32>) -> tensor<*xf32>
    "std.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal_like3
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf64>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       [[SEED:%.+]] = arith.constant
// CHECK-DAG:       "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf64>, index, f64, f64, f64) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf64>
}

//-----
  func @test_random_normal_like4(%arg0: tensor<3x4x?x?xf32>) -> tensor<*xf32> {
    %0 = "onnx.RandomNormalLike"(%arg0) {dtype = 1 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<3x4x?x?xf32>) -> tensor<*xf32>
    "std.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal_like4
// CHECK-DAG:       [[C2:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[DIM2:%.+]] = memref.dim %arg0, [[C2]] : memref<3x4x?x?xf32>
// CHECK-DAG:       [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[DIM3:%.+]] = memref.dim %arg0, [[C3]] : memref<3x4x?x?xf32>
// CHECK-DAG:       [[DYN_ALLOC:%.+]] = memref.alloc([[DIM2]], [[DIM3]]) {alignment = 16 : i64} : memref<3x4x?x?xf32>
// CHECK-DAG:       [[C12:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[C2:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[DIM2:%.+]] = memref.dim %arg0, [[C2]] : memref<3x4x?x?xf32>
// CHECK-DAG:       [[MUL1:%.+]] = arith.muli [[C12]], [[DIM2]] : index
// CHECK-DAG:       [[C3:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[DIM3:%.+]] = memref.dim %arg0, [[C3]] : memref<3x4x?x?xf32>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.muli [[MUL1]], [[DIM3]] : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[SEED:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       "krnl.random_normal"([[DYN_ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x?x?xf32>, index, f32, f32, f32) -> ()
// CHECK:           return [[DYN_ALLOC]] : memref<3x4x?x?xf32>
}

//-----
  func @test_random_normal_like5(%arg0: tensor<3x4x5xf32>) -> tensor<*xf32> {
    %0 = "onnx.RandomNormalLike"(%arg0) {mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<3x4x5xf32>) -> tensor<*xf32>
    "std.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal_like5
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[SEED:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf32>, index, f32, f32, f32) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf32>
}
