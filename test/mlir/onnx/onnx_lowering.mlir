// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl  --canonicalize %s -split-input-file | FileCheck %s

// -----

// TODO: Remove test_no_argument_1 from the test - empty function body is no longer
// supported in mlir: https://reviews.llvm.org/D91886
func.func private @test_no_argument_2() -> tensor<*xf32> {
  %0 = onnx.Constant {value =  dense<[[1.000000e+0, 2.000000e+00], [3.000000e+00, 4.000000e+00]]> : tensor<2x2xf32>} : tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL: test_no_argument_2
// CHECK: [[GLOBAL:%.+]] = "{{.*}}"({{.*}}) {{.*}} : ({{.*}}) -> memref<2x2xf32>
// CHECK: return [[GLOBAL]] : memref<2x2xf32>

// -----

func.func private @test_elementwise_op_with_scalar_values_1(%arg0 : tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Exp"(%arg0) : (tensor<f32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_elementwise_op_with_scalar_values_1
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<f32>
  // CHECK: [[EXP:%.+]] = math.exp [[LOAD]] : f32
  // CHECK: krnl.store [[EXP]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func.func private @test_elementwise_op_with_scalar_values_2(%arg0 : tensor<f32>, %arg1 : tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_elementwise_op_with_scalar_values_2
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<f32>
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[] : memref<f32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[] : memref<f32>
  // CHECK: [[ADD:%.+]] = arith.addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[ADD]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func.func private @test_elementwise_op_with_scalar_values_3(%arg0 : tensor<f32>, %arg1 : tensor<f32>, %arg2 : tensor<f32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1, %arg2) : (tensor<f32>, tensor<f32>, tensor<f32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

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

func.func private @test_add(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_add
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[IV:%.+]]:2 = krnl.get_induction_var_value([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: [[ADDF:%.+]] = arith.addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[ADDF]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func.func private @test_mul(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Mul"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_mul
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[IV:%.+]]:2 = krnl.get_induction_var_value([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: [[MULF:%.+]] = arith.mulf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[MULF]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func.func private @test_div(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Div"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_div
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[IV:%.+]]:2 = krnl.get_induction_var_value([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: [[DIVF:%.+]] = arith.divf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[DIVF]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func.func private @test_sub(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sub"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sub
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[IV:%.+]]:2 = krnl.get_induction_var_value([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: [[SUBF:%.+]] = arith.subf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[SUBF]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func.func private @test_and(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<*xi1> {
  %0 = "onnx.And"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  "func.return"(%0) : (tensor<*xi1>) -> ()

  // CHECK-LABEL: test_and
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xi1>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[IV:%.+]]:2 = krnl.get_induction_var_value([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1] : memref<10x10xi1>
  // CHECK: [[AND:%.+]] = arith.andi [[LOAD1]], [[LOAD2]] : i1
  // CHECK: krnl.store [[AND]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<10x10xi1>
  // CHECK: return [[RES]] : memref<10x10xi1>
}

// -----

func.func private @test_or(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<*xi1> {
  %0 = "onnx.Or"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  "func.return"(%0) : (tensor<*xi1>) -> ()

  // CHECK-LABEL: test_or
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xi1>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[IV:%.+]]:2 = krnl.get_induction_var_value([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1] : memref<10x10xi1>
  // CHECK: [[OR:%.+]] = arith.ori [[LOAD1]], [[LOAD2]] : i1
  // CHECK: krnl.store [[OR]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<10x10xi1>
  // CHECK: return [[RES]] : memref<10x10xi1>
}

// -----

func.func private @test_xor(%arg0 : tensor<10x10xi1>, %arg1 : tensor<10x10xi1>) -> tensor<*xi1> {
  %0 = "onnx.Xor"(%arg0, %arg1) : (tensor<10x10xi1>, tensor<10x10xi1>) -> tensor<*xi1>
  "func.return"(%0) : (tensor<*xi1>) -> ()

  // CHECK-LABEL: test_xor
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xi1>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[IV:%.+]]:2 = krnl.get_induction_var_value([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1] : memref<10x10xi1>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1] : memref<10x10xi1>
  // CHECK: [[XOR:%.+]] = arith.xori [[LOAD1]], [[LOAD2]] : i1
  // CHECK: krnl.store [[XOR]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<10x10xi1>
  // CHECK: return [[RES]] : memref<10x10xi1>
}

// -----

func.func private @test_bitwise_and(%arg0 : tensor<10x10xi8>, %arg1 : tensor<10x10xi8>) -> tensor<*xi8> {
  %0 = "onnx.BitwiseAnd"(%arg0, %arg1) : (tensor<10x10xi8>, tensor<10x10xi8>) -> tensor<*xi8>
  "func.return"(%0) : (tensor<*xi8>) -> ()

  // CHECK-LABEL: test_bitwise_and
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xi8>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[IV:%.+]]:2 = krnl.get_induction_var_value([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1] : memref<10x10xi8>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1] : memref<10x10xi8>
  // CHECK: [[AND:%.+]] = arith.andi [[LOAD1]], [[LOAD2]] : i8
  // CHECK: krnl.store [[AND]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<10x10xi8>
  // CHECK: return [[RES]] : memref<10x10xi8>
}

// -----

func.func private @test_bitwise_or(%arg0 : tensor<10x10xi16>, %arg1 : tensor<10x10xi16>) -> tensor<*xi16> {
  %0 = "onnx.BitwiseOr"(%arg0, %arg1) : (tensor<10x10xi16>, tensor<10x10xi16>) -> tensor<*xi16>
  "func.return"(%0) : (tensor<*xi16>) -> ()

  // CHECK-LABEL: test_bitwise_or
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xi16>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[IV:%.+]]:2 = krnl.get_induction_var_value([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1] : memref<10x10xi16>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1] : memref<10x10xi16>
  // CHECK: [[OR:%.+]] = arith.ori [[LOAD1]], [[LOAD2]] : i16
  // CHECK: krnl.store [[OR]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<10x10xi16>
  // CHECK: return [[RES]] : memref<10x10xi16>
}

// -----

func.func private @test_bitwise_xor(%arg0 : tensor<10x10xi32>, %arg1 : tensor<10x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.BitwiseXor"(%arg0, %arg1) : (tensor<10x10xi32>, tensor<10x10xi32>) -> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()

  // CHECK-LABEL: test_bitwise_xor
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xi32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[IV:%.+]]:2 = krnl.get_induction_var_value([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1] : memref<10x10xi32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1] : memref<10x10xi32>
  // CHECK: [[XOR:%.+]] = arith.xori [[LOAD1]], [[LOAD2]] : i32
  // CHECK: krnl.store [[XOR]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<10x10xi32>
  // CHECK: return [[RES]] : memref<10x10xi32>
}

// -----


func.func private @test_reshape_constant(%arg0 : tensor<1x10xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[2, 5]> : tensor<2xi64>
  %1 = "onnx.Reshape"(%arg0, %0) : (tensor<1x10xf32>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reshape_constant
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x10xf32>) -> memref<2x5xf32> {
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 5], strides: [5, 1] : memref<1x10xf32> to memref<2x5xf32>
// CHECK:           return [[VAR_reinterpret_cast_]] : memref<2x5xf32>
// CHECK:         }
}

// -----

// `Reshape` ops are lowerd to `reinterpret_cast` op. `reinterpret_cast` ops just change
// the view of input memref. So, input memref should not be deallocated if it is retuned.
// This test confirms the deallocation is not generated.

func.func private @test_reshape_constant_dealloc(%arg0 : tensor<10x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<10x1xf32>) -> tensor<*xf32>
  %1 = onnx.Constant dense<[2, 5]> : tensor<2xi64>
  %2 = "onnx.Reshape"(%0, %1) : (tensor<*xf32>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%2) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reshape_constant_dealloc
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x1xf32>) -> memref<2x5xf32> {
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [2, 5], strides: [5, 1] : memref<10x1xf32> to memref<2x5xf32>
// CHECK:           return [[VAR_reinterpret_cast_]] : memref<2x5xf32>
// CHECK:         }
}

// -----

func.func private @test_sum(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sum"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_sum
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[IV:%.+]]:2 = krnl.get_induction_var_value([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: [[ADD:%.+]] = arith.addf [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[ADD]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func.func private @test_max(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Max"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_max
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[IV:%.+]]:2 = krnl.get_induction_var_value([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: [[MAX:%.+]] = arith.cmpf ogt, [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[RELU_RES:%.+]] = arith.select [[MAX]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[RELU_RES]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----

func.func private @test_min(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Min"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_min
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10x10xf32>
  // CHECK: [[DEF_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) with ([[DEF_LOOPS]]#0 -> %arg2 = 0 to 10, [[DEF_LOOPS]]#1 -> %arg3 = 0 to 10){
  // CHECK: [[IV:%.+]]:2 = krnl.get_induction_var_value([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: [[MIN:%.+]] = arith.cmpf olt, [[LOAD1]], [[LOAD2]] : f32
  // CHECK: [[RELU_RES:%.+]] = arith.select [[MIN]], [[LOAD1]], [[LOAD2]] : f32
  // CHECK: krnl.store [[RELU_RES]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<10x10xf32>
  // CHECK: return [[RES]] : memref<10x10xf32>
}

// -----


func.func private @test_reducemax_v13(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducemax_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             krnl.store [[CST_0_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<3x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_4_:%.+]] = 0 to 2){
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[I_3_]], [[I_4_]]{{.}} : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[I_2_]], [[I_4_]]{{.}} : memref<3x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.cmpf ogt, [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_5_:%.+]] = arith.select [[VAR_4_]], [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[I_2_]], [[I_4_]]{{.}} : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----

func.func private @test_reducemax_v13_negative_inf_f32(%arg0 : tensor<2x3xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_reducemax_v13_negative_inf_f32
  // CHECK: arith.constant 0xFF800000 : f32
}

// -----

func.func private @test_reducemax_v13_negative_inf_f64(%arg0 : tensor<2x3xf64>) -> tensor<*xf64> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xf64>)-> tensor<*xf64>
  "func.return"(%0) : (tensor<*xf64>) -> ()
  // CHECK-LABEL: test_reducemax_v13_negative_inf_f64
  // CHECK: arith.constant 0xFFF0000000000000 : f64
}

// -----

func.func private @test_reducemax_v13_negative_inf_i8(%arg0 : tensor<2x3xi8>) -> tensor<*xi8> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xi8>)-> tensor<*xi8>
  "func.return"(%0) : (tensor<*xi8>) -> ()
  // CHECK-LABEL: test_reducemax_v13_negative_inf_i8
  // CHECK: arith.constant -128 : i8
}

// -----

func.func private @test_reducemax_v13_negative_inf_i32(%arg0 : tensor<2x3xi32>) -> tensor<*xi32> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xi32>)-> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()
  // CHECK-LABEL: test_reducemax_v13_negative_inf_i32
  // CHECK: arith.constant -2147483648 : i32
}

// -----

func.func private @test_reducemax_v13_negative_inf_i64(%arg0 : tensor<2x3xi64>) -> tensor<*xi64> {
  %0 ="onnx.ReduceMaxV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xi64>)-> tensor<*xi64>
  "func.return"(%0) : (tensor<*xi64>) -> ()
  // CHECK-LABEL: test_reducemax_v13_negative_inf_i64
  // CHECK: arith.constant -9223372036854775808 : i64
}

// -----


func.func private @test_reducemin_v13(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducemin_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             krnl.store [[CST_0_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<3x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_4_:%.+]] = 0 to 2){
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[I_3_]], [[I_4_]]{{.}} : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[I_2_]], [[I_4_]]{{.}} : memref<3x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.cmpf olt, [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_5_:%.+]] = arith.select [[VAR_4_]], [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[I_2_]], [[I_4_]]{{.}} : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----

func.func private @test_reducemin_v13_positive_inf_f32(%arg0 : tensor<2x3xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_reducemin_v13_positive_inf_f32
  // CHECK: arith.constant 0x7F800000 : f32
}

// -----

func.func private @test_reducemin_v13_positive_inf_f64(%arg0 : tensor<2x3xf64>) -> tensor<*xf64> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xf64>)-> tensor<*xf64>
  "func.return"(%0) : (tensor<*xf64>) -> ()
  // CHECK-LABEL: test_reducemin_v13_positive_inf_f64
  // CHECK: arith.constant 0x7FF0000000000000 : f64
}

// -----

func.func private @test_reducemin_v13_positive_inf_i8(%arg0 : tensor<2x3xi8>) -> tensor<*xi8> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xi8>)-> tensor<*xi8>
  "func.return"(%0) : (tensor<*xi8>) -> ()
  // CHECK-LABEL: test_reducemin_v13_positive_inf_i8
  // CHECK: arith.constant 127 : i8
}

// -----

func.func private @test_reducemin_v13_positive_inf_i32(%arg0 : tensor<2x3xi32>) -> tensor<*xi32> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xi32>)-> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()
  // CHECK-LABEL: test_reducemin_v13_positive_inf_i32
  // CHECK: arith.constant 2147483647 : i32
}

// -----

func.func private @test_reducemin_v13_positive_inf_i64(%arg0 : tensor<2x3xi64>) -> tensor<*xi64> {
  %0 ="onnx.ReduceMinV13"(%arg0) {axes=[0], keepdims = 0 : si64} : (tensor<2x3xi64>)-> tensor<*xi64>
  "func.return"(%0) : (tensor<*xi64>) -> ()
  // CHECK-LABEL: test_reducemin_v13_positive_inf_i64
  // CHECK: arith.constant 9223372036854775807 : i64
}

// -----


func.func private @test_reduceprod_v13(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceProdV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reduceprod_v13
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             krnl.store [[CST_1_dot_000000_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<3x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_4_:%.+]] = 0 to 2){
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[I_3_]], [[I_4_]]{{.}} : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[I_2_]], [[I_4_]]{{.}} : memref<3x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.mulf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[I_2_]], [[I_4_]]{{.}} : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----


func.func private @test_reducesum(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %cst = onnx.Constant dense<[1]> : tensor<1xi64>
  %0 ="onnx.ReduceSum"(%arg0, %cst) {keepdims = 0 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<1xi64>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducesum
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<3x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_4_:%.+]] = 0 to 2){
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[I_3_]], [[I_4_]]{{.}} : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[I_2_]], [[I_4_]]{{.}} : memref<3x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[I_2_]], [[I_4_]]{{.}} : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----


func.func private @test_reducesumV11(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceSumV11"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducesumV11
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<3x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_4_:%.+]] = 0 to 2){
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_2_]], [[I_3_]], [[I_4_]]{{.}} : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[I_2_]], [[I_4_]]{{.}} : memref<3x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[I_2_]], [[I_4_]]{{.}} : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----


func.func private @test_reducesum1(%arg0: tensor<3x2x2xf32>, %arg1: tensor<?xi64>) -> tensor<3x1x2xf32> {
  %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 1 : si64} : (tensor<3x2x2xf32>, tensor<?xi64>) -> tensor<3x1x2xf32>
  return %0 : tensor<3x1x2xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_reducesum1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>, [[PARAM_1_:%.+]]: memref<?xi64>) -> memref<3x1x2xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_1_]] : memref<?xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3xi1>
// CHECK:           krnl.store [[VAR_false_]], [[RES_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi1>
// CHECK:           krnl.store [[VAR_false_]], [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<3xi1>
// CHECK:           krnl.store [[VAR_false_]], [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi1>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_]]{{.}} : memref<?xi64>
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi slt, [[LOAD_PARAM_1_MEM_]], [[CST_0_]] : i64
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_]], [[CST_3_]] : i64
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_5_]], [[VAR_6_]], [[LOAD_PARAM_1_MEM_]] : i64
// CHECK:             [[VAR_8_:%.+]] = arith.index_cast [[VAR_7_]] : i64 to index
// CHECK:             krnl.store [[VAR_true_]], [[RES_]]{{.}}[[VAR_8_]]{{.}} : memref<3xi1>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x1x2xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 1, [[LOOP_1_]]#2 -> [[I_3_:%.+]] = 0 to 2){
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]]{{.}}[[I_1_]], [[I_2_]], [[I_3_]]{{.}} : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_2_]]#2 -> [[I_6_:%.+]] = 0 to 2){
// CHECK:             [[VAR_3_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi1>
// CHECK:             [[LOAD_PARAM_1_MEM_1_:%.+]] = arith.cmpi eq, [[VAR_3_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_5_1_:%.+]] = arith.select [[LOAD_PARAM_1_MEM_1_]], [[CST_0_1_]], [[I_4_]] : index
// CHECK-DAG:         [[VAR_6_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7_1_:%.+]] = arith.cmpi eq, [[VAR_6_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_8_1_:%.+]] = arith.select [[VAR_7_1_]], [[CST_0_1_]], [[I_5_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_10_:%.+]] = arith.cmpi eq, [[LOAD_RES_MEM_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.select [[VAR_10_]], [[CST_0_1_]], [[I_6_]] : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_4_]], [[I_5_]], [[I_6_]]{{.}} : memref<3x2x2xf32>
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_8_1_]], [[VAR_11_]]{{.}} : memref<3x1x2xf32>
// CHECK:             [[VAR_14_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_14_]], [[RES_1_]]{{.}}[[VAR_5_1_]], [[VAR_8_1_]], [[VAR_11_]]{{.}} : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<3x1x2xf32>
// CHECK:         }
}

// -----


func.func @test_reducesum2(%arg0: tensor<3x2x2xf32>, %arg1: tensor<?xi64>) -> tensor<3x1x2xf32> {
  %0 = "onnx.ReduceSum"(%arg0, %arg1) {keepdims = 1 : si64, noop_with_empty_axes = 0 : si64} : (tensor<3x2x2xf32>, tensor<?xi64>) -> tensor<3x1x2xf32>
  return %0 : tensor<3x1x2xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_reducesum2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>, [[PARAM_1_:%.+]]: memref<?xi64>) -> memref<3x1x2xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_1_]] : memref<?xi64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3xi1>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_1_]] : memref<?xi64>
// CHECK:           [[VAR_0_:%.+]] = arith.cmpi eq, [[VAR_dim_0_]], [[CST_0_1_]] : index
// CHECK:           krnl.store [[VAR_0_]], [[RES_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi1>
// CHECK:           krnl.store [[VAR_0_]], [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<3xi1>
// CHECK:           krnl.store [[VAR_0_]], [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi1>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_4_]]{{.}} : memref<?xi64>
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi slt, [[LOAD_PARAM_1_MEM_]], [[CST_0_]] : i64
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[LOAD_PARAM_1_MEM_]], [[CST_3_]] : i64
// CHECK:             [[VAR_8_:%.+]] = arith.select [[VAR_6_]], [[VAR_7_]], [[LOAD_PARAM_1_MEM_]] : i64
// CHECK:             [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : i64 to index
// CHECK:             krnl.store [[VAR_true_]], [[RES_]]{{.}}[[VAR_9_]]{{.}} : memref<3xi1>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x1x2xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 1, [[LOOP_1_]]#2 -> [[I_3_:%.+]] = 0 to 2){
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]]{{.}}[[I_1_]], [[I_2_]], [[I_3_]]{{.}} : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_2_]]#2 -> [[I_6_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi1>
// CHECK:             [[LOAD_PARAM_1_MEM_1_:%.+]] = arith.cmpi eq, [[VAR_4_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_6_1_:%.+]] = arith.select [[LOAD_PARAM_1_MEM_1_]], [[CST_0_1_]], [[I_4_]] : index
// CHECK-DAG:         [[VAR_7_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_8_1_:%.+]] = arith.cmpi eq, [[VAR_7_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_9_1_:%.+]] = arith.select [[VAR_8_1_]], [[CST_0_1_]], [[I_5_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_11_:%.+]] = arith.cmpi eq, [[LOAD_RES_MEM_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_0_1_]], [[I_6_]] : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_4_]], [[I_5_]], [[I_6_]]{{.}} : memref<3x2x2xf32>
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_6_1_]], [[VAR_9_1_]], [[VAR_12_]]{{.}} : memref<3x1x2xf32>
// CHECK:             [[VAR_15_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_15_]], [[RES_1_]]{{.}}[[VAR_6_1_]], [[VAR_9_1_]], [[VAR_12_]]{{.}} : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<3x1x2xf32>
// CHECK:         }
}

// -----

func.func private @test_unsqueeze(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[0, 3]> : tensor<2xi64>
  %1 = "onnx.Unsqueeze"(%arg0, %0) : (tensor<10x10xf32>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_unsqueeze
  // CHECK: [[RES:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 10, 10, 1], strides: [100, 10, 1, 1] : memref<10x10xf32> to memref<1x10x10x1xf32>
  // CHECK: return [[RES]] : memref<1x10x10x1xf32>
}

// -----

func.func private @test_unsqueezev11(%arg0 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.UnsqueezeV11"(%arg0) {axes=[0,3]} : (tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_unsqueezev11
  // CHECK: [[RES:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [1, 10, 10, 1], strides: [100, 10, 1, 1] : memref<10x10xf32> to memref<1x10x10x1xf32>
  // CHECK: return [[RES]] : memref<1x10x10x1xf32>
}

// -----

// `UnsqueezeV11` ops are lowerd to `reinterpret_cast` op. `reinterpret_cast` ops just
// change the view of input memref. So, input memref should not be deallocated if it
// is retuned. This test confirms the deallocation is not generated.

func.func private @test_unsqueeze_dealloc(%arg0 : tensor<10x20xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[0, 3]> : tensor<2xi64>
  %1 = "onnx.Transpose"(%arg0) : (tensor<10x20xf32>) -> tensor<*xf32>
  %2 = "onnx.Unsqueeze"(%1, %0) : (tensor<*xf32>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%2) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: func private @test_unsqueeze_dealloc
  // CHECK:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<20x10xf32>
  // CHECK:       [[VAR_2_:%.+]] = memref.reinterpret_cast [[VAR_0_]] to offset: [0], sizes: [1, 20, 10, 1], strides: [200, 10, 1, 1] : memref<20x10xf32> to memref<1x20x10x1xf32>
  // CHECK-NOT:   memref.dealloc [[VAR_0_]] : memref<20x10xf32>
  // CHECK:	  return [[VAR_2_]] : memref<1x20x10x1xf32>
}

// -----

func.func private @test_unsqueezev11_dealloc(%arg0 : tensor<10x20xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<10x20xf32>) -> tensor<*xf32>
  %1 = "onnx.UnsqueezeV11"(%0) {axes=[0,3]} : (tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: func private @test_unsqueezev11_dealloc
  // CHECK:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<20x10xf32>
  // CHECK:       [[VAR_2_:%.+]] = memref.reinterpret_cast [[VAR_0_]] to offset: [0], sizes: [1, 20, 10, 1], strides: [200, 10, 1, 1] : memref<20x10xf32> to memref<1x20x10x1xf32>
  // CHECK-NOT:   memref.dealloc [[VAR_0_]] : memref<20x10xf32>
  // CHECK:	  return [[VAR_2_]] : memref<1x20x10x1xf32>
}

// -----

// Test for multiple `reinterpret_cast` in a function. Only returned memrefs should not be deallocated.

func.func private @test_unsqueeze_squeeze_dealloc(%arg0 : tensor<10x20xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[1, -1]> : tensor<2xi64>
  %1 = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %2 = "onnx.Transpose"(%arg0) : (tensor<10x20xf32>) -> tensor<*xf32>
  %3 = "onnx.Unsqueeze"(%2, %0) : (tensor<*xf32>, tensor<2xi64>) -> tensor<*xf32>
  %4 = "onnx.Transpose"(%3) {perm = [0, 3, 1, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  %5 = "onnx.Squeeze"(%4, %1) : (tensor<*xf32>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%5) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_unsqueeze_squeeze_dealloc
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x20xf32>) -> memref<20x10xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<20x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 20){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<10x20xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#1, [[VAR_1_]]#0] : memref<20x10xf32>
// CHECK:           }
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [20, 10], strides: [10, 1] : memref<20x10xf32> to memref<20x10xf32>
// CHECK:           return [[VAR_reinterpret_cast_]] : memref<20x10xf32>
// CHECK:         }
}

// -----


func.func private @test_unsqueezev11_squeezev11_dealloc(%arg0 : tensor<10x20xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<10x20xf32>) -> tensor<*xf32>
  %1 = "onnx.UnsqueezeV11"(%0) { axes = [1, -1]} : (tensor<*xf32>) -> (tensor<*xf32>)
  %2 = "onnx.Transpose"(%1) {perm = [0, 3, 1, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  %3 = "onnx.SqueezeV11"(%2) {axes=[1, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%3) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_unsqueezev11_squeezev11_dealloc
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x20xf32>) -> memref<20x10xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<20x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 20){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<10x20xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#1, [[VAR_1_]]#0] : memref<20x10xf32>
// CHECK:           }
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [20, 10], strides: [10, 1] : memref<20x10xf32> to memref<20x10xf32>
// CHECK:           return [[VAR_reinterpret_cast_]] : memref<20x10xf32>
// CHECK:         }
}

// -----

func.func private @test_transpose(%arg0 : tensor<10x20x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<10x20x30x40xf32>) -> tensor<*xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_transpose
  // CHECK:       [[RES1:%.+]] = memref.alloc() {{.*}}: memref<40x30x20x10xf32>
  // CHECK:       [[DEF_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK:       krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to 10,
  // CHECK-SAME:               [[DEF_LOOPS]]#1 -> %arg2 = 0 to 20, [[DEF_LOOPS]]#2 -> %arg3 = 0 to 30, [[DEF_LOOPS]]#3 -> %arg4 = 0 to 40){
  // CHECK-NEXT:  [[IV:%.+]]:4 = krnl.get_induction_var_value([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) :
  // CHECK-SAME:     (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
  // CHECK:       [[LOAD:%.+]] = krnl.load %arg0{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3{{.}} : memref<10x20x30x40xf32>
  // CHECK:       krnl.store [[LOAD]], [[RES1]]{{.}}[[IV]]#3, [[IV]]#2, [[IV]]#1, [[IV]]#0{{.}} : memref<40x30x20x10xf32>
  // CHECK:       [[RES0:%.+]] = memref.alloc() {{.*}}: memref<40x10x30x20xf32>
  // CHECK:       [[DEF_LOOPS1:%.+]]:4 = krnl.define_loops 4
  // CHECK:       krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1, [[DEF_LOOPS1]]#2, [[DEF_LOOPS1]]#3) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 40,
  // CHECK-SAME:               [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 30, [[DEF_LOOPS1]]#2 -> %arg3 = 0 to 20, [[DEF_LOOPS1]]#3 -> %arg4 = 0 to 10){
  // CHECK-NEXT:  [[IV1:%.+]]:4 = krnl.get_induction_var_value([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1, [[DEF_LOOPS1]]#2, [[DEF_LOOPS1]]#3) :
  // CHECK-SAME:     (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
  // CHECK:       [[LOAD:%.+]] = krnl.load [[RES1]]{{.}}[[IV1]]#0, [[IV1]]#1, [[IV1]]#2, [[IV1]]#3{{.}} : memref<40x30x20x10xf32>
  // CHECK:       krnl.store [[LOAD]], [[RES0]]{{.}}[[IV1]]#0, [[IV1]]#3, [[IV1]]#1, [[IV1]]#2{{.}} : memref<40x10x30x20xf32>
  // CHECK:       return [[RES0]] : memref<40x10x30x20xf32>
}

// -----

// COM: Test whether the lowering is correct in the presence of dynamic dimensions.

func.func private @test_transpose_dynamic_dims(%arg0 : tensor<10x?x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<10x?x30x40xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_transpose_dynamic_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x?x30x40xf32>) -> memref<10x40x?x30xf32> {
// CHECK:           [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<10x?x30x40xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<10x40x?x30xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<10x?x30x40xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 30, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 40){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<10x?x30x40xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#3, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<10x40x?x30xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x40x?x30xf32>
// CHECK:         }
}

// -----

func.func private @test_identity(%arg0 : tensor<10x20x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Identity"(%arg0) : (tensor<10x20x30x40xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_identity
  // CHECK: return %arg0 : memref<10x20x30x40xf32>
}

// -----

func.func private @test_batchnorm_testmode_Nd(%arg0: tensor<1x2x1x3xf32>, %arg1: tensor<2xf32>, %arg2: tensor<2xf32>, %arg3: tensor<2xf32>, %arg4: tensor<2xf32>) -> tensor<1x2x1x3xf32> {
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


func.func private @test_batchnorm_testmode_1d(%arg0: tensor<10xf32>, %arg1: tensor<1xf32>, %arg2: tensor<1xf32>, %arg3: tensor<1xf32>, %arg4: tensor<1xf32>) -> tensor<10xf32> {
  %0 = "onnx.BatchNormalizationInferenceMode"(%arg0, %arg1, %arg2, %arg3, %arg4) : (tensor<10xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<10xf32>
  return %0 : tensor<10xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_batchnorm_testmode_1d
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10xf32>, [[PARAM_1_:%.+]]: memref<1xf32>, [[PARAM_2_:%.+]]: memref<1xf32>, [[PARAM_3_:%.+]]: memref<1xf32>, [[PARAM_4_:%.+]]: memref<1xf32>) -> memref<10xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_9_dot_99999974_:%.+]] = arith.constant 9.99999974E-6 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_3_MEM_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[CST_0_]]{{.}} : memref<1xf32>
// CHECK-DAG:       [[LOAD_PARAM_4_MEM_:%.+]] = krnl.load [[PARAM_4_]]{{.}}[[CST_0_]]{{.}} : memref<1xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 10){
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]]{{.}} : memref<10xf32>
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_3_MEM_]] : f32
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addf [[LOAD_PARAM_4_MEM_]], [[CST_9_dot_99999974_]] : f32
// CHECK:             [[VAR_8_:%.+]] = math.sqrt [[VAR_7_]] : f32
// CHECK:             [[VAR_9_:%.+]] = arith.divf [[VAR_6_]], [[VAR_8_]] : f32
// CHECK:             [[VAR_10_:%.+]] = arith.mulf [[LOAD_PARAM_1_MEM_]], [[VAR_9_]] : f32
// CHECK:             [[VAR_11_:%.+]] = arith.addf [[VAR_10_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             krnl.store [[VAR_11_]], [[RES_]]{{.}}[[I_0_]]{{.}} : memref<10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10xf32>
// CHECK:         }
}

// -----

func.func private @test_batchnorm_testmode_2d(%arg0: tensor<10x3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>, %arg3: tensor<3xf32>, %arg4: tensor<3xf32>) -> tensor<10x3xf32> {
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

func.func private @test_constant_dense_2d_value(%arg0: tensor<1xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant {value = dense<[[0.0, 0.0], [1.0, 1.1], [2.0, 2.1]]> : tensor<3x2xf32>} : tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_constant_dense_2d_value
  // CHECK: [[GLOBAL:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3, 2], value = dense<{{.*}}[0.000000e+00, 0.000000e+00], [1.000000e+00, 1.100000e+00], [2.000000e+00, 2.100000e+00]{{.*}}> : tensor<3x2xf32>} : () -> memref<3x2xf32>
  // CHECK: return [[GLOBAL]] : memref<3x2xf32>
}

// -----


func.func private @test_pool_general_computation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (0, d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (32, d0 + 2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>
// CHECK-LABEL:  func.func private @test_pool_general_computation
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x32x32xf32>) -> memref<1x3x31x31xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x3x31x31xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 31, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 31){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.max [[MAP_0_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.min [[MAP_1_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.max [[MAP_0_]]([[VAR_1_]]#3)
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.min [[MAP_1_]]([[VAR_1_]]#3)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.subi [[VAR_3_]], [[VAR_2_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.subi [[VAR_5_]], [[VAR_4_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to min [[MAP_2_]]([[VAR_1_]]#2){{.}}[[CST_32_]], [[CST_2_]], [[CST_0_]], [[CST_1_]], [[CST_1_]]{{.}}, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to min [[MAP_2_]]([[VAR_1_]]#3){{.}}[[CST_32_]], [[CST_2_]], [[CST_0_]], [[CST_1_]], [[CST_1_]]{{.}}){
// CHECK-DAG:           [[VAR_15_:%.+]] = arith.addi [[I_4_]], [[VAR_2_]] : index
// CHECK-DAG:           [[VAR_16_:%.+]] = arith.addi [[I_5_]], [[VAR_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]5, [[VAR_1_]]6] : memref<1x3x32x32xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:               krnl.store [[VAR_19_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<1x3x31x31xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<1x3x31x31xf32>
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.muli [[VAR_6_]], [[VAR_7_]] : index
// CHECK:             [[VAR_12_:%.+]] = arith.index_cast [[VAR_11_]] : index to i64
// CHECK:             [[VAR_13_:%.+]] = arith.sitofp [[VAR_12_]] : i64 to f32
// CHECK:             [[VAR_14_:%.+]] = arith.divf [[LOAD_RES_MEM_]], [[VAR_13_]] : f32
// CHECK:             krnl.store [[VAR_14_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<1x3x31x31xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x3x31x31xf32>
// CHECK:         }
}

// -----


func.func private @test_averagepool_identity_value(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (0, d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (32, d0 + 2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>
// CHECK-LABEL:  func.func private @test_averagepool_identity_value
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x32x32xf32>) -> memref<1x3x31x31xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x3x31x31xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 31, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 31){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.max [[MAP_0_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.min [[MAP_1_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.max [[MAP_0_]]([[VAR_1_]]#3)
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.min [[MAP_1_]]([[VAR_1_]]#3)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.subi [[VAR_3_]], [[VAR_2_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.subi [[VAR_5_]], [[VAR_4_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to min [[MAP_2_]]([[VAR_1_]]#2){{.}}[[CST_32_]], [[CST_2_]], [[CST_0_]], [[CST_1_]], [[CST_1_]]{{.}}, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to min [[MAP_2_]]([[VAR_1_]]#3){{.}}[[CST_32_]], [[CST_2_]], [[CST_0_]], [[CST_1_]], [[CST_1_]]{{.}}){
// CHECK-DAG:           [[VAR_15_:%.+]] = arith.addi [[I_4_]], [[VAR_2_]] : index
// CHECK-DAG:           [[VAR_16_:%.+]] = arith.addi [[I_5_]], [[VAR_4_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]5, [[VAR_1_]]6] : memref<1x3x32x32xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:               krnl.store [[VAR_19_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<1x3x31x31xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<1x3x31x31xf32>
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.muli [[VAR_6_]], [[VAR_7_]] : index
// CHECK:             [[VAR_12_:%.+]] = arith.index_cast [[VAR_11_]] : index to i64
// CHECK:             [[VAR_13_:%.+]] = arith.sitofp [[VAR_12_]] : i64 to f32
// CHECK:             [[VAR_14_:%.+]] = arith.divf [[LOAD_RES_MEM_]], [[VAR_13_]] : f32
// CHECK:             krnl.store [[VAR_14_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<1x3x31x31xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x3x31x31xf32>
// CHECK:         }
}

// -----


func.func private @test_maxpool_identity_value(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (0, d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>
// CHECK-LABEL:  func.func private @test_maxpool_identity_value
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x32x32xf32>) -> memref<1x3x31x31xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x3x31x31xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 31, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 31){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             krnl.store [[CST_0_]], [[RES_1_]][] : memref<f32>
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.max [[MAP_0_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.max [[MAP_0_]]([[VAR_1_]]#3)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to min [[MAP_1_]]([[VAR_1_]]#2){{.}}[[CST_32_]], [[CST_2_]], [[CST_0_1_]], [[CST_1_]], [[CST_1_]]{{.}}, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to min [[MAP_1_]]([[VAR_1_]]#3){{.}}[[CST_32_]], [[CST_2_]], [[CST_0_1_]], [[CST_1_]], [[CST_1_]]{{.}}){
// CHECK-DAG:           [[VAR_6_:%.+]] = arith.addi [[I_4_]], [[VAR_2_]] : index
// CHECK-DAG:           [[VAR_7_:%.+]] = arith.addi [[I_5_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_6_]], [[VAR_7_]]{{.}} : memref<1x3x32x32xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_10_:%.+]] = arith.cmpf ogt, [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:               [[VAR_11_:%.+]] = arith.select [[VAR_10_]], [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:               krnl.store [[VAR_11_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<1x3x31x31xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x3x31x31xf32>
// CHECK:         }
}

// -----

func.func private @test_averagepool_pooling_operation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_averagepool_pooling_operation
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<1x3x31x31xf32>

  // CHECK: [[REDUCTION_VAL:%.+]] = memref.alloca() : memref<f32>
  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31){
  // CHECK:   [[IV:%.+]]:4 = krnl.get_induction_var_value([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)

  // CHECK:   krnl.store {{.*}}, [[REDUCTION_VAL]][] : memref<f32>

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #{{.*}}([[IV]]#2)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #{{.*}}([[IV]]#3)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]){

  // CHECK:     [[INPUT_LOAD:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     [[OUTPUT_LOAD:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:     [[SUM:%.+]] = arith.addf [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     krnl.store [[SUM]], [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   }
  // CHECK:   [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   krnl.store [[LOAD_REDUCTION]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3] : memref<1x3x31x31xf32>

  // CHECK:   [[NUMERATOR:%.+]] = krnl.load [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3] : memref<1x3x31x31xf32>
  // CHECK:   [[AVERAGE:%.+]] = arith.divf [[NUMERATOR]], {{.*}} : f32
  // CHECK:   krnl.store [[AVERAGE]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3] : memref<1x3x31x31xf32>
  // CHECK: }
}

// -----


func.func private @test_maxpool_pooling_operation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (0, d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>
// CHECK-LABEL:  func.func private @test_maxpool_pooling_operation
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x32x32xf32>) -> memref<1x3x31x31xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x3x31x31xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 31, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 31){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             krnl.store [[CST_0_]], [[RES_1_]][] : memref<f32>
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.max [[MAP_0_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.max [[MAP_0_]]([[VAR_1_]]#3)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to min [[MAP_1_]]([[VAR_1_]]#2){{.}}[[CST_32_]], [[CST_2_]], [[CST_0_1_]], [[CST_1_]], [[CST_1_]]{{.}}, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to min [[MAP_1_]]([[VAR_1_]]#3){{.}}[[CST_32_]], [[CST_2_]], [[CST_0_1_]], [[CST_1_]], [[CST_1_]]{{.}}){
// CHECK-DAG:           [[VAR_6_:%.+]] = arith.addi [[I_4_]], [[VAR_2_]] : index
// CHECK-DAG:           [[VAR_7_:%.+]] = arith.addi [[I_5_]], [[VAR_3_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_6_]], [[VAR_7_]]{{.}} : memref<1x3x32x32xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_10_:%.+]] = arith.cmpf ogt, [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:               [[VAR_11_:%.+]] = arith.select [[VAR_10_]], [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:               krnl.store [[VAR_11_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<1x3x31x31xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x3x31x31xf32>
// CHECK:         }
}

// -----

func.func private @test_squeeze(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[1, -2]> : tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<16x1x32x1x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_squeeze
  // CHECK: [[RES:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 32, 64], strides: [2048, 64, 1] : memref<16x1x32x1x64xf32> to memref<16x32x64xf32>
  // CHECK: return [[RES]] : memref<16x32x64xf32>
}

// -----

func.func private @test_squeezev11(%arg0 : tensor<16x1x32x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) { axes = [1, -2]} : (tensor<16x1x32x1x64xf32>) -> (tensor<*xf32>)
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_squeezev11
  // CHECK: [[RES:%.+]] = memref.reinterpret_cast %arg0 to offset: [0], sizes: [16, 32, 64], strides: [2048, 64, 1] : memref<16x1x32x1x64xf32> to memref<16x32x64xf32>
  // CHECK: return [[RES]] : memref<16x32x64xf32>
}

// -----

// `SqueezeV11` ops are lowerd to `reinterpret_cast` op. `reinterpret_cast` ops just change the view of input memref. So, input memref should not be deallocated if it is retuned. This test confirms the deallocation is not generated.

func.func private @test_squeeze_dealloc(%arg0 : tensor<16x32x1x1x64xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[1, -2]> : tensor<2xi64>
  %1 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2, 4]} : (tensor<16x32x1x1x64xf32>) -> tensor<*xf32>
  %2 = "onnx.Squeeze"(%1, %0) : (tensor<*xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "func.return"(%2) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_squeeze_dealloc
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32x1x1x64xf32>) -> memref<16x32x64xf32> {
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [16, 32, 64], strides: [2048, 64, 1] : memref<16x32x1x1x64xf32> to memref<16x32x64xf32>
// CHECK:           return [[VAR_reinterpret_cast_]] : memref<16x32x64xf32>
// CHECK:         }
}

// -----


func.func private @test_squeezev11_dealloc(%arg0 : tensor<16x32x1x1x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2, 4]} : (tensor<16x32x1x1x64xf32>) -> tensor<*xf32>
  %1 = "onnx.SqueezeV11"(%0) { axes = [1, -2]} : (tensor<*xf32>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_squeezev11_dealloc
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32x1x1x64xf32>) -> memref<16x32x64xf32> {
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [16, 32, 64], strides: [2048, 64, 1] : memref<16x32x1x1x64xf32> to memref<16x32x64xf32>
// CHECK:           return [[VAR_reinterpret_cast_]] : memref<16x32x64xf32>
// CHECK:         }
}

// -----

func.func private @test_split_equal(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = 0 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 8)>
  // CHECK-LABEL: @test_split_equal
  // CHECK:     [[RES_0:%.+]] = memref.alloc() {{.*}}: memref<8x32x64xf32>
  // CHECK:     [[RES_1:%.+]] = memref.alloc() {{.*}}: memref<8x32x64xf32>
  // CHECK:     [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK:     krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 8,
  // CHECK-SAME:             [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64){
  // CHECK:       [[IV:%.+]]:3 = krnl.get_induction_var_value([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) :
  // CHECK-SAME:    (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:       [[LOAD_0:%.+]] = krnl.load %arg0{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:       krnl.store [[LOAD_0]], [[RES_0]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<8x32x64xf32>
  // CHECK:     }
  // CHECK:     [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK:     krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 8,
  // CHECK-SAME:             [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64){
  // CHECK:       [[IV:%.+]]:3 = krnl.get_induction_var_value([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) :
  // CHECK-SAME:    (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:       [[INDEX:%.+]] = affine.apply [[INDEX_MAP]]{{.}}[[IV]]#0{{.}}
  // CHECK:       [[LOAD_1:%.+]] = krnl.load %arg0{{.}}[[INDEX]], [[IV]]#1, [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:       krnl.store [[LOAD_1]], [[RES_1]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<8x32x64xf32>
  // CHECK:     }
  // CHECK:     return [[RES_0]], [[RES_1]] : memref<8x32x64xf32>, memref<8x32x64xf32>
}

// -----

func.func private @test_split_variable(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %split = onnx.Constant dense<[2, 30]> : tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) { axis = 1 : si64} : (tensor<16x32x64xf32>, tensor<2xi64>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 2)>
  // CHECK-LABEL: @test_split_variable

  // CHECK: [[RES_0:%.+]] = memref.alloc() {{.*}}: memref<16x2x64xf32>
  // CHECK: [[RES_1:%.+]] = memref.alloc() {{.*}}: memref<16x30x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 2, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[IV:%.+]]:3 = krnl.get_induction_var_value([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:   [[LOAD_0:%.+]] = krnl.load %arg0{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_0]], [[RES_0]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x2x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 30, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[IV:%.+]]:3 = krnl.get_induction_var_value([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:   [[INDEX:%.+]] = affine.apply [[INDEX_MAP]]{{.}}[[IV]]#1{{.}}
  // CHECK:   [[LOAD_1:%.+]] = krnl.load %arg0{{.}}[[IV]]#0, [[INDEX]], [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_1]], [[RES_1]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x30x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<16x2x64xf32>, memref<16x30x64xf32>
}

// -----

func.func private @test_splitv11_equal(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.SplitV11"(%arg0) { axis = 0 : si64} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 8)>
  // CHECK-LABEL: @test_splitv11_equal

  // CHECK: [[RES_0:%.+]] = memref.alloc() {{.*}}: memref<8x32x64xf32>
  // CHECK: [[RES_1:%.+]] = memref.alloc() {{.*}}: memref<8x32x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 8, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[IV:%.+]]:3 = krnl.get_induction_var_value([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:   [[LOAD_0:%.+]] = krnl.load %arg0{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_0]], [[RES_0]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<8x32x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 8, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[IV:%.+]]:3 = krnl.get_induction_var_value([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:   [[INDEX:%.+]] = affine.apply [[INDEX_MAP]]{{.}}[[IV]]#0{{.}}
  // CHECK:   [[LOAD_1:%.+]] = krnl.load %arg0{{.}}[[INDEX]], [[IV]]#1, [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_1]], [[RES_1]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<8x32x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<8x32x64xf32>, memref<8x32x64xf32>
}

// -----

func.func private @test_splitv11_variable(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.SplitV11"(%arg0) { axis = 1 : si64, split = [2, 30]} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 2)>
  // CHECK-LABEL: @test_splitv11_variable

  // CHECK: [[RES_0:%.+]] = memref.alloc() {{.*}}: memref<16x2x64xf32>
  // CHECK: [[RES_1:%.+]] = memref.alloc() {{.*}}: memref<16x30x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 2, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[IV:%.+]]:3 = krnl.get_induction_var_value([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:   [[LOAD_0:%.+]] = krnl.load %arg0{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_0]], [[RES_0]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x2x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 30, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[INDEX:%.+]] = affine.apply [[INDEX_MAP]]{{.}}[[IV]]#1{{.}}
  // CHECK:   [[LOAD_1:%.+]] = krnl.load %arg0{{.}}[[IV]]#0, [[INDEX]], [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_1]], [[RES_1]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x30x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<16x2x64xf32>, memref<16x30x64xf32>
}

// -----

func.func private @cast_lowering_sametype(%arg0: tensor<f32>) -> tensor<f32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<f32>) -> tensor<f32>
  "func.return"(%0) : (tensor<f32>) -> ()

  // CHECK-LABEL: cast_lowering_sametype
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<f32>
  // CHECK: krnl.store [[LOAD]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func.func private @cast_lowering_intfloat(%arg0: tensor<i64>) -> tensor<f32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<i64>) -> tensor<f32>
  "func.return"(%0) : (tensor<f32>) -> ()

  // CHECK-LABEL: cast_lowering_intfloat
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<i64>
  // CHECK: [[VAL:%.+]] = arith.sitofp [[LOAD]] : i64 to f32
  // CHECK: krnl.store [[VAL]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func.func private @cast_lowering_floatint(%arg0: tensor<f32>) -> tensor<i64> {
  %0 = "onnx.Cast"(%arg0) {to = i64} : (tensor<f32>) -> tensor<i64>
  "func.return"(%0) : (tensor<i64>) -> ()

  // CHECK-LABEL: cast_lowering_floatint
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<i64>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<f32>
  // CHECK: [[VAL:%.+]] = arith.fptosi [[LOAD]] : f32 to i64
  // CHECK: krnl.store [[VAL]], [[RES]][] : memref<i64>
  // CHECK: return [[RES]] : memref<i64>
}

// -----

func.func private @cast_lowering_f16f32(%arg0: tensor<f16>) -> tensor<f32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<f16>) -> tensor<f32>
  "func.return"(%0) : (tensor<f32>) -> ()

  // CHECK-LABEL: cast_lowering_f16f32
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<f16>
  // CHECK: [[VAL:%.+]] = arith.extf [[LOAD]] : f16 to f32
  // CHECK: krnl.store [[VAL]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func.func private @cast_lowering_f64f32(%arg0: tensor<f64>) -> tensor<f32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<f64>) -> tensor<f32>
  "func.return"(%0) : (tensor<f32>) -> ()

  // CHECK-LABEL: cast_lowering_f64f32
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<f32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<f64>
  // CHECK: [[VAL:%.+]] = arith.truncf [[LOAD]] : f64 to f32
  // CHECK: krnl.store [[VAL]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
}

// -----

func.func private @cast_lowering_f64f32_10(%arg0: tensor<10xf64>) -> tensor<*xf32> {
  %0 = "onnx.Cast"(%arg0) {to = f32} : (tensor<10xf64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: cast_lowering_f64f32_10
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<10xf32>
  // CHECK: [[DEF_LOOPS:%.+]] = krnl.define_loops 1
  // CHECK: krnl.iterate([[DEF_LOOPS]]) with ([[DEF_LOOPS]] -> %arg1 = 0 to 10){
  // CHECK: [[IV:%.+]] = krnl.get_induction_var_value([[DEF_LOOPS]]) : (!krnl.loop) -> index
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg0[[[IV]]] : memref<10xf64>
  // CHECK: [[FPTRUNC:%.+]] = arith.truncf [[LOAD1]] : f64 to f32
  // CHECK: krnl.store [[FPTRUNC]], [[RES]][[[IV]]] : memref<10xf32>
  // CHECK: return [[RES]] : memref<10xf32>
}

// -----

func.func private @cast_lowering_int_wider_int(%arg0: tensor<i32>) -> tensor<i64> {
  %0 = "onnx.Cast"(%arg0) {to = i64} : (tensor<i32>) -> tensor<i64>
  "func.return"(%0) : (tensor<i64>) -> ()

  // CHECK-LABEL: cast_lowering_int_wider_int
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<i64>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<i32>
  // CHECK: [[CAST:%.+]] = arith.extsi [[LOAD]] : i32 to i64
  // CHECK: krnl.store [[CAST]], [[RES]][] : memref<i64>
  // CHECK: return [[RES]] : memref<i64>
}

// -----

func.func private @cast_lowering_int_narrow_int(%arg0: tensor<i64>) -> tensor<i32> {
  %0 = "onnx.Cast"(%arg0) {to = i32 } : (tensor<i64>) -> tensor<i32>
  "func.return"(%0) : (tensor<i32>) -> ()

  // CHECK-LABEL: cast_lowering_int_narrow_int
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<i32>
  // CHECK: [[LOAD:%.+]] = krnl.load %arg0[] : memref<i64>
  // CHECK: [[CAST:%.+]] = arith.trunci [[LOAD]] : i64 to i32
  // CHECK: krnl.store [[CAST]], [[RES]][] : memref<i32>
  // CHECK: return [[RES]] : memref<i32>
}

// -----

func.func private @cast_lowering_int_to_bool(%arg0: tensor<i64>) -> tensor<i1> {
  %0 = "onnx.Cast"(%arg0) {to = i1 } : (tensor<i64>) -> tensor<i1>
  "func.return"(%0) : (tensor<i1>) -> ()

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

func.func private @cast_lowering_uint_to_bool(%arg0 : tensor<ui8>) -> tensor<i1> {
  %0 = "onnx.Cast"(%arg0) {to = i1} : (tensor<ui8>) -> tensor<i1>
  "func.return"(%0): (tensor<i1>) -> ()

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

func.func private @cast_lowering_float_to_bool(%arg0: tensor<f32>) -> tensor<i1> {
  %0 = "onnx.Cast"(%arg0) {to = i1 } : (tensor<f32>) -> tensor<i1>
  "func.return"(%0) : (tensor<i1>) -> ()

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

func.func private @cast_lowering_uint_narrow_uint(%arg0: tensor<ui64>) -> tensor<ui32> {
  %0 = "onnx.Cast"(%arg0) {to = ui32 } : (tensor<ui64>) -> tensor<ui32>
  "func.return"(%0) : (tensor<ui32>) -> ()
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

func.func private @cast_lowering_uint_wider_uint(%arg0: tensor<ui32>) -> tensor<ui64> {
  %0 = "onnx.Cast"(%arg0) {to = ui64 } : (tensor<ui32>) -> tensor<ui64>
  "func.return"(%0) : (tensor<ui64>) -> ()
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

func.func private @test_size_known(%arg0: tensor<2x2xf32>) -> tensor<i64> {
  %1 = "onnx.Size"(%arg0) : (tensor<2x2xf32>) -> tensor<i64>
  "func.return"(%1) : (tensor<i64>) -> ()

  // CHECK-LABEL: test_size_known
  // CHECK:      [[RES:%.+]] = memref.alloc() {{.*}}: memref<i64>
  // CHECK-NEXT  [[SIZE:%.+]] = arith.constant 4 : i64
  // CHECK-NEXT  krnl.store [[SIZE]], [[RES]][] : memref<i64>
  // CHECK-NEXT  return [[RES]] : memref<i64>

}

// -----


func.func private @test_size_unknown(%arg0 : tensor<?x2x?xf32>) -> tensor<i64> {
  %1 = "onnx.Size"(%arg0)  : (tensor<?x2x?xf32>) -> tensor<i64>
  "func.return"(%1) : (tensor<i64>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_size_unknown
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x2x?xf32>) -> memref<i64> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<i64>
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x2x?xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[CST_2_1_]] : i64
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x2x?xf32>
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           [[VAR_3_:%.+]] = arith.muli [[VAR_1_]], [[VAR_2_]] : i64
// CHECK:           krnl.store [[VAR_3_]], [[RES_]][] : memref<i64>
// CHECK:           return [[RES_]] : memref<i64>
// CHECK:         }
}

// -----

// Check the lowering of ConstantOfShape when:
//   - No value attribute.
//   - The input is an empty tensor.
// Expected emitted code:
//   - No need a Krnl iterate.
//   - The output is a scalar tensor.

func.func private @test_constant_of_shape_empty_tensor(%arg0 : tensor<0xi64>) -> tensor<*xf32> {
  %0 = "onnx.ConstantOfShape"(%arg0) : (tensor<0xi64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_constant_of_shape_empty_tensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<0xi64>) -> memref<f32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[CST_0_dot_000000_]], [[RES_]][] : memref<f32>
// CHECK:           return [[RES_]] : memref<f32>
// CHECK:         }
}

// -----

// Check the lowering of ConstantOfShape when:
//   - The input is not a arith.constant tensor.
// Expected emitted code:
//   - Emit code to compute output dimensions from the input's dimensions.
//   - Krnl iterates are used to set values to the output.

func.func private @test_constant_of_shape_dynamic_dims(%arg0 : tensor<3xi64>) -> tensor<*xf32> {
  %0 = "onnx.ConstantOfShape"(%arg0) {value = dense<[1.0]> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-LABEL:  func.func private @test_constant_of_shape_dynamic_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3xi64>) -> memref<?x?x?xf32> {
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[CST_0_]]{{.}} : memref<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[CST_1_]]{{.}} : memref<3xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_1_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[CST_2_]]{{.}} : memref<3xi64>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_2_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_1_]], [[VAR_3_]], [[VAR_5_]]) {{.*}}: memref<?x?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_1_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_1_]], [[VAR_3_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_2_]]([[VAR_1_]], [[VAR_3_]], [[VAR_5_]])){
// CHECK:             [[VAR_7_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[CST_1_dot_000000_]], [[RES_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1, [[VAR_7_]]#2] : memref<?x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?xf32>
// CHECK:         }
}

// -----

// Check the lowering of ConstantOfShape when:
//   - The input is a arith.constant tensor.
// Expected emitted code:
//   - Output dimensions are computed during compilation time.
//   - Krnl iterates are used to set values to the output.

func.func private @test_constant_of_shape_static_dims() -> tensor<*xf32> {
  %0 = onnx.Constant dense<[3, 4, 5]> : tensor<3xi64>
  %1 = "onnx.ConstantOfShape"(%0) {value = dense<[1.0]> : tensor<1xf32>} : (tensor<3xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_constant_of_shape_static_dims
// CHECK-SAME:   () -> memref<3x4x5xf32> {
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[CST_1_dot_000000_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x4x5xf32>
// CHECK:         }
}

// -----


func.func private @test_flatten0(%arg0 : tensor<2x3x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Flatten"(%arg0) {axis = 0 : si64} : (tensor<2x3x4xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 12 + d1 * 4 + d2)>
// CHECK-LABEL:  func.func private @test_flatten0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x3x4xf32>) -> memref<1x24xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x24xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<2x3x4xf32>
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[I_0_]], [[I_1_]], [[I_2_]])
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[CST_0_]], [[VAR_2_]]{{.}} : memref<1x24xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x24xf32>
// CHECK:         }
}

// -----

// test partially known input shape

func.func private @test_flatten1(%arg0 : tensor<2x?x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<2x?x4xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1)[s0] -> (d1 + d0 * s0)>
// CHECK-LABEL:  func.func private @test_flatten1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x?x4xf32>) -> memref<?x4xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<2x?x4xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.muli [[VAR_dim_]], [[CST_2_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<2x?x4xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[VAR_dim_0_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<2x?x4xf32>
// CHECK-DAG:         [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<2x?x4xf32>
// CHECK:             [[VAR_3_:%.+]] = affine.apply [[MAP_0_]]([[I_0_]], [[I_1_]]){{.}}[[VAR_dim_1_]]{{.}}
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_3_]], [[I_2_]]{{.}} : memref<?x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x4xf32>
// CHECK:         }
}

// -----


func.func private @test_less(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xi1>
  return %0 : tensor<3x4x5xi1>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_less
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf32>, [[PARAM_1_:%.+]]: memref<3x4x5xf32>) -> memref<3x4x5xi1> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.cmpf olt, [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xi1>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x4x5xi1>
// CHECK:         }
}

// -----


func.func private @test_less_broadcast(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5xf32>) -> tensor<3x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<5xf32>) -> tensor<3x4x5xi1>
  return %0 : tensor<3x4x5xi1>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_less_broadcast
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf32>, [[PARAM_1_:%.+]]: memref<5xf32>) -> memref<3x4x5xi1> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#2] : memref<5xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.cmpf olt, [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xi1>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x4x5xi1>
// CHECK:         }
}

// -----


func.func private @test_clip(%arg0: tensor<3xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<3xf32> attributes {input_names = ["x", "min", "max"], output_names = ["y"]} {
  %0 = "onnx.Clip"(%arg0, %arg1, %arg2) : (tensor<3xf32>, tensor<f32>, tensor<f32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_clip
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3xf32>, [[PARAM_1_:%.+]]: memref<f32>, [[PARAM_2_:%.+]]: memref<f32>) -> memref<3xf32> attributes {input_names = ["x", "min", "max"], output_names = ["y"]} {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<3xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]][] : memref<f32>
// CHECK:             [[VAR_5_:%.+]] = arith.cmpf olt, [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[LOAD_PARAM_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.cmpf olt, [[VAR_6_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_6_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             krnl.store [[VAR_8_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3xf32>
// CHECK:         }
}

// -----

func.func private @test_clip_default_min(%arg0: tensor<3xf32>, %arg1: tensor<f32>, %arg2: tensor<f32>) -> tensor<3xf32> attributes {input_names = ["x", "min", "max"], output_names = ["y"]} {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Clip"(%arg0, %cst, %arg2) : (tensor<3xf32>, none, tensor<f32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>

// CHECK-LABEL: test_clip_default_min
// CHECK-SAME:   ([[INPUT:%.+]]: memref<3xf32>, [[MIN:%.+]]: memref<f32>, [[MAX:%.+]]: memref<f32>) -> memref<3xf32> attributes {input_names = ["x", "min", "max"], output_names = ["y"]} {
// CHECK-DAG:       [[RES:%.+]] = memref.alloc() {{.*}}: memref<3xf32>
// CHECK-DAG:       [[LOOP_0:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0]]) with ([[LOOP_0]] -> [[I_0:%.+]] = 0 to 3){
// CHECK-NEXT:        [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_0]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_INPUT_MEM:%.+]] = krnl.load [[INPUT]]{{.}}[[IV]]{{.}} : memref<3xf32>
// CHECK-DAG:         [[LOAD_MAX_MEM:%.+]] = krnl.load [[MAX]][] : memref<f32>
// CHECK:             [[VAR_7:%.+]] = arith.cmpf olt, [[LOAD_INPUT_MEM]], [[LOAD_MAX_MEM]] : f32
// CHECK:             [[VAR_8:%.+]] = arith.select [[VAR_7]], [[LOAD_INPUT_MEM]], [[LOAD_MAX_MEM]] : f32
// CHECK:             krnl.store [[VAR_8]], [[RES]]{{.}}[[IV]]{{.}} : memref<3xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<3xf32>
// CHECK:         }
}

// -----

func.func private @test_pown(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> attributes {input_names = ["x", "y"], output_names = ["z"]} {
    %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xf32>
    return %0 : tensor<3x4x5xf32>
// CHECK-LABEL: test_pow
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x4x5xf32>, [[POWER_:%.+]]: memref<3x4x5xf32>) -> memref<3x4x5xf32> attributes {input_names = ["x", "y"], output_names = ["z"]} {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
// CHECK-DAG:         [[LOAD_POWER_MEM_:%.+]] = krnl.load [[POWER_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
// CHECK:             [[VAR_4_:%.+]] = math.powf [[LOAD_INPUT_MEM_]], [[LOAD_POWER_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x4x5xf32>
// CHECK:         }
}

// -----

// COM: Check float PRelu without broadcasting.

func.func @test_prelu_float(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_prelu_float
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf32>, [[PARAM_1_:%.+]]: memref<3x4x5xf32>) -> memref<3x4x5xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpf olt, [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.mulf [[LOAD_PARAM_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x4x5xf32>
// CHECK:         }
}

// -----

// COM: Check int PRelu without broadcasting.

func.func @test_prelu_int(%arg0: tensor<3x4x5xi32>, %arg1: tensor<3x4x5xi32>) -> tensor<*xi32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xi32>, tensor<3x4x5xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_prelu_int
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xi32>, [[PARAM_1_:%.+]]: memref<3x4x5xi32>) -> memref<3x4x5xi32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xi32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[LOAD_PARAM_0_MEM_]], [[CST_0_]] : i32
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.muli [[LOAD_PARAM_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : i32
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[LOAD_PARAM_0_MEM_]] : i32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xi32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x4x5xi32>
// CHECK:         }
}

// -----

// COM: Check PRelu with unidirectional broadcasting.

func.func @test_prelu_broadcast1(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_prelu_broadcast1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf32>, [[PARAM_1_:%.+]]: memref<5xf32>) -> memref<3x4x5xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#2] : memref<5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpf olt, [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.mulf [[LOAD_PARAM_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x4x5xf32>
// CHECK:         }
}

// -----

// COM: Check PRelu with unidirectional broadcasting.
// COM: Tensor slope should be unidirectional broadcastable to input tensor X

func.func @test_prelu_broadcast2(%arg0: tensor<3x4x5xf32>, %arg1: tensor<1x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<1x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_prelu_broadcast2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf32>, [[PARAM_1_:%.+]]: memref<1x5xf32>) -> memref<3x4x5xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_1_]]#2] : memref<1x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpf olt, [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.mulf [[LOAD_PARAM_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x4x5xf32>
// CHECK:         }
}

// -----

// COM: Check simple if lowering.

func.func @test_if_simple(%arg0: tensor<i1>, %arg1: tensor<i64>, %arg2: tensor<i64>) -> tensor<i64> {
  %0 = "onnx.If"(%arg0) ({
    onnx.Yield %arg1 : tensor<i64>
  }, {
    onnx.Yield %arg2 : tensor<i64>
  }) : (tensor<i1>) -> tensor<i64>
  return %0 : tensor<i64>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_if_simple
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<i1>, [[PARAM_1_:%.+]]: memref<i64>, [[PARAM_2_:%.+]]: memref<i64>) -> memref<i64> {
// CHECK:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][] : memref<i1>
// CHECK:           [[VAR_1_:%.+]] = arith.select [[LOAD_PARAM_0_MEM_]], [[PARAM_1_]], [[PARAM_2_]] : memref<i64>
// CHECK:           return [[VAR_1_]] : memref<i64>
// CHECK:         }
}

// -----

// COM: Check simple loop lowering.

func.func private @test_loop_simple_main_graph(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<1xi64>) -> tensor<1xi64> {
  %0 = "onnx.Loop"(%arg0, %arg1, %arg2) ({
    ^bb0(%body_arg0: tensor<i64>, %body_arg1: tensor<i1>, %body_arg2: tensor<1xi64>):
    %1 = "onnx.Add"(%body_arg2, %body_arg0) : (tensor<1xi64>, tensor<i64>) -> tensor<1xi64>
    onnx.Yield %body_arg1, %1 : tensor<i1>, tensor<1xi64>
  }) : (tensor<i64>, tensor<i1>, tensor<1xi64>) -> tensor<1xi64>
  return %0 : tensor<1xi64>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_loop_simple_main_graph
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<i64>, [[PARAM_1_:%.+]]: memref<i1>, [[PARAM_2_:%.+]]: memref<1xi64>) -> memref<1xi64> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 1){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_5_]]{{.}} : memref<1xi64>
// CHECK:             krnl.store [[LOAD_PARAM_2_MEM_]], [[RES_]]{{.}}[[VAR_5_]]{{.}} : memref<1xi64>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<i1>
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<i1>
// CHECK:           krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_1_]][] : memref<i1>
// CHECK:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][] : memref<i64>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_]] : i64 to index
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = [[CST_0_]] to [[VAR_3_]]){
// CHECK-DAG:         [[VAR_5_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<i1>
// CHECK:             scf.if [[LOAD_PARAM_2_MEM_1_]] {
// CHECK:               "krnl.region"() ({
// CHECK-DAG:             [[VAR_7_:%.+]] = arith.index_cast [[VAR_5_1_]] : index to i64
// CHECK-DAG:             [[RES_2_:%.+]] = memref.alloc() : memref<i64>
// CHECK:                 krnl.store [[VAR_7_]], [[RES_2_]][] : memref<i64>
// CHECK-DAG:             [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xi64>
// CHECK-DAG:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 1){
// CHECK-DAG:               [[VAR_11_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:               [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_0_]]{{.}} : memref<1xi64>
// CHECK-DAG:               [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<i64>
// CHECK:                   [[VAR_14_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[LOAD_RES_2_MEM_]] : i64
// CHECK:                   krnl.store [[VAR_14_]], [[RES_3_]]{{.}}[[VAR_11_]]{{.}} : memref<1xi64>
// CHECK:                 }
// CHECK:                 [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]][] : memref<i1>
// CHECK:                 krnl.store [[LOAD_PARAM_1_MEM_1_]], [[RES_1_]][] : memref<i1>
// CHECK:                 [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to 1){
// CHECK:                   [[VAR_11_1_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_11_1_]]{{.}} : memref<1xi64>
// CHECK:                   krnl.store [[LOAD_RES_MEM_1_]], [[RES_]]{{.}}[[VAR_11_1_]]{{.}} : memref<1xi64>
// CHECK:                 }
// CHECK:               }) : () -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1xi64>
// CHECK:         }
}

// -----


func.func @test_loop(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<?xf32>) -> (tensor<?x?xf32>) {
  %0 = "onnx.Loop"(%arg0, %arg1) ({
  ^bb0(%arg3: tensor<i64>, %arg4: tensor<i1>):
    %7 = "onnx.Add"(%arg2, %arg2) : (tensor<?xf32>, tensor<?xf32>) -> (tensor<?xf32>)
    onnx.Yield %arg4,  %7 : tensor<i1>, tensor<?xf32>
  }) : (tensor<i64>, tensor<i1>) -> tensor<?x?xf32>
  return  %0 : tensor<?x?xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_loop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<i64>, [[PARAM_1_:%.+]]: memref<i1>, [[PARAM_2_:%.+]]: memref<?xf32>) -> memref<?x?xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][] : memref<i64>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<?xmemref<?xf32>>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<i1>
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<i1>
// CHECK:           krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_1_]][] : memref<i1>
// CHECK:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]][] : memref<i64>
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_1_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = [[CST_0_]] to [[VAR_4_]]){
// CHECK-DAG:         [[VAR_8_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<i1>
// CHECK:             scf.if [[LOAD_RES_1_MEM_]] {
// CHECK:               "krnl.region"() ({
// CHECK-DAG:             [[VAR_10_:%.+]] = arith.index_cast [[VAR_8_]] : index to i64
// CHECK-DAG:             [[RES_2_:%.+]] = memref.alloc() : memref<i64>
// CHECK:                 krnl.store [[VAR_10_]], [[RES_2_]][] : memref<i64>
// CHECK-DAG:             [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_2_]], [[CST_0_]] : memref<?xf32>
// CHECK-DAG:             [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_2_]], [[CST_0_]] : memref<?xf32>
// CHECK:                 [[VAR_11_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_3_]], [[VAR_dim_4_]]{{.}}
// CHECK-DAG:             [[RES_3_:%.+]] = memref.alloc([[VAR_11_]]) {{.*}}: memref<?xf32>
// CHECK-DAG:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_3_]], [[VAR_dim_4_]], [[VAR_11_]])){
// CHECK:                   [[VAR_14_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:               [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_14_]]{{.}} : memref<?xf32>
// CHECK-DAG:               [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_14_]]{{.}} : memref<?xf32>
// CHECK:                   [[VAR_17_:%.+]] = arith.addf [[LOAD_PARAM_2_MEM_]], [[LOAD_PARAM_2_MEM_1_]] : f32
// CHECK:                   krnl.store [[VAR_17_]], [[RES_3_]]{{.}}[[VAR_14_]]{{.}} : memref<?xf32>
// CHECK:                 }
// CHECK:                 [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]][] : memref<i1>
// CHECK:                 krnl.store [[LOAD_PARAM_1_MEM_1_]], [[RES_1_]][] : memref<i1>
// CHECK:                 "krnl.seqstore"([[RES_3_]], [[RES_]], [[VAR_8_]]) : (memref<?xf32>, memref<?xmemref<?xf32>>, index) -> ()
// CHECK:               }) : () -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_0_]]{{.}} : memref<?xmemref<?xf32>>
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[LOAD_RES_MEM_]], [[CST_0_]] : memref<?xf32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc([[VAR_1_]], [[VAR_dim_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = [[CST_0_]] to [[VAR_4_]]){
// CHECK:             [[VAR_8_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:             "krnl.region"() ({
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = "krnl.seqextract"([[RES_]], [[VAR_8_1_]]) {copy = 0 : ui1} : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
// CHECK-DAG:           [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:               [[VAR_dim_2_:%.+]] = memref.dim [[LOAD_RES_1_MEM_1_]], [[CST_0_]] : memref<?xf32>
// CHECK:               krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_2_]])){
// CHECK:                 [[VAR_11_1_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOOP_1_:%.+]] = krnl.load [[LOAD_RES_1_MEM_1_]]{{.}}[[VAR_11_1_]]{{.}} : memref<?xf32>
// CHECK:                 krnl.store [[LOOP_1_]], [[RES_4_]]{{.}}[[VAR_8_1_]], [[VAR_11_1_]]{{.}} : memref<?x?xf32>
// CHECK:               }
// CHECK:             }) : () -> ()
// CHECK:           }
// CHECK:           return [[RES_4_]] : memref<?x?xf32>
// CHECK:         }
}

// -----


func.func @test_resize1(%arg0 : tensor<3x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = onnx.Constant dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>
  %1 = onnx.Constant dense<[1.000000e+00,  3.000000e+00]> : tensor<2xf32>
  %2 = "onnx.Resize"(%arg0, %0, %1, %cst) {coordinate_transformation_mode = "asymmetric", mode = "nearest", nearest_mode = "floor"} : (tensor<3x4xf32>, tensor<4xf32>, tensor<2xf32>, none) -> tensor<*xf32>
  "func.return"(%2) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_resize1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4xf32>) -> memref<3x12xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_3_dot_000000_:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x12xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 12){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_2_:%.+]] = arith.index_cast [[VAR_1_]]#0 : index to i64
// CHECK:             [[VAR_3_:%.+]] = arith.sitofp [[VAR_2_]] : i64 to f32
// CHECK:             [[VAR_4_:%.+]] = math.floor [[VAR_3_]] : f32
// CHECK:             [[VAR_5_:%.+]] = arith.fptosi [[VAR_4_]] : f32 to i64
// CHECK:             [[VAR_6_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_0_]] : i64
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[CST_0_]], [[VAR_5_]] : i64
// CHECK:             [[VAR_8_:%.+]] = arith.index_cast [[VAR_7_]] : i64 to index
// CHECK:             [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[VAR_8_]], [[CST_2_]] : index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.index_cast [[VAR_1_]]#1 : index to i64
// CHECK:             [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:             [[VAR_13_:%.+]] = arith.divf [[VAR_12_]], [[CST_3_dot_000000_]] : f32
// CHECK:             [[VAR_14_:%.+]] = math.floor [[VAR_13_]] : f32
// CHECK:             [[VAR_15_:%.+]] = arith.fptosi [[VAR_14_]] : f32 to i64
// CHECK:             [[VAR_16_:%.+]] = arith.cmpi slt, [[VAR_15_]], [[CST_0_]] : i64
// CHECK:             [[VAR_17_:%.+]] = arith.select [[VAR_16_]], [[CST_0_]], [[VAR_15_]] : i64
// CHECK:             [[VAR_18_:%.+]] = arith.index_cast [[VAR_17_]] : i64 to index
// CHECK:             [[VAR_19_:%.+]] = arith.cmpi slt, [[VAR_18_]], [[CST_4_]] : index
// CHECK:             [[VAR_20_:%.+]] = arith.select [[VAR_19_]], [[VAR_18_]], [[CST_3_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_10_]], [[VAR_20_]]{{.}} : memref<3x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<3x12xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x12xf32>
// CHECK:         }
}

// -----


func.func @test_resize2(%arg0 : tensor<3x4xf32>, %scale : tensor<2xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = onnx.Constant dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>
  %2 = "onnx.Resize"(%arg0, %0, %scale, %cst) {coordinate_transformation_mode = "asymmetric", mode = "nearest", nearest_mode = "floor"} : (tensor<3x4xf32>, tensor<4xf32>, tensor<2xf32>, none) -> tensor<*xf32>
  "func.return"(%2) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func @test_resize2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4xf32>, [[PARAM_1_:%.+]]: memref<2xf32>) -> memref<?x?xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_4_dot_000000_:%.+]] = arith.constant 4.000000e+00 : f32
// CHECK-DAG:       [[CST_3_dot_000000_:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<2xf32>
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_1_]]{{.}} : memref<2xf32>
// CHECK:           [[VAR_2_:%.+]] = arith.mulf [[LOAD_PARAM_1_MEM_]], [[CST_3_dot_000000_]] : f32
// CHECK:           [[VAR_3_:%.+]] = arith.fptosi [[VAR_2_]] : f32 to i64
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[VAR_3_]] : i64 to index
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.mulf [[LOAD_PARAM_1_MEM_1_]], [[CST_4_dot_000000_]] : f32
// CHECK:           [[VAR_6_:%.+]] = arith.fptosi [[VAR_5_]] : f32 to i64
// CHECK:           [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_4_]], [[VAR_7_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_4_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_4_]], [[VAR_7_]])){
// CHECK:             [[VAR_9_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_10_:%.+]] = arith.index_cast [[VAR_9_]]#0 : index to i64
// CHECK:             [[VAR_11_:%.+]] = arith.sitofp [[VAR_10_]] : i64 to f32
// CHECK:             [[VAR_12_:%.+]] = arith.divf [[VAR_11_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_13_:%.+]] = math.floor [[VAR_12_]] : f32
// CHECK:             [[VAR_14_:%.+]] = arith.fptosi [[VAR_13_]] : f32 to i64
// CHECK:             [[VAR_15_:%.+]] = arith.cmpi slt, [[VAR_14_]], [[CST_0_]] : i64
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[CST_0_]], [[VAR_14_]] : i64
// CHECK:             [[VAR_17_:%.+]] = arith.index_cast [[VAR_16_]] : i64 to index
// CHECK:             [[VAR_18_:%.+]] = arith.cmpi slt, [[VAR_17_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.select [[VAR_18_]], [[VAR_17_]], [[CST_2_]] : index
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.index_cast [[VAR_9_]]#1 : index to i64
// CHECK:             [[VAR_21_:%.+]] = arith.sitofp [[VAR_20_]] : i64 to f32
// CHECK:             [[VAR_22_:%.+]] = arith.divf [[VAR_21_]], [[LOAD_PARAM_1_MEM_1_]] : f32
// CHECK:             [[VAR_23_:%.+]] = math.floor [[VAR_22_]] : f32
// CHECK:             [[VAR_24_:%.+]] = arith.fptosi [[VAR_23_]] : f32 to i64
// CHECK:             [[VAR_25_:%.+]] = arith.cmpi slt, [[VAR_24_]], [[CST_0_]] : i64
// CHECK:             [[VAR_26_:%.+]] = arith.select [[VAR_25_]], [[CST_0_]], [[VAR_24_]] : i64
// CHECK:             [[VAR_27_:%.+]] = arith.index_cast [[VAR_26_]] : i64 to index
// CHECK:             [[VAR_28_:%.+]] = arith.cmpi slt, [[VAR_27_]], [[CST_4_]] : index
// CHECK:             [[VAR_29_:%.+]] = arith.select [[VAR_28_]], [[VAR_27_]], [[CST_3_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_19_]], [[VAR_29_]]{{.}} : memref<3x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_9_]]#0, [[VAR_9_]]#1] : memref<?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?xf32>
// CHECK:         }
}

// -----


func.func @test_resize3(%arg0 : tensor<?x?xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = onnx.Constant dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>
  %1 = onnx.Constant dense<[1.000000e+00,  3.000000e+00]> : tensor<2xf32>
  %2 = "onnx.Resize"(%arg0, %0, %1, %cst) {coordinate_transformation_mode = "asymmetric", mode = "nearest", nearest_mode = "floor"} : (tensor<?x?xf32>, tensor<4xf32>, tensor<2xf32>, none) -> tensor<*xf32>
  "func.return"(%2) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func @test_resize3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?xf32>) -> memref<?x?xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_3_dot_000000_:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_1_]] : memref<?x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?xf32>
// CHECK:           [[VAR_0_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           [[VAR_1_:%.+]] = arith.sitofp [[VAR_0_]] : i64 to f32
// CHECK:           [[VAR_2_:%.+]] = arith.mulf [[VAR_1_]], [[CST_3_dot_000000_]] : f32
// CHECK:           [[VAR_3_:%.+]] = arith.fptosi [[VAR_2_]] : f32 to i64
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[VAR_3_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_4_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_4_]])){
// CHECK:             [[VAR_6_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]]#0 : index to i64
// CHECK:             [[VAR_8_:%.+]] = arith.sitofp [[VAR_7_]] : i64 to f32
// CHECK:             [[VAR_9_:%.+]] = math.floor [[VAR_8_]] : f32
// CHECK:             [[VAR_10_:%.+]] = arith.fptosi [[VAR_9_]] : f32 to i64
// CHECK:             [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_10_]], [[CST_0_]] : i64
// CHECK:             [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_0_]], [[VAR_10_]] : i64
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.index_cast [[VAR_12_]] : i64 to index
// CHECK-DAG:         [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_1_]] : memref<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.cmpi slt, [[VAR_13_]], [[VAR_dim_1_]] : index
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.subi [[VAR_dim_1_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_16_:%.+]] = arith.select [[VAR_14_]], [[VAR_13_]], [[VAR_15_]] : index
// CHECK-DAG:         [[VAR_17_:%.+]] = arith.index_cast [[VAR_6_]]#1 : index to i64
// CHECK:             [[VAR_18_:%.+]] = arith.sitofp [[VAR_17_]] : i64 to f32
// CHECK:             [[VAR_19_:%.+]] = arith.divf [[VAR_18_]], [[CST_3_dot_000000_]] : f32
// CHECK:             [[VAR_20_:%.+]] = math.floor [[VAR_19_]] : f32
// CHECK:             [[VAR_21_:%.+]] = arith.fptosi [[VAR_20_]] : f32 to i64
// CHECK:             [[VAR_22_:%.+]] = arith.cmpi slt, [[VAR_21_]], [[CST_0_]] : i64
// CHECK:             [[VAR_23_:%.+]] = arith.select [[VAR_22_]], [[CST_0_]], [[VAR_21_]] : i64
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.index_cast [[VAR_23_]] : i64 to index
// CHECK-DAG:         [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.cmpi slt, [[VAR_24_]], [[VAR_dim_2_]] : index
// CHECK-DAG:         [[VAR_26_:%.+]] = arith.subi [[VAR_dim_2_]], [[CST_1_]] : index
// CHECK:             [[VAR_27_:%.+]] = arith.select [[VAR_25_]], [[VAR_24_]], [[VAR_26_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_16_]], [[VAR_27_]]{{.}} : memref<?x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_6_]]#0, [[VAR_6_]]#1] : memref<?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?xf32>
// CHECK:         }
}

// -----


func.func @test_resize2(%arg0 : tensor<3x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = onnx.Constant dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>
  %1 = onnx.Constant dense<[1.000000e+00,  3.000000e+00]> : tensor<2xf32>
  %2 = "onnx.Resize"(%arg0, %0, %1, %cst) {mode = "linear"} : (tensor<3x4xf32>, tensor<4xf32>, tensor<2xf32>, none) -> tensor<*xf32>
  "func.return"(%2) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_resize2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4xf32>) -> memref<3x12xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [2], value = dense<[1.000000e+00, 3.000000e+00]> : tensor<2xf32>} : () -> memref<2xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x12xf32>
// CHECK:           "krnl.call"([[RES_]], [[PARAM_0_]], [[VAR_0_]], [[VAR_1_]]) {funcName = "Resize_Scales", mode = "linear", nearest_mode = "round_prefer_floor", numOfOutput = 1 : si64} : (memref<3x12xf32>, memref<3x4xf32>, memref<4xf32>, memref<2xf32>) -> ()
// CHECK:           return [[RES_]] : memref<3x12xf32>
// CHECK:         }
}

// -----


func.func @test_gather_scalar(%arg0: tensor<4xi64>, %arg1: tensor<i64>) -> tensor<i64> {
  %0 = "onnx.Gather"(%arg0, %arg1) {axis = 0 : si64} : (tensor<4xi64>, tensor<i64>) -> tensor<i64>
  return %0 : tensor<i64>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_gather_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4xi64>, [[PARAM_1_:%.+]]: memref<i64>) -> memref<i64> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<i64>
// CHECK:           krnl.define_loops 0
// CHECK:           krnl.iterate() with (){
// CHECK:             krnl.get_induction_var_value() : () -> ()
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<i64>
// CHECK:             [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_2_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.addi [[VAR_1_]], [[CST_4_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.select [[VAR_2_]], [[VAR_3_]], [[VAR_1_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]]{{.}} : memref<4xi64>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][] : memref<i64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<i64>
// CHECK:         }
}

// -----


func.func @test_gather_elements(%arg0: tensor<4xi64>, %arg1: tensor<2xi64>) -> tensor<2xi64> {
  %0 = "onnx.GatherElements"(%arg0, %arg1) : (tensor<4xi64>, tensor<2xi64>) -> tensor<2xi64>
  return %0 : tensor<2xi64>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_gather_elements
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4xi64>, [[PARAM_1_:%.+]]: memref<2xi64>) -> memref<2xi64> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]{{.}} : memref<2xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.addi [[VAR_3_]], [[CST_4_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]]{{.}} : memref<4xi64>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<2xi64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2xi64>
// CHECK:         }
}

// -----

// COM: Test GatherND with indices_shape[-1] == rank(data) - batch_dims

func.func @test_gather_nd_1(%arg0 : tensor<2x2xf32>, %arg1 : tensor<2x2xi64>) -> tensor<2xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2xf32>, tensor<2x2xi64>) -> tensor<2xf32>
  "func.return"(%0) : (tensor<2xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_gather_nd_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x2xf32>, [[PARAM_1_:%.+]]: memref<2x2xi64>) -> memref<2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [1, 2, 2], strides: [4, 2, 1] : memref<2x2xi64> to memref<1x2x2xi64>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1, 2, 2], strides: [4, 2, 1] : memref<2x2xf32> to memref<1x2x2xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<2xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[CST_0_]]{{.}} : memref<1x2x2xi64>
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_]] : i64 to index
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[CST_1_]]{{.}} : memref<1x2x2xi64>
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_0_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_0_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_5_]]{{.}} : memref<1x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             krnl.store [[LOAD_VAR_reinterpret_cast_0_MEM_]], [[RES_]]{{.}}[[LOAD_RES_1_MEM_]]{{.}} : memref<2xf32>
// CHECK:             [[VAR_8_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_]] : index
// CHECK:             krnl.store [[VAR_8_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2], strides: [1] : memref<2xf32> to memref<2xf32>
// CHECK:           return [[VAR_reinterpret_cast_1_]] : memref<2xf32>
// CHECK:         }
}

// -----

// COM: Test GatherND with indices_shape[-1] < rank(data) - batch_dims

func.func @test_gather_nd_2(%arg0 : tensor<2x2x2xf32>, %arg1 : tensor<2x1x2xi64>) -> tensor<2x1x2xf32> {
  %0 = "onnx.GatherND"(%arg0, %arg1) {batch_dims = 0 : si64} : (tensor<2x2x2xf32>, tensor<2x1x2xi64>) -> tensor<2x1x2xf32>
  "func.return"(%0) : (tensor<2x1x2xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_gather_nd_2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x2x2xf32>, [[PARAM_1_:%.+]]: memref<2x1x2xi64>) -> memref<2x1x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_1_]] to offset: [0], sizes: [1, 2, 2], strides: [4, 2, 1] : memref<2x1x2xi64> to memref<1x2x2xi64>
// CHECK-DAG:       [[VAR_reinterpret_cast_0_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: [1, 2, 2, 2], strides: [8, 4, 2, 1] : memref<2x2x2xf32> to memref<1x2x2x2xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<4xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_1_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_VAR_reinterpret_cast_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[CST_0_]]{{.}} : memref<1x2x2xi64>
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_]] : i64 to index
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_MEM_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[CST_1_]]{{.}} : memref<1x2x2xi64>
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[LOAD_VAR_reinterpret_cast_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_0_MEM_:%.+]] = krnl.load [[VAR_reinterpret_cast_0_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_5_]], [[CST_0_]]{{.}} : memref<1x2x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             krnl.store [[LOAD_VAR_reinterpret_cast_0_MEM_]], [[RES_]]{{.}}[[LOAD_RES_1_MEM_]]{{.}} : memref<4xf32>
// CHECK:             [[VAR_8_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[CST_1_]] : index
// CHECK:             krnl.store [[VAR_8_]], [[RES_1_]][] : memref<index>
// CHECK-DAG:         [[LOAD_VAR_reinterpret_cast_0_MEM_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_0_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]], [[VAR_5_]], [[CST_1_]]{{.}} : memref<1x2x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<index>
// CHECK:             krnl.store [[LOAD_VAR_reinterpret_cast_0_MEM_1_]], [[RES_]]{{.}}[[LOAD_RES_1_MEM_1_]]{{.}} : memref<4xf32>
// CHECK:             [[VAR_11_:%.+]] = arith.addi [[LOAD_RES_1_MEM_1_]], [[CST_1_]] : index
// CHECK:             krnl.store [[VAR_11_]], [[RES_1_]][] : memref<index>
// CHECK:           }
// CHECK:           [[VAR_reinterpret_cast_1_:%.+]] = memref.reinterpret_cast [[RES_]] to offset: [0], sizes: [2, 1, 2], strides: [2, 2, 1] : memref<4xf32> to memref<2x1x2xf32>
// CHECK:           return [[VAR_reinterpret_cast_1_]] : memref<2x1x2xf32>
// CHECK:         }
}

// -----


func.func @test_reversesequence_1(%arg0: tensor<10x?xf32>, %arg1: tensor<10xi64>) -> tensor<*xf32> {
  %0 = "onnx.ReverseSequence"(%arg0, %arg1) {batch_axis = 1 : si64, time_axis = 0 : si64} : (tensor<10x?xf32>, tensor<10xi64>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_reversesequence_1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x?xf32>, [[PARAM_1_:%.+]]: memref<10xi64>) -> memref<10x?xf32> {
// CHECK:           [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<10x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<10x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#1] : memref<10xi64>
// CHECK:             [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[VAR_1_]]#0, [[VAR_3_]] : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.subi [[VAR_3_]], [[VAR_1_]]#0 : index
// CHECK:             [[VAR_6_:%.+]] = arith.subi [[VAR_5_]], [[CST_1_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_4_]], [[VAR_6_]], [[VAR_1_]]#0 : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_7_]], [[VAR_1_]]#1] : memref<10x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<10x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x?xf32>
// CHECK:         }
}

// -----

func.func @test_random_normal1() -> tensor<*xf32> {
  %0 = "onnx.RandomNormal"() {shape = [3, 4, 5], dtype = 1 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : () -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal1
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[SEED:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:           "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf32>, index, f32, f32, f32) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf32>
}

// -----

func.func @test_random_normal2() -> tensor<*xf32> {
  %0 = "onnx.RandomNormal"() {shape = [3, 4, 5], dtype = 2 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : () -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal2
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf64>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       [[SEED:%.+]] = arith.constant 2.000000e+00 : f64
// CHECK:           "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf64>, index, f64, f64, f64) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf64>
}

// -----

func.func @test_random_normal3() -> tensor<*xf32> {
  %0 = "onnx.RandomNormal"() {shape = [3, 4, 5], dtype = 2 : si64, mean = 0.0 :f32, scale = 1.0 : f32} : () -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal3
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf64>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       [[SEED:%.+]] = arith.constant
// CHECK:           "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf64>, index, f64, f64, f64) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf64>
}

// -----

func.func @test_random_normal_like1(%arg0: tensor<3x4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.RandomNormalLike"(%arg0) {dtype = 1 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<3x4x5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal_like1
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[SEED:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:           "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf32>, index, f32, f32, f32) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf32>
}

// -----
func.func @test_random_normal_like2(%arg0: tensor<3x4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.RandomNormalLike"(%arg0) {dtype = 2 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<3x4x5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal_like2
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf64>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       [[SEED:%.+]] = arith.constant 2.000000e+00 : f64
// CHECK:           "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf64>, index, f64, f64, f64) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf64>
}

// -----
func.func @test_random_normal_like3(%arg0: tensor<3x4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.RandomNormalLike"(%arg0) {dtype = 2 : si64, mean = 0.0 :f32, scale = 1.0 : f32} : (tensor<3x4x5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal_like3
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf64>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f64
// CHECK-DAG:       [[SEED:%.+]] = arith.constant
// CHECK:           "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf64>, index, f64, f64, f64) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf64>
}

// -----


func.func @test_random_normal_like4(%arg0: tensor<3x4x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.RandomNormalLike"(%arg0) {dtype = 1 : si64, mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<3x4x?x?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_random_normal_like4
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x?x?xf32>) -> memref<3x4x?x?xf32> {
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<3x4x?x?xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<3x4x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_2) {{.*}}: memref<3x4x?x?xf32>
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<3x4x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.muli [[VAR_dim_3_]], [[CST_12_]] : index
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<3x4x?x?xf32>
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[VAR_dim_4_]] : index
// CHECK:           "krnl.random_normal"([[RES_]], [[VAR_1_]], [[CST_0_dot_000000_]], [[CST_1_dot_000000_]], [[CST_2_dot_000000_]]) : (memref<3x4x?x?xf32>, index, f32, f32, f32) -> ()
// CHECK:           return [[RES_]] : memref<3x4x?x?xf32>
// CHECK:         }
}

// -----
func.func @test_random_normal_like5(%arg0: tensor<3x4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.RandomNormalLike"(%arg0) {mean = 0.0 :f32, scale = 1.0 : f32, seed = 2.0 : f32} : (tensor<3x4x5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// CHECK-LABEL:  @test_random_normal_like5
// CHECK-DAG:       [[ALLOC:%.+]] = memref.alloc() {alignment = 16 : i64} : memref<3x4x5xf32>
// CHECK-DAG:       [[ALL_VALUES:%.+]] = arith.constant 60 : index
// CHECK-DAG:       [[MEAN:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[SCALE:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[SEED:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK:           "krnl.random_normal"([[ALLOC]], [[ALL_VALUES]], [[MEAN]], [[SCALE]], [[SEED]]) : (memref<3x4x5xf32>, index, f32, f32, f32) -> ()
// CHECK:           return [[ALLOC]] : memref<3x4x5xf32>
}

// -----


func.func @test_scatter_elements1(%arg0: tensor<3x3xf32>, %arg1: tensor<3x2xi64>, %arg2: tensor<3x2xf32>) -> (tensor<*xf32>,tensor<*xf32>) {
  %0 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 0 : si64} : (tensor<3x3xf32>, tensor<3x2xi64>, tensor<3x2xf32>) -> tensor<*xf32>
  %1 = "onnx.ScatterElements"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<3x2xi64>, tensor<3x2xf32>) -> tensor<*xf32>
  return %0, %1 : tensor<*xf32>, tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_scatter_elements1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x3xf32>, [[PARAM_1_:%.+]]: memref<3x2xi64>, [[PARAM_2_:%.+]]: memref<3x2xf32>) -> (memref<3x3xf32>, memref<3x3xf32>) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_9_:%.+]] = arith.constant 9 : i64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x3xf32>
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_9_]], [[CST_0_]], [[CST_0_]]) : (memref<3x3xf32>, memref<3x3xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<3x2xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<3x2xi64>
// CHECK:             [[VAR_5_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.addi [[VAR_5_]], [[CST_3_]] : index
// CHECK:             [[VAR_8_:%.+]] = arith.select [[VAR_6_]], [[VAR_7_]], [[VAR_5_]] : index
// CHECK:             krnl.store [[LOAD_PARAM_2_MEM_]], [[RES_]]{{.}}[[VAR_8_]], [[VAR_2_]]#1] : memref<3x3xf32>
// CHECK:           }
// CHECK:           [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x3xf32>
// CHECK:           "krnl.memcpy"([[RES_1_]], [[PARAM_0_]], [[CST_9_]], [[CST_0_]], [[CST_0_]]) : (memref<3x3xf32>, memref<3x3xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<3x2xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1] : memref<3x2xi64>
// CHECK:             [[VAR_5_1_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_1_]] : i64 to index
// CHECK-DAG:         [[VAR_6_1_:%.+]] = arith.cmpi slt, [[VAR_5_1_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_7_1_:%.+]] = arith.addi [[VAR_5_1_]], [[CST_3_]] : index
// CHECK:             [[VAR_8_1_:%.+]] = arith.select [[VAR_6_1_]], [[VAR_7_1_]], [[VAR_5_1_]] : index
// CHECK:             krnl.store [[LOAD_PARAM_2_MEM_1_]], [[RES_1_]]{{.}}[[VAR_2_1_]]#0, [[VAR_8_1_]]{{.}} : memref<3x3xf32>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_0 : memref<3x3xf32>, memref<3x3xf32>
// CHECK:         }
}

// -----


func.func @test_scatter_nd1(%arg0: tensor<4x4x4xf32>, %arg1: tensor<2x1xi64>, %arg2: tensor<2x4x4xf32>) -> tensor<4x4x4xf32> {
  %0 = "onnx.ScatterND"(%arg0, %arg1, %arg2) : (tensor<4x4x4xf32>, tensor<2x1xi64>, tensor<2x4x4xf32>) -> tensor<4x4x4xf32>
  return %0 : tensor<4x4x4xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_scatter_nd1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x4x4xf32>, [[PARAM_1_:%.+]]: memref<2x1xi64>, [[PARAM_2_:%.+]]: memref<2x4x4xf32>) -> memref<4x4x4xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x4x4xf32>
// CHECK:           "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_64_]], [[CST_0_]], [[CST_0_]]) : (memref<4x4x4xf32>, memref<4x4x4xf32>, i64, index, index) -> ()
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[CST_0_]]{{.}} : memref<2x1xi64>
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x4x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_2_MEM_]], [[RES_]]{{.}}[[VAR_3_]], [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<4x4x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x4x4xf32>
// CHECK:         }
}

// -----


func.func @test_sequence_erase(%arg0: !onnx.Seq<tensor<?x4x5xf32>>) -> tensor<3xi64>  {
  %0 = onnx.Constant {value = dense<0> : tensor<i64>} : tensor<i64>
  %7 = "onnx.SequenceErase"(%arg0, %0) : (!onnx.Seq<tensor<?x4x5xf32>>, tensor<i64>) -> !onnx.Seq<tensor<?x4x5xf32>>
  %4 = "onnx.SequenceAt"(%7, %0) : (!onnx.Seq<tensor<?x4x5xf32>>, tensor<i64>) -> tensor<?x4x5xf32>
  %5 = "onnx.Shape"(%4) : (tensor<?x4x5xf32>) -> tensor<3xi64>
  return %5 : tensor<3xi64>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 - 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1] -> (s0)>
// CHECK-LABEL:  func.func @test_sequence_erase
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?xmemref<?x4x5xf32>>) -> memref<3xi64> {
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : i64
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : i64
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<0> : tensor<i64>} : () -> memref<i64>
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?xmemref<?x4x5xf32>>
// CHECK:           [[VAR_1_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.seqalloc"([[VAR_1_]]) : (index) -> memref<?xmemref<?x4x5xf32>>
// CHECK-DAG:       [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<i64>
// CHECK:           [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_]] : i64 to index
// CHECK-DAG:       [[VAR_5_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_]], [[VAR_4_]]{{.}}
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_5_]], [[VAR_4_]] : index
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_7_]]){
// CHECK:             [[VAR_18_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_18_]]{{.}} : memref<?xmemref<?x4x5xf32>>
// CHECK:             "krnl.seqstore"([[LOAD_PARAM_0_MEM_]], [[VAR_2_]], [[VAR_7_]]) : (memref<?x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()
// CHECK:           }
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.addi [[VAR_7_]], [[CST_1_]] : index
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = [[VAR_9_]] to [[MAP_2_]](){{.}}[[VAR_dim_]], [[VAR_4_]]{{.}}){
// CHECK:             [[VAR_18_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_18_1_]]{{.}} : memref<?xmemref<?x4x5xf32>>
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.subi [[VAR_18_1_]], [[CST_1_]] : index
// CHECK:             "krnl.seqstore"([[LOAD_PARAM_0_MEM_1_]], [[VAR_2_]], [[VAR_2_]]0) : (memref<?x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()
// CHECK:           }
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[VAR_2_]], [[CST_0_]] : memref<?xmemref<?x4x5xf32>>
// CHECK-DAG:       [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<i64>
// CHECK:           [[VAR_12_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.cmpi slt, [[VAR_12_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_14_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_12_]], [[VAR_dim_0_]]{{.}}
// CHECK:           [[VAR_15_:%.+]] = arith.select [[VAR_13_]], [[VAR_14_]], [[VAR_12_]] : index
// CHECK-DAG:       [[VAR_16_:%.+]] = "krnl.seqextract"([[VAR_2_]], [[VAR_15_]]) {copy = 1 : ui1} : (memref<?xmemref<?x4x5xf32>>, index) -> memref<?x4x5xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3xi64>
// CHECK:           [[VAR_dim_1_:%.+]] = memref.dim [[VAR_16_]], [[CST_0_]] : memref<?x4x5xf32>
// CHECK:           [[VAR_17_:%.+]] = arith.index_cast [[VAR_dim_1_]] : index to i64
// CHECK:           krnl.store [[VAR_17_]], [[RES_]]{{.}}[[CST_0_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[CST_4_]], [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[CST_5_]], [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi64>
// CHECK:           return [[RES_]] : memref<3xi64>
// CHECK:         }
}

// -----

//===----------------------------------------------------------------------===//
/// Test krnl lowering for IsInf.
//===----------------------------------------------------------------------===//

func.func @test_isinf(%arg0 : tensor<2x3x4xf32>) -> tensor<*xi1> {
  %0 = "onnx.IsInf"(%arg0) {detect_negative = 1 : si64, detect_positive = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_isinf
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x3x4xf32>) -> memref<2x3x4xi1> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xf32>
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.cmpf oeq, [[LOAD_PARAM_0_MEM_]], [[CST_0_]] : f32
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpf oeq, [[LOAD_PARAM_0_MEM_]], [[CST_0_1_]] : f32
// CHECK:             [[VAR_5_:%.+]] = arith.ori [[VAR_3_]], [[VAR_4_]] : i1
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xi1>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x4xi1>
// CHECK:         }
}

// -----


func.func @test_isinf_positive(%arg0 : tensor<2x3x4xf32>) -> tensor<*xi1> {
  %0 = "onnx.IsInf"(%arg0) {detect_negative = 0 : si64, detect_positive = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_isinf_positive
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x3x4xf32>) -> memref<2x3x4xi1> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xf32>
// CHECK:             [[VAR_3_:%.+]] = arith.cmpf oeq, [[LOAD_PARAM_0_MEM_]], [[CST_0_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xi1>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x4xi1>
// CHECK:         }
}

// -----


func.func @test_isinf_negative(%arg0 : tensor<2x3x4xf32>) -> tensor<*xi1> {
  %0 = "onnx.IsInf"(%arg0) {detect_negative = 1 : si64, detect_positive = 0 : si64} : (tensor<2x3x4xf32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_isinf_negative
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x3x4xf32>) -> memref<2x3x4xi1> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xf32>
// CHECK:             [[VAR_3_:%.+]] = arith.cmpf oeq, [[LOAD_PARAM_0_MEM_]], [[CST_0_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xi1>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x4xi1>
// CHECK:         }
}

// -----

//===----------------------------------------------------------------------===//
/// Test krnl lowering for IsNaN.
//===----------------------------------------------------------------------===//
func.func @test_isnan(%arg0 : tensor<2x3x4xf32>) -> tensor<*xi1> {
  %0 = "onnx.IsNaN"(%arg0) : (tensor<2x3x4xf32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>

// CHECK-LABEL isnan_function
// CHECK: [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<2x3x4xi1>
// CHECK: [[LOOP:%.+]]:3 = krnl.define_loops 3
// CHECK: krnl.iterate
// CHECK: [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK: [[LOAD:%.+]] = {{.*}}load %arg0[[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<2x3x4xf32>
// CHECK: [[ERF:%.+]]  = "krnl.isnan"([[LOAD]]) : (f32) -> i1
// CHECK: {{.*}}store [[ERF]], [[ALLOC]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<2x3x4xi1>
// CHECK: return [[ALLOC]] : memref<2x3x4xi1>
}
