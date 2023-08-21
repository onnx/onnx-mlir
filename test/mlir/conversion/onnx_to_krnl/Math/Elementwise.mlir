// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

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

func.func private @test_less(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x4x5xf32>) -> tensor<3x4x5xi1>
  return %0 : tensor<3x4x5xi1>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_less
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4x5xf32>, [[PARAM_1_:%.+]]: memref<3x4x5xf32>) -> memref<3x4x5xi1> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_5_1_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3_2_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_2_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_5_2_:%.+]] = arith.constant 5 : index
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
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_5_1_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_5_2_:%.+]] = arith.constant 5 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[CST_3_2_:%.+]] = arith.constant 3 : index
// CHECK-DAG:         [[CST_4_2_:%.+]] = arith.constant 4 : index
// CHECK-DAG:         [[CST_5_3_:%.+]] = arith.constant 5 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x4x5xf32>
// CHECK-DAG:         [[CST_5_4_:%.+]] = arith.constant 5 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_true_11_:%.+]] = arith.constant true
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
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
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3_1_:%.+]] = arith.constant 3 : index
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

func.func private @test_pow(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x4x5xf32>) -> tensor<3x4x5xf32> attributes {input_names = ["x", "y"], output_names = ["z"]} {
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

  // CHECK-LABEL: func @test_prelu_float
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5){
  // CHECK:   [[IV:%.+]]:3 = krnl.get_induction_var_value([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:   [[LOAD_X:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
  // CHECK:   [[LOAD_SLOPE:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
  // CHECK:   [[CST_0:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK:   [[LESS_THAN_ZERO:%.+]] = arith.cmpf olt, [[LOAD_X]], [[CST_0]] : f32
  // CHECK:   [[MUL:%.+]] = arith.mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:   [[SELECT:%.+]] = arith.select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:   krnl.store [[SELECT]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xf32>
}

// -----

// COM: Check int PRelu without broadcasting.
func.func @test_prelu_int(%arg0: tensor<3x4x5xi32>, %arg1: tensor<3x4x5xi32>) -> tensor<*xi32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xi32>, tensor<3x4x5xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>

  // CHECK-LABEL: func @test_prelu_int
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xi32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5){
  // CHECK:   [[IV:%.+]]:3 = krnl.get_induction_var_value([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:   [[LOAD_X:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xi32>
  // CHECK:   [[LOAD_SLOPE:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xi32>
  // CHECK:   [[CST_0:%.+]] = arith.constant 0 : i32
  // CHECK:   [[LESS_THAN_ZERO:%.+]] = arith.cmpi slt, [[LOAD_X]], [[CST_0]] : i32
  // CHECK:   [[MUL:%.+]] = arith.muli [[LOAD_SLOPE]], [[LOAD_X]] : i32
  // CHECK:   [[SELECT:%.+]] = arith.select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : i32
  // CHECK:   krnl.store [[SELECT]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xi32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xi32>
}

// -----

// COM: Check PRelu with unidirectional broadcasting.
func.func @test_prelu_broadcast1(%arg0: tensor<3x4x5xf32>, %arg1: tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_prelu_broadcast1
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5){
  // CHECK:       [[IV:%.+]]:3 = krnl.get_induction_var_value([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1[[[IV]]#2] : memref<5xf32>
  // CHECK-DAG:   [[CST_0:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = arith.cmpf olt, [[LOAD_X]], [[CST_0]] : f32
  // CHECK:       [[MUL:%.+]] = arith.mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT:%.+]] = arith.select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:       krnl.store [[SELECT]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xf32>
}

// -----

// COM: Check PRelu with unidirectional broadcasting.
// COM: Tensor slope should be unidirectional broadcastable to input tensor X
func.func @test_prelu_broadcast2(%arg0: tensor<3x4x5xf32>, %arg1: tensor<1x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<1x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_prelu_broadcast2
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5){
  // CHECK:       [[IV:%.+]]:3 = krnl.get_induction_var_value([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
  // CHECK-DAG:   [[ZERO_INDEX:%.+]] = arith.constant 0 : index
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1{{\[}}[[ZERO_INDEX]], [[IV]]#2] : memref<1x5xf32>
  // CHECK-DAG:   [[CST_0:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = arith.cmpf olt, [[LOAD_X]], [[CST_0]] : f32
  // CHECK:       [[MUL:%.+]] = arith.mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT:%.+]] = arith.select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:       krnl.store [[SELECT]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xf32>
}

// -----

func.func @test_isinf(%arg0 : tensor<2x3x4xf32>) -> tensor<*xi1> {
  %0 = "onnx.IsInf"(%arg0) {detect_negative = 1 : si64, detect_positive = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>

// CHECK-LABEL:  func.func @test_isinf
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x3x4xf32>) -> memref<2x3x4xi1> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_2_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_2_:%.+]] = arith.constant 4 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.cmpf oeq, [[LOAD_PARAM_0_MEM_]], [[CST_0_1_]] : f32
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpf oeq, [[LOAD_PARAM_0_MEM_]], [[CST_0_2_]] : f32
// CHECK:             [[VAR_5_:%.+]] = arith.ori [[VAR_4_]], [[VAR_3_]] : i1
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xi1>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x4xi1>
// CHECK:         }
}

// -----

func.func @test_isinf_positive(%arg0 : tensor<2x3x4xf32>) -> tensor<*xi1> {
  %0 = "onnx.IsInf"(%arg0) {detect_negative = 0 : si64, detect_positive = 1 : si64} : (tensor<2x3x4xf32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>

// CHECK-LABEL:  func.func @test_isinf_positive
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x3x4xf32>) -> memref<2x3x4xi1> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_2_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_2_:%.+]] = arith.constant 4 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK:             [[VAR_3_:%.+]] = arith.cmpf oeq, [[LOAD_PARAM_0_MEM_]], [[CST_0_2_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xi1>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x4xi1>
// CHECK:         }
}

// -----

func.func @test_isinf_negative(%arg0 : tensor<2x3x4xf32>) -> tensor<*xi1> {
  %0 = "onnx.IsInf"(%arg0) {detect_negative = 1 : si64, detect_positive = 0 : si64} : (tensor<2x3x4xf32>) -> tensor<*xi1>
  return %0 : tensor<*xi1>

// CHECK-LABEL:  func.func @test_isinf_negative
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x3x4xf32>) -> memref<2x3x4xi1> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_3_2_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_2_:%.+]] = arith.constant 4 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK:             [[VAR_3_:%.+]] = arith.cmpf oeq, [[LOAD_PARAM_0_MEM_]], [[CST_0_1_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xi1>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x4xi1>
// CHECK:         }
}

// -----

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
