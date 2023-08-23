// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// Check the lowering of ConstantOfShape when:
//   - No value attribute.
//   - The input is an empty tensor.
// Expected emitted code:
//   - No need a Krnl iterate.
//   - The output is a scalar tensor.
func.func private @test_constant_of_shape_empty_tensor(%arg0 : tensor<0xi64>) -> tensor<*xf32> {
  %0 = "onnx.ConstantOfShape"(%arg0) : (tensor<0xi64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_constant_of_shape_empty_tensor
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<f32>
  // CHECK: [[CST_VALUE:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-NOT: krnl.iterate
  // CHECK: krnl.store [[CST_VALUE]], [[RES]][] : memref<f32>
  // CHECK: return [[RES]] : memref<f32>
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
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-LABEL:  func.func private @test_constant_of_shape_dynamic_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3xi64>) -> memref<?x?x?xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[CST_0_]]{{.}} : memref<3xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_]] : i64 to index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[CST_1_]]{{.}} : memref<3xi64>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_1_]] : i64 to index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK:           [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[CST_2_]]{{.}} : memref<3xi64>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_2_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_1_]], [[VAR_3_]], [[VAR_5_]]) {{.*}}: memref<?x?x?xf32>
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_1_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_2_]]([[VAR_1_]], [[VAR_3_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_3_]]([[VAR_1_]], [[VAR_3_]], [[VAR_5_]])){
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

  // CHECK-LABEL: test_constant_of_shape_static_dims
  // CHECK: [[GLOBAL_CST:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3], value = dense<[3, 4, 5]> : tensor<3xi64>} : () -> memref<3xi64>
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
  // CHECK: [[CST_VALUE:%.+]] = arith.constant 1.000000e+00 : f32
  // CHECK: [[LOOP_DEF:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[LOOP_DEF]]#0, [[LOOP_DEF]]#1, [[LOOP_DEF]]#2) with ([[LOOP_DEF]]#0 -> %arg0 = 0 to 3, [[LOOP_DEF]]#1 -> %arg1 = 0 to 4, [[LOOP_DEF]]#2 -> %arg2 = 0 to 5){
  // CHECK:   [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_DEF]]#0, [[LOOP_DEF]]#1, [[LOOP_DEF]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:   krnl.store [[CST_VALUE]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xf32>
}

