// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// Check lowering transpose to a view op when the order of the dimensions whose
// value is not 1 is unchanged.
func.func @test_transpose_lowered_to_a_view_op(%arg0: tensor<?x1x1x384xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<?x1x1x384xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-LABEL:  func @test_transpose_lowered_to_a_view_op
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x1x384xf32>) -> memref<?x384x1x1xf32> {
  // CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
  // CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x1x1x384xf32>
  // CHECK:           [[VAR_1_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: {{.}}[[VAR_0_]], 384, 1, 1], strides: [384, 1, 1, 1] : memref<?x1x1x384xf32> to memref<?x384x1x1xf32>
  // CHECK:           return [[VAR_1_]] : memref<?x384x1x1xf32>
  // CHECK:         }
}

// -----

// Check lowering transpose to a view op when the order of the dimensions whose
// value is not 1 is unchanged.
// The order of the dimension whose value is not 1 is changed by transpose.

func.func @test_transpose_lowered_to_a_view_op_inv(%arg0: tensor<?x1x1x384xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [3, 0, 1, 2]} : (tensor<?x1x1x384xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_transpose_lowered_to_a_view_op_inv
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x1x384xf32>) -> memref<384x1x1x?xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x1x1x384xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<384x1x1x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x1x1x384xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 1, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 384){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<?x1x1x384xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#3, [[VAR_1_]]#2, [[VAR_1_]]#1, [[VAR_1_]]#0] : memref<384x1x1x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<384x1x1x?xf32>
// CHECK:         }
}

// -----


func.func @test_transpose_block_1_last_dim(%arg0: tensor<?x256x12x64xf32>) -> tensor<?x12x256x64xf32> {
    %1 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3] } : (tensor<?x256x12x64xf32>) -> tensor<?x12x256x64xf32>
    return %1 : tensor<?x12x256x64xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_transpose_block_1_last_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x256x12x64xf32>) -> memref<64x12x256x64xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<64x12x256x64xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x256x12x64xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 256, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 12, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 64){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<?x256x12x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#3, [[VAR_1_]]#2, [[VAR_1_]]#1, [[VAR_1_]]#0] : memref<64x12x256x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<64x12x256x64xf32>
// CHECK:         }
}

// -----


func.func @test_transpose_block_2_last_dims(%arg0: tensor<2x256x12x32x64xf32>) -> tensor<2x12x256x32x64xf32> {
    %1 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3, 4] } : (tensor<2x256x12x32x64xf32>) -> tensor<2x12x256x32x64xf32>
    return %1 : tensor<2x12x256x32x64xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_transpose_block_2_last_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x256x12x32x64xf32>) -> memref<2x12x256x32x64xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x12x256x32x64xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:5 = krnl.define_loops 5
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 256, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 12, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 32, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 64){
// CHECK:             [[VAR_1_:%.+]]:5 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3, [[VAR_1_]]#4] : memref<2x256x12x32x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#4, [[VAR_1_]]#3, [[VAR_1_]]#2, [[VAR_1_]]#1, [[VAR_1_]]#0] : memref<2x12x256x32x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x12x256x32x64xf32>
// CHECK:         }
}

