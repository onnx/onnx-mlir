// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// Fuse both Sqrt to Add

func.func @test_fuse_element3(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %1 = "onnx.Sqrt"(%0) : (tensor<1024xf32>) -> tensor<1024xf32>
    %2 = "onnx.Sqrt"(%1) : (tensor<1024xf32>) -> tensor<1024xf32>
    return %2 : tensor<1024xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_fuse_element3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1024xf32>, [[PARAM_1_:%.+]]: memref<1024xf32>) -> memref<1024xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1024xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 1024){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<1024xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]{{.}} : memref<1024xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_5_:%.+]] = math.sqrt [[VAR_4_]] : f32
// CHECK:             [[VAR_6_:%.+]] = math.sqrt [[VAR_5_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<1024xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1024xf32>
// CHECK:         }
}

// -----

// Stop fusion after the first Sqrt because it has more one user

func.func @test_fuse_element4(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %1 = "onnx.Sqrt"(%0) : (tensor<1024xf32>) -> tensor<1024xf32>
    %2 = "onnx.Sqrt"(%1) : (tensor<1024xf32>) -> tensor<1024xf32>
    %3 = "onnx.Add"(%1, %2) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    return %3 : tensor<1024xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_fuse_element4
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1024xf32>, [[PARAM_1_:%.+]]: memref<1024xf32>) -> memref<1024xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1024xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 1024){
// CHECK:             [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]{{.}} : memref<1024xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_]]{{.}} : memref<1024xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_7_:%.+]] = math.sqrt [[VAR_6_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_3_]]{{.}} : memref<1024xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1024xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 1024){
// CHECK:             [[VAR_3_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_3_1_]]{{.}} : memref<1024xf32>
// CHECK:             [[LOAD_PARAM_1_MEM_1_:%.+]] = math.sqrt [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_3_1_]]{{.}} : memref<1024xf32>
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1024xf32>
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 1024){
// CHECK:             [[VAR_3_2_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_3_2_]]{{.}} : memref<1024xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_3_2_]]{{.}} : memref<1024xf32>
// CHECK:             [[VAR_6_1_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_6_1_]], [[RES_2_]]{{.}}[[VAR_3_2_]]{{.}} : memref<1024xf32>
// CHECK:           }
// CHECK:           return [[RES_2_]] : memref<1024xf32>
// CHECK:         }
}

// -----

// Start from unary op and dynamic

func.func @test_fuse_element7(%arg0: tensor<?xf32>, %arg1: tensor<1xf32>) -> tensor<?xf32> {
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<?xf32>, tensor<1xf32>) -> tensor<?xf32>
  %1 = "onnx.Sqrt"(%0) : (tensor<?xf32>) -> tensor<?xf32>
  return %1 : tensor<?xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_fuse_element7
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?xf32>, [[PARAM_1_:%.+]]: memref<1xf32>) -> memref<?xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<?xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<1xf32>
// CHECK:             [[VAR_4_:%.+]] = math.powf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_5_:%.+]] = math.sqrt [[VAR_4_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?xf32>
// CHECK:         }
}

// -----

// Start from binary and last fusible Op has different element type

func.func @test_fuse_element8(%arg0: tensor<?xf32>, %arg1: tensor<1xf32>) -> tensor<?xi8> {
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<?xf32>, tensor<1xf32>) -> tensor<?xf32>
  %1 = "onnx.Cast"(%0) {to = i8} : (tensor<?xf32>) -> tensor<?xi8>
  return %1 : tensor<?xi8>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_fuse_element8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?xf32>, [[PARAM_1_:%.+]]: memref<1xf32>) -> memref<?xi8> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<?xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<1xf32>
// CHECK:             [[VAR_4_:%.+]] = math.powf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_5_:%.+]] = arith.fptosi [[VAR_4_]] : f32 to i32
// CHECK:             [[VAR_6_:%.+]] = arith.trunci [[VAR_5_]] : i32 to i8
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<?xi8>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?xi8>
// CHECK:         }
}

// -----

// fusible for binary with block argument input
func.func @fuse_element_10(%arg0: tensor<4x5xf32>, %arg1: tensor<4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<4x5xf32>) -> tensor<4x5xf32>
  %1 = "onnx.Add"(%0, %arg1) : (tensor<4x5xf32>, tensor<4x5xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
// CHECK-LABEL:  func.func @fuse_element_10
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x5xf32>, [[PARAM_1_:%.+]]: memref<4x5xf32>) -> memref<4x5xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 4, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<4x5xf32>
// CHECK-DAG:         [[VAR_3_:%.+]] = math.sqrt [[LOAD_PARAM_0_MEM_]] : f32
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<4x5xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[VAR_3_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<4x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x5xf32>
// CHECK:         }
}

// -----

// fusible binary with constant input
func.func @fuse_element_14(%arg0: tensor<5xf32>) -> tensor<*xf32> {
  %cst = onnx.Constant dense<[1.0, 2.0, 3.0, 4.0, 5.0]> : tensor<5xf32>
  %0 = "onnx.Sqrt"(%arg0) : (tensor<5xf32>) -> tensor<?xf32>
  %1 = "onnx.Add"(%0, %cst) : (tensor<?xf32>, tensor<5xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
// CHECK-LABEL:  func.func @fuse_element_14
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5xf32>) -> memref<5xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [5], value = dense<[1.000000e+00, 2.000000e+00, 3.000000e+00, 4.000000e+00, 5.000000e+00]> : tensor<5xf32>} : () -> memref<5xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 5){
// CHECK:             [[VAR_2_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_]]{{.}} : memref<5xf32>
// CHECK-DAG:         [[VAR_4_:%.+]] = math.sqrt [[LOAD_PARAM_0_MEM_]] : f32
// CHECK-DAG:         [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[VAR_2_]]{{.}} : memref<5xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[VAR_4_]], [[LOAD_VAR_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_2_]]{{.}} : memref<5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<5xf32>
// CHECK:         }
}

// -----

func.func @fuse_element_15(%arg0: tensor<4x5xf32>, %arg1: tensor<?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<4x5xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg1) : (tensor<*xf32>, tensor<?xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}
// CHECK-LABEL:  func.func @fuse_element_15
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x5xf32>, [[PARAM_1_:%.+]]: memref<?xf32>) -> memref<4x5xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 4, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<4x5xf32>
// CHECK-DAG:         [[VAR_3_:%.+]] = math.sqrt [[LOAD_PARAM_0_MEM_]] : f32
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#1] : memref<?xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[VAR_3_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<4x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x5xf32>
// CHECK:         }

// -----

func.func @fuse_element_16(%arg0: tensor<4x?xf32>, %arg1: tensor<?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<4x?xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg1) : (tensor<*xf32>, tensor<?xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-LABEL:  func.func @fuse_element_16
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x?xf32>, [[PARAM_1_:%.+]]: memref<?xf32>) -> memref<4x?xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<4x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<4x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<4x?xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 4, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]])){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<4x?xf32>
// CHECK:             [[VAR_5_:%.+]] = math.sqrt [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<4x?xf32>
// CHECK:           }
// CHECK:           [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?xf32>
// CHECK:           [[VAR_1_:%.+]] = affine.max [[MAP_1_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1]
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<4x?xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_1_]])){
// CHECK-DAG:         [[VAR_3_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = arith.cmpi sgt, [[VAR_dim_]], [[CST_1_]] : index
// CHECK:             [[VAR_5_1_:%.+]] = arith.select [[LOAD_PARAM_0_MEM_1_]], [[VAR_3_1_]]#1, [[CST_0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_3_1_]]#0, [[VAR_5_1_]]{{.}} : memref<4x?xf32>
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.cmpi sgt, [[VAR_dim_1_]], [[CST_1_]] : index
// CHECK:             [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_3_1_]]#1, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_8_]]{{.}} : memref<?xf32>
// CHECK:             [[VAR_10_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_10_]], [[RES_1_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1] : memref<4x?xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<4x?xf32>
// CHECK:         }

// -----

func.func @fuse_element_17(%arg0: tensor<?x5xf32>, %arg1: tensor<?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<?x5xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%0, %arg1) : (tensor<*xf32>, tensor<?xf32>) -> tensor<*xf32>
  return %1 : tensor<*xf32>
}
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @fuse_element_17
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x5xf32>, [[PARAM_1_:%.+]]: memref<?xf32>) -> memref<?x5xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x5xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x5xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x5xf32>
// CHECK-DAG:         [[VAR_3_:%.+]] = math.sqrt [[LOAD_PARAM_0_MEM_]] : f32
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#1] : memref<?xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[VAR_3_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x5xf32>
// CHECK:         }

// -----

func.func @fuse_element_20(%533: tensor<?x?x768xf32>, %537 : tensor<?x?x1xf32>,  %361: tensor<768xf32>, %360: tensor<768xf32>) -> tensor<?x?x768xf32> {
    %538 = "onnx.Div"(%533, %537) : (tensor<?x?x768xf32>, tensor<?x?x1xf32>) -> tensor<?x?x768xf32>
    %539 = "onnx.Mul"(%538, %361) : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    %540 = "onnx.Add"(%539, %360) : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    return %540 : tensor<?x?x768xf32>
}
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5)>
// CHECK-LABEL:  func.func @fuse_element_20
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>, [[PARAM_1_:%.+]]: memref<?x?x1xf32>, [[PARAM_2_:%.+]]: memref<768xf32>, [[PARAM_3_:%.+]]: memref<768xf32>) -> memref<?x?x768xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x1xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1]
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_0_]], [[VAR_dim_2_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_1_]]) {{.*}}: memref<?x?x768xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_dim_]]_0, [[VAR_dim_]]_2, [[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_dim_]]_0, [[VAR_dim_]]_2, [[VAR_0_]], [[VAR_1_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 768){
// CHECK-DAG:         [[VAR_3_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi sgt, [[VAR_dim_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.select [[VAR_4_]], [[VAR_3_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_3_]]#1, [[CST_0_]] : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_5_]], [[VAR_7_]], [[VAR_3_]]#2] : memref<?x?x768xf32>
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sgt, [[VAR_dim_1_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[VAR_3_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi sgt, [[VAR_dim_2_]], [[CST_1_]] : index
// CHECK:             [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[VAR_3_]]#1, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_10_]], [[VAR_12_]], [[CST_0_]]{{.}} : memref<?x?x1xf32>
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_3_]]#2] : memref<768xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_16_:%.+]] = arith.mulf [[VAR_14_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK-DAG:         [[LOAD_PARAM_3_MEM_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[VAR_3_]]#2] : memref<768xf32>
// CHECK:             [[VAR_18_:%.+]] = arith.addf [[VAR_16_]], [[LOAD_PARAM_3_MEM_]] : f32
// CHECK:             krnl.store [[VAR_18_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2] : memref<?x?x768xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x768xf32>
// CHECK:         }

// -----


func.func @test_fuse_element21(%arg0: tensor<?xf32>, %arg1: tensor<1xf32>, %arg2 : tensor<1xi8>) -> tensor<?xi8> {
  %0 = "onnx.Pow"(%arg0, %arg1) : (tensor<?xf32>, tensor<1xf32>) -> tensor<?xf32>
  %1 = "onnx.Cast"(%0) {to = i8} : (tensor<?xf32>) -> tensor<?xi8>
  %2 = "onnx.Add"(%1, %arg2) : (tensor<?xi8>, tensor<1xi8>) -> tensor<?xi8>
  return %2 : tensor<?xi8>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_fuse_element21
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?xf32>, [[PARAM_1_:%.+]]: memref<1xf32>, [[PARAM_2_:%.+]]: memref<1xi8>) -> memref<?xi8> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?xi8>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]])){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<?xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<1xf32>
// CHECK:             [[VAR_4_:%.+]] = math.powf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_5_:%.+]] = arith.fptosi [[VAR_4_]] : f32 to i32
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.trunci [[VAR_5_]] : i32 to i8
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[CST_0_]]{{.}} : memref<1xi8>
// CHECK:             [[VAR_8_:%.+]] = arith.addi [[VAR_6_]], [[LOAD_PARAM_2_MEM_]] : i8
// CHECK:             krnl.store [[VAR_8_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<?xi8>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?xi8>
// CHECK:         }
}

