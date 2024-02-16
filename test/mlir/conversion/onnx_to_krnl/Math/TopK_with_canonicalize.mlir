// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

func.func @top_k(%arg0: tensor<3x4xf32>, %arg1: tensor<1xi64>) -> (tensor<*xf32>, tensor<*xi64>) {
  %Values, %Indices = "onnx.TopK"(%arg0, %arg1) {axis = 1 : si64, largest = 1 : si64, sorted = 1 : si64} : (tensor<3x4xf32>, tensor<1xi64>) -> (tensor<*xf32>, tensor<*xi64>)
  return %Values, %Indices : tensor<*xf32>, tensor<*xi64>

// mlir2FileCheck.py -a'["X", "K"]'
// CHECK-LABEL:  func @top_k
// CHECK-SAME:   ([[X_:%.+]]: memref<3x4xf32>, [[K_:%.+]]: memref<1xi64>) -> (memref<3x?xf32>, memref<3x?xi64>) {
// CHECK:           [[VAR_c0_i64_:%.+]] = arith.constant 0 : i64
// CHECK:           [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_K_MEM_:%.+]] = krnl.load [[K_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_K_MEM_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<3x?xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<3x?xi64>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<3x4xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_8_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[VAR_8_]]#1, [[RES_2_]]{{.}}[[VAR_8_]]#0, [[VAR_8_]]#1] : memref<3x4xindex>
// CHECK:           }
// CHECK:           "krnl.call"([[RES_2_]], [[X_]], [[VAR_c1_i64_]], [[VAR_c0_i64_]]) {funcName = "omTensorSort", numOfOutput = 1 : si64} : (memref<3x4xindex>, memref<3x4xf32>, i64, i64) -> ()
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_5_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_6_:%.+]] = 0 to [[VAR_1_]]){
// CHECK:             [[VAR_8_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_8_2_]]#0, [[VAR_8_2_]]#1] : memref<3x4xindex>
// CHECK:             [[LOAD_X_MEM_2_:%.+]] = krnl.load [[X_]]{{.}}[[VAR_8_2_]]#0, [[LOAD_RES_2_MEM_2_]]{{.}} : memref<3x4xf32>
// CHECK:             krnl.store [[LOAD_X_MEM_2_]], [[RES_]]{{.}}[[VAR_8_2_]]#0, [[VAR_8_2_]]#1] : memref<3x?xf32>
// CHECK:             [[LOAD_RES_2_MEM_3_:%.+]] = arith.index_cast [[LOAD_RES_2_MEM_2_]] : index to i64
// CHECK:             krnl.store [[LOAD_RES_2_MEM_3_]], [[RES_1_]]{{.}}[[VAR_8_2_]]#0, [[VAR_8_2_]]#1] : memref<3x?xi64>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]] : memref<3x?xf32>, memref<3x?xi64>
// CHECK:         }
}

// -----

func.func @top_k_smallest(%arg0: tensor<3x4xf32>, %arg1: tensor<1xi64>) -> (tensor<*xf32>, tensor<*xi64>) {
  %Values, %Indices = "onnx.TopK"(%arg0, %arg1) {axis = 1 : si64, largest = 0 : si64, sorted = 1 : si64} : (tensor<3x4xf32>, tensor<1xi64>) -> (tensor<*xf32>, tensor<*xi64>)
  return %Values, %Indices : tensor<*xf32>, tensor<*xi64>

// mlir2FileCheck.py -a'["X", "K"]'
// CHECK-LABEL:  func @top_k_smallest
// CHECK-SAME:   ([[X_:%.+]]: memref<3x4xf32>, [[K_:%.+]]: memref<1xi64>) -> (memref<3x?xf32>, memref<3x?xi64>) {
// CHECK:           [[VAR_c1_i64_:%.+]] = arith.constant 1 : i64
// CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_K_MEM_:%.+]] = krnl.load [[K_]]{{.}}[[VAR_c0_]]{{.}} : memref<1xi64>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_K_MEM_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<3x?xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<3x?xi64>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<3x4xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4){
// CHECK:             [[VAR_8_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[VAR_8_]]#1, [[RES_2_]]{{.}}[[VAR_8_]]#0, [[VAR_8_]]#1] : memref<3x4xindex>
// CHECK:           }
// CHECK:           "krnl.call"([[RES_2_]], [[X_]], [[VAR_c1_i64_]], [[VAR_c1_i64_]]) {funcName = "omTensorSort", numOfOutput = 1 : si64} : (memref<3x4xindex>, memref<3x4xf32>, i64, i64) -> ()
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_5_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_6_:%.+]] = 0 to [[VAR_1_]]){
// CHECK:             [[VAR_8_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_RES_2_MEM_2_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_8_2_]]#0, [[VAR_8_2_]]#1] : memref<3x4xindex>
// CHECK:             [[LOAD_X_MEM_2_:%.+]] = krnl.load [[X_]]{{.}}[[VAR_8_2_]]#0, [[LOAD_RES_2_MEM_2_]]{{.}} : memref<3x4xf32>
// CHECK:             krnl.store [[LOAD_X_MEM_2_]], [[RES_]]{{.}}[[VAR_8_2_]]#0, [[VAR_8_2_]]#1] : memref<3x?xf32>
// CHECK:             [[LOAD_RES_2_MEM_3_:%.+]] = arith.index_cast [[LOAD_RES_2_MEM_2_]] : index to i64
// CHECK:             krnl.store [[LOAD_RES_2_MEM_3_]], [[RES_1_]]{{.}}[[VAR_8_2_]]#0, [[VAR_8_2_]]#1] : memref<3x?xi64>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]] : memref<3x?xf32>, memref<3x?xi64>
// CHECK:         }
}

// -----

func.func @top_k_unknown_dims(%arg0: tensor<?x?xf32>, %arg1: tensor<1xi64>) -> (tensor<*xf32>, tensor<*xi64>) {
  %Values, %Indices = "onnx.TopK"(%arg0, %arg1) {axis = 1 : si64, largest = 1 : si64, sorted = 1 : si64} : (tensor<?x?xf32>, tensor<1xi64>) -> (tensor<*xf32>, tensor<*xi64>)
  return %Values, %Indices : tensor<*xf32>, tensor<*xi64>

// mlir2FileCheck.py -a'["X", "K"]'
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0)[s0] -> (s0)>
// CHECK-LABEL:  func.func @top_k_unknown_dims
// CHECK-SAME:   ([[X_:%.+]]: memref<?x?xf32>, [[K_:%.+]]: memref<1xi64>) -> (memref<?x?xf32>, memref<?x?xi64>) {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i64
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_K_MEM_:%.+]] = krnl.load [[K_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_K_MEM_]] : i64 to index
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[X_]], [[CST_0_1_]] : memref<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_1_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_1_]]) {{.*}}: memref<?x?xi64>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[X_]], [[CST_0_1_]] : memref<?x?xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[X_]], [[CST_1_1_]] : memref<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc([[VAR_dim_1_]], [[VAR_dim_2_]]) {{.*}}: memref<?x?xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_1_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_1_]], [[VAR_dim_2_]])){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[VAR_4_]]#1, [[RES_2_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<?x?xindex>
// CHECK:           }
// CHECK:           "krnl.call"([[RES_2_]], [[X_]], [[CST_1_]], [[CST_0_]]) {funcName = "omTensorSort", numOfOutput = 1 : si64} : (memref<?x?xindex>, memref<?x?xf32>, i64, i64) -> ()
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]]), [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_]]){{.}}[[VAR_1_]]{{.}}){
// CHECK:             [[VAR_4_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_4_1_]]#0, [[VAR_4_1_]]#1] : memref<?x?xindex>
// CHECK:             [[LOAD_X_MEM_:%.+]] = krnl.load [[X_]]{{.}}[[VAR_4_1_]]#0, [[LOAD_RES_2_MEM_]]{{.}} : memref<?x?xf32>
// CHECK:             krnl.store [[LOAD_X_MEM_]], [[RES_]]{{.}}[[VAR_4_1_]]#0, [[VAR_4_1_]]#1] : memref<?x?xf32>
// CHECK:             [[VAR_7_:%.+]] = arith.index_cast [[LOAD_RES_2_MEM_]] : index to i64
// CHECK:             krnl.store [[VAR_7_]], [[RES_1_]]{{.}}[[VAR_4_1_]]#0, [[VAR_4_1_]]#1] : memref<?x?xi64>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_0 : memref<?x?xf32>, memref<?x?xi64>
// CHECK:         }
}
