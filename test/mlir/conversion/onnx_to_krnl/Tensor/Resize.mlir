// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func @test_resize1(%arg0 : tensor<3x4xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = onnx.Constant dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>
  %1 = onnx.Constant dense<[1.000000e+00,  3.000000e+00]> : tensor<2xf32>
  %2 = "onnx.Resize"(%arg0, %0, %1, %cst) {coordinate_transformation_mode = "asymmetric", mode = "nearest", nearest_mode = "floor"} : (tensor<3x4xf32>, tensor<4xf32>, tensor<2xf32>, none) -> tensor<*xf32>
  "func.return"(%2) : (tensor<*xf32>) -> ()
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
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[IV]]#0 : index to i64
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
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.index_cast [[IV]]#1 : index to i64
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
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<3x12xf32>
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
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func @test_resize2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4xf32>, [[PARAM_1_:%.+]]: memref<2xf32>) -> memref<?x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<2xf32>
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_1_]]{{.}} : memref<2xf32>
// CHECK-DAG:       [[CST_3_dot_000000_:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK:           [[VAR_4_:%.+]] = arith.mulf [[CST_3_dot_000000_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:           [[VAR_5_:%.+]] = arith.fptosi [[VAR_4_]] : f32 to i64
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[VAR_5_]] : i64 to index
// CHECK-DAG:       [[CST_4_dot_000000_:%.+]] = arith.constant 4.000000e+00 : f32
// CHECK:           [[VAR_7_:%.+]] = arith.mulf [[CST_4_dot_000000_]], [[LOAD_PARAM_1_MEM_1_]] : f32
// CHECK:           [[VAR_8_:%.+]] = arith.fptosi [[VAR_7_]] : f32 to i64
// CHECK:           [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_6_]], [[VAR_9_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_6_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_2_]]([[VAR_6_]], [[VAR_9_]])){
// CHECK:             [[VAR_11_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_12_:%.+]] = arith.index_cast [[VAR_11_]]#0 : index to i64
// CHECK:             [[VAR_13_:%.+]] = arith.sitofp [[VAR_12_]] : i64 to f32
// CHECK:             [[VAR_14_:%.+]] = arith.divf [[VAR_13_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_15_:%.+]] = math.floor [[VAR_14_]] : f32
// CHECK:             [[VAR_16_:%.+]] = arith.fptosi [[VAR_15_]] : f32 to i64
// CHECK:             [[VAR_17_:%.+]] = arith.cmpi slt, [[VAR_16_]], [[CST_0_1_]] : i64
// CHECK:             [[VAR_18_:%.+]] = arith.select [[VAR_17_]], [[CST_0_1_]], [[VAR_16_]] : i64
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.index_cast [[VAR_18_]] : i64 to index
// CHECK-DAG:         [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.cmpi slt, [[VAR_19_]], [[CST_3_1_]] : index
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.subi [[CST_3_1_]], [[CST_1_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.select [[VAR_20_]], [[VAR_19_]], [[VAR_21_]] : index
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.index_cast [[VAR_11_]]#1 : index to i64
// CHECK:             [[VAR_24_:%.+]] = arith.sitofp [[VAR_23_]] : i64 to f32
// CHECK:             [[VAR_25_:%.+]] = arith.divf [[VAR_24_]], [[LOAD_PARAM_1_MEM_1_]] : f32
// CHECK:             [[VAR_26_:%.+]] = math.floor [[VAR_25_]] : f32
// CHECK:             [[VAR_27_:%.+]] = arith.fptosi [[VAR_26_]] : f32 to i64
// CHECK:             [[VAR_28_:%.+]] = arith.cmpi slt, [[VAR_27_]], [[CST_0_1_]] : i64
// CHECK:             [[VAR_29_:%.+]] = arith.select [[VAR_28_]], [[CST_0_1_]], [[VAR_27_]] : i64
// CHECK-DAG:         [[VAR_30_:%.+]] = arith.index_cast [[VAR_29_]] : i64 to index
// CHECK-DAG:         [[CST_4_1_:%.+]] = arith.constant 4 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_31_:%.+]] = arith.cmpi slt, [[VAR_30_]], [[CST_4_1_]] : index
// CHECK-DAG:         [[VAR_32_:%.+]] = arith.subi [[CST_4_1_]], [[CST_1_1_]] : index
// CHECK:             [[VAR_33_:%.+]] = arith.select [[VAR_31_]], [[VAR_30_]], [[VAR_32_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_22_]], [[VAR_33_]]{{.}} : memref<3x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_11_]]#0, [[VAR_11_]]#1] : memref<?x?xf32>
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
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func @test_resize3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?xf32>) -> memref<?x?xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [2], value = dense<[1.000000e+00, 3.000000e+00]> : tensor<2xf32>} : () -> memref<2xf32>
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?xf32>
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?xf32>
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_3_dot_000000_:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK:           [[VAR_4_:%.+]] = arith.sitofp [[VAR_3_]] : i64 to f32
// CHECK:           [[VAR_5_:%.+]] = arith.mulf [[VAR_4_]], [[CST_3_dot_000000_]] : f32
// CHECK:           [[VAR_6_:%.+]] = arith.fptosi [[VAR_5_]] : f32 to i64
// CHECK:           [[VAR_7_:%.+]] = arith.index_cast [[VAR_6_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_7_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_]], [[VAR_7_]])){
// CHECK:             [[VAR_9_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_10_:%.+]] = arith.index_cast [[VAR_9_]]#0 : index to i64
// CHECK:             [[VAR_11_:%.+]] = arith.sitofp [[VAR_10_]] : i64 to f32
// CHECK:             [[VAR_12_:%.+]] = arith.divf [[VAR_11_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_13_:%.+]] = math.floor [[VAR_12_]] : f32
// CHECK:             [[VAR_14_:%.+]] = arith.fptosi [[VAR_13_]] : f32 to i64
// CHECK:             [[VAR_15_:%.+]] = arith.cmpi slt, [[VAR_14_]], [[CST_0_1_]] : i64
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[CST_0_1_]], [[VAR_14_]] : i64
// CHECK-DAG:         [[VAR_17_:%.+]] = arith.index_cast [[VAR_16_]] : i64 to index
// CHECK-DAG:         [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_dim_7_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_4_]] : memref<?x?xf32>
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.cmpi slt, [[VAR_17_]], [[VAR_dim_7_]] : index
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.subi [[VAR_dim_7_]], [[CST_1_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.select [[VAR_18_]], [[VAR_17_]], [[VAR_19_]] : index
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.index_cast [[VAR_9_]]#1 : index to i64
// CHECK:             [[VAR_22_:%.+]] = arith.sitofp [[VAR_21_]] : i64 to f32
// CHECK:             [[VAR_23_:%.+]] = arith.divf [[VAR_22_]], [[CST_3_dot_000000_]] : f32
// CHECK:             [[VAR_24_:%.+]] = math.floor [[VAR_23_]] : f32
// CHECK:             [[VAR_25_:%.+]] = arith.fptosi [[VAR_24_]] : f32 to i64
// CHECK:             [[VAR_26_:%.+]] = arith.cmpi slt, [[VAR_25_]], [[CST_0_1_]] : i64
// CHECK:             [[VAR_27_:%.+]] = arith.select [[VAR_26_]], [[CST_0_1_]], [[VAR_25_]] : i64
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.index_cast [[VAR_27_]] : i64 to index
// CHECK-DAG:         [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_dim_9_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_3_]] : memref<?x?xf32>
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.cmpi slt, [[VAR_28_]], [[VAR_dim_9_]] : index
// CHECK-DAG:         [[VAR_30_:%.+]] = arith.subi [[VAR_dim_9_]], [[CST_1_1_]] : index
// CHECK:             [[VAR_31_:%.+]] = arith.select [[VAR_29_]], [[VAR_28_]], [[VAR_30_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_20_]], [[VAR_31_]]{{.}} : memref<?x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_9_]]#0, [[VAR_9_]]#1] : memref<?x?xf32>
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
// CHECK-LABEL:  func.func @test_resize2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x4xf32>) -> memref<3x12xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [4], value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00]> : tensor<4xf32>} : () -> memref<4xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [2], value = dense<[1.000000e+00, 3.000000e+00]> : tensor<2xf32>} : () -> memref<2xf32>
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_3_dot_000000_:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK-DAG:       [[CST_4_dot_000000_:%.+]] = arith.constant 4.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_200000_:%.+]] = arith.constant 1.200000e+01 : f32
// CHECK-DAG:       [[CST_12_:%.+]] = arith.constant 12 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x12xf32>
// CHECK:           "krnl.call"([[RES_]], [[PARAM_0_]], [[VAR_1_]], [[VAR_2_]]) {funcName = "Resize_Scales", mode = "linear", nearest_mode = "round_prefer_floor", numOfOutput = 1 : si64} : (memref<3x12xf32>, memref<3x4xf32>, memref<4xf32>, memref<2xf32>) -> ()
// CHECK:           return [[RES_]] : memref<3x12xf32>
// CHECK:         }
}

