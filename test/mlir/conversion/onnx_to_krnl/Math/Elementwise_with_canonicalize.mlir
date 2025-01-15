// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// COM: Check the template for lowering binary operations whose output type can be different from its input type.
// With updated approach, no max is needed for the first dim as max(dim(arg0, 0), 1) is always dim(arg0, 0).
func.func @test_binary_elementwise_op_template_unknown_dims(%arg0: tensor<?x4x5xf32>, %arg1: tensor<1x?x1xf32>) -> tensor<?x4x5xi1> {
  %0 = "onnx.Less"(%arg0, %arg1) : (tensor<?x4x5xf32>, tensor<1x?x1xf32>) -> tensor<?x4x5xi1>
  return %0 : tensor<?x4x5xi1>
// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func @test_binary_elementwise_op_template_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x5xf32>, [[PARAM_1_:%.+]]: memref<1x?x1xf32>) -> memref<?x4x5xi1> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x4x5xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<1x?x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x4x5xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<?x4x5xf32>
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi sgt, [[VAR_1_]], [[CST_1_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[IV]]#1, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]], [[VAR_6_]], [[CST_0_]]{{.}} : memref<1x?x1xf32>
// CHECK:             [[VAR_8_:%.+]] = arith.cmpf olt, [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_8_]], [[RES_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<?x4x5xi1>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x4x5xi1>
// CHECK:         }
}

// -----

// COM: Check the template for lowering variadic operations and binary operations whose output type is the same as its input type: Min, Max, Add, Sub, etc.

func.func @test_variadic_elementwise_op_template_unknown_dims(%arg0: tensor<?x4x1xf32>, %arg1: tensor<?x?x5xf32>, %arg2: tensor<?x1x5xf32>) -> tensor<?x4x5xf32> {
  %0 = "onnx.Max"(%arg0, %arg1, %arg2) : (tensor<?x4x1xf32>, tensor<?x?x5xf32>, tensor<?x1x5xf32>) -> tensor<?x4x5xf32>
  return %0 : tensor<?x4x5xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-LABEL:  func.func @test_variadic_elementwise_op_template_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x1xf32>, [[PARAM_1_:%.+]]: memref<?x?x5xf32>, [[PARAM_2_:%.+]]: memref<?x1x5xf32>) -> memref<?x4x5xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x4x1xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x5xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x5xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_2_]], [[CST_0_]] : memref<?x1x5xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]
// CHECK:           [[VAR_1_:%.+]] = arith.cmpi sgt, [[VAR_dim_2_]], [[VAR_0_]] : index
// CHECK:           [[VAR_2_:%.+]] = arith.select [[VAR_1_]], [[VAR_dim_2_]], [[VAR_0_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_2_]]) {{.*}}: memref<?x4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_]]_0, [[VAR_2_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK-DAG:         [[VAR_4_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi sgt, [[VAR_dim_]], [[CST_1_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[VAR_4_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_4_]]#1, [[CST_0_]]{{.}} : memref<?x4x1xf32>
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.select [[VAR_8_]], [[VAR_4_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpi sgt, [[VAR_dim_1_]], [[CST_1_]] : index
// CHECK:             [[VAR_11_:%.+]] = arith.select [[VAR_10_]], [[VAR_4_]]#1, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_9_]], [[VAR_11_]], [[VAR_4_]]#2] : memref<?x?x5xf32>
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.maxnumf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.cmpi sgt, [[VAR_dim_2_]], [[CST_1_]] : index
// CHECK:             [[VAR_15_:%.+]] = arith.select [[VAR_14_]], [[VAR_4_]]#0, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_15_]], [[CST_0_]], [[VAR_4_]]#2] : memref<?x1x5xf32>
// CHECK:             [[VAR_17_:%.+]] = arith.maxnumf [[VAR_13_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             krnl.store [[VAR_17_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_4_]]#2] : memref<?x4x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x4x5xf32>
// CHECK:         }
}

// -----

func.func @test_gelu_none(%arg0 : tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.Gelu"(%arg0) {approximate = "none"} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_gelu_none
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x3x4xf32>) -> memref<2x3x4xf32> {
// CHECK-DAG:       [[CST_0_dot_707106769_:%.+]] = arith.constant 0.707106769 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xf32>
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_707106769_]] : f32
// CHECK:             [[VAR_5_:%.+]] = math.erf [[VAR_4_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[VAR_5_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.mulf [[VAR_3_]], [[VAR_6_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x4xf32>
// CHECK:         }
}

// -----

func.func @test_gelu_tanh(%arg0 : tensor<2x3x4xf32>) -> tensor<2x3x4xf32> {
  %0 = "onnx.Gelu"(%arg0) {approximate = "tanh"} : (tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
  return %0 : tensor<2x3x4xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_gelu_tanh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x3x4xf32>) -> memref<2x3x4xf32> {
// CHECK-DAG:       [[CST_0_dot_797884583_:%.+]] = arith.constant 0.797884583 : f32
// CHECK-DAG:       [[CST_4_dot_471500_:%.+]] = arith.constant 4.471500e-02 : f32
// CHECK-DAG:       [[CST_3_dot_000000_:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xf32>
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.mulf [[LOAD_PARAM_0_MEM_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_4_:%.+]] = math.powf [[LOAD_PARAM_0_MEM_]], [[CST_3_dot_000000_]] : f32
// CHECK:             [[VAR_5_:%.+]] = arith.mulf [[VAR_4_]], [[CST_4_dot_471500_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[VAR_5_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.mulf [[VAR_6_]], [[CST_0_dot_797884583_]] : f32
// CHECK:             [[VAR_8_:%.+]] = math.tanh [[VAR_7_]] : f32
// CHECK:             [[VAR_9_:%.+]] = arith.addf [[VAR_8_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_10_:%.+]] = arith.mulf [[VAR_3_]], [[VAR_9_]] : f32
// CHECK:             krnl.store [[VAR_10_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<2x3x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x4xf32>
// CHECK:         }
}

// -----

// COM: Check PRelu with unidirectional broadcasting and unknown dimensions.
// COM: Because of unidirectional broadcasting, always get constant dimensions from X even thought their values are 1.

func.func @test_prelu_broadcast_unknown_dims(%arg0: tensor<3x1x5xf32>, %arg1: tensor<3x?x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x1x5xf32>, tensor<3x?x1xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_prelu_broadcast_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x1x5xf32>, [[PARAM_1_:%.+]]: memref<3x?x1xf32>) -> memref<3x1x5xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x1x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[CST_0_]], [[VAR_1_]]#2] : memref<3x1x5xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[CST_0_]]{{.}} : memref<3x?x1xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpf olt, [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.mulf [[LOAD_PARAM_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<3x1x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x1x5xf32>
// CHECK:         }
}

// -----

// COM: Check PRelu with unidirectional broadcasting and unknown dimensions.
// COM: If X's dimensions are unknown, get dimensions from slope whenever they are non-zero constants.
func.func @test_prelu_broadcast_unknown_dims1(%arg0: tensor<?x2x?xf32>, %arg1: tensor<?x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<?x2x?xf32>, tensor<?x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
  // CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
  // CHECK-LABEL: @test_prelu_broadcast_unknown_dims1
  // CHECK-DAG: [[CST0:%.+]] = arith.constant 0 : index
  // CHECK-DAG: [[CST1:%.+]] = arith.constant 1 : index
  // CHECK-DAG: [[CST0_f32:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK:     [[DIM0_X:%.+]] = memref.dim %arg0, [[CST0]] : memref<?x2x?xf32>
  // CHECK:     [[DIM0_SLOPE:%.+]] = memref.dim %arg1, [[CST0]] : memref<?x5xf32>
  // CHECK:     [[RES:%.+]] = memref.alloc([[DIM0_X]]) {{.*}} : memref<?x2x5xf32>
  // CHECK:     [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK:     krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to [[MAP_0_]]([[DIM0_X]]), [[MAIN_LOOP]]#1 -> %arg3 = 0 to 2, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5){
  // CHECK:       [[IV:%.+]]:3 = krnl.get_induction_var_value([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<?x2x?xf32>
  // CHECK:       [[GREATER_THAN_ONE:%.+]] = arith.cmpi sgt, [[DIM0_SLOPE]], [[CST1]] : index
  // CHECK:       [[SELECT1:%.+]] = arith.select [[GREATER_THAN_ONE]], [[IV]]#1, [[CST0]] : index
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1{{\[}}[[SELECT1]], [[IV]]#2] : memref<?x5xf32>
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = arith.cmpf olt, [[LOAD_X]], [[CST0_f32]] : f32
  // CHECK:       [[MUL:%.+]] = arith.mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT2:%.+]] = arith.select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:       krnl.store [[SELECT2]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<?x2x5xf32>
  // CHECK:     }
  // CHECK:     return [[RES]] : memref<?x2x5xf32>
}

// -----

// dim analysis detect that Prelu has the same size inputs, and thus issue no broadcast.
func.func @test_prelu_broadcast_ruled_out_by_dim_analysis(%arg0: tensor<?x4x5xi32>, %arg1: tensor<?x4x5xi32>) -> tensor<*xi32> {
  %0 = "onnx.PRelu"(%arg0, %arg0) : (tensor<?x4x5xi32>, tensor<?x4x5xi32>) -> tensor<*xi32>
  return %0 : tensor<*xi32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_prelu_broadcast_ruled_out_by_dim_analysis
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x5xi32>, [[PARAM_1_:%.+]]: memref<?x4x5xi32>) -> memref<?x4x5xi32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_1_]] : memref<?x4x5xi32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x4x5xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<?x4x5xi32>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<?x4x5xi32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi slt, [[LOAD_PARAM_0_MEM_]], [[CST_0_]] : i32
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.muli [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_0_MEM_]] : i32
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[LOAD_PARAM_0_MEM_]] : i32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<?x4x5xi32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x4x5xi32>
// CHECK:         }
}

// -----
// COM: Check PRelu with unidirectional broadcasting.
// COM: Tensor slope should be unidirectional broadcastable to input tensor X
func.func @test_prelu_broadcast3(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x1x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x1x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_prelu_broadcast3
  // CHECK-DAG: [[ZERO_INDEX:%.+]] = arith.constant 0 : index
  // CHECK-DAG: [[CST_0:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5){
  // CHECK:       [[IV:%.+]]:3 = krnl.get_induction_var_value([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1[[[IV]]#0, [[ZERO_INDEX]], [[IV]]#2] : memref<3x1x5xf32>
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
func.func @test_prelu_broadcast4(%arg0: tensor<3x4x5xf32>, %arg1: tensor<3x1x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.PRelu"(%arg0, %arg1) : (tensor<3x4x5xf32>, tensor<3x1x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

  // CHECK-LABEL: func @test_prelu_broadcast4
  // CHECK-DAG: [[ZERO_INDEX:%.+]] = arith.constant 0 : index
  // CHECK-DAG: [[CST_0:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-DAG: [[RES:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
  // CHECK: [[MAIN_LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) with ([[MAIN_LOOP]]#0 -> %arg2 = 0 to 3, [[MAIN_LOOP]]#1 -> %arg3 = 0 to 4, [[MAIN_LOOP]]#2 -> %arg4 = 0 to 5){
  // CHECK:       [[IV:%.+]]:3 = krnl.get_induction_var_value([[MAIN_LOOP]]#0, [[MAIN_LOOP]]#1, [[MAIN_LOOP]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:       [[LOAD_X:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
  // CHECK:       [[LOAD_SLOPE:%.+]] = krnl.load %arg1[[[IV]]#0, [[ZERO_INDEX]], [[IV]]#2] : memref<3x1x5xf32>
  // CHECK:       [[LESS_THAN_ZERO:%.+]] = arith.cmpf olt, [[LOAD_X]], [[CST_0]] : f32
  // CHECK:       [[MUL:%.+]] = arith.mulf [[LOAD_SLOPE]], [[LOAD_X]] : f32
  // CHECK:       [[SELECT:%.+]] = arith.select [[LESS_THAN_ZERO]], [[MUL]], [[LOAD_X]] : f32
  // CHECK:       krnl.store [[SELECT]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x4x5xf32>
  // CHECK: }
  // CHECK: return [[RES]] : memref<3x4x5xf32>
}

// -----

func.func @test_nonzero(%arg0: tensor<2x2xi1>) -> tensor<*xi64> {
    %0 = "onnx.NonZero"(%arg0) : (tensor<2x2xi1>) -> tensor<*xi64>
    return %0 : tensor<*xi64>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_nonzero
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x2xi1>) -> memref<2x?xi64> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[CST_0_]], [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = [[CST_0_]] to [[CST_2_]]){
// CHECK:             [[VAR_9_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             krnl.store [[CST_0_]], [[RES_1_]]{{.}}[[VAR_9_]]{{.}} : memref<2xindex>
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<2xindex>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = [[CST_0_]] to [[CST_2_]]){
// CHECK:             [[VAR_9_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             krnl.store [[CST_0_]], [[RES_2_]]{{.}}[[VAR_9_1_]]{{.}} : memref<2xindex>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 2){
// CHECK:             [[VAR_9_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_9_2_]]#0, [[VAR_9_2_]]#1] : memref<2x2xi1>
// CHECK:             [[VAR_11_:%.+]] = arith.cmpi eq, [[LOAD_INPUT_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_0_]], [[CST_1_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_14_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_12_]] : index
// CHECK:             krnl.store [[VAR_14_]], [[RES_]][] : memref<index>
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_9_2_]]#0] : memref<2xindex>
// CHECK:             [[VAR_16_:%.+]] = arith.addi [[LOAD_RES_1_MEM_]], [[VAR_12_]] : index
// CHECK:             krnl.store [[VAR_16_]], [[RES_1_]]{{.}}[[VAR_9_2_]]#0] : memref<2xindex>
// CHECK:             [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_9_2_]]#1] : memref<2xindex>
// CHECK:             [[VAR_18_:%.+]] = arith.addi [[LOAD_RES_2_MEM_]], [[VAR_12_]] : index
// CHECK:             krnl.store [[VAR_18_]], [[RES_2_]]{{.}}[[VAR_9_2_]]#1] : memref<2xindex>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<2x?xi64>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloca() : memref<index>
// CHECK-DAG:       [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_4_:%.+]] = [[CST_0_]] to [[LOAD_RES_MEM_1_]]){
// CHECK-DAG:         [[VAR_9_3_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:             krnl.store [[CST_minus_1_]], [[RES_4_]][] : memref<index>
// CHECK:             krnl.store [[CST_0_]], [[RES_5_]][] : memref<index>
// CHECK:             [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_5_:%.+]] = [[CST_0_]] to [[CST_2_]]){
// CHECK:               [[VAR_20_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_20_]]{{.}} : memref<2xindex>
// CHECK-DAG:           [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:           [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:               [[VAR_24_:%.+]] = arith.addi [[LOAD_RES_5_MEM_]], [[LOAD_RES_1_MEM_1_]] : index
// CHECK-DAG:           [[VAR_25_:%.+]] = arith.cmpi slt, [[VAR_9_3_]], [[VAR_24_]] : index
// CHECK-DAG:           [[VAR_26_:%.+]] = arith.cmpi eq, [[LOAD_RES_4_MEM_]], [[CST_minus_1_]] : index
// CHECK:               [[VAR_27_:%.+]] = arith.andi [[VAR_25_]], [[VAR_26_]] : i1
// CHECK:               [[VAR_28_:%.+]] = arith.select [[VAR_27_]], [[VAR_20_]], [[LOAD_RES_4_MEM_]] : index
// CHECK:               krnl.store [[VAR_28_]], [[RES_4_]][] : memref<index>
// CHECK:               krnl.store [[VAR_24_]], [[RES_5_]][] : memref<index>
// CHECK:             }
// CHECK:             [[LOAD_RES_4_MEM_1_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:             [[VAR_14_1_:%.+]] = arith.index_cast [[LOAD_RES_4_MEM_1_]] : index to i64
// CHECK:             krnl.store [[VAR_14_1_]], [[RES_3_]]{{.}}[[CST_0_]], [[VAR_9_3_]]{{.}} : memref<2x?xi64>
// CHECK:             krnl.store [[CST_minus_1_]], [[RES_4_]][] : memref<index>
// CHECK:             krnl.store [[CST_0_]], [[RES_5_]][] : memref<index>
// CHECK:             [[LOOP_5_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_5_]]) with ([[LOOP_5_]] -> [[I_6_:%.+]] = [[CST_0_]] to [[CST_2_]]){
// CHECK:               [[VAR_20_1_:%.+]] = krnl.get_induction_var_value([[LOOP_5_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_20_1_]]{{.}} : memref<2xindex>
// CHECK-DAG:           [[LOAD_RES_5_MEM_1_:%.+]] = krnl.load [[RES_5_]][] : memref<index>
// CHECK-DAG:           [[LOAD_RES_4_MEM_2_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:               [[VAR_24_1_:%.+]] = arith.addi [[LOAD_RES_5_MEM_1_]], [[LOAD_RES_2_MEM_1_]] : index
// CHECK-DAG:           [[VAR_25_1_:%.+]] = arith.cmpi slt, [[VAR_9_3_]], [[VAR_24_1_]] : index
// CHECK-DAG:           [[VAR_26_1_:%.+]] = arith.cmpi eq, [[LOAD_RES_4_MEM_2_]], [[CST_minus_1_]] : index
// CHECK:               [[VAR_27_1_:%.+]] = arith.andi [[VAR_25_1_]], [[VAR_26_1_]] : i1
// CHECK:               [[VAR_28_1_:%.+]] = arith.select [[VAR_27_1_]], [[VAR_20_1_]], [[LOAD_RES_4_MEM_2_]] : index
// CHECK:               krnl.store [[VAR_28_1_]], [[RES_4_]][] : memref<index>
// CHECK:               krnl.store [[VAR_24_1_]], [[RES_5_]][] : memref<index>
// CHECK:             }
// CHECK:             [[VAR_18_1_:%.+]] = krnl.load [[RES_4_]][] : memref<index>
// CHECK:             [[VAR_19_:%.+]] = arith.index_cast [[VAR_18_1_]] : index to i64
// CHECK:             krnl.store [[VAR_19_]], [[RES_3_]]{{.}}[[CST_1_]], [[VAR_9_3_]]{{.}} : memref<2x?xi64>
// CHECK:           }
// CHECK:           return [[RES_3_]] : memref<2x?xi64>
// CHECK:         }
}

// -----

func.func @test_mod_fp32(%arg0: tensor<6xf32>, %arg1: tensor<6xf32>) -> tensor<*xf32> {
  %0 = "onnx.Mod"(%arg0, %arg1) {fmod = 1 : si64} : (tensor<6xf32>, tensor<6xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>
// mlir2FileCheck.py -a'["a", "b"]'
// CHECK-LABEL:  func @test_mod_fp32
// CHECK-SAME:   ([[A_:%.+]]: memref<6xf32>, [[B_:%.+]]: memref<6xf32>) -> memref<6xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<6xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 6){
// CHECK:             [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[IV]]{{.}} : memref<6xf32>
// CHECK-DAG:         [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[IV]]{{.}} : memref<6xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.remf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:             [[VAR_5_:%.+]] = math.copysign [[VAR_4_]], [[LOAD_A_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[IV]]{{.}} : memref<6xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6xf32>
// CHECK:         }
}

// -----

func.func @test_mean(%arg0: tensor<3xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>) -> tensor<*xf32>  {
    %0 = "onnx.Mean"(%arg0, %arg1, %arg2) : (tensor<3xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>
// mlir2FileCheck.py -a'["a", "b", "c"]'
// CHECK-LABEL:  func @test_mean
// CHECK-SAME:   ([[A_:%.+]]: memref<3xf32>, [[B_:%.+]]: memref<3xf32>, [[C_:%.+]]: memref<3xf32>) -> memref<3xf32> {
// CHECK-DAG:       [[CST_3_dot_000000_:%.+]] = arith.constant 3.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]][[[IV]]] : memref<3xf32>
// CHECK-DAG:         [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]][[[IV]]] : memref<3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.addf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK-DAG:         [[LOAD_C_MEM_:%.+]] = krnl.load [[C_]][[[IV]]] : memref<3xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[VAR_4_]], [[LOAD_C_MEM_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.divf [[VAR_6_]], [[CST_3_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]][[[IV]]] : memref<3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3xf32>
// CHECK:         }
}

// -----

func.func @where(%arg0: tensor<2x2xi1>, %arg1: tensor<2x2xf32>, %arg2: tensor<2x2xf32>) -> tensor<2x2xf32> {
  %0 = "onnx.Where"(%arg0, %arg1, %arg2) : (tensor<2x2xi1>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
  return %0 : tensor<2x2xf32>

// mlir2FileCheck.py -a'["condition", "x", "y"]'
// CHECK-LABEL:  func @where
// CHECK-SAME:   ([[CONDITION_:%.+]]: memref<2x2xi1>, [[X_:%.+]]: memref<2x2xf32>, [[Y_:%.+]]: memref<2x2xf32>) -> memref<2x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]][[[IV]]#0, [[IV]]#1] : memref<2x2xi1>
// CHECK-DAG:         [[LOAD_X_MEM_:%.+]] = krnl.load [[X_]][[[IV]]#0, [[IV]]#1] : memref<2x2xf32>
// CHECK-DAG:         [[LOAD_Y_MEM_:%.+]] = krnl.load [[Y_]][[[IV]]#0, [[IV]]#1] : memref<2x2xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.select [[LOAD_CONDITION_MEM_]], [[LOAD_X_MEM_]], [[LOAD_Y_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<2x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2xf32>
// CHECK:         }
}

// -----


func.func @round(%arg0: tensor<15xf32>) -> tensor<*xf32> {
  %0 = "onnx.Round"(%arg0) : (tensor<15xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @round
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<15xf32>) -> memref<15xf32> {
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<15xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 15){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<15xf32>
// CHECK:             [[VAR_3_:%.+]] = math.floor [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_4_:%.+]] = arith.subf [[LOAD_PARAM_0_MEM_]], [[VAR_3_]] : f32
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpf ogt, [[VAR_4_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addf [[VAR_3_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.select [[VAR_5_]], [[VAR_6_]], [[VAR_3_]] : f32
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.mulf [[VAR_3_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_9_:%.+]] = math.floor [[VAR_8_]] : f32
// CHECK:             [[VAR_10_:%.+]] = arith.mulf [[VAR_9_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_11_:%.+]] = arith.subf [[VAR_3_]], [[VAR_10_]] : f32
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.cmpf oeq, [[VAR_11_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.addf [[VAR_3_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_12_]], [[VAR_13_]], [[VAR_3_]] : f32
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.cmpf oeq, [[VAR_4_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[VAR_14_]], [[VAR_7_]] : f32
// CHECK:             krnl.store [[VAR_16_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<15xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<15xf32>
// CHECK:         }
}

// -----

func.func @roberta_partial_simd_1dim_v1(%arg0: tensor<?x?x768xf32>, %arg1: tensor<?x?x768xf32>) -> tensor<?x?x768xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5)>
// CHECK-LABEL:  func.func @roberta_partial_simd_1dim_v1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>, [[PARAM_1_:%.+]]: memref<?x?x768xf32>) -> memref<?x?x768xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x768xf32>
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
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_10_]], [[VAR_12_]], [[VAR_3_]]#2] : memref<?x?x768xf32>
// CHECK:             [[VAR_14_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_14_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2] : memref<?x?x768xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x768xf32>
// CHECK:         }
}

// -----

func.func @roberta_partial_simd_1dim_v2(%arg0: tensor<?x?x768xf32>, %arg1: tensor<768xf32>) -> tensor<?x?x768xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func @roberta_partial_simd_1dim_v2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>, [[PARAM_1_:%.+]]: memref<768xf32>) -> memref<?x?x768xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?x768xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_]]_0), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 768){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<?x?x768xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#2] : memref<768xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<?x?x768xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x768xf32>
// CHECK:         }
}

// -----

func.func @roberta_partial_simd_1dim_scalar(%arg0: tensor<?x?x768xf32>, %arg1: tensor<1xf32>) -> tensor<?x?x768xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<1xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func @roberta_partial_simd_1dim_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>, [[PARAM_1_:%.+]]: memref<1xf32>) -> memref<?x?x768xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x768xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x768xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_0) {{.*}}: memref<?x?x768xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_]]_0), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 768){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<?x?x768xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<1xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2] : memref<?x?x768xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x768xf32>
// CHECK:         }
}

// -----

// has ?x? in the first 2 dims for both params

func.func @roberta_partial_simd_2dim_v1(%arg0: tensor<?x?x96x8xf32>, %arg1: tensor<?x?x96x8xf32>) -> tensor<?x?x96x8xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x96x8xf32>, tensor<?x?x96x8xf32>) -> tensor<?x?x96x8xf32>
    return %0 : tensor<?x?x96x8xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5)>
// CHECK-LABEL:  func.func @roberta_partial_simd_2dim_v1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x96x8xf32>, [[PARAM_1_:%.+]]: memref<?x?x96x8xf32>) -> memref<?x?x96x8xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x96x8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1]
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_0_]], [[VAR_dim_2_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_1_]]) {{.*}}: memref<?x?x96x8xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_dim_]]_0, [[VAR_dim_]]_2, [[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_dim_]]_0, [[VAR_dim_]]_2, [[VAR_0_]], [[VAR_1_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 96, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 8){
// CHECK-DAG:         [[VAR_3_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi sgt, [[VAR_dim_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.select [[VAR_4_]], [[VAR_3_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_3_]]#1, [[CST_0_]] : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_5_]], [[VAR_7_]], [[VAR_3_]]#2, [[VAR_3_]]#3] : memref<?x?x96x8xf32>
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sgt, [[VAR_dim_1_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[VAR_3_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi sgt, [[VAR_dim_2_]], [[CST_1_]] : index
// CHECK:             [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[VAR_3_]]#1, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_10_]], [[VAR_12_]], [[VAR_3_]]#2, [[VAR_3_]]#3] : memref<?x?x96x8xf32>
// CHECK:             [[VAR_14_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_14_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2, [[VAR_3_]]#3] : memref<?x?x96x8xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x96x8xf32>
// CHECK:         }
}

// -----

// has ?x2 and ?x? in the first 2 dims

func.func @roberta_partial_simd_2dim_v2(%arg0: tensor<?x2x96x8xf32>, %arg1: tensor<?x?x96x8xf32>) -> tensor<?x?x96x8xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x2x96x8xf32>, tensor<?x?x96x8xf32>) -> tensor<?x?x96x8xf32>
    return %0 : tensor<?x?x96x8xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-LABEL:  func.func @roberta_partial_simd_2dim_v2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x2x96x8xf32>, [[PARAM_1_:%.+]]: memref<?x?x96x8xf32>) -> memref<?x2x96x8xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x2x96x8xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x96x8xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x96x8xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x2x96x8xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_]]_0, [[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 96, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 8){
// CHECK-DAG:         [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.cmpi sgt, [[VAR_dim_]], [[CST_1_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[VAR_2_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]], [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<?x2x96x8xf32>
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_2_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpi sgt, [[VAR_dim_1_]], [[CST_1_]] : index
// CHECK:             [[VAR_9_:%.+]] = arith.select [[VAR_8_]], [[VAR_2_]]#1, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_7_]], [[VAR_9_]], [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<?x?x96x8xf32>
// CHECK:             [[VAR_11_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_11_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<?x2x96x8xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x2x96x8xf32>
// CHECK:         }
}

// -----

// has ?x2 and ?x1 in the first 2 dims

func.func @roberta_partial_simd_2dim_v3(%arg0: tensor<?x2x96x8xf32>, %arg1: tensor<?x1x96x8xf32>) -> tensor<?x?x96x8xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x2x96x8xf32>, tensor<?x1x96x8xf32>) -> tensor<?x?x96x8xf32>
    return %0 : tensor<?x?x96x8xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-LABEL:  func.func @roberta_partial_simd_2dim_v3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x2x96x8xf32>, [[PARAM_1_:%.+]]: memref<?x1x96x8xf32>) -> memref<?x2x96x8xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x2x96x8xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x1x96x8xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_0]
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x2x96x8xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_]]_0, [[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 96, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 8){
// CHECK-DAG:         [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.cmpi sgt, [[VAR_dim_]], [[CST_1_]] : index
// CHECK:             [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[VAR_2_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]], [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<?x2x96x8xf32>
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_2_]]#0, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_7_]], [[CST_0_]], [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<?x1x96x8xf32>
// CHECK:             [[VAR_9_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_9_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<?x2x96x8xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x2x96x8xf32>
// CHECK:         }
}

// -----

func.func @roberta_partial_simd_2dim_not_0_mod_vl(%arg0: tensor<?x?x95x7xf32>, %arg1: tensor<?x?x95x7xf32>) -> tensor<?x?x95x7xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x95x7xf32>, tensor<?x?x95x7xf32>) -> tensor<?x?x95x7xf32>
    return %0 : tensor<?x?x95x7xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5)>
// CHECK-LABEL:  func.func @roberta_partial_simd_2dim_not_0_mod_vl
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x95x7xf32>, [[PARAM_1_:%.+]]: memref<?x?x95x7xf32>) -> memref<?x?x95x7xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x95x7xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x95x7xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_1_]], [[CST_0_]] : memref<?x?x95x7xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_1_]], [[CST_1_]] : memref<?x?x95x7xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_]], [[VAR_dim_]]_1]
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_dim_0_]], [[VAR_dim_2_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_1_]]) {{.*}}: memref<?x?x95x7xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_dim_]]_0, [[VAR_dim_]]_2, [[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_dim_]]_0, [[VAR_dim_]]_2, [[VAR_0_]], [[VAR_1_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 95, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 7){
// CHECK-DAG:         [[VAR_3_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpi sgt, [[VAR_dim_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.select [[VAR_4_]], [[VAR_3_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.cmpi sgt, [[VAR_dim_0_]], [[CST_1_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_3_]]#1, [[CST_0_]] : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_5_]], [[VAR_7_]], [[VAR_3_]]#2, [[VAR_3_]]#3] : memref<?x?x95x7xf32>
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.cmpi sgt, [[VAR_dim_1_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[VAR_3_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi sgt, [[VAR_dim_2_]], [[CST_1_]] : index
// CHECK:             [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[VAR_3_]]#1, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_10_]], [[VAR_12_]], [[VAR_3_]]#2, [[VAR_3_]]#3] : memref<?x?x95x7xf32>
// CHECK:             [[VAR_14_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_14_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2, [[VAR_3_]]#3] : memref<?x?x95x7xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x95x7xf32>
}

// -----

// Test whether lowering is correct for a string tensor input.

func.func @test_equal_string(%arg0: tensor<2x2x!onnx.String>, %arg1: tensor<2x2x!onnx.String>) -> tensor<*xi1> {
  %0 = "onnx.Equal"(%arg0, %arg1) : (tensor<2x2x!onnx.String>, tensor<2x2x!onnx.String>) -> tensor<*xi1>
   return %0 : tensor<*xi1>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_equal_string
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x2x!krnl.string>, [[PARAM_1_:%.+]]: memref<2x2x!krnl.string>) -> memref<2x2xi1> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<2x2x!krnl.string>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<2x2x!krnl.string>
// CHECK:             [[VAR_4_:%.+]] = "krnl.strlen"([[LOAD_PARAM_0_MEM_]]) : (!krnl.string) -> i64
// CHECK:             [[VAR_5_:%.+]] = "krnl.strncmp"([[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]], [[VAR_4_]]) : (!krnl.string, !krnl.string, i64) -> i32
// CHECK:             [[VAR_6_:%.+]] = arith.cmpi eq, [[VAR_5_]], [[CST_0_]] : i32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<2x2xi1>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2xi1>
// CHECK:         }
}

// -----

// Test whether lowering is correct for a int64_t tensor input.

func.func @test_equal_int64(%arg0: tensor<2x2xi64>, %arg1: tensor<2x2xi64>) -> tensor<*xi1> {
  %0 = "onnx.Equal"(%arg0, %arg1) : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<*xi1>
   return %0 : tensor<*xi1>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_equal_int64
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x2xi64>, [[PARAM_1_:%.+]]: memref<2x2xi64>) -> memref<2x2xi1> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<2x2xi64>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<2x2xi64>
// CHECK:             [[VAR_4_:%.+]] = arith.cmpi eq, [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : i64
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<2x2xi1>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2xi1>
// CHECK:         }
}

// -----

// Test whether lowering is correct for a f32 tensor input.

func.func @test_equal_float32(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<*xi1> {
  %0 = "onnx.Equal"(%arg0, %arg1) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<*xi1>
   return %0 : tensor<*xi1>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_equal_float32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x2xf32>, [[PARAM_1_:%.+]]: memref<2x2xf32>) -> memref<2x2xi1> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2xi1>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<2x2xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<2x2xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.cmpf oeq, [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<2x2xi1>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2xi1>
// CHECK:         }
}

// -----

func.func @test_erf(%arg0: tensor<?x10xf32>) -> (tensor<*xf32>) attributes {} {
  %0 = "onnx.Erf"(%arg0): (tensor<?x10xf32>) -> (tensor<*xf32>)
  return %0 : tensor<*xf32>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_erf
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.erf [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func @add_partial_splat(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<3x1x1xf32>) -> tensor<*xf32> {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x3x4x5xf32>, tensor<3x1x1xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @add_partial_splat
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x3x4x5xf32>, [[PARAM_1_:%.+]]: memref<3x1x1xf32>) -> memref<2x3x4x5xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<2x3x4x5xf32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]#1, [[CST_0_]], [[CST_0_]]{{.}} : memref<3x1x1xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<2x3x4x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x4x5xf32>
// CHECK:         }
}

// -----


func.func private @test_exp(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Exp"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_exp
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.exp [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_tanh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Tanh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_tanh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.tanh [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_sinh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sinh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_sinh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_1_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK-DAG:         [[VAR_4_:%.+]] = math.exp [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_5_:%.+]] = math.exp [[VAR_3_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.subf [[VAR_4_]], [[VAR_5_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.divf [[VAR_6_]], [[CST_2_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_cosh(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Cosh"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_cosh
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_1_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK-DAG:         [[VAR_3_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK-DAG:         [[VAR_4_:%.+]] = math.exp [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_5_:%.+]] = math.exp [[VAR_3_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[VAR_4_]], [[VAR_5_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.divf [[VAR_6_]], [[CST_2_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_cos(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Cos"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_cos
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.cos [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_sin(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sin"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_sin
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.sin [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_log(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Log"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_log
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.log [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_sigmoid(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sigmoid"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_sigmoid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_1_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_4_:%.+]] = math.exp [[VAR_3_]] : f32
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[VAR_4_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[VAR_5_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_relu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Relu"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_relu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = arith.maxnumf [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_elu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Elu"(%arg0) {alpha=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_elu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_2_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK-DAG:         [[VAR_3_:%.+]] = math.exp [[LOAD_PARAM_0_MEM_]] : f32
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpf olt, [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_5_:%.+]] = arith.subf [[VAR_3_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.mulf [[VAR_5_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_4_]], [[VAR_6_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_leakyrelu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.LeakyRelu"(%arg0) {alpha=1.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_leakyrelu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_selu(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Selu"(%arg0) {alpha=1.0:f32, gamma=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_selu
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_2_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK-DAG:         [[VAR_3_:%.+]] = math.exp [[LOAD_PARAM_0_MEM_]] : f32
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.cmpf ogt, [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_5_:%.+]] = arith.subf [[VAR_3_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[LOAD_PARAM_0_MEM_]], [[VAR_5_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.mulf [[VAR_6_]], [[CST_2_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_hardsigmoid(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.HardSigmoid"(%arg0) {alpha=1.0:f32, beta=2.0:f32} : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_hardsigmoid
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_2_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_4_:%.+]] = arith.maxnumf [[VAR_3_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_5_:%.+]] = arith.minnumf [[VAR_4_]], [[CST_1_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_reciprocal(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Reciprocal"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_reciprocal
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = arith.divf [[CST_1_dot_000000_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_softplus(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softplus"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_softplus
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.exp [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[VAR_3_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_5_:%.+]] = math.log [[VAR_4_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_softsign(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softsign"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_softsign
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.absf [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[VAR_3_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_5_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_]], [[VAR_4_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_sqrt(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sqrt"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_sqrt
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.sqrt [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_sign_f(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Sign"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_sign_f
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[CST_minus_1_dot_000000_:%.+]] = arith.constant -1.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_2_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = arith.cmpf ogt, [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[CST_1_dot_000000_]], [[CST_minus_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpf oeq, [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[CST_0_dot_000000_]], [[VAR_4_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_sign_i(%arg0 : tensor<?x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.Sign"(%arg0) : (tensor<?x10xi32>) -> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_sign_i
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xi32>) -> memref<?x10xi32> {
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_1_]] : memref<?x10xi32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_1_]] : memref<?x10xi32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xi32>
// CHECK:             [[VAR_3_:%.+]] = arith.cmpi sgt, [[LOAD_PARAM_0_MEM_]], [[CST_0_]] : i32
// CHECK-DAG:         [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[CST_1_]], [[CST_minus_1_]] : i32
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi eq, [[LOAD_PARAM_0_MEM_]], [[CST_0_]] : i32
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[CST_0_]], [[VAR_4_]] : i32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xi32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xi32>
// CHECK:         }
}

// -----

func.func private @test_abs_float(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Abs"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_abs_float
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.absf [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----


func.func private @test_abs_int(%arg0 : tensor<?x10xi32>) -> tensor<*xi32> {
  %0 = "onnx.Abs"(%arg0) : (tensor<?x10xi32>) -> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_abs_int
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xi32>) -> memref<?x10xi32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xi32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xi32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xi32>
// CHECK:             [[VAR_3_:%.+]] = math.absi [[LOAD_PARAM_0_MEM_]] : i32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xi32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xi32>
// CHECK:         }
}

// -----

func.func private @test_floor(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Floor"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_floor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.floor [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

func.func private @test_ceil(%arg0 : tensor<?x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Ceil"(%arg0) : (tensor<?x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_ceil
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>) -> memref<?x10xf32> {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:             [[VAR_3_:%.+]] = math.ceil [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_3_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}
