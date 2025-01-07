// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

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
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
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
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = [[CST_0_1_]] to [[VAR_3_]]){
// CHECK-DAG:         [[VAR_5_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<i1>
// CHECK:             scf.if [[LOAD_PARAM_2_MEM_1_]] {
// CHECK:               "krnl.region"() ({
// CHECK-DAG:             [[VAR_7_:%.+]] = arith.index_cast [[VAR_5_1_]] : index to i64
// CHECK-DAG:             [[RES_2_:%.+]] = memref.alloc() : memref<i64>
// CHECK:                 krnl.store [[VAR_7_]], [[RES_2_]][] : memref<i64>
// CHECK-DAG:             [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:             [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK-DAG:             [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK-DAG:             [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<1xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_8_:%.+]] = builtin.unrealized_conversion_cast [[RES_3_]] : memref<1xi64> to tensor<1xi64>
// CHECK-DAG:             [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_0_2_]]{{.}} : memref<1xi64>
// CHECK-DAG:             [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<i64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_11_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[LOAD_RES_2_MEM_]] : i64
// CHECK-DAG:             [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK:                 krnl.store [[VAR_11_]], [[RES_3_]]{{.}}[[CST_0_3_]]{{.}} : memref<1xi64>
// CHECK-DAG:             [[VAR_12_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<i1> to memref<i1>
// CHECK-DAG:             [[VAR_13_:%.+]] = builtin.unrealized_conversion_cast [[VAR_8_]] : tensor<1xi64> to memref<1xi64>
// CHECK:                 [[LOAD_VAR_12_MEM_:%.+]] = krnl.load [[VAR_12_]][] : memref<i1>
// CHECK:                 krnl.store [[LOAD_VAR_12_MEM_]], [[RES_1_]][] : memref<i1>
// CHECK-DAG:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK-DAG:             [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:             [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK:                 krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 1){
// CHECK:                   [[VAR_16_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_VAR_13_MEM_:%.+]] = krnl.load [[VAR_13_]]{{.}}[[VAR_16_]]{{.}} : memref<1xi64>
// CHECK:                   krnl.store [[LOAD_VAR_13_MEM_]], [[RES_]]{{.}}[[VAR_16_]]{{.}} : memref<1xi64>
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
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_loop
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<i64>, [[PARAM_1_:%.+]]: memref<i1>, [[PARAM_2_:%.+]]: memref<?xf32>) -> memref<?x?xf32> {
// CHECK:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][] : memref<i64>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<?xmemref<?xf32>>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<i1>
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<i1>
// CHECK:           krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_1_]][] : memref<i1>
// CHECK:           [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]][] : memref<i64>
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_1_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = [[CST_0_]] to [[VAR_4_]]){
// CHECK-DAG:         [[VAR_8_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<i1>
// CHECK:             scf.if [[LOAD_RES_1_MEM_]] {
// CHECK:               "krnl.region"() ({
// CHECK-DAG:             [[VAR_10_:%.+]] = arith.index_cast [[VAR_8_]] : index to i64
// CHECK-DAG:             [[RES_2_:%.+]] = memref.alloc() : memref<i64>
// CHECK:                 krnl.store [[VAR_10_]], [[RES_2_]][] : memref<i64>
// CHECK-DAG:             [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:             [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[VAR_dim_7_:%.+]] = memref.dim [[PARAM_2_]], [[CST_0_1_]] : memref<?xf32>
// CHECK-DAG:             [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK:                 [[VAR_dim_9_:%.+]] = memref.dim [[PARAM_2_]], [[CST_0_2_]] : memref<?xf32>
// CHECK-DAG:             [[VAR_11_:%.+]] = affine.max [[MAP_0_]]([[VAR_dim_7_]], [[VAR_dim_9_]])
// CHECK-DAG:             [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:                 [[RES_3_:%.+]] = memref.alloc([[VAR_11_]]) {{.*}}: memref<?xf32>
// CHECK-DAG:             [[VAR_12_:%.+]] = builtin.unrealized_conversion_cast [[RES_3_]] : memref<?xf32> to tensor<?xf32>
// CHECK-DAG:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:             [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:             [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK:                 krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_7_]], [[VAR_dim_9_]], [[VAR_11_]])){
// CHECK:                   [[VAR_17_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:               [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_17_]]{{.}} : memref<?xf32>
// CHECK-DAG:               [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_17_]]{{.}} : memref<?xf32>
// CHECK:                   [[VAR_20_:%.+]] = arith.addf [[LOAD_PARAM_2_MEM_]], [[LOAD_PARAM_2_MEM_1_]] : f32
// CHECK:                   krnl.store [[VAR_20_]], [[RES_3_]]{{.}}[[VAR_17_]]{{.}} : memref<?xf32>
// CHECK:                 }
// CHECK-DAG:             [[VAR_14_:%.+]] = builtin.unrealized_conversion_cast [[PARAM_1_]] : memref<i1> to memref<i1>
// CHECK-DAG:             [[VAR_15_:%.+]] = builtin.unrealized_conversion_cast [[VAR_12_]] : tensor<?xf32> to memref<?xf32>
// CHECK:                 [[LOAD_VAR_14_MEM_:%.+]] = krnl.load [[VAR_14_]][] : memref<i1>
// CHECK:                 krnl.store [[LOAD_VAR_14_MEM_]], [[RES_1_]][] : memref<i1>
// CHECK:                 "krnl.seqstore"([[VAR_15_]], [[RES_]], [[VAR_8_]]) : (memref<?xf32>, memref<?xmemref<?xf32>>, index) -> ()
// CHECK:               }) : () -> ()
// CHECK:             }
// CHECK:           }
// CHECK:           [[CST_0_5_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_0_5_]]{{.}} : memref<?xmemref<?xf32>>
// CHECK-DAG:       [[CST_0_6_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_7_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[LOAD_RES_MEM_]], [[CST_0_7_]] : memref<?xf32>
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc([[VAR_1_]], [[VAR_dim_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = [[CST_0_]] to [[VAR_4_]]){
// CHECK:             [[VAR_8_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:             "krnl.region"() ({
// CHECK-DAG:           [[LOAD_RES_1_MEM_1_:%.+]] = "krnl.seqextract"([[RES_]], [[VAR_8_1_]]) {copy = 0 : ui1} : (memref<?xmemref<?xf32>>, index) -> memref<?xf32>
// CHECK-DAG:           [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK-DAG:           [[CST_0_8_:%.+]] = arith.constant 0 : index
// CHECK-DAG:           [[CST_0_9_:%.+]] = arith.constant 0 : index
// CHECK:               [[VAR_dim_7_1_:%.+]] = memref.dim [[LOAD_RES_1_MEM_1_]], [[CST_0_9_]] : memref<?xf32>
// CHECK:               krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_3_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_7_1_]])){
// CHECK:                 [[VAR_11_1_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:                 [[VAR_12_1_:%.+]] = krnl.load [[LOAD_RES_1_MEM_1_]]{{.}}[[VAR_11_1_]]{{.}} : memref<?xf32>
// CHECK:                 krnl.store [[VAR_12_1_]], [[RES_4_]]{{.}}[[VAR_8_1_]], [[VAR_11_1_]]{{.}} : memref<?x?xf32>
// CHECK:               }
// CHECK:             }) : () -> ()
// CHECK:           }
// CHECK:           return [[RES_4_]] : memref<?x?xf32>
}