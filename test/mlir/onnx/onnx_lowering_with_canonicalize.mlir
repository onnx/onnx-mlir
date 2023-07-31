// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// -----

// Focus on accumulated offset for the store op in each loop
func.func @test_concat_5(%arg0 : tensor<?x?x?xf32>, %arg1 : tensor<?x3x32xf32>, %arg2 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = -2 : si64} : (tensor<?x?x?xf32>, tensor<?x3x32xf32>, tensor<?x?x?xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// mlir2FileCheck.py
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 3)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK-DAG: [[MAP_3_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d2)>
// CHECK-DAG: [[MAP_4_:#.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-DAG: [[MAP_5_:#.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d2)>
// CHECK-DAG: [[MAP_6_:#.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5)>
// CHECK-DAG: [[MAP_7_:#.+]] = affine_map<(d0)[s0] -> (d0 + s0 + 3)>
// CHECK-LABEL:  func.func @test_concat_5
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>, [[PARAM_1_:%.+]]: memref<?x3x32xf32>, [[PARAM_2_:%.+]]: memref<?x?x?xf32>) -> memref<?x?x32xf32> {
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_2_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK:           [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_0_]], [[VAR_dim_1_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_0_]]) {{.*}}: memref<?x?x32xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_]], [[VAR_dim_]]_2), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 32){
// CHECK:             [[VAR_4_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_4_]]#2] : memref<?x?x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_4_]]#2] : memref<?x?x32xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_]], [[VAR_dim_]]_2), [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 32){
// CHECK:             [[VAR_4_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_4_]]([[VAR_4_1_]]#1){{.}}[[VAR_dim_3_]]{{.}}
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_4_1_]]#0, [[VAR_4_1_]]#1, [[VAR_4_1_]]#2] : memref<?x3x32xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_4_1_]]#0, [[LOAD_PARAM_0_MEM_1_]], [[VAR_4_1_]]#2] : memref<?x?x32xf32>
// CHECK:           }
// CHECK-DAG:       [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[PARAM_2_]], [[VAR_c1_]] : memref<?x?x?xf32>
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_6_:%.+]] = 0 to [[MAP_5_]]([[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_]], [[VAR_dim_]]_2, [[VAR_dim_]]_3), [[LOOP_2_]]#1 -> [[I_7_:%.+]] = 0 to [[MAP_6_]]([[VAR_dim_0_]], [[VAR_dim_1_]], [[VAR_dim_]], [[VAR_dim_]]_2, [[VAR_dim_]]_3, [[VAR_dim_]]_4), [[LOOP_2_]]#2 -> [[I_8_:%.+]] = 0 to 32){
// CHECK:             [[VAR_4_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_7_]]([[VAR_4_2_]]#1){{.}}[[VAR_dim_3_]]{{.}}
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_4_2_]]#0, [[VAR_4_2_]]#1, [[VAR_4_2_]]#2] : memref<?x?x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_1_]], [[RES_]]{{.}}[[VAR_4_2_]]#0, [[LOAD_PARAM_0_MEM_1_]], [[VAR_4_2_]]#2] : memref<?x?x32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x32xf32>
// CHECK:         }
}

// -----

// Please check the loop bounds for each input: should be same for dynamic
func.func @test_concat_4(%arg0 : tensor<?x1x?xf32>, %arg1 : tensor<?x3x32xf32>, %arg2 : tensor<?x5x?xf32>) -> tensor<*xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = -2 : si64} : (tensor<?x1x?xf32>, tensor<?x3x32xf32>, tensor<?x5x?xf32>)  -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
// mlir2FileCheck.py
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 + 4)>
// CHECK-LABEL:  func.func @test_concat_4
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x?xf32>, [[PARAM_1_:%.+]]: memref<?x3x32xf32>, [[PARAM_2_:%.+]]: memref<?x5x?xf32>) -> memref<?x9x32xf32> {
// CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x1x?xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x9x32xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 32){
// CHECK:             [[VAR_3_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2] : memref<?x1x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_3_]]#2] : memref<?x9x32xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 32){
// CHECK:             [[VAR_3_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_1_]]([[VAR_3_1_]]#1)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1, [[VAR_3_1_]]#2] : memref<?x3x32xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_]], [[RES_]]{{.}}[[VAR_3_1_]]#0, [[LOAD_PARAM_0_MEM_1_]], [[VAR_3_1_]]#2] : memref<?x9x32xf32>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_6_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_2_]]#1 -> [[I_7_:%.+]] = 0 to 5, [[LOOP_2_]]#2 -> [[I_8_:%.+]] = 0 to 32){
// CHECK:             [[VAR_3_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP_2_]]([[VAR_3_2_]]#1)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_3_2_]]#0, [[VAR_3_2_]]#1, [[VAR_3_2_]]#2] : memref<?x5x?xf32>
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_1_]], [[RES_]]{{.}}[[VAR_3_2_]]#0, [[LOAD_PARAM_0_MEM_1_]], [[VAR_3_2_]]#2] : memref<?x9x32xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x9x32xf32>
// CHECK:         }
}

// -----

func.func @test_sequence_insert(%arg0: tensor<?x4x5xf32>, %arg1:tensor<3x4x5xf32>) -> tensor<3xi64>  {
  %0 = onnx.Constant {value = dense<0> : tensor<i64>} : tensor<i64>
  %1 = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<*xf32>>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.SequenceInsert"(%1, %arg0, %0) : (!onnx.Seq<tensor<*xf32>>, tensor<?x4x5xf32>, tensor<i64>) -> !onnx.Seq<tensor<?x4x5xf32>>
  %6 = "onnx.SequenceInsert"(%3, %arg1, %2) : (!onnx.Seq<tensor<?x4x5xf32>>, tensor<3x4x5xf32>, none) -> !onnx.Seq<tensor<?x4x5xf32>>
  %4 = "onnx.SequenceAt"(%6, %0) : (!onnx.Seq<tensor<?x4x5xf32>>, tensor<i64>) -> tensor<?x4x5xf32>
  %5 = "onnx.Shape"(%4) {start = 0 : si64} : (tensor<?x4x5xf32>) -> tensor<3xi64>
  return %5 : tensor<3xi64>
// mlir2FileCheck.py
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 2)>
// CHECK-LABEL:  func.func @test_sequence_insert
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x5xf32>, [[PARAM_1_:%.+]]: memref<3x4x5xf32>) -> memref<3xi64> {
// CHECK-DAG:       [[VAR_c5_i64_:%.+]] = arith.constant 5 : i64
// CHECK-DAG:       [[VAR_c4_i64_:%.+]] = arith.constant 4 : i64
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "{{.+}}, shape = [], value = dense<0> : tensor<i64>} : () -> memref<i64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.seqalloc"([[VAR_c1_]]) : (index) -> memref<1xmemref<?x4x5xf32>>
// CHECK-DAG:       [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<i64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_]] : i64 to index
// CHECK:           "krnl.seqstore"([[PARAM_0_]], [[VAR_1_]], [[VAR_3_]]) : (memref<?x4x5xf32>, memref<1xmemref<?x4x5xf32>>, index) -> ()
// CHECK-DAG:       [[VAR_4_:%.+]] = "krnl.seqalloc"([[VAR_c2_]]) : (index) -> memref<2xmemref<?x4x5xf32>>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 1){
// CHECK:             [[VAR_14_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[VAR_1_]]4] : memref<1xmemref<?x4x5xf32>>
// CHECK:             "krnl.seqstore"([[LOAD_VAR_1_MEM_]], [[VAR_4_]], [[VAR_c1_]]) : (memref<?x4x5xf32>, memref<2xmemref<?x4x5xf32>>, index) -> ()
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 2 to 1){
// CHECK:             [[VAR_14_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_VAR_1_MEM_1_:%.+]] = krnl.load [[VAR_1_]]{{.}}[[VAR_1_]]4] : memref<1xmemref<?x4x5xf32>>
// CHECK-DAG:         [[VAR_16_:%.+]] = arith.addi [[VAR_14_1_]], [[VAR_c1_]] : index
// CHECK:             "krnl.seqstore"([[LOAD_VAR_1_MEM_1_]], [[VAR_4_]], [[VAR_16_]]) : (memref<?x4x5xf32>, memref<2xmemref<?x4x5xf32>>, index) -> ()
// CHECK:           }
// CHECK:           "krnl.seqstore"([[PARAM_1_]], [[VAR_4_]], [[VAR_c1_]]) : (memref<3x4x5xf32>, memref<2xmemref<?x4x5xf32>>, index) -> ()
// CHECK:           [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<i64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.cmpi slt, [[VAR_8_]], [[VAR_c0_]] : index
// CHECK-DAG:       [[VAR_10_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_8_]]{{.}}
// CHECK:           [[VAR_11_:%.+]] = arith.select [[VAR_9_]], [[VAR_10_]], [[VAR_8_]] : index
// CHECK-DAG:       [[VAR_12_:%.+]] = "krnl.seqextract"([[VAR_4_]], [[VAR_11_]]) {copy = 1 : ui1} : (memref<2xmemref<?x4x5xf32>>, index) -> memref<?x4x5xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3xi64>
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[VAR_12_]], [[VAR_c0_]] : memref<?x4x5xf32>
// CHECK:           [[VAR_13_:%.+]] = arith.index_cast [[VAR_dim_]] : index to i64
// CHECK:           krnl.store [[VAR_13_]], [[RES_]]{{.}}[[VAR_c0_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[VAR_c4_i64_]], [[RES_]]{{.}}[[VAR_c1_]]{{.}} : memref<3xi64>
// CHECK:           krnl.store [[VAR_c5_i64_]], [[RES_]]{{.}}[[VAR_c2_]]{{.}} : memref<3xi64>
// CHECK:           return [[RES_]] : memref<3xi64>
// CHECK:         }
}

// -----

// Check nested if lowering (function computes scalar Sign).
func.func @test_if_sign(%arg0: tensor<f32>) -> tensor<i32> {
  %zero = onnx.Constant {value = dense<0> : tensor<i32>} : tensor<i32>
  %plus = onnx.Constant {value = dense<1> : tensor<i32>} : tensor<i32>
  %minus = onnx.Constant {value = dense<-1> : tensor<i32>} : tensor<i32>
  %0 = onnx.Constant {value = dense<0.0> : tensor<f32>} : tensor<f32>
  %1 = "onnx.Less"(%arg0, %0) : (tensor<f32>, tensor<f32>) -> tensor<i1>
  %2 = "onnx.If"(%1) ({
    onnx.Yield %minus : tensor<i32>
  }, {
    %3 = "onnx.Greater"(%arg0, %0) : (tensor<f32>, tensor<f32>) -> tensor<i1>
    %4 = "onnx.If"(%3) ({
      onnx.Yield %plus : tensor<i32>
    }, {
      onnx.Yield %zero : tensor<i32>
    }) : (tensor<i1>) -> tensor<i32>
    onnx.Yield %4 : tensor<i32>
  }) : (tensor<i1>) -> tensor<i32>
  return %2 : tensor<i32>
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @test_if_sign
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<f32>) -> memref<i32> {
// CHECK-DAG:       [[CONSTANT_1_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<0> : tensor<i32>} : () -> memref<i32>
// CHECK-DAG:       [[CONSTANT_2_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<1> : tensor<i32>} : () -> memref<i32>
// CHECK-DAG:       [[CONSTANT_3_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<-1> : tensor<i32>} : () -> memref<i32>
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [], value = dense<0.000000e+00> : tensor<f32>} : () -> memref<f32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<i1>
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.cmpf olt, [[LOAD_PARAM_0_MEM_]], [[LOAD_VAR_0_MEM_]] : f32
// CHECK-DAG:       krnl.store [[VAR_3_]], [[RES_]][] : memref<i1>
// CHECK-DAG:       [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<i1>
// CHECK-DAG:       [[VAR_5_:%.+]] = scf.if [[LOAD_RES_MEM_]] -> (memref<i32>) {
// CHECK-DAG:         scf.yield [[CONSTANT_3_]] : memref<i32>
// CHECK-DAG:       } else {
// CHECK-DAG:         [[RES_1_:%.+]] = memref.alloc() : memref<i1>
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.cmpf ogt, [[LOAD_PARAM_0_MEM_1_]], [[LOAD_VAR_0_MEM_1_]] : f32
// CHECK-DAG:         krnl.store [[VAR_8_]], [[RES_1_]][] : memref<i1>
// CHECK-DAG:         [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<i1>
// CHECK:             [[VAR_10_:%.+]] = arith.select [[LOAD_RES_1_MEM_]], [[CONSTANT_2_]], [[CONSTANT_1_]] : memref<i32>
// CHECK:             scf.yield [[VAR_10_]] : memref<i32>
// CHECK:           }
// CHECK:           return [[VAR_5_]] : memref<i32>
// CHECK:         }
}

// -----

func.func private @test_squeezev11_unknown_dimensions(%arg0 : tensor<?x1x32x?x64xf32>) -> tensor<*xf32> {
  %0 = "onnx.SqueezeV11"(%arg0) { axes = [1,-2]} : (tensor<?x1x32x?x64xf32>) -> (tensor<*xf32>)
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_squeezev11_unknown_dimensions
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x32x?x64xf32>) -> memref<?x32x64xf32> {
// CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x1x32x?x64xf32>
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: {{.}}[[VAR_dim_]], 32, 64], strides: [2048, 64, 1] : memref<?x1x32x?x64xf32> to memref<?x32x64xf32>
// CHECK:           return [[VAR_reinterpret_cast_]] : memref<?x32x64xf32>
// CHECK:         }
}

// -----

func.func private @test_squeeze_unknown_dimensions(%arg0 : tensor<?x1x32x?x64xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[1, -2]> : tensor<2xi64>
  %1 = "onnx.Squeeze"(%arg0, %0) : (tensor<?x1x32x?x64xf32>, tensor<2xi64>) -> (tensor<*xf32>)
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_squeeze_unknown_dimensions
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x32x?x64xf32>) -> memref<?x32x64xf32> {
// CHECK:           [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?x1x32x?x64xf32>
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: {{.}}[[VAR_dim_]], 32, 64], strides: [2048, 64, 1] : memref<?x1x32x?x64xf32> to memref<?x32x64xf32>
// CHECK:           return [[VAR_reinterpret_cast_]] : memref<?x32x64xf32>
// CHECK:         }
}

// -----

// Slice where all the parameters are constant.
func.func @test_slice_constant_default_axes(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = "onnx.NoValue"() {value} : () -> none
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, none, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:  func @test_slice_constant_default_axes
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_constant_default_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %steps = "onnx.NoValue"() {value} : () -> none
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, none) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func @test_slice_constant_default_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x3xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[IV]]#1] : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x3xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, 3]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:  func @test_slice_all_constant
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant_negative(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = onnx.Constant dense<[0, -1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, -1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:  func @test_slice_all_constant_negative
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant_end_outofbound(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 0]> : tensor<2xi64>
  %ends = onnx.Constant dense<[5, 3]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 2]> : tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * 2)>
// CHECK-LABEL:  func @test_slice_all_constant_end_outofbound
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

func.func @test_slice_all_constant_negative_steps(%arg0 : tensor<2x4xf32>) -> tensor<*xf32> {
  %axes = onnx.Constant dense<[0, 1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 3]> : tensor<2xi64>
  %ends = onnx.Constant dense<[2, 0]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, -2]> : tensor<2xi64>
  %1 = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<2x4xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * -2 + 3)>
// CHECK-LABEL:  func @test_slice_all_constant_negative_steps
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x4xf32>) -> memref<1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.apply [[MAP_0_]]([[IV]]#0)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_6_]], [[VAR_7_]]{{.}} : memref<2x4xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1] : memref<1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2xf32>
// CHECK:         }
}

// -----

// Slice where the data is dyn sized along a non-sliced dim
func.func @dyntest_slice_constant_dynshape_not_spliced(%arg0 : tensor<?x4x5xf32>) -> tensor<*xf32> {
  // %data = onnx.Constant dense<[ [ [ 0, 1, 2, 3, 4 ], [ 10, 11, 12, 13, 14 ], [ 20, 21, 22, 23, 24 ], [ 30, 31, 32, 33, 34 ] ], [ [ 100, 101, 102, 103, 104 ], [ 110, 111, 112, 113, 114 ], [ 120, 121, 122, 123, 124 ], [ 130, 131, 132, 133, 134 ] ], [ [ 200, 201, 202, 203, 204 ], [ 210, 211, 212, 213, 214 ], [ 220, 221, 222, 223, 224 ], [ 230, 231, 232, 233, 234 ] ] ] > : tensor<3x4x5xi64>
  // slice * 1-3 1-4 with neg numbers
  %axes = onnx.Constant dense<[2, 1]> : tensor<2xi64>
  %starts = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %ends = onnx.Constant dense<[-1, -1]> : tensor<2xi64>
  %steps = onnx.Constant dense<[1, 1]> : tensor<2xi64>
  %res = "onnx.Slice"(%arg0, %starts, %ends, %axes, %steps) : (tensor<?x4x5xf32>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<*xf32>
  "func.return"(%res) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-LABEL:  func @dyntest_slice_constant_dynshape_not_spliced
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x4x5xf32>) -> memref<?x2x3xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[DIM_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x4x5xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[DIM_0_]]) {{.*}} : memref<?x2x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[DIM_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#1)
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.apply [[MAP_1_]]([[IV]]#2)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[IV]]#0, [[VAR_7_]], [[VAR_8_]]{{.}} : memref<?x4x5xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<?x2x3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x2x3xf32>
// CHECK:         }
}

// -----

// Check where all is dynamic except input size and axis. The code was verified
// using a procedure simioar to mlir-run and by manually adding code to print the
// output as a vector

func.func @compute_slice_all_dyn(%arg0 : tensor<2xi64>, %arg1 : tensor<2xi64>, %arg2 : tensor<2xi64>) -> tensor<3x?x?xi64> {
   %data = onnx.Constant dense<[ [ [ 0, 1, 2, 3, 4 ], [ 10, 11, 12, 13, 14 ], [ 20, 21, 22, 23, 24 ], [ 30, 31, 32, 33, 34 ] ], [ [ 100, 101, 102, 103, 104 ], [ 110, 111, 112, 113, 114 ], [ 120, 121, 122, 123, 124 ], [ 130, 131, 132, 133, 134 ] ], [ [ 200, 201, 202, 203, 204 ], [ 210, 211, 212, 213, 214 ], [ 220, 221, 222, 223, 224 ], [ 230, 231, 232, 233, 234 ] ] ] > : tensor<3x4x5xi64>

  // slice * 1-3 1-4 with neg numbers
  %axes = onnx.Constant dense<[2, 1]> : tensor<2xi64>
  %res = "onnx.Slice"(%data, %arg0, %arg1, %axes, %arg2) : (tensor<3x4x5xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>, tensor<2xi64>) -> tensor<3x?x?xi64>
  return %res : tensor<3x?x?xi64>

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 5)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 + 4)>
// CHECK-LABEL:  func @compute_slice_all_dyn
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2xi64>, [[PARAM_1_:%.+]]: memref<2xi64>, [[PARAM_2_:%.+]]: memref<2xi64>) -> memref<3x?x?xi64> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_minus_2147483648_:%.+]] = arith.constant -2147483648 : index
// CHECK-DAG:       [[CST_2147483647_:%.+]] = arith.constant 2147483647 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [3, 4, 5], value = dense<{{.}}{{.}}[0, 1, 2, 3, 4], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24], [30, 31, 32, 33, 34]{{.}}, {{.}}[100, 101, 102, 103, 104], [110, 111, 112, 113, 114], [120, 121, 122, 123, 124], [130, 131, 132, 133, 134]{{.}}, {{.}}[200, 201, 202, 203, 204], [210, 211, 212, 213, 214], [220, 221, 222, 223, 224], [230, 231, 232, 233, 234]{{.}}{{.}}> : tensor<3x4x5xi64>} : () -> memref<3x4x5xi64>
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{\[}}[[CST_0_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{\[}}[[CST_0_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{\[}}[[CST_0_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.index_cast [[LOAD_PARAM_2_MEM_]] : i64 to index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.cmpi slt, [[VAR_3_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_3_]]{{.}}
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_8_]], [[VAR_9_]], [[VAR_3_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_10_]], [[CST_0_]] : index
// CHECK:           [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_0_]], [[VAR_10_]] : index
// CHECK:           [[VAR_13_:%.+]] = arith.cmpi sgt, [[VAR_12_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[CST_4_]], [[VAR_12_]] : index
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.cmpi slt, [[VAR_10_]], [[CST_0_]] : index
// CHECK:           [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[CST_0_]], [[VAR_10_]] : index
// CHECK:           [[VAR_17_:%.+]] = arith.cmpi sgt, [[VAR_16_]], [[CST_5_]] : index
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.select [[VAR_17_]], [[CST_5_]], [[VAR_16_]] : index
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.select [[VAR_19_]], [[VAR_14_]], [[VAR_18_]] : index
// CHECK-DAG:       [[VAR_21_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_22_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_5_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_23_:%.+]] = arith.select [[VAR_21_]], [[VAR_22_]], [[VAR_5_]] : index
// CHECK-DAG:       [[VAR_24_:%.+]] = arith.cmpi sle, [[VAR_5_]], [[CST_minus_2147483648_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_25_:%.+]] = arith.select [[VAR_24_]], [[CST_minus_1_]], [[VAR_23_]] : index
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_2147483647_]] : index
// CHECK:           [[VAR_27_:%.+]] = arith.select [[VAR_26_]], [[CST_5_]], [[VAR_25_]] : index
// CHECK:           [[VAR_28_:%.+]] = arith.cmpi slt, [[VAR_27_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_29_:%.+]] = arith.select [[VAR_28_]], [[CST_minus_1_]], [[VAR_27_]] : index
// CHECK:           [[VAR_30_:%.+]] = arith.cmpi sgt, [[VAR_29_]], [[CST_5_]] : index
// CHECK-DAG:       [[VAR_31_:%.+]] = arith.select [[VAR_30_]], [[CST_5_]], [[VAR_29_]] : index
// CHECK-DAG:       [[VAR_32_:%.+]] = arith.cmpi slt, [[VAR_27_]], [[CST_0_]] : index
// CHECK:           [[VAR_33_:%.+]] = arith.select [[VAR_32_]], [[CST_0_]], [[VAR_27_]] : index
// CHECK:           [[VAR_34_:%.+]] = arith.cmpi sgt, [[VAR_33_]], [[CST_5_]] : index
// CHECK-DAG:       [[VAR_35_:%.+]] = arith.select [[VAR_34_]], [[CST_5_]], [[VAR_33_]] : index
// CHECK-DAG:       [[VAR_36_:%.+]] = arith.cmpi slt, [[VAR_7_]], [[CST_0_]] : index
// CHECK:           [[VAR_37_:%.+]] = arith.select [[VAR_36_]], [[VAR_31_]], [[VAR_35_]] : index
// CHECK:           [[VAR_38_:%.+]] = arith.subi [[VAR_37_]], [[VAR_20_]] : index
// CHECK:           [[VAR_39_:%.+]] = arith.ceildivsi [[VAR_38_]], [[VAR_7_]] : index
// CHECK:           [[VAR_40_:%.+]] = arith.cmpi slt, [[VAR_39_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_41_:%.+]] = arith.select [[VAR_40_]], [[CST_0_]], [[VAR_39_]] : index
// CHECK-DAG:       [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{\[}}[[CST_1_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_43_:%.+]] = arith.index_cast [[LOAD_PARAM_0_MEM_1_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{\[}}[[CST_1_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_45_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_1_]] : i64 to index
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{\[}}[[CST_1_]]{{\]}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_47_:%.+]] = arith.index_cast [[LOAD_PARAM_2_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_48_:%.+]] = arith.cmpi slt, [[VAR_43_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_49_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_43_]]{{.}}
// CHECK:           [[VAR_50_:%.+]] = arith.select [[VAR_48_]], [[VAR_49_]], [[VAR_43_]] : index
// CHECK:           [[VAR_51_:%.+]] = arith.cmpi slt, [[VAR_50_]], [[CST_0_]] : index
// CHECK:           [[VAR_52_:%.+]] = arith.select [[VAR_51_]], [[CST_0_]], [[VAR_50_]] : index
// CHECK:           [[VAR_53_:%.+]] = arith.cmpi sgt, [[VAR_52_]], [[CST_3_]] : index
// CHECK-DAG:       [[VAR_54_:%.+]] = arith.select [[VAR_53_]], [[CST_3_]], [[VAR_52_]] : index
// CHECK-DAG:       [[VAR_55_:%.+]] = arith.cmpi slt, [[VAR_50_]], [[CST_0_]] : index
// CHECK:           [[VAR_56_:%.+]] = arith.select [[VAR_55_]], [[CST_0_]], [[VAR_50_]] : index
// CHECK:           [[VAR_57_:%.+]] = arith.cmpi sgt, [[VAR_56_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_58_:%.+]] = arith.select [[VAR_57_]], [[CST_4_]], [[VAR_56_]] : index
// CHECK-DAG:       [[VAR_59_:%.+]] = arith.cmpi slt, [[VAR_47_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_60_:%.+]] = arith.select [[VAR_59_]], [[VAR_54_]], [[VAR_58_]] : index
// CHECK-DAG:       [[VAR_61_:%.+]] = arith.cmpi slt, [[VAR_45_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_62_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_45_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_63_:%.+]] = arith.select [[VAR_61_]], [[VAR_62_]], [[VAR_45_]] : index
// CHECK-DAG:       [[VAR_64_:%.+]] = arith.cmpi sle, [[VAR_45_]], [[CST_minus_2147483648_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_65_:%.+]] = arith.select [[VAR_64_]], [[CST_minus_1_]], [[VAR_63_]] : index
// CHECK-DAG:       [[VAR_66_:%.+]] = arith.cmpi sge, [[VAR_45_]], [[CST_2147483647_]] : index
// CHECK:           [[VAR_67_:%.+]] = arith.select [[VAR_66_]], [[CST_4_]], [[VAR_65_]] : index
// CHECK:           [[VAR_68_:%.+]] = arith.cmpi slt, [[VAR_67_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_69_:%.+]] = arith.select [[VAR_68_]], [[CST_minus_1_]], [[VAR_67_]] : index
// CHECK:           [[VAR_70_:%.+]] = arith.cmpi sgt, [[VAR_69_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_71_:%.+]] = arith.select [[VAR_70_]], [[CST_4_]], [[VAR_69_]] : index
// CHECK-DAG:       [[VAR_72_:%.+]] = arith.cmpi slt, [[VAR_67_]], [[CST_0_]] : index
// CHECK:           [[VAR_73_:%.+]] = arith.select [[VAR_72_]], [[CST_0_]], [[VAR_67_]] : index
// CHECK:           [[VAR_74_:%.+]] = arith.cmpi sgt, [[VAR_73_]], [[CST_4_]] : index
// CHECK-DAG:       [[VAR_75_:%.+]] = arith.select [[VAR_74_]], [[CST_4_]], [[VAR_73_]] : index
// CHECK-DAG:       [[VAR_76_:%.+]] = arith.cmpi slt, [[VAR_47_]], [[CST_0_]] : index
// CHECK:           [[VAR_77_:%.+]] = arith.select [[VAR_76_]], [[VAR_71_]], [[VAR_75_]] : index
// CHECK:           [[VAR_78_:%.+]] = arith.subi [[VAR_77_]], [[VAR_60_]] : index
// CHECK:           [[VAR_79_:%.+]] = arith.ceildivsi [[VAR_78_]], [[VAR_47_]] : index
// CHECK:           [[VAR_80_:%.+]] = arith.cmpi slt, [[VAR_79_]], [[CST_0_]] : index
// CHECK:           [[VAR_81_:%.+]] = arith.select [[VAR_80_]], [[CST_0_]], [[VAR_79_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_81_]], [[VAR_41_]]) {{.*}} : memref<3x?x?xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[VAR_81_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[VAR_41_]]){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[VAR_84_:%.+]] = arith.muli [[VAR_47_]], [[IV]]#1 : index
// CHECK-DAG:         [[VAR_85_:%.+]] = arith.addi [[VAR_84_]], [[VAR_60_]] : index
// CHECK-DAG:         [[VAR_86_:%.+]] = arith.muli [[VAR_7_]], [[IV]]#2 : index
// CHECK:             [[VAR_87_:%.+]] = arith.addi [[VAR_86_]], [[VAR_20_]] : index
// CHECK:             [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][[[IV]]#0, [[VAR_85_]], [[VAR_87_]]{{.}} : memref<3x4x5xi64>
// CHECK:             krnl.store [[LOAD_VAR_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x?x?xi64>
// CHECK:           }
// CHECK:           return
// CHECK:         }
}

// -----

// GEMM with everything constant
func.func @test_gemm(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<5x10xf32>, [[PARAM_1_:%.+]]: memref<5x10xf32>, [[PARAM_2_:%.+]]: memref<10xf32>) -> memref<10x10xf32> {
}

// -----

// Gemm with all dimensions dynamic
func.func @test_gemm_all_dyn(%arg0 : tensor<?x?xf32>, %arg1 : tensor<?x?xf32>, %arg2: tensor<?xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<?x?xf32>, tensor<?x?xf32>, tensor<?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm_all_dyn
}

// -----

// A[10, *] * B[*, 10] result in constant size output but dyn reduction.
func.func @test_gemm_k_dyn(%arg0 : tensor<?x10xf32>, %arg1 : tensor<?x10xf32>, %arg2: tensor<10xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<?x10xf32>, tensor<?x10xf32>, tensor<10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm_k_dyn
}

// -----

// Broadcast bias C is dym, so we don't know if its 1 -> broadcast or 10. Dyn test for that.
func.func @test_gemm_c_dyn(%arg0 : tensor<5x10xf32>, %arg1 : tensor<5x10xf32>, %arg2: tensor<?xf32>) -> tensor<*xf32> {
  %0 ="onnx.Gemm"(%arg0, %arg1, %arg2) {alpha = 1.0 : f32, beta = 5.0 : f32, transA = 1 : si64, transB = 0 : si64} : (tensor<5x10xf32>, tensor<5x10xf32>, tensor<?xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func @test_gemm_c_dyn
}

// -----

// Test tile with constant repeats
func.func @test_tile1(%arg0 : tensor<4x8xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[3, 2]> : tensor<2xi64>
  %1 = "onnx.Tile"(%arg0, %0) : (tensor<4x8xf32>, tensor<2xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-DAG: [[MAP0:#map.*]] = affine_map<(d0) -> (d0 mod 4)>
// CHECK-DAG: [[MAP1:#map.+]] = affine_map<(d0) -> (d0 mod 8)>
// CHECK-LABEL:  func @test_tile1
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<4x8xf32>) -> memref<12x16xf32> {
// CHECK-DAG:       [[RES:%.+]] = memref.alloc() {{.*}}: memref<12x16xf32>
// CHECK-DAG:       [[LOOP_0:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 12, [[LOOP_0]]#1 -> [[I_1:%.+]] = 0 to 16){
// CHECK-NEXT:        [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_3:%.+]] = affine.apply [[MAP0]]([[IV]]#0)
// CHECK-DAG:         [[VAR_4:%.+]] = affine.apply [[MAP1]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM:%.+]] = krnl.load [[PARAM_0]]{{.}}[[VAR_3]], [[VAR_4]]{{.}} : memref<4x8xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM]], [[RES]]{{.}}[[IV]]#0, [[IV]]#1{{.}} : memref<12x16xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<12x16xf32>
// CHECK:         }
}

// -----

// Test tile without arith.constant repeats
func.func @test_tile2(%arg0 : tensor<8xf32>, %arg1 : tensor<1xi64>) -> tensor<*xf32> {
  %1 = "onnx.Tile"(%arg0, %arg1) : (tensor<8xf32>, tensor<1xi64>) -> tensor<*xf32>
  return %1 : tensor<*xf32>

// CHECK-DAG: [[MAP0:#map.*]] = affine_map<()[s0] -> (s0 * 8)>
// CHECK-DAG: [[MAP1:#map.+]] = affine_map<(d0) -> (d0 mod 8)>
// CHECK-LABEL:  func @test_tile2
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<8xf32>, [[PARAM_1:%.+]]: memref<1xi64>) -> memref<?xf32> {
// CHECK-DAG:       [[CST_0:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PARAM_1_MEM:%.+]] = krnl.load [[PARAM_1]]{{\[}}[[CST_0]]{{\]}} : memref<1xi64>
// CHECK:           [[VAR_1:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM]] : i64 to index
// CHECK:           [[VAR_2:%.+]] = affine.apply [[MAP0]](){{.}}[[VAR_1]]{{.}}
// CHECK-DAG:       [[RES:%.+]] = memref.alloc([[VAR_2]]) {{.*}} : memref<?xf32>
// CHECK-DAG:       [[LOOP_0:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0]]) with ([[LOOP_0]] -> [[I_0:%.+]] = 0 to [[MAP0]](){{.}}[[VAR_1]]{{.}}){
// CHECK-NEXT:        [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_0]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_5:%.+]] = affine.apply [[MAP1]]([[IV]])
// CHECK:             [[LOAD_PARAM_0_MEM:%.+]] = krnl.load [[PARAM_0]]{{.}}[[VAR_5]]{{.}} : memref<8xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM]], [[RES]]{{.}}[[IV]]{{.}} : memref<?xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<?xf32>
// CHECK:         }
}

// -----

// Test gather along axis 0, first example in ONNX for Gather. Positive indices, so no select.
func.func @test_gather_axis0(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = onnx.Constant dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "func.return"(%0) : (tensor<2x2x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_axis0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2xf32>) -> memref<2x2x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2, 2], value = dense<{{.}}[0, 1], [1, 2]{{.}}> : tensor<2x2xi64>} : () -> memref<2x2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][[[IV]]#0, [[IV]]#1] : memref<2x2xi64>
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_1_MEM_]] : i64 to index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[VAR_4_]], [[IV]]#2] : memref<3x2xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]][[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<2x2x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x2xf32>
// CHECK:         }
}

// -----

// Test gather along axis 0, first example in ONNX for Gather. Positive indices, so no select.
func.func @test_gather_axis0neg(%arg0 : tensor<3x2xf32>) -> tensor<2x2x2xf32> {
  %indices = onnx.Constant dense<[[0, -1], [1, 2]]> : tensor<2x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2x2xf32>
  "func.return"(%0) : (tensor<2x2x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_axis0neg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2xf32>) -> memref<2x2x2xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2, 2], value = dense<{{.}}[0, -1], [1, 2]{{.}}> : tensor<2x2xi64>} : () -> memref<2x2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_:%.+]]#0, [[LOOP_0_:%.+]]#1, [[LOOP_0_:%.+]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][[[IV]]#0, [[IV]]#1] : memref<2x2xi64>
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_1_MEM_]] : i64 to index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.addi [[VAR_4_]], [[CST_3_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_5_]], [[VAR_6_]], [[VAR_4_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[VAR_7_]], [[IV]]#2] : memref<3x2xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<2x2x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x2x2xf32>
// CHECK:         }
}

// -----

// Test gather along axis 1, second example in ONNX for Gather.
func.func @test_gather_axis1(%arg0 : tensor<3x3xf32>) -> tensor<3x1x2xf32> {
  %indices = onnx.Constant dense<[[0, 2]]> : tensor<1x2xi64>
  %0 = "onnx.Gather"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x3xf32>, tensor<1x2xi64>) -> tensor<3x1x2xf32>
  "func.return"(%0) : (tensor<3x1x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_axis1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x3xf32>) -> memref<3x1x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x1x2xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [1, 2], value = dense<{{.}}[0, 2]{{.}}> : tensor<1x2xi64>} : () -> memref<1x2xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][[[IV]]#1, [[IV]]#2] : memref<1x2xi64>
// CHECK:             [[VAR_4_:%.+]] = arith.index_cast [[LOAD_VAR_1_MEM_]] : i64 to index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[IV]]#0, [[VAR_4_]]] : memref<3x3xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2] : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x1x2xf32>
// CHECK:         }
}

// -----

// COM: Test GatherElements along axis 0. Positive indices, so no select.
func.func @test_gather_elements_axis0(%arg0 : tensor<3x3xf32>) -> tensor<2x3xf32> {
  %indices = onnx.Constant dense<[[1, 2, 0], [2, 0, 0]]> : tensor<2x3xi64>
  %0 = "onnx.GatherElements"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x3xf32>, tensor<2x3xi64>) -> tensor<2x3xf32>
  "func.return"(%0) : (tensor<2x3xf32>) -> ()

// CHECK-LABEL:  func @test_gather_elements_axis0
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<3x3xf32>) -> memref<2x3xf32> {
// CHECK-DAG:       [[RES:%.+]] = memref.alloc() {{.*}}: memref<2x3xf32>
// CHECK-DAG:       [[INDICES:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2, 3], value = dense<{{.}}[1, 2, 0], [2, 0, 0]{{.}}> : tensor<2x3xi64>} : () -> memref<2x3xi64>
// CHECK-DAG:       [[LOOP_0:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 2, [[LOOP_0]]#1 -> [[I_1:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[INDEX:%.+]] = krnl.load [[INDICES]][[[IV]]#0, [[IV]]#1] : memref<2x3xi64>
// CHECK:             [[INDEX_CAST:%.+]] = arith.index_cast [[INDEX]] : i64 to index
// CHECK:             [[DATA_VAL:%.+]] = krnl.load [[PARAM_0]][[[INDEX_CAST]], [[IV]]#1] : memref<3x3xf32>
// CHECK:             krnl.store [[DATA_VAL]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<2x3xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<2x3xf32>
// CHECK:         }
}

// -----

// Test GatherElements along axis 0. Negative indices.
func.func @test_gather_elements_axis0neg(%arg0 : tensor<3x2xf32>) -> tensor<2x2xf32> {
  %indices = onnx.Constant dense<[[0, -1], [1, 2]]> : tensor<2x2xi64>
  %0 = "onnx.GatherElements"(%arg0, %indices) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2xf32>
  "func.return"(%0) : (tensor<2x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_elements_axis0neg
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2xf32>) -> memref<2x2xf32> {
// CHECK-DAG:       [[CST_0:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_3:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES:%.+]] = memref.alloc() {{.*}}: memref<2x2xf32>
// CHECK-DAG:       [[INDICES:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2, 2], value = dense<{{.}}[0, -1], [1, 2]{{.}}> : tensor<2x2xi64>} : () -> memref<2x2xi64>
// CHECK-DAG:       [[LOOP_0:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 2, [[LOOP_0]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0:%.+]]#0, [[LOOP_0:%.+]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[INDEX:%.+]] = krnl.load [[INDICES]][[[IV]]#0, [[IV]]#1] : memref<2x2xi64>
// CHECK:             [[INDEX_CAST:%.+]] = arith.index_cast [[INDEX]] : i64 to index
// CHECK-DAG:         [[CMP:%.+]] = arith.cmpi slt, [[INDEX_CAST]], [[CST_0]] : index
// CHECK-DAG:         [[VAR_1:%.+]] = arith.addi [[INDEX_CAST]], [[CST_3]] : index
// CHECK:             [[SEL:%.+]] = arith.select [[CMP]], [[VAR_1]], [[INDEX_CAST]] : index
// CHECK:             [[DATA_VAL:%.+]] = krnl.load [[PARAM_0_]][[[SEL]], [[IV]]#1] : memref<3x2xf32>
// CHECK:             krnl.store [[DATA_VAL]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<2x2xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<2x2xf32>
// CHECK:         }
}

// -----

// COM: Test GatherElements along axis 1. Positive indices, so no select.
func.func @test_gather_elements_axis1(%arg0 : tensor<3x2xf32>) -> tensor<2x2xf32> {
  %indices = onnx.Constant dense<[[0, 0], [1, 0]]> : tensor<2x2xi64>
  %0 = "onnx.GatherElements"(%arg0, %indices) {axis = 1 : si64} : (tensor<3x2xf32>, tensor<2x2xi64>) -> tensor<2x2xf32>
  "func.return"(%0) : (tensor<2x2xf32>) -> ()

// CHECK-LABEL:  func @test_gather_elements_axis1
// CHECK-SAME:   ([[PARAM_0:%.+]]: memref<3x2xf32>) -> memref<2x2xf32> {
// CHECK-DAG:       [[RES:%.+]] = memref.alloc() {{.*}}: memref<2x2xf32>
// CHECK-DAG:       [[INDICES:%.+]] = "krnl.global"() {name = {{.*}}, shape = [2, 2], value = dense<{{.}}[0, 0], [1, 0]{{.}}> : tensor<2x2xi64>} : () -> memref<2x2xi64>
// CHECK-DAG:       [[LOOP_0:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 2, [[LOOP_0]]#1 -> [[I_1:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[INDEX:%.+]] = krnl.load [[INDICES]][[[IV]]#0, [[IV]]#1] : memref<2x2xi64>
// CHECK:             [[INDEX_CAST:%.+]] = arith.index_cast [[INDEX]] : i64 to index
// CHECK:             [[DATA_VAL:%.+]] = krnl.load [[PARAM_0]][[[IV]]#0, [[INDEX_CAST]]] : memref<3x2xf32>
// CHECK:             krnl.store [[DATA_VAL]], [[RES]][[[IV]]#0, [[IV]]#1] : memref<2x2xf32>
// CHECK:           }
// CHECK:           return [[RES]] : memref<2x2xf32>
// CHECK:         }
}

// -----

// COM: test split with unknown dimensions and explicit split.
func.func @test_split_unknown_dimension(%arg0 : tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %split = onnx.Constant dense<[2, 30]> : tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) { axis = 1 : si64} : (tensor<?x?x64xf32>, tensor<2xi64>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

// CHECK:        [[MAP0:#.+]] = affine_map<(d0) -> (d0)>
// CHECK:        [[MAP1:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-LABEL:  func @test_split_unknown_dimension
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x64xf32>) -> (memref<?x2x64xf32>, memref<?x30x64xf32>) {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[DIM_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[DIM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[DIM_0_]]) {{.*}} : memref<?x2x64xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[DIM_1_]]) {{.*}} : memref<?x30x64xf32>
// CHECK-DAG:       [[LOOP_0:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) with ([[LOOP_0]]#0 -> %arg1 = 0 to [[MAP0]]([[DIM_0_]]), [[LOOP_0]]#1 -> %arg2 = 0 to 2, [[LOOP_0]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x2x64xf32>
// CHECK:           }
// CHECK:           [[LOOP_1:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) with ([[LOOP_1]]#0 -> %arg1 = 0 to [[MAP0]]([[DIM_1_]]), [[LOOP_1]]#1 -> %arg2 = 0 to 30, [[LOOP_1]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP1]]{{.}}[[IV]]#1{{.}}
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[LOAD_PARAM_0_MEM_1_]], [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_1_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x30x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]] : memref<?x2x64xf32>, memref<?x30x64xf32>
// CHECK:         }
}

// -----

// COM: test split with unknown dimensions and default split.
func.func @test_split_unknown_dimension_equal_split(%arg0 : tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = 1 : si64 } : (tensor<?x?x64xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

// CHECK:       [[MAP0:#.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CHECK:       [[MAP1:#.+]] = affine_map<(d0) -> (d0)>
// CHECK:       [[MAP2:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK:       [[MAP3:#.+]] = affine_map<(d0)[s0] -> (d0 + s0 ceildiv 2)>
// CHECK-LABEL: func @test_split_unknown_dimension_equal_split
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x64xf32>) -> (memref<?x?x64xf32>, memref<?x?x64xf32>) {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[DIM_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP0]](){{.}}[[DIM_0_]]{{.}}
// CHECK-DAG:       [[VAR_5_:%.+]] = affine.apply [[MAP0]](){{.}}[[DIM_0_]]{{.}}
// CHECK-DAG:       [[DIM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[DIM_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[DIM_1_]], [[VAR_3_]]) {{.*}} : memref<?x?x64xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[DIM_2_]], [[VAR_5_]]) {{.*}} : memref<?x?x64xf32>
// CHECK-DAG:       [[LOOP_0:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) with ([[LOOP_0]]#0 -> %arg1 = 0 to [[MAP1]]([[DIM_1_]]), [[LOOP_0]]#1 -> %arg2 = 0 to [[MAP2]]([[DIM_1_]], [[VAR_3_]]), [[LOOP_0]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:           }
// CHECK:           [[LOOP_1:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) with ([[LOOP_1]]#0 -> %arg1 = 0 to [[MAP1]]([[DIM_2_]]), [[LOOP_1]]#1 -> %arg2 = 0 to [[MAP2]]([[DIM_2_]], [[VAR_5_]]), [[LOOP_1]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP3]]([[IV]]#1){{.}}[[DIM_0_]]{{.}}
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[LOAD_PARAM_0_MEM_1_]], [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_1_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]] : memref<?x?x64xf32>, memref<?x?x64xf32>
// CHECK:         }
}

// -----

// COM: test split with unknown dimensions and explicit split.
func.func @test_splitv11_unknown_dimension(%arg0 : tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.SplitV11"(%arg0) { axis = 1 : si64, split = [2, 30]} : (tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

// CHECK:        [[MAP0:#.+]] = affine_map<(d0) -> (d0)>
// CHECK:        [[MAP1:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-LABEL:  func @test_splitv11_unknown_dimension
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x64xf32>) -> (memref<?x2x64xf32>, memref<?x30x64xf32>) {
// CHECK:           [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[DIM_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[DIM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[DIM_0_]]) {{.*}} : memref<?x2x64xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[DIM_1_]]) {{.*}} : memref<?x30x64xf32>
// CHECK-DAG:       [[LOOP_0:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) with ([[LOOP_0]]#0 -> %arg1 = 0 to [[MAP0]]([[DIM_0_]]), [[LOOP_0]]#1 -> %arg2 = 0 to 2, [[LOOP_0]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x2x64xf32>
// CHECK:           }
// CHECK:           [[LOOP_1:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) with ([[LOOP_1]]#0 -> %arg1 = 0 to [[MAP0]]([[DIM_1_]]), [[LOOP_1]]#1 -> %arg2 = 0 to 30, [[LOOP_1]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP1]]([[IV]]#1)
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[LOAD_PARAM_0_MEM_1_]], [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_1_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x30x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]] : memref<?x2x64xf32>, memref<?x30x64xf32>
// CHECK:         }
}

// -----

// COM: test splitv11 with unknown dimensions and default split.
func.func @test_splitv11_unknown_dimension_equal_split(%arg0 : tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.SplitV11"(%arg0) { axis = 1 : si64 } : (tensor<?x?x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

// CHECK:       [[MAP0:#.+]] = affine_map<()[s0] -> (s0 ceildiv 2)>
// CHECK:       [[MAP1:#.+]] = affine_map<(d0) -> (d0)>
// CHECK:       [[MAP2:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK:       [[MAP3:#.+]] = affine_map<(d0)[s0] -> (d0 + s0 ceildiv 2)>
// CHECK-LABEL: func @test_splitv11_unknown_dimension_equal_split
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x64xf32>) -> (memref<?x?x64xf32>, memref<?x?x64xf32>) {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK:           [[DIM_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP0]](){{.}}[[DIM_0_]]{{.}}
// CHECK-DAG:       [[VAR_5_:%.+]] = affine.apply [[MAP0]](){{.}}[[DIM_0_]]{{.}}
// CHECK-DAG:       [[DIM_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-DAG:       [[DIM_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x64xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[DIM_1_]], [[VAR_3_]]) {{.*}} : memref<?x?x64xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[DIM_2_]], [[VAR_5_]]) {{.*}} : memref<?x?x64xf32>
// CHECK-DAG:       [[LOOP_0:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) with ([[LOOP_0]]#0 -> %arg1 = 0 to [[MAP1]]([[DIM_1_]]), [[LOOP_0]]#1 -> %arg2 = 0 to [[MAP2]]([[DIM_1_]], [[VAR_3_]]), [[LOOP_0]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:           }
// CHECK:           [[LOOP_1:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) with ([[LOOP_1]]#0 -> %arg1 = 0 to [[MAP1]]([[DIM_2_]]), [[LOOP_1]]#1 -> %arg2 = 0 to [[MAP2]]([[DIM_2_]], [[VAR_5_]]), [[LOOP_1]]#2 -> %arg3 = 0 to 64){
// CHECK:             [[IV:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1]]#0, [[LOOP_1]]#1, [[LOOP_1]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = affine.apply [[MAP3]]([[IV]]#1){{.}}[[DIM_0_]]{{.}}
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[LOAD_PARAM_0_MEM_1_]], [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_2_]], [[RES_1_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<?x?x64xf32>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_1_]] : memref<?x?x64xf32>, memref<?x?x64xf32>
// CHECK:         }
}

// -----

/// Check computing the divisor in ReduceMean
/// when the input has unknown dimensions and is of i32.

func.func @test_reducemean_v13_i32_unknown_dims(%arg0 : tensor<3x?x2xi32>) -> tensor<*xi32> {
  %0 ="onnx.ReduceMeanV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x?x2xi32>)-> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_reducemean_v13_i32_unknown_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x?x2xi32>) -> memref<3x2xi32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_]], [[RES_]]{{.}}[[VAR_5_]]#0, [[VAR_5_]]#1] : memref<3x2xi32>
// CHECK:           }
// CHECK-DAG:       [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<3x?x2xi32>
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_1_]]#2 -> [[I_4_:%.+]] = 0 to 2){
// CHECK:             [[VAR_5_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_5_1_]]#0, [[VAR_5_1_]]#1, [[VAR_5_1_]]#2] : memref<3x?x2xi32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_5_1_]]#0, [[VAR_5_1_]]#2] : memref<3x2xi32>
// CHECK:             [[VAR_8_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : i32
// CHECK:             krnl.store [[VAR_8_]], [[RES_]]{{.}}[[VAR_5_1_]]#0, [[VAR_5_1_]]#2] : memref<3x2xi32>
// CHECK:           }
// CHECK:           [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<3x?x2xi32>
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[VAR_dim_0_]] : index to i64
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.trunci [[VAR_2_]] : i64 to i32
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_5_:%.+]] = 0 to 3, [[LOOP_2_]]#1 -> [[I_6_:%.+]] = 0 to 2){
// CHECK:             [[VAR_5_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_5_2_]]#0, [[VAR_5_2_]]#1] : memref<3x2xi32>
// CHECK:             [[LOAD_RES_MEM_2_:%.+]] = arith.divsi [[LOAD_RES_MEM_1_]], [[VAR_3_]] : i32
// CHECK:             krnl.store [[LOAD_RES_MEM_2_]], [[RES_]]{{.}}[[VAR_5_2_]]#0, [[VAR_5_2_]]#1] : memref<3x2xi32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xi32>
// CHECK:         }
}

// -----

/// Check computing the divisor in ReduceMean
/// when the input has unknown dimensions and is of f32.
func.func @test_reducemean_v13_f32_unknown_dims(%arg0 : tensor<3x?x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMeanV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x?x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
  // CHECK-LABEL: test_reducemean_v13_f32_unknown_dims
  // CHECK: [[ONE:%.+]] = arith.constant 1 : index
  // CHECK: krnl.iterate
  // CHECK: krnl.iterate
  // CHECK: [[DIM:%.+]] = memref.dim %arg0, [[ONE]] : memref<3x?x2xf32>
  // CHECK: [[UNKNOWN_DIM_i64:%.+]] = arith.index_cast [[DIM]] : index to i64
  // CHECK: [[DIVISOR:%.+]] = arith.uitofp [[UNKNOWN_DIM_i64]] : i64 to f32
  // CHECK: krnl.iterate
}

// -----

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
// CHECK:             [[VAR_13_:%.+]] = arith.cmpf ogt, [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.cmpi sgt, [[VAR_dim_2_]], [[CST_1_]] : index
// CHECK:             [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[VAR_4_]]#0, [[CST_0_]] : index
// CHECK:             [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_16_]], [[CST_0_]], [[VAR_4_]]#2] : memref<?x1x5xf32>
// CHECK:             [[VAR_18_:%.+]] = arith.cmpf ogt, [[VAR_14_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             [[VAR_19_:%.+]] = arith.select [[VAR_18_]], [[VAR_14_]], [[LOAD_PARAM_2_MEM_]] : f32
// CHECK:             krnl.store [[VAR_19_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_4_]]#2] : memref<?x4x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x4x5xf32>
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

/// Check ReduceMeanV13 with f32.

func.func private @test_reducemean_v13_f32(%arg0 : tensor<3x2x2xf32>) -> tensor<*xf32> {
  %0 ="onnx.ReduceMeanV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xf32>)-> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducemean_v13_f32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xf32>) -> memref<3x2xf32> {
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<3x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_4_:%.+]] = 0 to 2){
// CHECK:             [[VAR_3_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1, [[VAR_3_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_5_:%.+]] = 0 to 3, [[LOOP_2_]]#1 -> [[I_6_:%.+]] = 0 to 2){
// CHECK:             [[VAR_3_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_3_2_]]#0, [[VAR_3_2_]]#1] : memref<3x2xf32>
// CHECK:             [[LOAD_RES_MEM_2_:%.+]] = arith.divf [[LOAD_RES_MEM_1_]], [[CST_2_dot_000000_]] : f32
// CHECK:             krnl.store [[LOAD_RES_MEM_2_]], [[RES_]]{{.}}[[VAR_3_2_]]#0, [[VAR_3_2_]]#1] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

// -----

/// Check ReduceMeanV13 with i32.

func.func private @test_reducemean_v13_i32(%arg0 : tensor<3x2x2xi32>) -> tensor<*xi32> {
  %0 ="onnx.ReduceMeanV13"(%arg0) {axes=[1], keepdims = 0 : si64} : (tensor<3x2x2xi32>)-> tensor<*xi32>
  "func.return"(%0) : (tensor<*xi32>) -> ()

// mlir2FileCheck.py
// CHECK-LABEL:  func.func private @test_reducemean_v13_i32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x2x2xi32>) -> memref<3x2xi32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : i32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x2xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<3x2xi32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_4_:%.+]] = 0 to 2){
// CHECK:             [[VAR_3_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1, [[VAR_3_1_]]#2] : memref<3x2x2xi32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#2] : memref<3x2xi32>
// CHECK:             [[VAR_6_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : i32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#2] : memref<3x2xi32>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_5_:%.+]] = 0 to 3, [[LOOP_2_]]#1 -> [[I_6_:%.+]] = 0 to 2){
// CHECK:             [[VAR_3_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_3_2_]]#0, [[VAR_3_2_]]#1] : memref<3x2xi32>
// CHECK:             [[LOAD_RES_MEM_2_:%.+]] = arith.divsi [[LOAD_RES_MEM_1_]], [[CST_2_]] : i32
// CHECK:             krnl.store [[LOAD_RES_MEM_2_]], [[RES_]]{{.}}[[VAR_3_2_]]#0, [[VAR_3_2_]]#1] : memref<3x2xi32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xi32>
// CHECK:         }
}

// -----

func.func private @test_concat_1(%arg0 : tensor<5x5x1x32xf32>, %arg1 : tensor<5x5x3x32xf32>, %arg2 : tensor<5x5x5x32xf32>) -> tensor<5x5x9x32xf32> {
  %1 = "onnx.Concat"(%arg0, %arg1, %arg2) { axis = 2 : si64} : (tensor<5x5x1x32xf32>, tensor<5x5x3x32xf32>, tensor<5x5x5x32xf32>)  -> tensor<5x5x9x32xf32>
  "func.return"(%1) : (tensor<5x5x9x32xf32>) -> ()

  // CHECK-LABEL: test_concat_1
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<5x5x9x32xf32>
  // CHECK: [[DEF_LOOPS0:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS0]]#0, [[DEF_LOOPS0]]#1, [[DEF_LOOPS0]]#2, [[DEF_LOOPS0]]#3) with ([[DEF_LOOPS0]]#0 -> %arg3 = 0 to 5, [[DEF_LOOPS0]]#1 -> %arg4 = 0 to 5, [[DEF_LOOPS0]]#2 -> %arg5 = 0 to 1, [[DEF_LOOPS0]]#3 -> %arg6 = 0 to 32){
  // CHECK: [[IV:%.+]]:4 = krnl.get_induction_var_value([[DEF_LOOPS0]]#0, [[DEF_LOOPS0]]#1, [[DEF_LOOPS0]]#2, [[DEF_LOOPS0]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
  // CHECK: [[LOAD0:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3] :  memref<5x5x1x32xf32>
  // CHECK: krnl.store [[LOAD0]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3] : memref<5x5x9x32xf32>

  // CHECK: [[DEF_LOOPS1:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1, [[DEF_LOOPS1]]#2, [[DEF_LOOPS1]]#3) with ([[DEF_LOOPS1]]#0 -> %arg3 = 0 to 5, [[DEF_LOOPS1]]#1 -> %arg4 = 0 to 5, [[DEF_LOOPS1]]#2 -> %arg5 = 0 to 3, [[DEF_LOOPS1]]#3 -> %arg6 = 0 to 32){
  // CHECK: [[AFFINE_APPLY1:%.+]] = affine.apply #{{.*}}([[IV]]#2)
  // CHECK: [[LOAD1:%.+]] = krnl.load %arg1[[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3] :  memref<5x5x3x32xf32>
  // CHECK: krnl.store [[LOAD1]], [[RES]][[[IV]]#0, [[IV]]#1, [[AFFINE_APPLY1]], [[IV]]#3] : memref<5x5x9x32xf32>

  // CHECK: [[DEF_LOOPS2:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[DEF_LOOPS2]]#0, [[DEF_LOOPS2]]#1, [[DEF_LOOPS2]]#2, [[DEF_LOOPS2]]#3) with ([[DEF_LOOPS2]]#0 -> %arg3 = 0 to 5, [[DEF_LOOPS2]]#1 -> %arg4 = 0 to 5, [[DEF_LOOPS2]]#2 -> %arg5 = 0 to 5, [[DEF_LOOPS2]]#3 -> %arg6 = 0 to 32){
  // CHECK: [[AFFINE_APPLY2:%.+]] = affine.apply #{{.*}}([[IV]]#2)
  // CHECK: [[LOAD2:%.+]] = krnl.load %arg2[[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3] :  memref<5x5x5x32xf32>
  // CHECK: krnl.store [[LOAD2]], [[RES]][[[IV]]#0, [[IV]]#1, [[AFFINE_APPLY2]], [[IV]]#3] : memref<5x5x9x32xf32>

  // CHECK: return [[RES]] :  memref<5x5x9x32xf32>
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

// COM: 2D matmul.

func.func private @test_matmul1(%arg0 : tensor<16x16xf32>, %arg1 : tensor<16x16xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<16x16xf32>, tensor<16x16xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a '["A","B"]'
// CHECK-LABEL:  func.func private @test_matmul1
// CHECK-SAME:   ([[A_:%.+]]: memref<16x16xf32>, [[B_:%.+]]: memref<16x16xf32>) -> memref<16x16xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x16xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 16, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 16){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_0_]]#2) with (){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#2) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_1_]]#0, [[VAR_3_]]{{.}} : memref<16x16xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_3_]], [[VAR_1_]]#1] : memref<16x16xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_7_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_8_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_7_]] : f32
// CHECK:               krnl.store [[VAR_8_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<16x16xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<16x16xf32>
// CHECK:         }
}

// -----

// 2-D x N-D

func.func private @test_matmul2(%arg0 : tensor<10x5xf32>, %arg1 : tensor<2x3x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<10x5xf32>, tensor<2x3x5x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a '["A","B"]' -n'{"0":"RES"}'
// CHECK-LABEL:  func.func private @test_matmul2
// CHECK-SAME:   ([[A_:%.+]]: memref<10x5xf32>, [[B_:%.+]]: memref<2x3x5x10xf32>) -> memref<2x3x10x10xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x10x10xf32>
// CHECK-DAG:       [[RES_1_:%.+]]:5 = krnl.define_loops 5
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[RES_1_]]#0, [[RES_1_]]#1, [[RES_1_]]#2, [[RES_1_]]#3) with ([[RES_1_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[RES_1_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[RES_1_]]#2 -> [[I_2_:%.+]] = 0 to 10, [[RES_1_]]#3 -> [[I_3_:%.+]] = 0 to 10, [[RES_1_]]#4 -> [[I_4_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[RES_1_]]#0, [[RES_1_]]#1, [[RES_1_]]#2, [[RES_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_2_]][] : memref<f32>
// CHECK:             krnl.iterate([[RES_1_]]#4) with (){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[RES_1_]]#4) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_1_]]#2, [[VAR_3_]]{{.}} : memref<10x5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_3_]], [[VAR_1_]]#3] : memref<2x3x5x10xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK:               [[VAR_7_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_8_:%.+]] = arith.addf [[LOAD_RES_2_MEM_]], [[VAR_7_]] : f32
// CHECK:               krnl.store [[VAR_8_]], [[RES_2_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<2x3x10x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x10x10xf32>
// CHECK:         }
}

// -----

// N-D x N-D

func.func private @test_matmul3(%arg0 : tensor<2x3x10x5xf32>, %arg1 : tensor<2x3x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<2x3x10x5xf32>, tensor<2x3x5x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a '["A","B"]' -n'{"0":"RES"}'
// CHECK-LABEL:  func.func private @test_matmul3
// CHECK-SAME:   ([[A_:%.+]]: memref<2x3x10x5xf32>, [[B_:%.+]]: memref<2x3x5x10xf32>) -> memref<2x3x10x10xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x10x10xf32>
// CHECK-DAG:       [[RES_1_:%.+]]:5 = krnl.define_loops 5
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[RES_1_]]#0, [[RES_1_]]#1, [[RES_1_]]#2, [[RES_1_]]#3) with ([[RES_1_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[RES_1_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[RES_1_]]#2 -> [[I_2_:%.+]] = 0 to 10, [[RES_1_]]#3 -> [[I_3_:%.+]] = 0 to 10, [[RES_1_]]#4 -> [[I_4_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[RES_1_]]#0, [[RES_1_]]#1, [[RES_1_]]#2, [[RES_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_2_]][] : memref<f32>
// CHECK:             krnl.iterate([[RES_1_]]#4) with (){
// CHECK:               [[VAR_3_:%.+]] = krnl.get_induction_var_value([[RES_1_]]#4) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_3_]]{{.}} : memref<2x3x10x5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_3_]], [[VAR_1_]]#3] : memref<2x3x5x10xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK:               [[VAR_7_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_8_:%.+]] = arith.addf [[LOAD_RES_2_MEM_]], [[VAR_7_]] : f32
// CHECK:               krnl.store [[VAR_8_]], [[RES_2_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<2x3x10x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x10x10xf32>
// CHECK:         }
}

// -----

// 1-D x 2-D
func.func private @test_matmul4(%arg0 : tensor<5xf32>, %arg1 : tensor<5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<5x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["A", "B"]' -n'{"0": "RES"}'
// CHECK-LABEL:  func private @test_matmul4
// CHECK-SAME:   ([[A_:%.+]]: memref<5xf32>, [[B_:%.+]]: memref<5x10xf32>) -> memref<10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#0) : (!krnl.loop) -> index
// CHECK:             krnl.store [[VAR_cst_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_0_]]#1) with (){
// CHECK:               [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#1) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_5_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_5_]], [[VAR_3_]]{{.}} : memref<5x10xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_9_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_10_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_9_]] : f32
// CHECK:               krnl.store [[VAR_10_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_3_]]{{.}} : memref<10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10xf32>
// CHECK:         }
}

// -----

// 1-D x N-D
func.func private @test_matmul5(%arg0 : tensor<5xf32>, %arg1 : tensor<?x5x10xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<?x5x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["A", "B"]' -n'{"1": "RES"}'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func private @test_matmul5
// CHECK-SAME:   ([[A_:%.+]]: memref<5xf32>, [[B_:%.+]]: memref<?x5x10xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[B_]], [[VAR_c0_]] : memref<?x5x10xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[VAR_cst_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_0_]]#2) with (){
// CHECK:               [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#2) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_6_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_4_]]#0, [[VAR_6_]], [[VAR_4_]]#1] : memref<?x5x10xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_10_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_11_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_10_]] : f32
// CHECK:               krnl.store [[VAR_11_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

// N-D x 1-D
func.func private @test_matmul6(%arg0 : tensor<?x10x5xf32>, %arg1 : tensor<5xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<?x10x5xf32>, tensor<5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["A", "B"]' -n'{"1": "RES"}'
// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func private @test_matmul6
// CHECK-SAME:   ([[A_:%.+]]: memref<?x10x5xf32>, [[B_:%.+]]: memref<5xf32>) -> memref<?x10xf32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[A_]], [[VAR_c0_]] : memref<?x10x5xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_0_]]) {{.*}}: memref<?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 10, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[VAR_cst_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_0_]]#2) with (){
// CHECK:               [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]#2) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1, [[VAR_6_]]{{.}} : memref<?x10x5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_6_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_10_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_11_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_10_]] : f32
// CHECK:               krnl.store [[VAR_11_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x10xf32>
// CHECK:         }
}

// -----

// 1-D x 1-D results in scalar
func.func private @test_matmul7(%arg0 : tensor<5xf32>, %arg1 : tensor<5xf32>) -> tensor<*xf32> {
  %0 ="onnx.MatMul"(%arg0, %arg1) : (tensor<5xf32>, tensor<5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
// mlir2FileCheck.py -a'["A", "B"]' -n'{"1": "RES"}'
// CHECK-LABEL:  func private @test_matmul7
// CHECK-SAME:   ([[A_:%.+]]: memref<5xf32>, [[B_:%.+]]: memref<5xf32>) -> memref<f32> {
// CHECK-DAG:       [[VAR_cst_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloca() : memref<f32>
// CHECK:           krnl.iterate() with ([[RES_1_]] -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             krnl.get_induction_var_value() : () -> ()
// CHECK:             krnl.store [[VAR_cst_]], [[RES_2_]][] : memref<f32>
// CHECK:             krnl.iterate([[RES_1_]]) with (){
// CHECK:               [[VAR_5_:%.+]] = krnl.get_induction_var_value([[RES_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_A_MEM_:%.+]] = krnl.load [[A_]]{{.}}[[VAR_5_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_B_MEM_:%.+]] = krnl.load [[B_]]{{.}}[[VAR_5_]]{{.}} : memref<5xf32>
// CHECK-DAG:           [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK:               [[VAR_9_:%.+]] = arith.mulf [[LOAD_A_MEM_]], [[LOAD_B_MEM_]] : f32
// CHECK:               [[VAR_10_:%.+]] = arith.addf [[LOAD_RES_2_MEM_]], [[VAR_9_]] : f32
// CHECK:               krnl.store [[VAR_10_]], [[RES_2_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_2_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_2_MEM_1_]], [[RES_]][] : memref<f32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<f32>
// CHECK:         }
}

// -----

func.func private @test_pool_unknown_dimensions(%arg0 : tensor<1x3x?x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x?x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 - 1)>
// CHECK-DAG: [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG: [[MAP_2_:#.+]] = affine_map<(d0) -> (0, d0)>
// CHECK-DAG: [[MAP_3_:#.+]] = affine_map<(d0)[s0] -> (s0, d0 + 2)>
// CHECK-DAG: [[MAP_4_:#.+]] = affine_map<(d0) -> (32, d0 + 2)>
// CHECK-DAG: [[MAP_5_:#.+]] = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>
// CHECK-LABEL:  func private @test_pool_unknown_dimensions
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x?x32xf32>) -> memref<1x3x?x31xf32> {
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x3x?x32xf32>
// CHECK:           [[VAR_1_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_0_]]{{.}}
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<1x3x?x31xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_1_]]([[VAR_1_]]), [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 31){
// CHECK:             [[IV:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[VAR_4_]][] : memref<f32>
// CHECK-DAG:         [[VAR_5_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<1x3x?x32xf32>
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.max [[MAP_2_]]([[IV]]#2)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.min [[MAP_3_]]([[IV]]#2){{.}}[[VAR_5_]]{{.}}
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.max [[MAP_2_]]([[IV]]#3)
// CHECK-DAG:         [[VAR_9_:%.+]] = affine.min [[MAP_4_]]([[IV]]#3)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.subi [[VAR_7_]], [[VAR_6_]] : index
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.subi [[VAR_9_]], [[VAR_8_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to min [[MAP_5_]]([[IV]]#2){{.}}[[VAR_5_]], [[CST_2_]], [[CST_0_]], [[CST_1_]], [[CST_1_]]{{.}}, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to min [[MAP_5_]]([[IV]]#3){{.}}[[CST_32_]], [[CST_2_]], [[CST_0_]], [[CST_1_]], [[CST_1_]]{{.}}){
// CHECK-DAG:           [[VAR_19_:%.+]] = arith.addi [[I_4_]], [[VAR_6_]] : index
// CHECK-DAG:           [[VAR_20_:%.+]] = arith.addi [[I_5_]], [[VAR_8_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[IV]]#0, [[IV]]#1, [[VAR_19_]], [[VAR_20_]]{{.}} : memref<1x3x?x32xf32>
// CHECK-DAG:           [[LOAD_VAR_4_MEM_:%.+]] = krnl.load [[VAR_4_]][] : memref<f32>
// CHECK:               [[VAR_23_:%.+]] = arith.addf [[LOAD_VAR_4_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:               krnl.store [[VAR_23_]], [[VAR_4_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_VAR_4_MEM_1_:%.+]] = krnl.load [[VAR_4_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_VAR_4_MEM_1_]], [[VAR_2_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3{{.}} : memref<1x3x?x31xf32>
// CHECK-DAG:         [[LOAD_VAR_2_MEM_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3{{.}} : memref<1x3x?x31xf32>
// CHECK-DAG:         [[VAR_15_:%.+]] = arith.muli [[VAR_10_]], [[VAR_11_]] : index
// CHECK:             [[VAR_16_:%.+]] = arith.index_cast [[VAR_15_]] : index to i64
// CHECK:             [[VAR_17_:%.+]] = arith.sitofp [[VAR_16_]] : i64 to f32
// CHECK:             [[VAR_18_:%.+]] = arith.divf [[LOAD_VAR_2_MEM_]], [[VAR_17_]] : f32
// CHECK:             krnl.store [[VAR_18_]], [[VAR_2_]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3{{.}} : memref<1x3x?x31xf32>
// CHECK:           }
// CHECK:           return [[VAR_2_]] : memref<1x3x?x31xf32>
// CHECK:         }
}

// -----

func.func private @test_conv_unknown_dimensions(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<5x2x6x7xf32>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<?x?x?x?xf32>, tensor<5x2x6x7xf32>, tensor<5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()



// mlir2FileCheck.py -a'["image", "filter", "bias"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 - 5)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 - 6)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (d0 * 5 + d1)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1)[s0] -> (s0)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (s1)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0) -> (-d0, 0)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0)[s0] -> (-d0 + s0, 6)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0, d1)[s0] -> (-d1, 0)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (-d1 + s1, 7)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 2)>
// CHECK-DAG:   [[MAP_11_:#.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL:  func private @test_conv_unknown_dimensions
// CHECK-SAME:   ([[IMAGE_:%.+]]: memref<?x?x?x?xf32>, [[FILTER_:%.+]]: memref<5x2x6x7xf32>, [[BIAS_:%.+]]: memref<5xf32>) -> memref<?x5x?x?xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.dim [[IMAGE_]], [[CST_0_]] : memref<?x?x?x?xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[IMAGE_]], [[CST_2_]] : memref<?x?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]]{{.}}
// CHECK-DAG:       [[VAR_3_:%.+]] = memref.dim [[IMAGE_]], [[CST_3_]] : memref<?x?x?x?xf32>
// CHECK:           [[VAR_4_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_3_]]{{.}}
// CHECK-DAG:       [[VAR_5_:%.+]] = memref.alloc([[VAR_0_]], [[VAR_2_]], [[VAR_4_]]) {{.*}}: memref<?x5x?x?xf32>
// CHECK-DAG:       [[VAR_11_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_2_]]([[VAR_1_]], [[VAR_3_]], [[VAR_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_7_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.apply [[MAP_3_]]([[VAR_7_]]#1, [[VAR_7_]]#2)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to [[MAP_4_]]([[VAR_7_]]#1, [[VAR_7_]]#2){{.}}[[VAR_2_]]{{.}}, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to [[MAP_5_]]([[VAR_7_]]#1, [[VAR_7_]]#2){{.}}[[VAR_2_]], [[VAR_4_]]{{.}}){
// CHECK-DAG:           [[VAR_10_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               krnl.store [[CST_0_dot_000000_]], [[VAR_11_]][] : memref<f32>
// CHECK-DAG:           [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:           [[VAR_13_:%.+]] = memref.dim [[IMAGE_]], [[CST_2_]] : memref<?x?x?x?xf32>
// CHECK-DAG:           [[VAR_14_:%.+]] = memref.dim [[IMAGE_]], [[CST_3_]] : memref<?x?x?x?xf32>
// CHECK:               krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_6_:%.+]] = max [[MAP_6_]]([[VAR_10_]]#0) to min [[MAP_7_]]([[VAR_10_]]#0){{.}}[[VAR_13_]]{{.}}, [[LOOP_2_]]#2 -> [[I_7_:%.+]] = max [[MAP_8_]]([[VAR_10_]]#0, [[VAR_10_]]#1){{.}}[[VAR_13_]]{{.}} to min [[MAP_9_]]([[VAR_10_]]#0, [[VAR_10_]]#1){{.}}[[VAR_13_]], [[VAR_14_]]{{.}}){
// CHECK:                 [[VAR_18_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:             [[VAR_19_:%.+]] = affine.apply [[MAP_10_]]([[VAR_18_]]#0, [[VAR_7_]]#1)
// CHECK-DAG:             [[VAR_20_:%.+]] = affine.apply [[MAP_11_]]([[VAR_18_]]#1, [[VAR_10_]]#0)
// CHECK-DAG:             [[VAR_21_:%.+]] = affine.apply [[MAP_11_]]([[VAR_18_]]#2, [[VAR_10_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_IMAGE_MEM_:%.+]] = krnl.load [[IMAGE_]]{{.}}[[VAR_7_]]#0, [[VAR_19_]], [[VAR_20_]], [[VAR_21_]]{{.}} : memref<?x?x?x?xf32>
// CHECK-DAG:             [[LOAD_FILTER_MEM_:%.+]] = krnl.load [[FILTER_]]{{.}}[[VAR_8_]], [[VAR_18_]]#0, [[VAR_18_]]#1, [[VAR_18_]]#2] : memref<5x2x6x7xf32>
// CHECK-DAG:             [[LOAD_VAR_11_MEM_:%.+]] = krnl.load [[VAR_11_]][] : memref<f32>
// CHECK:                 [[VAR_25_:%.+]] = arith.mulf [[LOAD_IMAGE_MEM_]], [[LOAD_FILTER_MEM_]] : f32
// CHECK:                 [[VAR_26_:%.+]] = arith.addf [[LOAD_VAR_11_MEM_]], [[VAR_25_]] : f32
// CHECK:                 krnl.store [[VAR_26_]], [[VAR_11_]][] : memref<f32>
// CHECK:               }
// CHECK-DAG:           [[LOAD_VAR_11_MEM_1_:%.+]] = krnl.load [[VAR_11_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_BIAS_MEM_:%.+]] = krnl.load [[BIAS_]]{{.}}[[VAR_8_]]{{.}} : memref<5xf32>
// CHECK:               [[VAR_17_:%.+]] = arith.addf [[LOAD_VAR_11_MEM_1_]], [[LOAD_BIAS_MEM_]] : f32
// CHECK:               krnl.store [[VAR_17_]], [[VAR_5_]]{{.}}[[VAR_7_]]#0, [[VAR_8_]], [[VAR_10_]]#0, [[VAR_10_]]#1] : memref<?x5x?x?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_5_]] : memref<?x5x?x?xf32>
// CHECK:         }
}

// -----

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
func.func private @test_reshape(%arg0 : tensor<?x10xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<4xi64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func private @test_reshape
// CHECK:          ([[PARAM_0_:%.+]]: memref<?x10xf32>, [[PARAM_1_:%.+]]: memref<4xi64>) -> memref<?x?x?x?xf32> {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_0_]]{{.}}
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:       [[VAR_5_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.cmpi eq, [[VAR_3_]], [[CST_0_]] : index
// CHECK:           [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_6_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_1_]], [[VAR_6_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_1_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_10_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_1_]] : i64 to index
// CHECK:           [[VAR_11_:%.+]] = arith.cmpi eq, [[VAR_10_]], [[CST_0_]] : index
// CHECK:           [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_10_]], [[VAR_10_]] : index
// CHECK:           [[VAR_13_:%.+]] = arith.cmpi eq, [[VAR_12_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[CST_1_]], [[VAR_12_]] : index
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.muli [[VAR_8_]], [[VAR_14_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_2_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_2_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_17_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_2_]] : i64 to index
// CHECK:           [[VAR_18_:%.+]] = arith.cmpi eq, [[VAR_17_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_19_:%.+]] = arith.select [[VAR_18_]], [[CST_1_]], [[VAR_17_]] : index
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.muli [[VAR_15_]], [[VAR_19_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_3_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_3_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_22_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_3_]] : i64 to index
// CHECK:           [[VAR_23_:%.+]] = arith.cmpi eq, [[VAR_22_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_24_:%.+]] = arith.select [[VAR_23_]], [[CST_1_]], [[VAR_22_]] : index
// CHECK-DAG:       [[VAR_25_:%.+]] = arith.muli [[VAR_20_]], [[VAR_24_]] : index
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.cmpi eq, [[VAR_6_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_27_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_25_]] : index
// CHECK-DAG:       [[VAR_29_:%.+]] = arith.cmpi eq, [[VAR_12_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_30_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_25_]] : index
// CHECK-DAG:       [[VAR_28_:%.+]] = arith.select [[VAR_26_]], [[VAR_27_]], [[VAR_6_]] : index
// CHECK-DAG:       [[VAR_31_:%.+]] = arith.select [[VAR_29_]], [[VAR_30_]], [[VAR_12_]] : index
// CHECK-DAG:       [[VAR_32_:%.+]] = arith.cmpi eq, [[VAR_17_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_33_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_25_]] : index
// CHECK-DAG:       [[VAR_34_:%.+]] = arith.select [[VAR_32_]], [[VAR_33_]], [[VAR_17_]] : index
// CHECK-DAG:       [[VAR_36_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_25_]] : index
// CHECK-DAG:       [[VAR_35_:%.+]] = arith.cmpi eq, [[VAR_22_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_37_:%.+]] = arith.select [[VAR_35_]], [[VAR_36_]], [[VAR_22_]] : index
// CHECK:           [[VAR_38_:%.+]] = arith.muli [[VAR_37_]], [[VAR_34_]] : index
// CHECK:           [[VAR_39_:%.+]] = arith.muli [[VAR_38_]], [[VAR_31_]] : index
// CHECK:           [[VAR_40_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: {{.}}[[VAR_28_]], [[VAR_31_]], [[VAR_34_]], [[VAR_37_]]{{.}}, strides: {{.}}[[VAR_39_]], [[VAR_38_]], [[VAR_37_]], 1] : memref<?x10xf32> to memref<?x?x?x?xf32>
// CHECK:           return [[VAR_40_]] : memref<?x?x?x?xf32>
// CHECK:         }
}

// -----

func.func private @test_conv_no_bias_no_pad(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, none) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["image", "filter", "bias"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 * 5 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (-d0, 0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (-d0 + 32, 6)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (-d1, 0)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1) -> (-d1 + 64, 7)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL:  func private @test_conv_no_bias_no_pad
// CHECK-SAME:   ([[IMAGE_:%.+]]: memref<1x2x32x64xf32>, [[FILTER_:%.+]]: memref<5x2x6x7xf32>) -> memref<1x5x27x58xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<1x5x27x58xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[BIAS_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_0_]]([[VAR_2_]]#1, [[VAR_2_]]#2)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 27, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 58){
// CHECK-DAG:           [[VAR_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               krnl.store [[CST_0_dot_000000_]], [[VAR_6_]][] : memref<f32>
// CHECK:               [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:               krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = max [[MAP_1_]]([[VAR_5_]]#0) to min [[MAP_2_]]([[VAR_5_]]#0), [[LOOP_2_]]#2 -> [[I_6_:%.+]] = max [[MAP_3_]]([[VAR_5_]]#0, [[VAR_5_]]#1) to min [[MAP_4_]]([[VAR_5_]]#0, [[VAR_5_]]#1)){
// CHECK:                 [[VAR_9_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:             [[VAR_10_:%.+]] = affine.apply [[MAP_5_]]([[VAR_9_]]#0, [[VAR_2_]]#1)
// CHECK-DAG:             [[VAR_11_:%.+]] = affine.apply [[MAP_6_]]([[VAR_9_]]#1, [[VAR_5_]]#0)
// CHECK-DAG:             [[VAR_12_:%.+]] = affine.apply [[MAP_6_]]([[VAR_9_]]#2, [[VAR_5_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_IMAGE_MEM_:%.+]] = krnl.load [[IMAGE_]]{{.}}[[VAR_2_]]#0, [[VAR_10_]], [[VAR_11_]], [[VAR_12_]]{{.}} : memref<1x2x32x64xf32>
// CHECK-DAG:             [[LOAD_FILTER_MEM_:%.+]] = krnl.load [[FILTER_]]{{.}}[[VAR_3_]], [[VAR_9_]]#0, [[VAR_9_]]#1, [[VAR_9_]]#2] : memref<5x2x6x7xf32>
// CHECK-DAG:             [[LOAD_VAR_6_MEM_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK:                 [[VAR_16_:%.+]] = arith.mulf [[LOAD_IMAGE_MEM_]], [[LOAD_FILTER_MEM_]] : f32
// CHECK:                 [[VAR_17_:%.+]] = arith.addf [[LOAD_VAR_6_MEM_]], [[VAR_16_]] : f32
// CHECK:                 krnl.store [[VAR_17_]], [[VAR_6_]][] : memref<f32>
// CHECK:               }
// CHECK:               [[LOAD_VAR_6_MEM_1_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_6_MEM_1_]], [[VAR_0_]]{{.}}[[VAR_2_]]#0, [[VAR_3_]], [[VAR_5_]]#0, [[VAR_5_]]#1] : memref<1x5x27x58xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_0_]] : memref<1x5x27x58xf32>
// CHECK:         }
}

// -----

func.func private @test_conv_bias_no_pad(%arg0 : tensor<1x2x32x64xf32>, %arg1 : tensor<5x2x6x7xf32>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<1x2x32x64xf32>, tensor<5x2x6x7xf32>, tensor<5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["image", "filter", "bias"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 * 5 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (-d0, 0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (-d0 + 32, 6)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (-d1, 0)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1) -> (-d1 + 64, 7)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL:  func private @test_conv_bias_no_pad
// CHECK-SAME:   ([[IMAGE_:%.+]]: memref<1x2x32x64xf32>, [[FILTER_:%.+]]: memref<5x2x6x7xf32>, [[BIAS_:%.+]]: memref<5xf32>) -> memref<1x5x27x58xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<1x5x27x58xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_0_]]([[VAR_2_]]#1, [[VAR_2_]]#2)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 27, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 58){
// CHECK-DAG:           [[VAR_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               krnl.store [[CST_0_dot_000000_]], [[VAR_6_]][] : memref<f32>
// CHECK:               [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:               krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_6_:%.+]] = max [[MAP_1_]]([[VAR_5_]]#0) to min [[MAP_2_]]([[VAR_5_]]#0), [[LOOP_2_]]#2 -> [[I_7_:%.+]] = max [[MAP_3_]]([[VAR_5_]]#0, [[VAR_5_]]#1) to min [[MAP_4_]]([[VAR_5_]]#0, [[VAR_5_]]#1)){
// CHECK:                 [[VAR_11_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:             [[VAR_12_:%.+]] = affine.apply [[MAP_5_]]([[VAR_11_]]#0, [[VAR_2_]]#1)
// CHECK-DAG:             [[VAR_13_:%.+]] = affine.apply [[MAP_6_]]([[VAR_11_]]#1, [[VAR_5_]]#0)
// CHECK-DAG:             [[VAR_14_:%.+]] = affine.apply [[MAP_6_]]([[VAR_11_]]#2, [[VAR_5_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_IMAGE_MEM_:%.+]] = krnl.load [[IMAGE_]]{{.}}[[VAR_2_]]#0, [[VAR_12_]], [[VAR_13_]], [[VAR_14_]]{{.}} : memref<1x2x32x64xf32>
// CHECK-DAG:             [[LOAD_FILTER_MEM_:%.+]] = krnl.load [[FILTER_]]{{.}}[[VAR_3_]], [[VAR_11_]]#0, [[VAR_11_]]#1, [[VAR_11_]]#2] : memref<5x2x6x7xf32>
// CHECK-DAG:             [[LOAD_VAR_6_MEM_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK:                 [[VAR_18_:%.+]] = arith.mulf [[LOAD_IMAGE_MEM_]], [[LOAD_FILTER_MEM_]] : f32
// CHECK:                 [[VAR_19_:%.+]] = arith.addf [[LOAD_VAR_6_MEM_]], [[VAR_18_]] : f32
// CHECK:                 krnl.store [[VAR_19_]], [[VAR_6_]][] : memref<f32>
// CHECK:               }
// CHECK-DAG:           [[LOAD_VAR_6_MEM_1_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK-DAG:           [[LOAD_BIAS_MEM_:%.+]] = krnl.load [[BIAS_]]{{.}}[[VAR_3_]]{{.}} : memref<5xf32>
// CHECK:               [[VAR_10_:%.+]] = arith.addf [[LOAD_VAR_6_MEM_1_]], [[LOAD_BIAS_MEM_]] : f32
// CHECK:               krnl.store [[VAR_10_]], [[VAR_0_]]{{.}}[[VAR_2_]]#0, [[VAR_3_]], [[VAR_5_]]#0, [[VAR_5_]]#1] : memref<1x5x27x58xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_0_]] : memref<1x5x27x58xf32>
// CHECK:         }
}

// -----

func.func private @test_conv_no_bias_no_pad_w_group(%arg0 : tensor<1x9x32x64xf32>, %arg1 : tensor<6x3x6x7xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 3 : si64} : (tensor<1x9x32x64xf32>, tensor<6x3x6x7xf32>, none) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["image", "filter", "bias"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 * 2 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (-d0, 0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (-d0 + 32, 6)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (-d1, 0)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1) -> (-d1 + 64, 7)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 3)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + d1)>
// CHECK-LABEL:  func private @test_conv_no_bias_no_pad_w_group
// CHECK-SAME:   ([[IMAGE_:%.+]]: memref<1x9x32x64xf32>, [[FILTER_:%.+]]: memref<6x3x6x7xf32>) -> memref<1x6x27x58xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<1x6x27x58xf32>
// CHECK-DAG:         [[VAR_6_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[BIAS_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_0_]]([[VAR_2_]]#1, [[VAR_2_]]#2)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 27, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 58){
// CHECK-DAG:           [[VAR_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               krnl.store [[CST_0_dot_000000_]], [[VAR_6_]][] : memref<f32>
// CHECK:               [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:               krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = max [[MAP_1_]]([[VAR_5_]]#0) to min [[MAP_2_]]([[VAR_5_]]#0), [[LOOP_2_]]#2 -> [[I_6_:%.+]] = max [[MAP_3_]]([[VAR_5_]]#0, [[VAR_5_]]#1) to min [[MAP_4_]]([[VAR_5_]]#0, [[VAR_5_]]#1)){
// CHECK:                 [[VAR_9_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:             [[VAR_10_:%.+]] = affine.apply [[MAP_5_]]([[VAR_9_]]#0, [[VAR_2_]]#1)
// CHECK-DAG:             [[VAR_11_:%.+]] = affine.apply [[MAP_6_]]([[VAR_9_]]#1, [[VAR_5_]]#0)
// CHECK-DAG:             [[VAR_12_:%.+]] = affine.apply [[MAP_6_]]([[VAR_9_]]#2, [[VAR_5_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_IMAGE_MEM_:%.+]] = krnl.load [[IMAGE_]]{{.}}[[VAR_2_]]#0, [[VAR_10_]], [[VAR_11_]], [[VAR_12_]]{{.}} : memref<1x9x32x64xf32>
// CHECK-DAG:             [[LOAD_FILTER_MEM_:%.+]] = krnl.load [[FILTER_]]{{.}}[[VAR_3_]], [[VAR_9_]]#0, [[VAR_9_]]#1, [[VAR_9_]]#2] : memref<6x3x6x7xf32>
// CHECK-DAG:             [[LOAD_VAR_6_MEM_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK:                 [[VAR_16_:%.+]] = arith.mulf [[LOAD_IMAGE_MEM_]], [[LOAD_FILTER_MEM_]] : f32
// CHECK:                 [[VAR_17_:%.+]] = arith.addf [[LOAD_VAR_6_MEM_]], [[VAR_16_]] : f32
// CHECK:                 krnl.store [[VAR_17_]], [[VAR_6_]][] : memref<f32>
// CHECK:               }
// CHECK:               [[LOAD_VAR_6_MEM_1_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_6_MEM_1_]], [[VAR_0_]]{{.}}[[VAR_2_]]#0, [[VAR_3_]], [[VAR_5_]]#0, [[VAR_5_]]#1] : memref<1x6x27x58xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_0_]] : memref<1x6x27x58xf32>
// CHECK:         }
}

// -----

func.func private @test_conv_no_bias_no_pad_w_strides(%arg0 : tensor<1x9x32x64xf32>, %arg1 : tensor<5x9x6x7xf32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Conv"(%arg0, %arg1, %cst) {auto_pad = "NOTSET", group = 1 : si64, strides = [2, 2]} : (tensor<1x9x32x64xf32>, tensor<5x9x6x7xf32>, none) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["image", "filter", "bias"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1) -> (d0 * 5 + d1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 * -2, 0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (d0 * -2 + 32, 6)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1) -> (d1 * -2, 0)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1) -> (d1 * -2 + 64, 7)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 9)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (d0 + d1 * 2)>
// CHECK-LABEL:  func private @test_conv_no_bias_no_pad_w_strides
// CHECK-SAME:   ([[IMAGE_:%.+]]: memref<1x9x32x64xf32>, [[FILTER_:%.+]]: memref<5x9x6x7xf32>) -> memref<1x5x14x29xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<1x5x14x29xf32>
// CHECK-DAG:       [[VAR_6_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[BIAS_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_0_]]([[VAR_2_]]#1, [[VAR_2_]]#2)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 14, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 29){
// CHECK-DAG:           [[VAR_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               krnl.store [[CST_0_dot_000000_]], [[VAR_6_]][] : memref<f32>
// CHECK:               [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:               krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 9, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = max [[MAP_1_]]([[VAR_5_]]#0) to min [[MAP_2_]]([[VAR_5_]]#0), [[LOOP_2_]]#2 -> [[I_6_:%.+]] = max [[MAP_3_]]([[VAR_5_]]#0, [[VAR_5_]]#1) to min [[MAP_4_]]([[VAR_5_]]#0, [[VAR_5_]]#1)){
// CHECK:                 [[VAR_9_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:             [[VAR_10_:%.+]] = affine.apply [[MAP_5_]]([[VAR_9_]]#0, [[VAR_2_]]#1)
// CHECK-DAG:             [[VAR_11_:%.+]] = affine.apply [[MAP_6_]]([[VAR_9_]]#1, [[VAR_5_]]#0)
// CHECK-DAG:             [[VAR_12_:%.+]] = affine.apply [[MAP_6_]]([[VAR_9_]]#2, [[VAR_5_]]#1)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:             [[LOAD_IMAGE_MEM_:%.+]] = krnl.load [[IMAGE_]]{{.}}[[VAR_2_]]#0, [[VAR_10_]], [[VAR_11_]], [[VAR_12_]]{{.}} : memref<1x9x32x64xf32>
// CHECK-DAG:             [[LOAD_FILTER_MEM_:%.+]] = krnl.load [[FILTER_]]{{.}}[[VAR_3_]], [[VAR_9_]]#0, [[VAR_9_]]#1, [[VAR_9_]]#2] : memref<5x9x6x7xf32>
// CHECK-DAG:             [[LOAD_VAR_6_MEM_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK:                 [[VAR_16_:%.+]] = arith.mulf [[LOAD_IMAGE_MEM_]], [[LOAD_FILTER_MEM_]] : f32
// CHECK:                 [[VAR_17_:%.+]] = arith.addf [[LOAD_VAR_6_MEM_]], [[VAR_16_]] : f32
// CHECK:                 krnl.store [[VAR_17_]], [[VAR_6_]][] : memref<f32>
// CHECK:               }
// CHECK:               [[LOAD_VAR_6_MEM_1_:%.+]] = krnl.load [[VAR_6_]][] : memref<f32>
// CHECK:               krnl.store [[LOAD_VAR_6_MEM_1_]], [[VAR_0_]]{{.}}[[VAR_2_]]#0, [[VAR_3_]], [[VAR_5_]]#0, [[VAR_5_]]#1] : memref<1x5x14x29xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_0_]] : memref<1x5x14x29xf32>
// CHECK:         }
}

// -----

// COM: Lower Softmax opset 11.
func.func private @test_softmax_v11(%arg0 : tensor<10x20x30xf32>) -> tensor<*xf32> {
  %0 = "onnx.SoftmaxV11"(%arg0) {axis=1: si64} : (tensor<10x20x30xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK:         func private @test_softmax_v11([[arg0_:%.+]]: memref<10x20x30xf32>) -> memref<10x20x30xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.alloc() {{.*}}: memref<10x20x30xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<f32>
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 10){
// CHECK:             [[VAR_4_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[VAR_1_]][] : memref<f32>
// CHECK:             krnl.store [[CST_0_]], [[VAR_0_]][] : memref<f32>
// CHECK:             [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 20, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 30){
// CHECK-DAG:           [[VAR_10_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK:               [[LOAD_arg0_MEM_:%.+]] = krnl.load [[arg0_]]{{.}}[[VAR_4_]], [[VAR_10_]]#0, [[VAR_10_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.cmpf ogt, [[LOAD_VAR_0_MEM_]], [[LOAD_arg0_MEM_]] : f32
// CHECK:               [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[LOAD_VAR_0_MEM_]], [[LOAD_arg0_MEM_]] : f32
// CHECK:               krnl.store [[VAR_14_]], [[VAR_0_]][] : memref<f32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 20, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 30){
// CHECK-DAG:           [[VAR_10_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_VAR_0_MEM_2_:%.+]] = krnl.load [[VAR_1_]][] : memref<f32>
// CHECK:               [[LOAD_arg0_MEM_1_:%.+]] = krnl.load [[arg0_]]{{.}}[[VAR_4_]], [[VAR_10_1_]]#0, [[VAR_10_1_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_13_1_:%.+]] = arith.subf [[LOAD_arg0_MEM_1_]], [[LOAD_VAR_0_MEM_1_]] : f32
// CHECK:               [[VAR_14_1_:%.+]] = math.exp [[VAR_13_1_]] : f32
// CHECK:               [[VAR_15_:%.+]] = arith.addf [[LOAD_VAR_0_MEM_2_]], [[VAR_14_1_]] : f32
// CHECK:               krnl.store [[VAR_15_]], [[VAR_1_]][] : memref<f32>
// CHECK:               krnl.store [[VAR_14_1_]], [[VAR_2_]]{{.}}[[VAR_4_]], [[VAR_10_1_]]#0, [[VAR_10_1_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][] : memref<f32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 20, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 30){
// CHECK:               [[VAR_10_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_VAR_0_MEM_2_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[VAR_4_]], [[VAR_10_2_]]#0, [[VAR_10_2_]]#1] : memref<10x20x30xf32>
// CHECK:               [[LOAD_arg0_MEM_1_:%.+]] = arith.divf [[LOAD_VAR_0_MEM_2_]], [[LOAD_VAR_1_MEM_]] : f32
// CHECK:               krnl.store [[LOAD_arg0_MEM_1_]], [[VAR_2_]]{{.}}[[VAR_4_]], [[VAR_10_2_]]#0, [[VAR_10_2_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_2_]] : memref<10x20x30xf32>
// CHECK:         }
}

// -----

// COM: Lower Softmax opset 13.

func.func private @test_softmax_v13(%arg0 : tensor<10x20x30xf32>) -> tensor<*xf32> {
  %0 = "onnx.Softmax"(%arg0) {axis=1: si64} : (tensor<10x20x30xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK:         func private @test_softmax_v13([[arg0_:%.+]]: memref<10x20x30xf32>) -> memref<10x20x30xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[VAR_2_:%.+]] = memref.alloc() {{.*}}: memref<10x20x30xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.alloc() {{.*}}: memref<f32>
// CHECK-DAG:       [[VAR_0_:%.+]] = memref.alloc() {{.*}}: memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 30){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[VAR_1_]][] : memref<f32>
// CHECK:             krnl.store [[CST_0_]], [[VAR_0_]][] : memref<f32>
// CHECK:             [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 20){
// CHECK-DAG:           [[VAR_10_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK:               [[LOAD_arg0_MEM_:%.+]] = krnl.load [[arg0_]]{{.}}[[VAR_4_]]#0, [[VAR_10_]], [[VAR_4_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.cmpf ogt, [[LOAD_VAR_0_MEM_]], [[LOAD_arg0_MEM_]] : f32
// CHECK:               [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[LOAD_VAR_0_MEM_]], [[LOAD_arg0_MEM_]] : f32
// CHECK:               krnl.store [[VAR_14_]], [[VAR_0_]][] : memref<f32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<f32>
// CHECK-DAG:         [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_3_:%.+]] = 0 to 20){
// CHECK-DAG:           [[VAR_10_1_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:           [[LOAD_VAR_0_MEM_2_:%.+]] = krnl.load [[VAR_1_]][] : memref<f32>
// CHECK:               [[LOAD_arg0_MEM_1_:%.+]] = krnl.load [[arg0_]]{{.}}[[VAR_4_]]#0, [[VAR_10_1_]], [[VAR_4_]]#1] : memref<10x20x30xf32>
// CHECK:               [[VAR_13_1_:%.+]] = arith.subf [[LOAD_arg0_MEM_1_]], [[LOAD_VAR_0_MEM_1_]] : f32
// CHECK:               [[VAR_14_1_:%.+]] = math.exp [[VAR_13_1_]] : f32
// CHECK:               [[VAR_15_:%.+]] = arith.addf [[LOAD_VAR_0_MEM_2_]], [[VAR_14_1_]] : f32
// CHECK:               krnl.store [[VAR_15_]], [[VAR_1_]][] : memref<f32>
// CHECK:               krnl.store [[VAR_14_1_]], [[VAR_2_]]{{.}}[[VAR_4_]]#0, [[VAR_10_1_]], [[VAR_4_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK-DAG:         [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][] : memref<f32>
// CHECK-DAG:         [[LOOP_3_:%.+]] = krnl.define_loops 1
// CHECK:             krnl.iterate([[LOOP_3_]]) with ([[LOOP_3_]] -> [[I_4_:%.+]] = 0 to 20){
// CHECK:               [[VAR_10_2_:%.+]] = krnl.get_induction_var_value([[LOOP_3_]]) : (!krnl.loop) -> index
// CHECK:               [[LOAD_VAR_0_MEM_2_:%.+]] = krnl.load [[VAR_2_]]{{.}}[[VAR_4_]]#0, [[VAR_10_2_]], [[VAR_4_]]#1] : memref<10x20x30xf32>
// CHECK:               [[LOAD_arg0_MEM_1_:%.+]] = arith.divf [[LOAD_VAR_0_MEM_2_]], [[LOAD_VAR_1_MEM_]] : f32
// CHECK:               krnl.store [[LOAD_arg0_MEM_1_]], [[VAR_2_]]{{.}}[[VAR_4_]]#0, [[VAR_10_2_]], [[VAR_4_]]#1] : memref<10x20x30xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[VAR_2_]] : memref<10x20x30xf32>
// CHECK:         }
}

// -----

func.func @instance_norm(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<3xf32>, %arg2: tensor<3xf32>) -> tensor<2x3x4x5xf32> attributes {input_names = ["x", "s", "bias"], output_names = ["y"]} {
  %0 = "onnx.InstanceNormalization"(%arg0, %arg1, %arg2) {epsilon = 0.00999999977 : f32} : (tensor<2x3x4x5xf32>, tensor<3xf32>, tensor<3xf32>) -> tensor<2x3x4x5xf32>
  return %0 : tensor<2x3x4x5xf32>

// mlir2FileCheck.py -a'["input", "scale", "bias"]'
// CHECK-LABEL:  func @instance_norm
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3x4x5xf32>, [[SCALE_:%.+]]: memref<3xf32>, [[BIAS_:%.+]]: memref<3xf32>) -> memref<2x3x4x5xf32> attributes {input_names = ["x", "s", "bias"], output_names = ["y"]} {
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 2.000000e+01 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_00999999977_:%.+]] = arith.constant 0.00999999977 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4x5xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:         [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK:             [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 5){
// CHECK-DAG:           [[VAR_17_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_17_]]#0, [[VAR_17_]]#1] : memref<2x3x4x5xf32>
// CHECK:               [[VAR_20_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_INPUT_MEM_]] : f32
// CHECK:               krnl.store [[VAR_20_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             [[VAR_8_:%.+]] = arith.divf [[LOAD_RES_1_MEM_1_]], [[CST_20_]] : f32
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]][] : memref<f32>
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 4, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 5){
// CHECK-DAG:           [[VAR_17_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[LOAD_INPUT_MEM_1_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_17_1_]]#0, [[VAR_17_1_]]#1] : memref<2x3x4x5xf32>
// CHECK:               [[VAR_20_1_:%.+]] = arith.subf [[LOAD_INPUT_MEM_1_]], [[VAR_8_]] : f32
// CHECK:               [[VAR_21_:%.+]] = arith.mulf [[VAR_20_1_]], [[VAR_20_1_]] : f32
// CHECK:               [[VAR_22_:%.+]] = arith.addf [[LOAD_RES_1_MEM_2_]], [[VAR_21_]] : f32
// CHECK:               krnl.store [[VAR_22_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_3_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             [[VAR_10_:%.+]] = arith.divf [[LOAD_RES_1_MEM_3_]], [[CST_20_]] : f32
// CHECK:             [[VAR_11_:%.+]] = arith.addf [[VAR_10_]], [[CST_0_dot_00999999977_]] : f32
// CHECK-DAG:         [[VAR_12_:%.+]] = math.sqrt [[VAR_11_]] : f32
// CHECK-DAG:         [[LOAD_SCALE_MEM_:%.+]] = krnl.load [[SCALE_]]{{.}}[[VAR_3_]]#1] : memref<3xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_14_:%.+]] = arith.divf [[LOAD_SCALE_MEM_]], [[VAR_12_]] : f32
// CHECK-DAG:         [[LOAD_BIAS_MEM_:%.+]] = krnl.load [[BIAS_]]{{.}}[[VAR_3_]]#1] : memref<3xf32>
// CHECK-DAG:         [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_6_:%.+]] = 0 to 4, [[LOOP_3_]]#1 -> [[I_7_:%.+]] = 0 to 5){
// CHECK:               [[VAR_17_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_INPUT_MEM_2_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_17_2_]]#0, [[VAR_17_2_]]#1] : memref<2x3x4x5xf32>
// CHECK:               [[LOAD_INPUT_MEM_1_:%.+]] = arith.subf [[LOAD_INPUT_MEM_2_]], [[VAR_8_]] : f32
// CHECK:               [[VAR_20_2_:%.+]] = arith.mulf [[VAR_14_]], [[LOAD_INPUT_MEM_1_]] : f32
// CHECK:               [[VAR_21_1_:%.+]] = arith.addf [[VAR_20_2_]], [[LOAD_BIAS_MEM_]] : f32
// CHECK:               krnl.store [[VAR_21_1_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1, [[VAR_17_2_]]#0, [[VAR_17_2_]]#1] : memref<2x3x4x5xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3x4x5xf32>
// CHECK:         }
}

// -----

func.func @test_nonzero(%arg0: tensor<2x2xi1>) -> tensor<*xi64> attributes {input_names = ["condition"], output_names = ["result"]} {
    %0 = "onnx.NonZero"(%arg0) : (tensor<2x2xi1>) -> tensor<*xi64>
    return %0 : tensor<*xi64>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_nonzero
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x2xi1>) -> memref<2x?xi64> attributes {input_names = ["condition"], output_names = ["result"]} {
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
// CHECK-LABEL:  func @round
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<15xf32>) -> memref<15xf32> {
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<15xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 15){
// CHECK:             [[IV:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]][[[IV]]] : memref<15xf32>
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
// CHECK:             krnl.store [[VAR_16_]], [[RES_]][[[IV]]] : memref<15xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<15xf32>
// CHECK:         }
}

// -----

func.func @pad_constant_mode(%arg0: tensor<1x3x4x5xf32>, %arg1: tensor<8xi64>, %arg2: tensor<f32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %arg1, %arg2, %cst) {mode = "constant"} : (tensor<1x3x4x5xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<*xf32>
  return %0 : tensor<*xf32>

//  use arg names: ['data', 'pad', 'constant_value']
// mlir2FileCheck.py -a'["data","pad","constant_value"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 3)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 4)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0)[s0] -> (d0 + s0)>
// CHECK-LABEL:  func @pad_constant_mode
// CHECK-SAME:   ([[DATA_:%.+]]: memref<1x3x4x5xf32>, [[PAD_:%.+]]: memref<8xi64>, [[CONSTANT_VALUE_:%.+]]: memref<f32>) -> memref<?x?x?x?xf32> {
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PAD_MEM_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_0_]]{{.}} : memref<8xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_1_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_4_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]], [[VAR_3_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_2_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_1_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_2_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_3_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_5_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_3_]] : i64 to index
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_6_]], [[VAR_8_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_4_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_2_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_4_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_5_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_6_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_13_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_5_]] : i64 to index
// CHECK-DAG:       [[VAR_14_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_11_]], [[VAR_13_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_6_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_3_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_6_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_7_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_7_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_18_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_7_]] : i64 to index
// CHECK:           [[VAR_19_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_16_]], [[VAR_18_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_4_]], [[VAR_9_]], [[VAR_14_]], [[VAR_19_]]) {{.*}}: memref<?x?x?x?xf32>
// CHECK-DAG:       [[LOAD_CONSTANT_VALUE_MEM_:%.+]] = krnl.load [[CONSTANT_VALUE_]][] : memref<f32>
// CHECK:           krnl.memset [[RES_]], [[LOAD_CONSTANT_VALUE_MEM_]] : memref<?x?x?x?xf32>
// CHECK:           [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 5){
// CHECK:             [[VAR_23_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_24_:%.+]] = affine.apply [[MAP_4_]]([[VAR_23_]]#0){{.}}[[VAR_1_]]{{.}}
// CHECK-DAG:         [[VAR_25_:%.+]] = affine.apply [[MAP_4_]]([[VAR_23_]]#1){{.}}[[VAR_6_]]{{.}}
// CHECK-DAG:         [[VAR_26_:%.+]] = affine.apply [[MAP_4_]]([[VAR_23_]]#2){{.}}[[VAR_11_]]{{.}}
// CHECK-DAG:         [[VAR_27_:%.+]] = affine.apply [[MAP_4_]]([[VAR_23_]]#3){{.}}[[VAR_16_]]{{.}}
// CHECK-DAG:         [[LOAD_DATA_MEM_:%.+]] = krnl.load [[DATA_]]{{.}}[[VAR_23_]]#0, [[VAR_23_]]#1, [[VAR_23_]]#2, [[VAR_23_]]#3] : memref<1x3x4x5xf32>
// CHECK:             krnl.store [[LOAD_DATA_MEM_]], [[RES_]]{{.}}[[VAR_24_]], [[VAR_25_]], [[VAR_26_]], [[VAR_27_]]{{.}} : memref<?x?x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?x?xf32>
// CHECK:         }
}

// -----

func.func @pad_edge_mode(%arg0: tensor<1x3x4x5xf32>, %arg1: tensor<8xi64>, %arg2: tensor<f32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %arg1, %arg2, %cst) {mode = "edge"} : (tensor<1x3x4x5xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a'["data","pad","constant_value"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 3)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 4)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d0)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d1)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d2)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d3)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0)[s0] -> (d0 - s0)>
// CHECK-LABEL:  func @pad_edge_mode
// CHECK-SAME:   ([[DATA_:%.+]]: memref<1x3x4x5xf32>, [[PAD_:%.+]]: memref<8xi64>, [[CONSTANT_VALUE_:%.+]]: memref<f32>) -> memref<?x?x?x?xf32> {
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PAD_MEM_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_0_]]{{.}} : memref<8xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_1_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_4_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]], [[VAR_3_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_2_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_1_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_2_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_3_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_5_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_3_]] : i64 to index
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_6_]], [[VAR_8_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_4_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_2_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_4_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_5_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_6_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_13_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_5_]] : i64 to index
// CHECK-DAG:       [[VAR_14_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_11_]], [[VAR_13_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_6_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_3_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_6_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_7_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_7_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_18_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_7_]] : i64 to index
// CHECK:           [[VAR_19_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_16_]], [[VAR_18_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_4_]], [[VAR_9_]], [[VAR_14_]], [[VAR_19_]]) {{.*}}: memref<?x?x?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_4_]]([[VAR_4_]]){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_5_]]([[VAR_4_]], [[VAR_9_]]){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_6_]]([[VAR_4_]], [[VAR_9_]], [[VAR_1_]]4){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to [[MAP_7_]]([[VAR_4_]], [[VAR_9_]], [[VAR_1_]]4, [[VAR_1_]]9){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8]){
// CHECK:             [[VAR_22_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.cmpi sle, [[VAR_22_]]#0, [[VAR_1_]] : index
// CHECK-DAG:         [[VAR_24_:%.+]] = affine.apply [[MAP_8_]]([[VAR_22_]]#0){{.}}[[VAR_1_]]{{.}}
// CHECK:             [[VAR_25_:%.+]] = arith.select [[VAR_23_]], [[CST_0_]], [[VAR_24_]] : index
// CHECK:             [[VAR_26_:%.+]] = arith.cmpi sge, [[VAR_25_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.select [[VAR_26_]], [[CST_0_]], [[VAR_25_]] : index
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.cmpi sle, [[VAR_22_]]#1, [[VAR_6_]] : index
// CHECK-DAG:         [[VAR_29_:%.+]] = affine.apply [[MAP_8_]]([[VAR_22_]]#1){{.}}[[VAR_6_]]{{.}}
// CHECK:             [[VAR_30_:%.+]] = arith.select [[VAR_28_]], [[CST_0_]], [[VAR_29_]] : index
// CHECK:             [[VAR_31_:%.+]] = arith.cmpi sge, [[VAR_30_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_32_:%.+]] = arith.select [[VAR_31_]], [[CST_2_]], [[VAR_30_]] : index
// CHECK-DAG:         [[VAR_33_:%.+]] = arith.cmpi sle, [[VAR_22_]]#2, [[VAR_11_]] : index
// CHECK-DAG:         [[VAR_34_:%.+]] = affine.apply [[MAP_8_]]([[VAR_22_]]#2){{.}}[[VAR_11_]]{{.}}
// CHECK:             [[VAR_35_:%.+]] = arith.select [[VAR_33_]], [[CST_0_]], [[VAR_34_]] : index
// CHECK:             [[VAR_36_:%.+]] = arith.cmpi sge, [[VAR_35_]], [[CST_4_]] : index
// CHECK-DAG:         [[VAR_37_:%.+]] = arith.select [[VAR_36_]], [[CST_3_]], [[VAR_35_]] : index
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.cmpi sle, [[VAR_22_]]#3, [[VAR_16_]] : index
// CHECK-DAG:         [[VAR_39_:%.+]] = affine.apply [[MAP_8_]]([[VAR_22_]]#3){{.}}[[VAR_16_]]{{.}}
// CHECK:             [[VAR_40_:%.+]] = arith.select [[VAR_38_]], [[CST_0_]], [[VAR_39_]] : index
// CHECK:             [[VAR_41_:%.+]] = arith.cmpi sge, [[VAR_40_]], [[CST_5_]] : index
// CHECK:             [[VAR_42_:%.+]] = arith.select [[VAR_41_]], [[CST_4_]], [[VAR_40_]] : index
// CHECK:             [[LOAD_DATA_MEM_:%.+]] = krnl.load [[DATA_]]{{.}}[[VAR_27_]], [[VAR_32_]], [[VAR_37_]], [[VAR_42_]]{{.}} : memref<1x3x4x5xf32>
// CHECK:             krnl.store [[LOAD_DATA_MEM_]], [[RES_]]{{.}}[[VAR_22_]]#0, [[VAR_22_]]#1, [[VAR_22_]]#2, [[VAR_22_]]#3] : memref<?x?x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?x?xf32>
// CHECK:         }
}

// -----

func.func @pad_reflect_mode(%arg0: tensor<1x3x4x5xf32>, %arg1: tensor<8xi64>, %arg2: tensor<f32>) -> tensor<*xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = "onnx.Pad"(%arg0, %arg1, %arg2, %cst) {mode = "reflect"} : (tensor<1x3x4x5xf32>, tensor<8xi64>, tensor<f32>, none) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a'["data","pad","constant_value"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 3)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 4)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<()[s0, s1] -> (s0 + s1 + 5)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d0)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d1)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d2)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1, d2, d3)[s0, s1, s2, s3, s4, s5, s6, s7] -> (d3)>
// CHECK-LABEL:  func.func @pad_reflect_mode
// CHECK-SAME:   ([[DATA_:%.+]]: memref<1x3x4x5xf32>, [[PAD_:%.+]]: memref<8xi64>, [[CONSTANT_VALUE_:%.+]]: memref<f32>) -> memref<?x?x?x?xf32> {
// CHECK-DAG:       [[CST_8_:%.+]] = arith.constant 8 : index
// CHECK-DAG:       [[CST_7_:%.+]] = arith.constant 7 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_6_:%.+]] = arith.constant 6 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_5_:%.+]] = arith.constant 5 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_PAD_MEM_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_0_]]{{.}} : memref<8xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_1_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_4_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]], [[VAR_3_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_2_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_1_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_2_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_3_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_5_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_3_]] : i64 to index
// CHECK-DAG:       [[VAR_9_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_6_]], [[VAR_8_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_4_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_2_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_4_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_5_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_6_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_13_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_5_]] : i64 to index
// CHECK-DAG:       [[VAR_14_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_11_]], [[VAR_13_]]{{.}}
// CHECK-DAG:       [[LOAD_PAD_MEM_6_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_3_]]{{.}} : memref<8xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_16_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_6_]] : i64 to index
// CHECK-DAG:       [[LOAD_PAD_MEM_7_:%.+]] = krnl.load [[PAD_]]{{.}}[[CST_7_]]{{.}} : memref<8xi64>
// CHECK:           [[VAR_18_:%.+]] = arith.index_cast [[LOAD_PAD_MEM_7_]] : i64 to index
// CHECK:           [[VAR_19_:%.+]] = affine.apply [[MAP_3_]](){{.}}[[VAR_16_]], [[VAR_18_]]{{.}}
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_4_]], [[VAR_9_]], [[VAR_14_]], [[VAR_19_]]) {{.*}}: memref<?x?x?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_4_]]([[VAR_4_]]){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_5_]]([[VAR_4_]], [[VAR_9_]]){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_6_]]([[VAR_4_]], [[VAR_9_]], [[VAR_1_]]4){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8], [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to [[MAP_7_]]([[VAR_4_]], [[VAR_9_]], [[VAR_1_]]4, [[VAR_1_]]9){{.}}[[VAR_1_]], [[VAR_3_]], [[VAR_6_]], [[VAR_8_]], [[VAR_1_]]1, [[VAR_1_]]3, [[VAR_1_]]6, [[VAR_1_]]8]){
// CHECK:             [[VAR_22_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.cmpi slt, [[VAR_22_]]#0, [[VAR_1_]] : index
// CHECK:             [[VAR_26_:%.+]] = arith.select [[VAR_23_]], {{.*}}, {{.*}} : index
// CHECK-DAG:         [[VAR_27_:%.+]] = arith.cmpi sge, [[VAR_26_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.subi [[CST_0_]], [[VAR_26_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.select [[VAR_27_]], [[VAR_28_]], [[VAR_26_]] : index
// CHECK-DAG:         [[VAR_30_:%.+]] = arith.cmpi slt, [[VAR_22_]]#1, [[VAR_6_]] : index
// CHECK:             [[VAR_33_:%.+]] = arith.select [[VAR_30_]], {{.*}}, {{.*}} : index
// CHECK-DAG:         [[VAR_34_:%.+]] = arith.cmpi sge, [[VAR_33_]], [[CST_3_]] : index
// CHECK-DAG:         [[VAR_35_:%.+]] = arith.subi [[CST_4_]], [[VAR_33_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_36_:%.+]] = arith.select [[VAR_34_]], [[VAR_35_]], [[VAR_33_]] : index
// CHECK-DAG:         [[VAR_37_:%.+]] = arith.cmpi slt, [[VAR_22_]]#2, [[VAR_11_]] : index
// CHECK:             [[VAR_40_:%.+]] = arith.select [[VAR_37_]], {{.*}}, {{.*}} : index
// CHECK-DAG:         [[VAR_41_:%.+]] = arith.cmpi sge, [[VAR_40_]], [[CST_4_]] : index
// CHECK-DAG:         [[VAR_42_:%.+]] = arith.subi [[CST_6_]], [[VAR_40_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_43_:%.+]] = arith.select [[VAR_41_]], [[VAR_42_]], [[VAR_40_]] : index
// CHECK-DAG:         [[VAR_44_:%.+]] = arith.cmpi slt, [[VAR_22_]]#3, [[VAR_16_]] : index
// CHECK:             [[VAR_47_:%.+]] = arith.select [[VAR_44_]], {{.*}}, {{.*}} : index
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.cmpi sge, [[VAR_47_]], [[CST_5_]] : index
// CHECK-DAG:         [[VAR_49_:%.+]] = arith.subi [[CST_8_]], [[VAR_47_]] : index
// CHECK:             [[VAR_50_:%.+]] = arith.select [[VAR_48_]], [[VAR_49_]], [[VAR_47_]] : index
// CHECK:             [[LOAD_DATA_MEM_:%.+]] = krnl.load [[DATA_]]{{.}}[[VAR_29_]], [[VAR_36_]], [[VAR_43_]], [[VAR_50_]]{{.}} : memref<1x3x4x5xf32>
// CHECK:             krnl.store [[LOAD_DATA_MEM_]], [[RES_]]{{.}}[[VAR_22_]]#0, [[VAR_22_]]#1, [[VAR_22_]]#2, [[VAR_22_]]#3] : memref<?x?x?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?x?xf32>
// CHECK:         }
}

// -----

func.func @pad_constant_mode_constant_pads(%arg0: tensor<16x16xf32>) -> tensor<18x20xf32> {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0 = onnx.Constant dense<[0, 3, 2, 1]> : tensor<4xi64>
  %1 = onnx.Constant dense<0.000000e+00> : tensor<1xf32>
  %2 = "onnx.Pad"(%arg0, %0, %1, %cst) {mode = "constant"} : (tensor<16x16xf32>, tensor<4xi64>, tensor<1xf32>, none) -> tensor<18x20xf32>
  return %2 : tensor<18x20xf32>

// mlir2FileCheck.py -a'["data"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 3)>
// CHECK-LABEL:  func @pad_constant_mode_constant_pads
// CHECK-SAME:   ([[DATA_:%.+]]: memref<16x16xf32>) -> memref<18x20xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = "constant_{{[0-9]+}}", shape = [1], value = dense<0.000000e+00> : tensor<1xf32>} : () -> memref<1xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<18x20xf32>
// CHECK:           [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]]{{.}}[[CST_0_]]{{.}} : memref<1xf32>
// CHECK:           krnl.memset [[RES_]], [[LOAD_VAR_0_MEM_]] : memref<18x20xf32>
// CHECK:           [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 16){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_0_]]([[VAR_3_]]#1)
// CHECK-DAG:         [[LOAD_DATA_MEM_:%.+]] = krnl.load [[DATA_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<16x16xf32>
// CHECK:             krnl.store [[LOAD_DATA_MEM_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_4_]]{{.}} : memref<18x20xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<18x20xf32>
// CHECK:         }
}

// -----

func.func @test_expand_with_arith_constant(%arg0 : tensor<2x1x6x1xf32>) -> tensor<*xf32> {
  %0 = onnx.Constant dense<[7, 1, 5]> : tensor<3xi64>
  %1 = "onnx.Expand"(%arg0, %0) : (tensor<2x1x6x1xf32>, tensor<3xi64>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py -a'["input", "shape"]'
// CHECK-LABEL:  func @test_expand_with_arith_constant
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x1x6x1xf32>) -> memref<2x7x6x5xf32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x7x6x5xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[SHAPE_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 7, [[LOOP_0_]]#2 -> [[I_1_:%.+]] = 0 to 6, [[LOOP_0_]]#3 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_2_]]#0, [[CST_0_]], [[VAR_2_]]#2, [[CST_0_]]{{.}} : memref<2x1x6x1xf32>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<2x7x6x5xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x7x6x5xf32>
// CHECK:         }
}

// -----

  func.func @expand_dyn(%arg0: tensor<?x?xf32>, %arg1: tensor<2xi64>) -> tensor<?x?xf32>  {
    %0 = "onnx.Expand"(%arg0, %arg1) : (tensor<?x?xf32>, tensor<2xi64>) -> tensor<?x?xf32>
    return %0 : tensor<?x?xf32>
// mlir2FileCheck.py -a'["input", "shape"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-LABEL:  func @expand_dyn
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<?x?xf32>, [[SHAPE_:%.+]]: memref<2xi64>) -> memref<?x?xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[LOAD_SHAPE_MEM_:%.+]] = krnl.load [[SHAPE_]]{{.}}[[CST_0_]]{{.}} : memref<2xi64>
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_SHAPE_MEM_]] : i64 to index
// CHECK-DAG:       [[LOAD_SHAPE_MEM_1_:%.+]] = krnl.load [[SHAPE_]]{{.}}[[CST_1_]]{{.}} : memref<2xi64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.index_cast [[LOAD_SHAPE_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_4_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?xf32>
// CHECK-DAG:       [[VAR_5_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_1_]], [[VAR_4_]]{{.}}
// CHECK-DAG:       [[VAR_7_:%.+]] = affine.max [[MAP_0_]](){{.}}[[VAR_3_]], [[VAR_5_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_6_]], [[VAR_7_]]) {{.*}}: memref<?x?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[VAR_6_]], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[VAR_7_]]){
// CHECK-DAG:         [[VAR_10_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.cmpi sgt, [[VAR_4_]], [[CST_1_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[VAR_10_]]#0, [[CST_0_]] : index
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.cmpi sgt, [[VAR_5_]], [[CST_1_]] : index
// CHECK:             [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[VAR_10_]]#1, [[CST_0_]] : index
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_12_]], [[VAR_14_]]{{.}} : memref<?x?xf32>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_]]{{.}}[[VAR_10_]]#0, [[VAR_10_]]#1] : memref<?x?xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?xf32>
// CHECK:         }
}

// -----

func.func @test_cumsum_constant_axis(%arg0: tensor<2x3xf64>) -> tensor<*xf64> {
  %axis = onnx.Constant dense<1> : tensor<i32>
  %0 = "onnx.CumSum"(%arg0, %axis) : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_cumsum_constant_axis
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[AXIS_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : index to i64
// CHECK:             [[VAR_6_:%.+]] = arith.sitofp [[LOAD_INPUT_MEM_1_]] : i64 to f32
// CHECK:             [[VAR_7_:%.+]] = math.exp2 [[VAR_6_]] : f32
// CHECK:             [[VAR_8_:%.+]] = arith.fptosi [[VAR_7_]] : f32 to i64
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.subi [[VAR_12_]]#1, [[VAR_9_]] : index
// CHECK:               [[VAR_15_:%.+]] = arith.cmpi sge, [[VAR_14_]], [[CST_0_]] : index
// CHECK:               [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[VAR_14_]], [[VAR_12_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_16_]]] : memref<2x3xf64>
// CHECK:               [[VAR_18_:%.+]] = arith.select [[VAR_15_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_18_]] : f64
// CHECK:               krnl.store [[VAR_19_]], [[RES_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_constant_axis_reverse_mode(%arg0: tensor<2x3xf64>) -> tensor<*xf64> {
  %axis = onnx.Constant dense<1> : tensor<i32>
  %0 = "onnx.CumSum"(%arg0, %axis) {reverse = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_cumsum_constant_axis_reverse_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[AXIS_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : index to i64
// CHECK:             [[VAR_6_:%.+]] = arith.sitofp [[LOAD_INPUT_MEM_1_]] : i64 to f32
// CHECK:             [[VAR_7_:%.+]] = math.exp2 [[VAR_6_]] : f32
// CHECK:             [[VAR_8_:%.+]] = arith.fptosi [[VAR_7_]] : f32 to i64
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.addi [[VAR_12_]]#1, [[VAR_9_]] : index
// CHECK:               [[VAR_15_:%.+]] = arith.cmpi slt, [[VAR_14_]], [[CST_3_]] : index
// CHECK:               [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[VAR_14_]], [[VAR_12_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_16_]]] : memref<2x3xf64>
// CHECK:               [[VAR_18_:%.+]] = arith.select [[VAR_15_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_18_]] : f64
// CHECK:               krnl.store [[VAR_19_]], [[RES_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----


func.func @test_cumsum_constant_axis_exclusive_mode(%arg0: tensor<2x3xf64>) -> tensor<*xf64> {
  %axis = onnx.Constant dense<1> : tensor<i32>
  %0 = "onnx.CumSum"(%arg0, %axis) {exclusive = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_cumsum_constant_axis_exclusive_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[AXIS_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_5_:%.+]] = arith.subi [[VAR_4_]]#1, [[CST_1_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.cmpi sge, [[VAR_5_]], [[CST_0_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_5_]], [[VAR_4_]]#1 : index
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_4_]]#0, [[VAR_7_]]{{.}} : memref<2x3xf64>
// CHECK:             [[VAR_9_:%.+]] = arith.select [[VAR_6_]], [[LOAD_INPUT_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:             krnl.store [[VAR_9_]], [[RES_1_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_5_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : index to i64
// CHECK:             [[VAR_6_1_:%.+]] = arith.sitofp [[VAR_5_1_]] : i64 to f32
// CHECK:             [[VAR_7_1_:%.+]] = math.exp2 [[VAR_6_1_]] : f32
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.fptosi [[VAR_7_1_]] : f32 to i64
// CHECK-DAG:         [[VAR_9_1_:%.+]] = arith.index_cast [[LOAD_INPUT_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.subi [[VAR_12_]]#1, [[VAR_9_1_]] : index
// CHECK:               [[VAR_15_:%.+]] = arith.cmpi sge, [[VAR_14_]], [[CST_0_]] : index
// CHECK:               [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[VAR_14_]], [[VAR_12_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_16_]]] : memref<2x3xf64>
// CHECK:               [[VAR_18_:%.+]] = arith.select [[VAR_15_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_18_]] : f64
// CHECK:               krnl.store [[VAR_19_]], [[RES_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----


func.func @test_cumsum_constant_axis_exclusive_reverse_mode(%arg0: tensor<2x3xf64>) -> tensor<*xf64> {
  %axis = onnx.Constant dense<1> : tensor<i32>
  %0 = "onnx.CumSum"(%arg0, %axis) {exclusive = 1 : si64, reverse = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input"]'
// CHECK-LABEL:  func @test_cumsum_constant_axis_exclusive_reverse_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[AXIS_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_4_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_5_:%.+]] = arith.addi [[VAR_4_]]#1, [[CST_1_]] : index
// CHECK:             [[VAR_6_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[CST_3_]] : index
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_5_]], [[VAR_4_]]#1 : index
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_4_]]#0, [[VAR_7_]]{{.}} : memref<2x3xf64>
// CHECK:             [[VAR_9_:%.+]] = arith.select [[VAR_6_]], [[LOAD_INPUT_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:             krnl.store [[VAR_9_]], [[RES_1_]]{{.}}[[VAR_4_]]#0, [[VAR_4_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_4_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_5_1_:%.+]] = arith.index_cast [[VAR_4_1_]] : index to i64
// CHECK:             [[VAR_6_1_:%.+]] = arith.sitofp [[VAR_5_1_]] : i64 to f32
// CHECK:             [[VAR_7_1_:%.+]] = math.exp2 [[VAR_6_1_]] : f32
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.fptosi [[VAR_7_1_]] : f32 to i64
// CHECK-DAG:         [[VAR_9_1_:%.+]] = arith.index_cast [[LOAD_INPUT_MEM_1_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_3_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.addi [[VAR_12_]]#1, [[VAR_9_1_]] : index
// CHECK:               [[VAR_15_:%.+]] = arith.cmpi slt, [[VAR_14_]], [[CST_3_]] : index
// CHECK:               [[VAR_16_:%.+]] = arith.select [[VAR_15_]], [[VAR_14_]], [[VAR_12_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_16_]]] : memref<2x3xf64>
// CHECK:               [[VAR_18_:%.+]] = arith.select [[VAR_15_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_19_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_18_]] : f64
// CHECK:               krnl.store [[VAR_19_]], [[RES_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_5_:%.+]] = 0 to 3){
// CHECK:               [[VAR_12_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_12_1_]]#0, [[VAR_12_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_12_]]#0, [[VAR_12_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_dynamic_axis(%arg0: tensor<2x3xf64>, %arg1:tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input", "axis"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1 + 1)>
// CHECK-LABEL:  func @test_cumsum_dynamic_axis
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>, [[AXIS_:%.+]]: memref<i32>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOAD_AXIS_MEM_:%.+]] = krnl.load [[AXIS_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_AXIS_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_2_]], [[VAR_3_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_2_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_3_]], [[VAR_8_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:           [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:           [[VAR_13_:%.+]] = math.log2 [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.fptosi [[VAR_13_]] : f32 to i64
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[VAR_14_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[MAP_1_]](){{.}}[[VAR_1_]], [[VAR_15_]]]){
// CHECK:             [[VAR_18_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.index_cast [[VAR_18_1_]] : index to i64
// CHECK:             [[VAR_20_:%.+]] = arith.sitofp [[LOAD_INPUT_MEM_1_]] : i64 to f32
// CHECK:             [[VAR_21_:%.+]] = math.exp2 [[VAR_20_]] : f32
// CHECK:             [[VAR_22_:%.+]] = arith.fptosi [[VAR_21_]] : f32 to i64
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.index_cast [[VAR_22_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_28_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.subi [[VAR_26_]]#0, [[VAR_23_]] : index
// CHECK:               [[VAR_30_:%.+]] = arith.cmpi sge, [[VAR_29_]], [[CST_0_]] : index
// CHECK:               [[VAR_31_:%.+]] = arith.andi [[VAR_28_]], [[VAR_30_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.select [[VAR_31_]], [[VAR_29_]], [[VAR_26_]]#0 : index
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.subi [[VAR_26_]]#1, [[VAR_23_]] : index
// CHECK:               [[VAR_35_:%.+]] = arith.cmpi sge, [[VAR_34_]], [[CST_0_]] : index
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_33_]], [[VAR_35_]] : i1
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.ori [[VAR_36_]], [[VAR_31_]] : i1
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.select [[VAR_36_]], [[VAR_34_]], [[VAR_26_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_]], [[VAR_38_]]{{.}} : memref<2x3xf64>
// CHECK:               [[VAR_40_:%.+]] = arith.select [[VAR_37_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_40_]] : f64
// CHECK:               krnl.store [[VAR_41_]], [[RES_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_dynamic_axis_reverse_mode(%arg0: tensor<2x3xf64>, %arg1:tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) {reverse = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input", "axis"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1 + 1)>
// CHECK-LABEL:  func @test_cumsum_dynamic_axis_reverse_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>, [[AXIS_:%.+]]: memref<i32>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOAD_AXIS_MEM_:%.+]] = krnl.load [[AXIS_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_AXIS_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_2_]], [[VAR_3_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_2_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_3_]], [[VAR_8_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:           [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:           [[VAR_13_:%.+]] = math.log2 [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.fptosi [[VAR_13_]] : f32 to i64
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[VAR_14_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[MAP_1_]](){{.}}[[VAR_1_]], [[VAR_15_]]]){
// CHECK:             [[VAR_18_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.index_cast [[VAR_18_1_]] : index to i64
// CHECK:             [[VAR_20_:%.+]] = arith.sitofp [[LOAD_INPUT_MEM_1_]] : i64 to f32
// CHECK:             [[VAR_21_:%.+]] = math.exp2 [[VAR_20_]] : f32
// CHECK:             [[VAR_22_:%.+]] = arith.fptosi [[VAR_21_]] : f32 to i64
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.index_cast [[VAR_22_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_28_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_29_:%.+]] = arith.addi [[VAR_26_]]#0, [[VAR_23_]] : index
// CHECK:               [[VAR_30_:%.+]] = arith.cmpi slt, [[VAR_29_]], [[CST_2_]] : index
// CHECK:               [[VAR_31_:%.+]] = arith.andi [[VAR_28_]], [[VAR_30_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.select [[VAR_31_]], [[VAR_29_]], [[VAR_26_]]#0 : index
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.addi [[VAR_26_]]#1, [[VAR_23_]] : index
// CHECK:               [[VAR_35_:%.+]] = arith.cmpi slt, [[VAR_34_]], [[CST_3_]] : index
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_33_]], [[VAR_35_]] : i1
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.ori [[VAR_36_]], [[VAR_31_]] : i1
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.select [[VAR_36_]], [[VAR_34_]], [[VAR_26_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_]], [[VAR_38_]]{{.}} : memref<2x3xf64>
// CHECK:               [[VAR_40_:%.+]] = arith.select [[VAR_37_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_40_]] : f64
// CHECK:               krnl.store [[VAR_41_]], [[RES_]]{{.}}[[VAR_26_]]#0, [[VAR_26_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_dynamic_axis_exclusive_mode(%arg0: tensor<2x3xf64>, %arg1:tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) {exclusive = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input", "axis"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1 + 1)>
// CHECK-LABEL:  func @test_cumsum_dynamic_axis_exclusive_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>, [[AXIS_:%.+]]: memref<i32>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOAD_AXIS_MEM_:%.+]] = krnl.load [[AXIS_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_AXIS_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_2_]], [[VAR_3_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_2_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_3_]], [[VAR_8_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:           [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:           [[VAR_13_:%.+]] = math.log2 [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.fptosi [[VAR_13_]] : f32 to i64
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[VAR_14_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:         [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK:             [[VAR_20_:%.+]] = arith.subi [[VAR_18_]]#0, [[CST_1_]] : index
// CHECK:             [[VAR_21_:%.+]] = arith.cmpi sge, [[VAR_20_]], [[CST_0_]] : index
// CHECK:             [[VAR_22_:%.+]] = arith.andi [[VAR_19_]], [[VAR_21_]] : i1
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.select [[VAR_22_]], [[VAR_20_]], [[VAR_18_]]#0 : index
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.subi [[VAR_18_]]#1, [[CST_1_]] : index
// CHECK:             [[VAR_26_:%.+]] = arith.cmpi sge, [[VAR_25_]], [[CST_0_]] : index
// CHECK:             [[VAR_27_:%.+]] = arith.andi [[VAR_24_]], [[VAR_26_]] : i1
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.ori [[VAR_27_]], [[VAR_22_]] : i1
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.select [[VAR_27_]], [[VAR_25_]], [[VAR_18_]]#1 : index
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_23_]], [[VAR_29_]]{{.}} : memref<2x3xf64>
// CHECK:             [[VAR_31_:%.+]] = arith.select [[VAR_28_]], [[LOAD_INPUT_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:             krnl.store [[VAR_31_]], [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[MAP_1_]](){{.}}[[VAR_1_]], [[VAR_15_]]]){
// CHECK:             [[VAR_18_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_19_1_:%.+]] = arith.index_cast [[VAR_18_1_]] : index to i64
// CHECK:             [[VAR_20_1_:%.+]] = arith.sitofp [[VAR_19_1_]] : i64 to f32
// CHECK:             [[VAR_21_1_:%.+]] = math.exp2 [[VAR_20_1_]] : f32
// CHECK:             [[VAR_22_1_:%.+]] = arith.fptosi [[VAR_21_1_]] : f32 to i64
// CHECK-DAG:         [[VAR_23_1_:%.+]] = arith.index_cast [[VAR_22_1_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_27_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_28_1_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_29_1_:%.+]] = arith.subi [[VAR_26_1_]]#0, [[VAR_23_1_]] : index
// CHECK:               [[LOAD_INPUT_MEM_1_:%.+]] = arith.cmpi sge, [[VAR_29_1_]], [[CST_0_]] : index
// CHECK:               [[VAR_31_1_:%.+]] = arith.andi [[VAR_28_1_]], [[LOAD_INPUT_MEM_1_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.select [[VAR_31_1_]], [[VAR_29_1_]], [[VAR_26_1_]]#0 : index
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.subi [[VAR_26_1_]]#1, [[VAR_23_1_]] : index
// CHECK:               [[VAR_35_:%.+]] = arith.cmpi sge, [[VAR_34_]], [[CST_0_]] : index
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_33_]], [[VAR_35_]] : i1
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.ori [[VAR_36_]], [[VAR_31_1_]] : i1
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.select [[VAR_36_]], [[VAR_34_]], [[VAR_26_1_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_]], [[VAR_38_]]{{.}} : memref<2x3xf64>
// CHECK:               [[VAR_40_:%.+]] = arith.select [[VAR_37_]], [[LOAD_RES_1_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[VAR_27_1_]], [[VAR_40_]] : f64
// CHECK:               krnl.store [[VAR_41_]], [[RES_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[VAR_27_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_26_2_]]#0, [[VAR_26_2_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[VAR_27_1_]], [[RES_1_]]{{.}}[[VAR_26_2_]]#0, [[VAR_26_2_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----

func.func @test_cumsum_dynamic_axis_exclusive_reverse_mode(%arg0: tensor<2x3xf64>, %arg1:tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) {exclusive = 1 : si64, reverse = 1 : si64} : (tensor<2x3xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a'["input", "axis"]'
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0, s1] -> (s1 + 1)>
// CHECK-LABEL:  func @test_cumsum_dynamic_axis_exclusive_reverse_mode
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<2x3xf64>, [[AXIS_:%.+]]: memref<i32>) -> memref<2x3xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOAD_AXIS_MEM_:%.+]] = krnl.load [[AXIS_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_AXIS_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_3_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_1_]]{{.}}
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_2_]], [[VAR_3_]], [[VAR_1_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<2x3xf64>
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_2_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_9_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_3_]], [[VAR_8_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.index_cast [[VAR_10_]] : index to i64
// CHECK:           [[VAR_12_:%.+]] = arith.sitofp [[VAR_11_]] : i64 to f32
// CHECK:           [[VAR_13_:%.+]] = math.log2 [[VAR_12_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.fptosi [[VAR_13_]] : f32 to i64
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.index_cast [[VAR_14_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3){
// CHECK-DAG:         [[VAR_18_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK:             [[VAR_20_:%.+]] = arith.addi [[VAR_18_]]#0, [[CST_1_]] : index
// CHECK:             [[VAR_21_:%.+]] = arith.cmpi slt, [[VAR_20_]], [[CST_2_]] : index
// CHECK:             [[VAR_22_:%.+]] = arith.andi [[VAR_19_]], [[VAR_21_]] : i1
// CHECK-DAG:         [[VAR_23_:%.+]] = arith.select [[VAR_22_]], [[VAR_20_]], [[VAR_18_]]#0 : index
// CHECK-DAG:         [[VAR_24_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK-DAG:         [[VAR_25_:%.+]] = arith.addi [[VAR_18_]]#1, [[CST_1_]] : index
// CHECK:             [[VAR_26_:%.+]] = arith.cmpi slt, [[VAR_25_]], [[CST_3_]] : index
// CHECK:             [[VAR_27_:%.+]] = arith.andi [[VAR_24_]], [[VAR_26_]] : i1
// CHECK-DAG:         [[VAR_28_:%.+]] = arith.ori [[VAR_27_]], [[VAR_22_]] : i1
// CHECK-DAG:         [[VAR_29_:%.+]] = arith.select [[VAR_27_]], [[VAR_25_]], [[VAR_18_]]#1 : index
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_23_]], [[VAR_29_]]{{.}} : memref<2x3xf64>
// CHECK:             [[VAR_31_:%.+]] = arith.select [[VAR_28_]], [[LOAD_INPUT_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:             krnl.store [[VAR_31_]], [[RES_1_]]{{.}}[[VAR_18_]]#0, [[VAR_18_]]#1] : memref<2x3xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[MAP_1_]](){{.}}[[VAR_1_]], [[VAR_15_]]]){
// CHECK:             [[VAR_18_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[VAR_19_1_:%.+]] = arith.index_cast [[VAR_18_1_]] : index to i64
// CHECK:             [[VAR_20_1_:%.+]] = arith.sitofp [[VAR_19_1_]] : i64 to f32
// CHECK:             [[VAR_21_1_:%.+]] = math.exp2 [[VAR_20_1_]] : f32
// CHECK:             [[VAR_22_1_:%.+]] = arith.fptosi [[VAR_21_1_]] : f32 to i64
// CHECK-DAG:         [[VAR_23_1_:%.+]] = arith.index_cast [[VAR_22_1_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[VAR_27_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK-DAG:           [[VAR_28_1_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_29_1_:%.+]] = arith.addi [[VAR_26_1_]]#0, [[VAR_23_1_]] : index
// CHECK:               [[LOAD_INPUT_MEM_1_:%.+]] = arith.cmpi slt, [[VAR_29_1_]], [[CST_2_]] : index
// CHECK:               [[VAR_31_1_:%.+]] = arith.andi [[VAR_28_1_]], [[LOAD_INPUT_MEM_1_]] : i1
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.select [[VAR_31_1_]], [[VAR_29_1_]], [[VAR_26_1_]]#0 : index
// CHECK-DAG:           [[VAR_33_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK-DAG:           [[VAR_34_:%.+]] = arith.addi [[VAR_26_1_]]#1, [[VAR_23_1_]] : index
// CHECK:               [[VAR_35_:%.+]] = arith.cmpi slt, [[VAR_34_]], [[CST_3_]] : index
// CHECK:               [[VAR_36_:%.+]] = arith.andi [[VAR_33_]], [[VAR_35_]] : i1
// CHECK-DAG:           [[VAR_37_:%.+]] = arith.ori [[VAR_36_]], [[VAR_31_1_]] : i1
// CHECK-DAG:           [[VAR_38_:%.+]] = arith.select [[VAR_36_]], [[VAR_34_]], [[VAR_26_1_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_32_]], [[VAR_38_]]{{.}} : memref<2x3xf64>
// CHECK:               [[VAR_40_:%.+]] = arith.select [[VAR_37_]], [[LOAD_RES_1_MEM_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_41_:%.+]] = arith.addf [[VAR_27_1_]], [[VAR_40_]] : f64
// CHECK:               krnl.store [[VAR_41_]], [[RES_]]{{.}}[[VAR_26_1_]]#0, [[VAR_26_1_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 3){
// CHECK:               [[VAR_26_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[VAR_27_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_26_2_]]#0, [[VAR_26_2_]]#1] : memref<2x3xf64>
// CHECK:               krnl.store [[VAR_27_1_]], [[RES_1_]]{{.}}[[VAR_26_2_]]#0, [[VAR_26_2_]]#1] : memref<2x3xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x3xf64>
// CHECK:         }
}

// -----


func.func @test_cumsum_dynamic_dims(%arg0: tensor<?x?xf64>, %arg1:tensor<i32>) -> tensor<*xf64> {
  %0 = "onnx.CumSum"(%arg0, %arg1) : (tensor<?x?xf64>, tensor<i32>) -> tensor<*xf64>
  return %0 : tensor<*xf64>

// mlir2FileCheck.py -a '["input","axis"]'
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<()[s0] -> (s0 + 2)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0)[s0, s1] -> (d0)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (d1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (s1 + 1)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (d0)>
// CHECK-LABEL:  func.func @test_cumsum_dynamic_dims
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<?x?xf64>, [[AXIS_:%.+]]: memref<i32>) -> memref<?x?xf64> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f64
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?xf64>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?xf64>
// CHECK-DAG:       [[LOAD_AXIS_MEM_:%.+]] = krnl.load [[AXIS_]][] : memref<i32>
// CHECK:           [[VAR_1_:%.+]] = arith.index_cast [[LOAD_AXIS_MEM_]] : i32 to index
// CHECK-DAG:       [[VAR_2_:%.+]] = affine.apply [[MAP_2_]](){{.}}[[VAR_1_]]{{.}}
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.cmpi slt, [[VAR_1_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[VAR_2_]], [[VAR_1_]] : index
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?xf64>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?xf64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_1_]], [[VAR_dim_2_]]) {{.*}}: memref<?x?xf64>
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?xf64>
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?xf64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_dim_3_]], [[VAR_dim_4_]]) {{.*}}: memref<?x?xf64>
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[VAR_dim_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK:           [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_dim_0_]], [[VAR_6_]] : index
// CHECK:           [[VAR_9_:%.+]] = arith.index_cast [[VAR_8_]] : index to i64
// CHECK:           [[VAR_10_:%.+]] = arith.sitofp [[VAR_9_]] : i64 to f32
// CHECK:           [[VAR_11_:%.+]] = math.log2 [[VAR_10_]] : f32
// CHECK:           [[VAR_12_:%.+]] = arith.fptosi [[VAR_11_]] : f32 to i64
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.index_cast [[VAR_12_]] : i64 to index
// CHECK-DAG:       [[VAR_dim_6_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?xf64>
// CHECK-DAG:       [[VAR_dim_7_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?xf64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_6_]]){{.}}[[VAR_1_]], [[VAR_1_]]3], [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_4_]]([[VAR_dim_6_]], [[VAR_dim_7_]]){{.}}[[VAR_1_]], [[VAR_1_]]3]){
// CHECK:             [[VAR_16_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_16_]]#0, [[VAR_16_]]#1] : memref<?x?xf64>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_16_]]#0, [[VAR_16_]]#1] : memref<?x?xf64>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to [[MAP_5_]]([[VAR_dim_6_]], [[VAR_dim_7_]]){{.}}[[VAR_1_]], [[VAR_1_]]3]){
// CHECK:             [[VAR_16_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = arith.index_cast [[VAR_16_1_]] : index to i64
// CHECK:             [[VAR_18_:%.+]] = arith.sitofp [[LOAD_INPUT_MEM_1_]] : i64 to f32
// CHECK:             [[VAR_19_:%.+]] = math.exp2 [[VAR_18_]] : f32
// CHECK:             [[VAR_20_:%.+]] = arith.fptosi [[VAR_19_]] : f32 to i64
// CHECK-DAG:         [[VAR_21_:%.+]] = arith.index_cast [[VAR_20_]] : i64 to index
// CHECK-DAG:         [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to [[MAP_6_]]([[VAR_dim_6_]], [[VAR_dim_7_]]){{.}}[[VAR_1_]], [[VAR_1_]]3], [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to [[MAP_4_]]([[VAR_dim_6_]], [[VAR_dim_7_]]){{.}}[[VAR_1_]], [[VAR_1_]]3]){
// CHECK:               [[VAR_24_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_24_]]#0, [[VAR_24_]]#1] : memref<?x?xf64>
// CHECK-DAG:           [[VAR_26_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_0_]] : index
// CHECK-DAG:           [[VAR_27_:%.+]] = arith.subi [[VAR_24_]]#0, [[VAR_21_]] : index
// CHECK:               [[VAR_28_:%.+]] = arith.cmpi sge, [[VAR_27_]], [[CST_0_]] : index
// CHECK:               [[VAR_29_:%.+]] = arith.andi [[VAR_26_]], [[VAR_28_]] : i1
// CHECK-DAG:           [[VAR_30_:%.+]] = arith.select [[VAR_29_]], [[VAR_27_]], [[VAR_24_]]#0 : index
// CHECK-DAG:           [[VAR_31_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_1_]] : index
// CHECK-DAG:           [[VAR_32_:%.+]] = arith.subi [[VAR_24_]]#1, [[VAR_21_]] : index
// CHECK:               [[VAR_33_:%.+]] = arith.cmpi sge, [[VAR_32_]], [[CST_0_]] : index
// CHECK:               [[VAR_34_:%.+]] = arith.andi [[VAR_31_]], [[VAR_33_]] : i1
// CHECK-DAG:           [[VAR_35_:%.+]] = arith.ori [[VAR_34_]], [[VAR_29_]] : i1
// CHECK-DAG:           [[VAR_36_:%.+]] = arith.select [[VAR_34_]], [[VAR_32_]], [[VAR_24_]]#1 : index
// CHECK:               [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_30_]], [[VAR_36_]]{{.}} : memref<?x?xf64>
// CHECK:               [[VAR_38_:%.+]] = arith.select [[VAR_35_]], [[LOAD_RES_1_MEM_1_]], [[CST_0_dot_000000_]] : f64
// CHECK:               [[VAR_39_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[VAR_38_]] : f64
// CHECK:               krnl.store [[VAR_39_]], [[RES_]]{{.}}[[VAR_24_]]#0, [[VAR_24_]]#1] : memref<?x?xf64>
// CHECK:             }
// CHECK:             [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to [[MAP_6_]]([[VAR_dim_6_]], [[VAR_dim_7_]]){{.}}[[VAR_1_]], [[VAR_1_]]3], [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to [[MAP_4_]]([[VAR_dim_6_]], [[VAR_dim_7_]]){{.}}[[VAR_1_]], [[VAR_1_]]3]){
// CHECK:               [[VAR_24_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:               [[LOAD_RES_1_MEM_2_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<?x?xf64>
// CHECK:               krnl.store [[LOAD_RES_1_MEM_2_]], [[RES_1_]]{{.}}[[VAR_24_1_]]#0, [[VAR_24_1_]]#1] : memref<?x?xf64>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?xf64>
// CHECK:         }
}

// -----
// Compress on axis 0, with enough conditions, test elided

func.func @compress_axis0(%arg0: tensor<3x2xf32>, %arg1: tensor<3xi1>) -> tensor<?x2xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
//  use arg names: ['input', 'condition']
// mlir2FileCheck.py -a'["input", "condition"]'
// CHECK-LABEL:  func @compress_axis0
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x2xf32>, [[CONDITION_:%.+]]: memref<3xi1>) -> memref<?x2xf32> {
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_c1_]], [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_10_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_8_]] : index
// CHECK:             krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x2xf32>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[VAR_5_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_1_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7_1_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_1_]], [[VAR_false_]] : i1
// CHECK:             scf.if [[VAR_7_1_]] {
// CHECK-DAG:           [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 2){
// CHECK:                 [[VAR_11_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_5_1_]], [[VAR_11_]]{{.}} : memref<3x2xf32>
// CHECK:                 krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[LOAD_RES_MEM_2_]], [[VAR_11_]]{{.}} : memref<?x2xf32>
// CHECK:               }
// CHECK:               [[VAR_10_1_:%.+]] = arith.addi [[LOAD_RES_MEM_2_]], [[VAR_c1_]] : index
// CHECK:               krnl.store [[VAR_10_1_]], [[RES_]][] : memref<index>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?x2xf32>
// CHECK:         }
}

// -----
// Compress on axis 0, with not enough conditions, test not elided

func.func @compress_axis0_not_enough(%arg0: tensor<3x2xf32>, %arg1: tensor<2xi1>) -> tensor<?x2xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) {axis = 0 : si64} : (tensor<3x2xf32>, tensor<2xi1>) -> tensor<?x2xf32>
  return %0 : tensor<?x2xf32>
// mlir2FileCheck.py -a'["input", "condition"]'
// CHECK-LABEL:  func @compress_axis0_not_enough
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x2xf32>, [[CONDITION_:%.+]]: memref<2xi1>) -> memref<?x2xf32> {
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 2){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_]]{{.}} : memref<2xi1>
// CHECK:             [[VAR_7_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_c1_]], [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_10_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_8_]] : index
// CHECK:             krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?x2xf32>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 3){
// CHECK:             [[VAR_5_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_1_:%.+]] = arith.cmpi slt, [[VAR_5_1_]], [[VAR_c2_]] : index
// CHECK:             scf.if [[LOAD_CONDITION_MEM_1_]] {
// CHECK:               [[LOAD_CONDITION_MEM_2_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_1_]]{{.}} : memref<2xi1>
// CHECK:               [[VAR_8_1_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_2_]], [[VAR_false_]] : i1
// CHECK:               scf.if [[VAR_8_1_]] {
// CHECK-DAG:             [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:             [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:                 krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 2){
// CHECK:                   [[VAR_12_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:                   [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_5_1_]], [[VAR_12_]]{{.}} : memref<3x2xf32>
// CHECK:                   krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[LOAD_RES_MEM_2_]], [[VAR_12_]]{{.}} : memref<?x2xf32>
// CHECK:                 }
// CHECK:                 [[VAR_11_:%.+]] = arith.addi [[LOAD_RES_MEM_2_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_11_]], [[RES_]][] : memref<index>
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?x2xf32>
// CHECK:         }
}

// -----
// Compress on axis 1, with enough conditions, test elided

func.func @compress_axis1(%arg0: tensor<3x2xf32>, %arg1: tensor<3xi1>) -> tensor<3x?xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) {axis = 1 : si64} : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<3x?xf32>
  return %0 : tensor<3x?xf32>
// mlir2FileCheck.py -a'["input", "condition"]'
// CHECK-LABEL:  func @compress_axis1
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x2xf32>, [[CONDITION_:%.+]]: memref<3xi1>) -> memref<3x?xf32> {
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_c1_]], [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_10_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_8_]] : index
// CHECK:             krnl.store [[VAR_10_]], [[RES_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<3x?xf32>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_5_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_1_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_5_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_7_1_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_1_]], [[VAR_false_]] : i1
// CHECK:             scf.if [[VAR_7_1_]] {
// CHECK-DAG:           [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK-DAG:           [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK:               krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 3){
// CHECK:                 [[VAR_11_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK:                 [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_11_]], [[VAR_5_1_]]{{.}} : memref<3x2xf32>
// CHECK:                 krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[VAR_11_]], [[LOAD_RES_MEM_2_]]{{.}} : memref<3x?xf32>
// CHECK:               }
// CHECK:               [[VAR_10_1_:%.+]] = arith.addi [[LOAD_RES_MEM_2_]], [[VAR_c1_]] : index
// CHECK:               krnl.store [[VAR_10_1_]], [[RES_]][] : memref<index>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<3x?xf32>
// CHECK:         }
}

// -----
// Compress witn no axis , with not enough conditions, test not elided

func.func @compress_no_axis_not_elided(%arg0: tensor<3x2xf32>, %arg1: tensor<3xi1>) -> tensor<?xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) : (tensor<3x2xf32>, tensor<3xi1>) -> tensor<?xf32>
  return %0 : tensor<?xf32>

// CHECK-LABEL:  func @compress_no_axis_not_elided
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x2xf32>, [[CONDITION_:%.+]]: memref<3xi1>) -> memref<?xf32> {
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 3){
// CHECK:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_6_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_8_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.select [[VAR_8_]], [[VAR_c1_]], [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_11_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_9_]] : index
// CHECK:             krnl.store [[VAR_11_]], [[RES_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?xf32>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_6_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_CONDITION_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[VAR_8_1_:%.+]] = arith.cmpi slt, [[LOAD_CONDITION_MEM_1_]], [[VAR_c3_]] : index
// CHECK:             scf.if [[VAR_8_1_]] {
// CHECK:               [[LOAD_CONDITION_MEM_2_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[LOAD_CONDITION_MEM_1_]]{{.}} : memref<3xi1>
// CHECK:               [[LOAD_RES_MEM_2_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_2_]], [[VAR_false_]] : i1
// CHECK:               scf.if [[LOAD_RES_MEM_2_]] {
// CHECK-DAG:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_6_1_]]#0, [[VAR_6_1_]]#1] : memref<3x2xf32>
// CHECK-DAG:             [[LOAD_RES_MEM_3_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:                 krnl.store [[LOAD_INPUT_MEM_]], [[RES_1_]]{{.}}[[LOAD_RES_MEM_3_]]{{.}} : memref<?xf32>
// CHECK:                 [[VAR_14_:%.+]] = arith.addi [[LOAD_RES_MEM_3_]], [[VAR_c1_]] : index
// CHECK:                 krnl.store [[VAR_14_]], [[RES_]][] : memref<index>
// CHECK:               }
// CHECK:               [[VAR_11_1_:%.+]] = arith.addi [[LOAD_CONDITION_MEM_1_]], [[VAR_c1_]] : index
// CHECK:               krnl.store [[VAR_11_1_]], [[RES_2_]][] : memref<index>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?xf32>
// CHECK:         }
}

// -----
// Compress witn no axis , with enough conditions, test elided

func.func @compress_no_axis_enough_cond(%arg0: tensor<3x2xf32>, %arg1: tensor<6xi1>) -> tensor<?xf32> {
  %0 = "onnx.Compress"(%arg0, %arg1) : (tensor<3x2xf32>, tensor<6xi1>) -> tensor<?xf32>
  return %0 : tensor<?xf32>
// CHECK-LABEL:  func @compress_no_axis_enough_cond
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x2xf32>, [[CONDITION_:%.+]]: memref<6xi1>) -> memref<?xf32> {
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_false_:%.+]] = arith.constant false
// CHECK-DAG:       [[RES_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 6){
// CHECK:             [[VAR_6_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_CONDITION_MEM_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[VAR_6_]]{{.}} : memref<6xi1>
// CHECK:             [[VAR_8_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_]], [[VAR_false_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.select [[VAR_8_]], [[VAR_c1_]], [[VAR_c0_]] : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:             [[VAR_11_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[VAR_9_]] : index
// CHECK:             krnl.store [[VAR_11_]], [[RES_]][] : memref<index>
// CHECK:           }
// CHECK:           [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[LOAD_RES_MEM_1_]]) {{.*}}: memref<?xf32>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_]][] : memref<index>
// CHECK:           [[RES_2_:%.+]] = memref.alloca() : memref<index>
// CHECK:           krnl.store [[VAR_c0_]], [[RES_2_]][] : memref<index>
// CHECK:           [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_2_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_6_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_CONDITION_MEM_1_:%.+]] = krnl.load [[RES_2_]][] : memref<index>
// CHECK:             [[LOAD_CONDITION_MEM_2_:%.+]] = krnl.load [[CONDITION_]]{{.}}[[LOAD_CONDITION_MEM_1_]]{{.}} : memref<6xi1>
// CHECK:             [[VAR_9_1_:%.+]] = arith.cmpi ne, [[LOAD_CONDITION_MEM_2_]], [[VAR_false_]] : i1
// CHECK:             scf.if [[VAR_9_1_]] {
// CHECK-DAG:           [[VAR_11_1_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_6_1_]]#0, [[VAR_6_1_]]#1] : memref<3x2xf32>
// CHECK-DAG:           [[LOAD_RES_MEM_2_:%.+]] = krnl.load [[RES_]][] : memref<index>
// CHECK:               krnl.store [[VAR_11_1_]], [[RES_1_]]{{.}}[[LOAD_RES_MEM_2_]]{{.}} : memref<?xf32>
// CHECK:               [[VAR_13_:%.+]] = arith.addi [[LOAD_RES_MEM_2_]], [[VAR_c1_]] : index
// CHECK:               krnl.store [[VAR_13_]], [[RES_]][] : memref<index>
// CHECK:             }
// CHECK:             [[LOAD_RES_MEM_3_:%.+]] = arith.addi [[LOAD_CONDITION_MEM_1_]], [[VAR_c1_]] : index
// CHECK:             krnl.store [[LOAD_RES_MEM_3_]], [[RES_2_]][] : memref<index>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?xf32>
// CHECK:         }
}

// -----


func.func @test_hardmax_axis_1(%arg0: tensor<3x4x5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Hardmax"(%arg0) {axis = 1 : si64} : (tensor<3x4x5xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a '["input"]'
// CHECK-LABEL:  func.func @test_hardmax_axis_1
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<3x4x5xf32>) -> memref<3x4x5xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3x4x5xf32>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<3x1x5xindex>
// CHECK:           krnl.memset [[RES_1_]], [[CST_0_]] : memref<3x1x5xindex>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 4, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 5){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_2_]]#0, [[CST_0_]], [[VAR_2_]]#2] : memref<3x1x5xindex>
// CHECK-DAG:         [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_2_]]#0, [[LOAD_RES_1_MEM_]], [[VAR_2_]]#2] : memref<3x4x5xf32>
// CHECK-DAG:         [[LOAD_INPUT_MEM_1_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2] : memref<3x4x5xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.cmpf ogt, [[LOAD_INPUT_MEM_1_]], [[LOAD_INPUT_MEM_]] : f32
// CHECK:             scf.if [[VAR_6_]] {
// CHECK:               krnl.store [[VAR_2_]]#1, [[RES_1_]]{{.}}[[VAR_2_]]#0, [[CST_0_]], [[VAR_2_]]#2] : memref<3x1x5xindex>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to 4, [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to 5){
// CHECK:             [[VAR_2_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_2_1_]]#0, [[CST_0_]], [[VAR_2_1_]]#2] : memref<3x1x5xindex>
// CHECK:             [[LOAD_INPUT_MEM_2_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_1_]], [[VAR_2_1_]]#1 : index
// CHECK:             scf.if [[LOAD_INPUT_MEM_2_]] {
// CHECK:               krnl.store [[CST_1_dot_000000_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2] : memref<3x4x5xf32>
// CHECK:             } else {
// CHECK:               krnl.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2] : memref<3x4x5xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x4x5xf32>
// CHECK:         }
}

// -----


func.func @test_hardmax_unknown_dims(%arg0: tensor<?x?x?xf32>) -> tensor<*xf32> {
  %0 = "onnx.Hardmax"(%arg0) {axis = 1 : si64} : (tensor<?x?x?xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// mlir2FileCheck.py -a '["input"]'
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-LABEL:  func.func @test_hardmax_unknown_dims
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<?x?x?xf32>) -> memref<?x?x?xf32> {
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[INPUT_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_dim_]]_2) {{.*}}: memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_3_:%.+]] = memref.dim [[INPUT_]], [[CST_0_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_4_:%.+]] = memref.dim [[INPUT_]], [[CST_1_]] : memref<?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_5_:%.+]] = memref.dim [[INPUT_]], [[CST_2_]] : memref<?x?x?xf32>
// CHECK:           [[RES_1_:%.+]] = memref.alloc([[VAR_dim_3_]], [[VAR_dim_5_]]) {{.*}}: memref<?x1x?xindex>
// CHECK:           krnl.memset [[RES_1_]], [[CST_0_]] : memref<?x1x?xindex>
// CHECK:           [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_3_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_3_]], [[VAR_dim_4_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_3_]], [[VAR_dim_4_]], [[VAR_dim_5_]])){
// CHECK:             [[VAR_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_2_]]#0, [[CST_0_]], [[VAR_2_]]#2] : memref<?x1x?xindex>
// CHECK-DAG:         [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_2_]]#0, [[LOAD_RES_1_MEM_]], [[VAR_2_]]#2] : memref<?x?x?xf32>
// CHECK-DAG:         [[LOAD_INPUT_MEM_1_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2] : memref<?x?x?xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.cmpf ogt, [[LOAD_INPUT_MEM_1_]], [[LOAD_INPUT_MEM_]] : f32
// CHECK:             scf.if [[VAR_6_]] {
// CHECK:               krnl.store [[VAR_2_]]#1, [[RES_1_]]{{.}}[[VAR_2_]]#0, [[CST_0_]], [[VAR_2_]]#2] : memref<?x1x?xindex>
// CHECK:             }
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_3_:%.+]] = 0 to [[MAP_1_]]([[VAR_dim_]]), [[LOOP_1_]]#1 -> [[I_4_:%.+]] = 0 to [[MAP_2_]]([[VAR_dim_]], [[VAR_dim_]]_1), [[LOOP_1_]]#2 -> [[I_5_:%.+]] = 0 to [[MAP_3_]]([[VAR_dim_]], [[VAR_dim_]]_1, [[VAR_dim_]]_2)){
// CHECK:             [[VAR_2_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_2_1_]]#0, [[CST_0_]], [[VAR_2_1_]]#2] : memref<?x1x?xindex>
// CHECK:             [[LOAD_INPUT_MEM_2_:%.+]] = arith.cmpi eq, [[LOAD_RES_1_MEM_1_]], [[VAR_2_1_]]#1 : index
// CHECK:             scf.if [[LOAD_INPUT_MEM_2_]] {
// CHECK:               krnl.store [[CST_1_dot_000000_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2] : memref<?x?x?xf32>
// CHECK:             } else {
// CHECK:               krnl.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2] : memref<?x?x?xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x?x?xf32>
// CHECK:         }
}

// -----

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

// -----

func.func @test_loop_tiny_yolo() -> tensor<?xi32> {
    %0 = onnx.Constant dense<7> : tensor<i64>
    %1 = onnx.Constant dense<true> : tensor<i1>
    %2 = onnx.Constant dense<0> : tensor<i32>
    %3:2 = "onnx.Loop"(%0, %1, %2) ( {
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i1>, %arg2: tensor<i32>):  // no predecessors
      %4 = onnx.Constant dense<1> : tensor<i32>
      %5 = "onnx.Add"(%arg2, %4) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      onnx.Yield %arg1, %5, %arg2 : tensor<i1>, tensor<i32>, tensor<i32>
    }) {input_names = ["i", "cond", "prev"], output_names = ["cond_out", "current", "range"]} : (tensor<i64>, tensor<i1>, tensor<i32>) -> (tensor<i32>, tensor<?xi32>)
    return %3#1 : tensor<?xi32>

// CHECK-LABEL:  func @test_loop_tiny_yolo
// CHECK-SAME:   () -> memref<?xi32> {
// CHECK-DAG:       [[ZERO:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[ONE_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<1> : tensor<i32>} : () -> memref<i32>
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<7> : tensor<i64>} : () -> memref<i64>
// CHECK-DAG:       [[VAR_1_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<true> : tensor<i1>} : () -> memref<i1>
// CHECK-DAG:       [[VAR_2_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<0> : tensor<i32>} : () -> memref<i32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() : memref<i32>
// CHECK-DAG:       [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<i64>
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_]] : i64 to index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_5_]]) {{.*}}: memref<?xi32>
// CHECK-DAG:       [[LOAD_VAR_2_MEM_:%.+]] = krnl.load [[VAR_2_]][] : memref<i32>
// CHECK-DAG:       krnl.store [[LOAD_VAR_2_MEM_]], [[RES_]][] : memref<i32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<i1>
// CHECK-DAG:       [[LOAD_VAR_1_MEM_:%.+]] = krnl.load [[VAR_1_]][] : memref<i1>
// CHECK-DAG:       krnl.store [[LOAD_VAR_1_MEM_]], [[RES_2_]][] : memref<i1>
// CHECK-DAG:       [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<i64>
// CHECK-DAG:       [[VAR_12_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_1_]] : i64 to index
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> %arg0 = [[ZERO]] to [[VAR_12_]]){
// CHECK-DAG:         [[I_0_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_RES_2_MEM_:%.+]] = krnl.load [[RES_2_]][] : memref<i1>
// CHECK-DAG:         scf.if [[LOAD_RES_2_MEM_]] {
// CHECK-DAG:           [[VAR_14_:%.+]] = arith.index_cast [[I_0_]] : index to i64
// CHECK-DAG:           [[RES_3_:%.+]] = memref.alloc() : memref<i64>
// CHECK-DAG:           krnl.store [[VAR_14_]], [[RES_3_]][] : memref<i64>
// CHECK-DAG:           [[RES_4_:%.+]] = memref.alloc() : memref<i32>
// CHECK-DAG:           [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]][] : memref<i32>
// CHECK:               [[LOAD_ONE_MEM_:%.+]] = krnl.load [[ONE_]][] : memref<i32>
// CHECK:               [[VAR_20_:%.+]] = arith.addi [[LOAD_RES_MEM_]], [[LOAD_ONE_MEM_]] : i32
// CHECK:               krnl.store [[VAR_20_]], [[RES_4_]][] : memref<i32>
// CHECK:               [[LOAD_VAR_1_MEM_1_:%.+]] = krnl.load [[VAR_1_]][] : memref<i1>
// CHECK:               krnl.store [[LOAD_VAR_1_MEM_1_]], [[RES_2_]][] : memref<i1>
// CHECK:               [[LOAD_RES_MEM_1_:%.+]] = krnl.load [[RES_]][] : memref<i32>
// CHECK:               krnl.store [[LOAD_RES_MEM_1_]], [[RES_1_]]{{.}}[[I_0_]]{{.}} : memref<?xi32>
// CHECK:               [[LOAD_RES_4_MEM_:%.+]] = krnl.load [[RES_4_]][] : memref<i32>
// CHECK:               krnl.store [[LOAD_RES_4_MEM_]], [[RES_]][] : memref<i32>
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<?xi32>
// CHECK:         }
}

// -----

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
  // CHECK-LABEL:  func @test_transpose_lowered_to_a_view_op_inv
  // CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x1x1x384xf32>) -> memref<384x?x1x1xf32> {
  // CHECK-NOT:       memref.reinterpret_cast
}

// -----


func.func @test_transpose_block_1_last_dim(%arg0: tensor<?x256x12x64xf32>) -> tensor<?x12x256x64xf32> {
    %1 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3] } : (tensor<?x256x12x64xf32>) -> tensor<?x12x256x64xf32>
    return %1 : tensor<?x12x256x64xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 196608 + d1 * 768 + d2 * 64)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 196608 + d1 * 64 + d2 * 16384)>
// CHECK-LABEL:  func.func @test_transpose_block_1_last_dim
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x256x12x64xf32>) -> memref<?x12x256x64xf32> {
// CHECK-DAG:       [[CST_64_:%.+]] = arith.constant 64 : i64
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x256x12x64xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x12x256x64xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x256x12x64xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_0_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 256, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 12){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2)
// CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_64_]], [[VAR_3_]], [[VAR_2_]]) : (memref<?x12x256x64xf32>, memref<?x256x12x64xf32>, i64, index, index) -> ()
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x12x256x64xf32>
// CHECK:         }
}

// -----

func.func @test_transpose_block_2_last_dims(%arg0: tensor<2x256x12x32x64xf32>) -> tensor<2x12x256x32x64xf32> {
    %1 = "onnx.Transpose"(%arg0) {perm = [0, 2, 1, 3, 4] } : (tensor<2x256x12x32x64xf32>) -> tensor<2x12x256x32x64xf32>
    return %1 : tensor<2x12x256x32x64xf32>

// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 6291456 + d1 * 2048 + d2 * 524288)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2) -> (d0 * 6291456 + d1 * 24576 + d2 * 2048)>
// CHECK-LABEL:  func.func @test_transpose_block_2_last_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x256x12x32x64xf32>) -> memref<2x12x256x32x64xf32> {
// CHECK-DAG:       [[CST_2048_:%.+]] = arith.constant 2048 : i64
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<2x12x256x32x64xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 256, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 12){
// CHECK:             [[VAR_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_2_]]([[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2)
// CHECK:             "krnl.memcpy"([[RES_]], [[PARAM_0_]], [[CST_2048_]], [[VAR_2_]], [[VAR_3_]]) : (memref<2x12x256x32x64xf32>, memref<2x256x12x32x64xf32>, i64, index, index) -> ()
// CHECK:           }
// CHECK:           return [[RES_]] : memref<2x12x256x32x64xf32>
// CHECK:         }
}

// -----

// Check lowering of onnx layout transform op and its introduction of maps for the mapped data.
// onnx-mlir-opt bibi.mlir -convert-onnx-to-krnl -canonicalize -convert-krnl-to-affine --normalize-memrefs

module {
  func.func @test_onnx_layout_transform(%arg0: tensor<5x3x32x32xf32>) -> tensor<5x3x32x32xf32> {
    %0 = "onnx.LayoutTransform"(%arg0) {target_layout = #onnx.layout<{dataLayout = "NCHW4C"}>} : (tensor<5x3x32x32xf32>) -> tensor<5x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>
    %1 = "onnx.LayoutTransform"(%0) : (tensor<5x3x32x32xf32, #onnx.layout<{dataLayout = "NCHW4C"}>>) -> tensor<5x3x32x32xf32>
    return %1 : tensor<5x3x32x32xf32>
  }

// mlir2FileCheck.py -a '["input"]'
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d0, d1 floordiv 4, d2, d3, d1 mod 4)>
// CHECK-LABEL:  func.func @test_onnx_layout_transform
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<5x3x32x32xf32>) -> memref<5x3x32x32xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<5x3x32x32xf32, [[MAP_1_]]>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 5, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 32, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 32){
// CHECK:             [[VAR_2_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_INPUT_MEM_:%.+]] = krnl.load [[INPUT_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<5x3x32x32xf32>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1, [[VAR_2_]]#2, [[VAR_2_]]#3] : memref<5x3x32x32xf32, [[MAP_1_]]>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<5x3x32x32xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 5, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 3, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to 32, [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 32){
// CHECK:             [[VAR_2_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_INPUT_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_2_1_]]#3] : memref<5x3x32x32xf32, [[MAP_1_]]>
// CHECK:             krnl.store [[LOAD_INPUT_MEM_1_]], [[RES_1_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2, [[VAR_2_1_]]#3] : memref<5x3x32x32xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<5x3x32x32xf32>
// CHECK:         }
}

// -----

#map = affine_map<(d0, d1) -> (d1, d0)>
module {
  func.func @test_shape_transform(%arg0: tensor<3x5xf32>) -> tensor<5x3xf32> {
    %0 = "onnx.ShapeTransform"(%arg0) {index_map = #map} : (tensor<3x5xf32>) -> tensor<5x3xf32>
    return %0 : tensor<5x3xf32>

// CHECK-LABEL:  func.func @test_shape_transform
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<3x5xf32>) -> memref<5x3xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<5x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 3, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<3x5xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#1, [[VAR_1_]]#0] : memref<5x3xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<5x3xf32>
// CHECK:         }
  }
}

// -----

func.func @test_dequantizelinear_i8(%arg0: tensor<4xi8>, %arg1: tensor<f32>, %arg2: tensor<i8>) -> tensor<4xf32> {
  %0 = "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<4xi8>, tensor<f32>, tensor<i8>) -> tensor<4xf32>
  return %0 : tensor<4xf32>

// CHECK-LABEL:  func.func @test_dequantizelinear_i8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4xi8>, [[PARAM_1_:%.+]]: memref<f32>, [[PARAM_2_:%.+]]: memref<i8>) -> memref<4xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<4xi8>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]][] : memref<i8>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.sitofp [[LOAD_PARAM_2_MEM_]] : i8 to f32
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.sitofp [[LOAD_PARAM_0_MEM_]] : i8 to f32
// CHECK:             [[VAR_7_:%.+]] = arith.subf [[VAR_6_]], [[VAR_5_]] : f32
// CHECK:             [[VAR_8_:%.+]] = arith.mulf [[VAR_7_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_8_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4xf32>
// CHECK:         }
}

// -----

func.func @test_dequantizelinear_ui8(%arg0: tensor<4xui8>, %arg1: tensor<f32>, %arg2: tensor<ui8>) -> tensor<4xf32> {
  %0 = "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<4xui8>, tensor<f32>, tensor<ui8>) -> tensor<4xf32>
  return %0 : tensor<4xf32>

// CHECK-LABEL:  func.func @test_dequantizelinear_ui8
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4xui8>, [[PARAM_1_:%.+]]: memref<f32>, [[PARAM_2_:%.+]]: memref<ui8>) -> memref<4xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<4xui8>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]][] : memref<ui8>
// CHECK:             [[VAR_5_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_2_MEM_]] : ui8 to i8
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.uitofp [[VAR_5_]] : i8 to f32
// CHECK-DAG:         [[VAR_7_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_]] : ui8 to i8
// CHECK:             [[VAR_8_:%.+]] = arith.uitofp [[VAR_7_]] : i8 to f32
// CHECK:             [[VAR_9_:%.+]] = arith.subf [[VAR_8_]], [[VAR_6_]] : f32
// CHECK:             [[VAR_10_:%.+]] = arith.mulf [[VAR_9_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_10_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4xf32>
// CHECK:         }
}

// -----

func.func @test_dequantizelinear_i32(%arg0: tensor<4xi32>, %arg1: tensor<f32>, %arg2: tensor<i32>) -> tensor<4xf32> {
  %0 = "onnx.DequantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<4xi32>, tensor<f32>, tensor<i32>) -> tensor<4xf32>
  return %0 : tensor<4xf32>

// CHECK-LABEL:  func.func @test_dequantizelinear_i32
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4xi32>, [[PARAM_1_:%.+]]: memref<f32>, [[PARAM_2_:%.+]]: memref<i32>) -> memref<4xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 4){
// CHECK:             [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<4xi32>
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<f32>
// CHECK-DAG:         [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]][] : memref<i32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.sitofp [[LOAD_PARAM_2_MEM_]] : i32 to f32
// CHECK-DAG:         [[VAR_6_:%.+]] = arith.sitofp [[LOAD_PARAM_0_MEM_]] : i32 to f32
// CHECK:             [[VAR_7_:%.+]] = arith.subf [[VAR_6_]], [[VAR_5_]] : f32
// CHECK:             [[VAR_8_:%.+]] = arith.mulf [[VAR_7_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             krnl.store [[VAR_8_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4xf32>
// CHECK:         }
}

// -----


func.func @test_dynamic_quantize_linear(%arg0: tensor<?x2xf32>) -> (tensor<?x2xui8>, tensor<f32>, tensor<ui8>) {
  %y, %y_scale, %y_zero_point = "onnx.DynamicQuantizeLinear"(%arg0) : (tensor<?x2xf32>) -> (tensor<?x2xui8>, tensor<f32>, tensor<ui8>)
  return %y, %y_scale, %y_zero_point:  tensor<?x2xui8>, tensor<f32>, tensor<ui8>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func @test_dynamic_quantize_linear
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x2xf32>) -> (memref<?x2xui8>, memref<f32>, memref<ui8>) {
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0x7F800000 : f32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_2_]] : memref<?x2xf32>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<?x2xui8>
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() : memref<f32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() : memref<ui8>
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[CST_2_dot_550000_]], [[RES_3_]][] : memref<f32>
// CHECK:           [[RES_4_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.store [[CST_0_dot_000000_]], [[RES_4_]][] : memref<f32>
// CHECK:           [[RES_5_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.define_loops 0
// CHECK:           krnl.iterate() with (){
// CHECK:             krnl.get_induction_var_value() : () -> ()
// CHECK:             krnl.store [[CST_0_1_]], [[RES_5_]][] : memref<f32>
// CHECK:           }
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_11_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_2_]] : memref<?x2xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_11_]]), [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2){
// CHECK:             [[VAR_33_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_33_]]#0, [[VAR_33_]]#1] : memref<?x2xf32>
// CHECK-DAG:         [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]][] : memref<f32>
// CHECK:             [[VAR_36_:%.+]] = arith.cmpf ogt, [[LOAD_RES_5_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_37_:%.+]] = arith.select [[VAR_36_]], [[LOAD_RES_5_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_37_]], [[RES_5_]][] : memref<f32>
// CHECK:           }
// CHECK:           [[RES_6_:%.+]] = memref.alloc() : memref<f32>
// CHECK:           krnl.define_loops 0
// CHECK:           krnl.iterate() with (){
// CHECK:             krnl.get_induction_var_value() : () -> ()
// CHECK:             krnl.store [[CST_0_]], [[RES_6_]][] : memref<f32>
// CHECK:           }
// CHECK-DAG:       [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK-DAG:       [[VAR_dim_13_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_2_]] : memref<?x2xf32>
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_13_]]), [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2){
// CHECK:             [[VAR_33_1_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_33_1_]]#0, [[VAR_33_1_]]#1] : memref<?x2xf32>
// CHECK-DAG:         [[LOAD_RES_5_MEM_1_:%.+]] = krnl.load [[RES_6_]][] : memref<f32>
// CHECK:             [[VAR_36_1_:%.+]] = arith.cmpf olt, [[LOAD_RES_5_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             [[VAR_37_1_:%.+]] = arith.select [[VAR_36_1_]], [[LOAD_RES_5_MEM_1_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_37_1_]], [[RES_6_]][] : memref<f32>
// CHECK:           }
// CHECK-DAG:       [[LOAD_RES_5_MEM_2_:%.+]] = krnl.load [[RES_5_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_RES_6_MEM_:%.+]] = krnl.load [[RES_6_]][] : memref<f32>
// CHECK:           [[VAR_4_:%.+]] = arith.cmpf ogt, [[LOAD_RES_5_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_5_:%.+]] = arith.select [[VAR_4_]], [[LOAD_RES_5_MEM_2_]], [[CST_0_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.cmpf olt, [[LOAD_RES_6_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[LOAD_RES_6_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_8_:%.+]] = arith.subf [[VAR_5_]], [[VAR_7_]] : f32
// CHECK:           [[VAR_9_:%.+]] = arith.divf [[VAR_8_]], [[CST_2_dot_550000_]] : f32
// CHECK:           krnl.store [[VAR_9_]], [[RES_1_]][] : memref<f32>
// CHECK:           [[VAR_10_:%.+]] = arith.subf [[CST_0_dot_000000_]], [[VAR_7_]] : f32
// CHECK:           [[VAR_11_:%.+]] = arith.divf [[VAR_10_]], [[VAR_9_]] : f32
// CHECK:           [[VAR_12_:%.+]] = arith.cmpf olt, [[VAR_11_]], [[CST_0_dot_000000_]] : f32
// CHECK:           [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[CST_0_dot_000000_]], [[VAR_11_]] : f32
// CHECK:           [[VAR_14_:%.+]] = arith.cmpf olt, [[VAR_13_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_15_:%.+]] = arith.select [[VAR_14_]], [[VAR_13_]], [[CST_2_dot_550000_]] : f32
// CHECK:           [[VAR_16_:%.+]] = math.floor [[VAR_15_]] : f32
// CHECK:           [[VAR_17_:%.+]] = arith.subf [[VAR_15_]], [[VAR_16_]] : f32
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.cmpf ogt, [[VAR_17_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_19_:%.+]] = arith.addf [[VAR_16_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.select [[VAR_18_]], [[VAR_19_]], [[VAR_16_]] : f32
// CHECK-DAG:       [[VAR_21_:%.+]] = arith.mulf [[VAR_16_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_22_:%.+]] = math.floor [[VAR_21_]] : f32
// CHECK:           [[VAR_23_:%.+]] = arith.mulf [[VAR_22_]], [[CST_2_dot_000000_]] : f32
// CHECK:           [[VAR_24_:%.+]] = arith.subf [[VAR_16_]], [[VAR_23_]] : f32
// CHECK-DAG:       [[VAR_25_:%.+]] = arith.cmpf oeq, [[VAR_24_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.addf [[VAR_16_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_27_:%.+]] = arith.select [[VAR_25_]], [[VAR_26_]], [[VAR_16_]] : f32
// CHECK-DAG:       [[VAR_28_:%.+]] = arith.cmpf oeq, [[VAR_17_]], [[CST_5_dot_000000_]] : f32
// CHECK:           [[VAR_29_:%.+]] = arith.select [[VAR_28_]], [[VAR_27_]], [[VAR_20_]] : f32
// CHECK:           [[VAR_30_:%.+]] = arith.fptoui [[VAR_29_]] : f32 to i8
// CHECK:           [[VAR_31_:%.+]] = builtin.unrealized_conversion_cast [[VAR_30_]] : i8 to ui8
// CHECK:           krnl.store [[VAR_31_]], [[RES_2_]][] : memref<ui8>
// CHECK:           [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_2_]]#1 -> [[I_5_:%.+]] = 0 to 2){
// CHECK:             [[VAR_33_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_2_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_33_2_]]#0, [[VAR_33_2_]]#1] : memref<?x2xf32>
// CHECK:             [[LOAD_RES_5_MEM_1_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_2_]], [[VAR_9_]] : f32
// CHECK:             [[VAR_36_2_:%.+]] = math.floor [[LOAD_RES_5_MEM_1_]] : f32
// CHECK:             [[VAR_37_2_:%.+]] = arith.subf [[LOAD_RES_5_MEM_1_]], [[VAR_36_2_]] : f32
// CHECK-DAG:         [[VAR_38_:%.+]] = arith.cmpf ogt, [[VAR_37_2_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_39_:%.+]] = arith.addf [[VAR_36_2_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_40_:%.+]] = arith.select [[VAR_38_]], [[VAR_39_]], [[VAR_36_2_]] : f32
// CHECK-DAG:         [[VAR_41_:%.+]] = arith.mulf [[VAR_36_2_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_42_:%.+]] = math.floor [[VAR_41_]] : f32
// CHECK:             [[VAR_43_:%.+]] = arith.mulf [[VAR_42_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_44_:%.+]] = arith.subf [[VAR_36_2_]], [[VAR_43_]] : f32
// CHECK-DAG:         [[VAR_45_:%.+]] = arith.cmpf oeq, [[VAR_44_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_46_:%.+]] = arith.addf [[VAR_36_2_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_47_:%.+]] = arith.select [[VAR_45_]], [[VAR_46_]], [[VAR_36_2_]] : f32
// CHECK-DAG:         [[VAR_48_:%.+]] = arith.cmpf oeq, [[VAR_37_2_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_49_:%.+]] = arith.select [[VAR_48_]], [[VAR_47_]], [[VAR_40_]] : f32
// CHECK:             [[VAR_50_:%.+]] = arith.addf [[VAR_49_]], [[VAR_29_]] : f32
// CHECK:             [[VAR_51_:%.+]] = arith.cmpf olt, [[VAR_50_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_52_:%.+]] = arith.select [[VAR_51_]], [[CST_0_dot_000000_]], [[VAR_50_]] : f32
// CHECK:             [[VAR_53_:%.+]] = arith.cmpf olt, [[VAR_52_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_54_:%.+]] = arith.select [[VAR_53_]], [[VAR_52_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_55_:%.+]] = arith.fptoui [[VAR_54_]] : f32 to i8
// CHECK:             [[VAR_56_:%.+]] = builtin.unrealized_conversion_cast [[VAR_55_]] : i8 to ui8
// CHECK:             krnl.store [[VAR_56_]], [[RES_]]{{.}}[[VAR_33_2_]]#0, [[VAR_33_2_]]#1] : memref<?x2xui8>
// CHECK:           }
// CHECK:           return [[RES_]], [[RES_]]_6, [[RES_]]_7 : memref<?x2xui8>, memref<f32>, memref<ui8>
// CHECK:         }
}

// -----

func.func @test_quantize_linear(%arg0: tensor<6xf32>, %arg1: tensor<f32>, %arg2: tensor<ui8>) -> tensor<6xui8> {
  %0 = "onnx.QuantizeLinear"(%arg0, %arg1, %arg2) {axis = 1 : si64} : (tensor<6xf32>, tensor<f32>, tensor<ui8>) -> tensor<6xui8>
  return %0 : tensor<6xui8>

// CHECK-LABEL:  func.func @test_quantize_linear
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<6xf32>, [[PARAM_1_:%.+]]: memref<f32>, [[PARAM_2_:%.+]]: memref<ui8>) -> memref<6xui8> {
// CHECK-DAG:       [[CST_5_dot_000000_:%.+]] = arith.constant 5.000000e-01 : f32
// CHECK-DAG:       [[CST_2_dot_000000_:%.+]] = arith.constant 2.000000e+00 : f32
// CHECK-DAG:       [[CST_1_dot_000000_:%.+]] = arith.constant 1.000000e+00 : f32
// CHECK-DAG:       [[CST_0_dot_000000_:%.+]] = arith.constant 0.000000e+00 : f32
// CHECK-DAG:       [[CST_2_dot_550000_:%.+]] = arith.constant 2.550000e+02 : f32
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<6xui8>
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<f32>
// CHECK-DAG:       [[LOAD_PARAM_2_MEM_:%.+]] = krnl.load [[PARAM_2_]][] : memref<ui8>
// CHECK:           [[VAR_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_2_MEM_]] : ui8 to i8
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.uitofp [[VAR_2_]] : i8 to f32
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 6){
// CHECK:             [[VAR_5_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_5_]]{{.}} : memref<6xf32>
// CHECK:             [[VAR_7_:%.+]] = arith.divf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_8_:%.+]] = math.floor [[VAR_7_]] : f32
// CHECK:             [[VAR_9_:%.+]] = arith.subf [[VAR_7_]], [[VAR_8_]] : f32
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.cmpf ogt, [[VAR_9_]], [[CST_5_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_11_:%.+]] = arith.addf [[VAR_8_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.select [[VAR_10_]], [[VAR_11_]], [[VAR_8_]] : f32
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.mulf [[VAR_8_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_14_:%.+]] = math.floor [[VAR_13_]] : f32
// CHECK:             [[VAR_15_:%.+]] = arith.mulf [[VAR_14_]], [[CST_2_dot_000000_]] : f32
// CHECK:             [[VAR_16_:%.+]] = arith.subf [[VAR_8_]], [[VAR_15_]] : f32
// CHECK-DAG:         [[VAR_17_:%.+]] = arith.cmpf oeq, [[VAR_16_]], [[CST_1_dot_000000_]] : f32
// CHECK-DAG:         [[VAR_18_:%.+]] = arith.addf [[VAR_8_]], [[CST_1_dot_000000_]] : f32
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.select [[VAR_17_]], [[VAR_18_]], [[VAR_8_]] : f32
// CHECK-DAG:         [[VAR_20_:%.+]] = arith.cmpf oeq, [[VAR_9_]], [[CST_5_dot_000000_]] : f32
// CHECK:             [[VAR_21_:%.+]] = arith.select [[VAR_20_]], [[VAR_19_]], [[VAR_12_]] : f32
// CHECK:             [[VAR_22_:%.+]] = arith.addf [[VAR_21_]], [[VAR_3_]] : f32
// CHECK:             [[VAR_23_:%.+]] = arith.cmpf olt, [[VAR_22_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_24_:%.+]] = arith.select [[VAR_23_]], [[CST_0_dot_000000_]], [[VAR_22_]] : f32
// CHECK:             [[VAR_25_:%.+]] = arith.cmpf olt, [[VAR_24_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_26_:%.+]] = arith.select [[VAR_25_]], [[VAR_24_]], [[CST_2_dot_550000_]] : f32
// CHECK:             [[VAR_27_:%.+]] = arith.fptoui [[VAR_26_]] : f32 to i8
// CHECK:             [[VAR_28_:%.+]] = builtin.unrealized_conversion_cast [[VAR_27_]] : i8 to ui8
// CHECK:             krnl.store [[VAR_28_]], [[RES_]]{{.}}[[VAR_5_]]{{.}} : memref<6xui8>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<6xui8>
// CHECK:         }
}

// -----

func.func @roberta_partial_simd_1dim_v1(%arg0: tensor<?x?x768xf32>, %arg1: tensor<?x?x768xf32>) -> tensor<?x?x768xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<?x?x768xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5)>
// CHECK-LABEL:  func.func @roberta_partial_simd_1dim_v1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>, [[PARAM_1_:%.+]]: memref<?x?x768xf32>) -> memref<?x?x768xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
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

func.func @roberta_partial_simd_1dim_v2(%arg0: tensor<?x?x768xf32>, %arg1: tensor<768xf32>) -> tensor<?x?x768xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<768xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func @roberta_partial_simd_1dim_v2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>, [[PARAM_1_:%.+]]: memref<768xf32>) -> memref<?x?x768xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
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

func.func @roberta_partial_simd_1dim_scalar(%arg0: tensor<?x?x768xf32>, %arg1: tensor<1xf32>) -> tensor<?x?x768xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x768xf32>, tensor<1xf32>) -> tensor<?x?x768xf32>
    return %0 : tensor<?x?x768xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK-LABEL:  func.func @roberta_partial_simd_1dim_scalar
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x768xf32>, [[PARAM_1_:%.+]]: memref<1xf32>) -> memref<?x?x768xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
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

func.func @roberta_partial_simd_2dim_v1(%arg0: tensor<?x?x96x8xf32>, %arg1: tensor<?x?x96x8xf32>) -> tensor<?x?x96x8xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x96x8xf32>, tensor<?x?x96x8xf32>) -> tensor<?x?x96x8xf32>
    return %0 : tensor<?x?x96x8xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5)>
// CHECK-LABEL:  func.func @roberta_partial_simd_2dim_v1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x96x8xf32>, [[PARAM_1_:%.+]]: memref<?x?x96x8xf32>) -> memref<?x?x96x8xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
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

func.func @roberta_partial_simd_2dim_v2(%arg0: tensor<?x2x96x8xf32>, %arg1: tensor<?x?x96x8xf32>) -> tensor<?x?x96x8xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x2x96x8xf32>, tensor<?x?x96x8xf32>) -> tensor<?x?x96x8xf32>
    return %0 : tensor<?x?x96x8xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-LABEL:  func.func @roberta_partial_simd_2dim_v2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x2x96x8xf32>, [[PARAM_1_:%.+]]: memref<?x?x96x8xf32>) -> memref<?x2x96x8xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
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

func.func @roberta_partial_simd_2dim_v3(%arg0: tensor<?x2x96x8xf32>, %arg1: tensor<?x1x96x8xf32>) -> tensor<?x?x96x8xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x2x96x8xf32>, tensor<?x1x96x8xf32>) -> tensor<?x?x96x8xf32>
    return %0 : tensor<?x?x96x8xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-LABEL:  func.func @roberta_partial_simd_2dim_v3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x2x96x8xf32>, [[PARAM_1_:%.+]]: memref<?x1x96x8xf32>) -> memref<?x2x96x8xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
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

func.func @roberta_partial_simd_2dim_not_0_mod_vl(%arg0: tensor<?x?x95x7xf32>, %arg1: tensor<?x?x95x7xf32>) -> tensor<?x?x95x7xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x?x95x7xf32>, tensor<?x?x95x7xf32>) -> tensor<?x?x95x7xf32>
    return %0 : tensor<?x?x95x7xf32>

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0, s1] -> (s1, s0)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0, d1, d2, d3, d4) -> (d4)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0, d1, d2, d3, d4, d5) -> (d5)>
// CHECK-LABEL:  func.func @roberta_partial_simd_2dim_not_0_mod_vl
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x95x7xf32>, [[PARAM_1_:%.+]]: memref<?x?x95x7xf32>) -> memref<?x?x95x7xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
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

func.func @test_matmulinteger_per_tensor(%arg0: tensor<16x32xui8>, %arg1: tensor<32x64xui8>, %arg2: tensor<1xui8>, %arg3: tensor<1xui8>) -> tensor<16x64xi32> {
  %0 = "onnx.MatMulInteger"(%arg0, %arg1, %arg2, %arg3) : (tensor<16x32xui8>, tensor<32x64xui8>, tensor<1xui8>, tensor<1xui8>) -> tensor<16x64xi32>
  return %0 : tensor<16x64xi32>

// CHECK-LABEL:  func.func @test_matmulinteger_per_tensor
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32xui8>, [[PARAM_1_:%.+]]: memref<32x64xui8>, [[PARAM_2_:%.+]]: memref<1xui8>, [[PARAM_3_:%.+]]: memref<1xui8>) -> memref<16x64xi32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_7_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1] : memref<16x32xui8>
// CHECK:             [[VAR_9_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_]] : ui8 to i8
// CHECK:             [[VAR_10_:%.+]] = arith.extui [[VAR_9_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_]], [[RES_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 1){
// CHECK:             [[VAR_7_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_7_1_]]{{.}} : memref<1xui8>
// CHECK:             [[VAR_9_1_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_]] : ui8 to i8
// CHECK:             [[VAR_10_1_:%.+]] = arith.extui [[VAR_9_1_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_1_]], [[RES_1_]]{{.}}[[VAR_7_1_]]{{.}} : memref<1xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 16, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 32){
// CHECK:             [[VAR_7_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1] : memref<16x32xi32>
// CHECK-DAG:         [[VAR_9_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_10_2_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_]], [[VAR_9_1_]] : i32
// CHECK:             krnl.store [[VAR_10_2_]], [[RES_2_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 32, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 64){
// CHECK:             [[VAR_7_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1] : memref<32x64xui8>
// CHECK:             [[VAR_9_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_1_]] : ui8 to i8
// CHECK:             [[VAR_10_3_:%.+]] = arith.extui [[VAR_9_2_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_3_]], [[RES_3_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 1){
// CHECK:             [[VAR_7_4_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[VAR_7_4_]]{{.}} : memref<1xui8>
// CHECK:             [[VAR_9_3_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_1_]] : ui8 to i8
// CHECK:             [[VAR_10_4_:%.+]] = arith.extui [[VAR_9_3_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_4_]], [[RES_4_]]{{.}}[[VAR_7_4_]]{{.}} : memref<1xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_8_:%.+]] = 0 to 32, [[LOOP_5_]]#1 -> [[I_9_:%.+]] = 0 to 64){
// CHECK:             [[VAR_7_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_7_5_]]#0, [[VAR_7_5_]]#1] : memref<32x64xi32>
// CHECK-DAG:         [[VAR_9_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_10_5_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_1_1_]], [[VAR_9_3_]] : i32
// CHECK:             krnl.store [[VAR_10_5_]], [[RES_5_]]{{.}}[[VAR_7_5_]]#0, [[VAR_7_5_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[LOOP_6_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloca() : memref<i32>
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = 0 to 16, [[LOOP_6_]]#1 -> [[I_11_:%.+]] = 0 to 64, [[LOOP_6_]]#2 -> [[I_12_:%.+]] = 0 to 32){
// CHECK:             [[VAR_7_6_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_]], [[RES_7_]][] : memref<i32>
// CHECK:             krnl.iterate([[LOOP_6_]]#2) with (){
// CHECK:               [[VAR_9_4_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]#2) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_10_5_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_7_6_]]#0, [[VAR_9_4_]]{{.}} : memref<16x32xi32>
// CHECK-DAG:           [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_9_4_]], [[VAR_7_6_]]#1] : memref<32x64xi32>
// CHECK-DAG:           [[LOAD_RES_7_MEM_:%.+]] = krnl.load [[RES_7_]][] : memref<i32>
// CHECK:               [[VAR_13_:%.+]] = arith.muli [[VAR_10_5_]], [[LOAD_RES_5_MEM_]] : i32
// CHECK:               [[VAR_14_:%.+]] = arith.addi [[LOAD_RES_7_MEM_]], [[VAR_13_]] : i32
// CHECK:               krnl.store [[VAR_14_]], [[RES_7_]][] : memref<i32>
// CHECK:             }
// CHECK:             [[LOAD_RES_7_MEM_1_:%.+]] = krnl.load [[RES_7_]][] : memref<i32>
// CHECK:             krnl.store [[LOAD_RES_7_MEM_1_]], [[RES_6_]]{{.}}[[VAR_7_6_]]#0, [[VAR_7_6_]]#1] : memref<16x64xi32>
// CHECK:           }
// CHECK:           return [[RES_6_]] : memref<16x64xi32>
// CHECK:         }
}

// -----

func.func @test_matmulinteger_per_row_a(%arg0: tensor<16x32xui8>, %arg1: tensor<32x64xui8>, %arg2: tensor<16xui8>, %arg3: tensor<1xui8>) -> tensor<16x64xi32> {
  %0 = "onnx.MatMulInteger"(%arg0, %arg1, %arg2, %arg3) : (tensor<16x32xui8>, tensor<32x64xui8>, tensor<16xui8>, tensor<1xui8>) -> tensor<16x64xi32>
  return %0 : tensor<16x64xi32>

// CHECK-LABEL:  func.func @test_matmulinteger_per_row
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<16x32xui8>, [[PARAM_1_:%.+]]: memref<32x64xui8>, [[PARAM_2_:%.+]]: memref<16xui8>, [[PARAM_3_:%.+]]: memref<1xui8>) -> memref<16x64xi32> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i32
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 16, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 32){
// CHECK:             [[VAR_7_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1] : memref<16x32xui8>
// CHECK:             [[VAR_9_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_]] : ui8 to i8
// CHECK:             [[VAR_10_:%.+]] = arith.extui [[VAR_9_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_]], [[RES_]]{{.}}[[VAR_7_]]#0, [[VAR_7_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<16xi32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_2_:%.+]] = 0 to 16){
// CHECK:             [[VAR_7_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_2_]]{{.}}[[VAR_7_1_]]{{.}} : memref<16xui8>
// CHECK:             [[VAR_9_1_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_]] : ui8 to i8
// CHECK:             [[VAR_10_1_:%.+]] = arith.extui [[VAR_9_1_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_1_]], [[RES_1_]]{{.}}[[VAR_7_1_]]{{.}} : memref<16xi32>
// CHECK:           }
// CHECK-DAG:       [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[RES_1_]] to offset: [0], sizes: [16, 1], strides: [1, 1] : memref<16xi32> to memref<16x1xi32>
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<16x32xi32>
// CHECK-DAG:       [[LOOP_2_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1) with ([[LOOP_2_]]#0 -> [[I_3_:%.+]] = 0 to 16, [[LOOP_2_]]#1 -> [[I_4_:%.+]] = 0 to 32){
// CHECK:             [[VAR_7_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1] : memref<16x32xi32>
// CHECK-DAG:         [[VAR_9_1_:%.+]] = krnl.load [[VAR_reinterpret_cast_]]{{.}}[[VAR_7_2_]]#0, [[CST_0_1_]]{{.}} : memref<16x1xi32>
// CHECK:             [[VAR_10_2_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_]], [[VAR_9_1_]] : i32
// CHECK:             krnl.store [[VAR_10_2_]], [[RES_2_]]{{.}}[[VAR_7_2_]]#0, [[VAR_7_2_]]#1] : memref<16x32xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_3_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_3_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_3_]]#0, [[LOOP_3_]]#1) with ([[LOOP_3_]]#0 -> [[I_5_:%.+]] = 0 to 32, [[LOOP_3_]]#1 -> [[I_6_:%.+]] = 0 to 64){
// CHECK:             [[VAR_7_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_3_]]#0, [[LOOP_3_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1] : memref<32x64xui8>
// CHECK:             [[VAR_9_2_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_1_]] : ui8 to i8
// CHECK:             [[VAR_10_3_:%.+]] = arith.extui [[VAR_9_2_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_3_]], [[RES_3_]]{{.}}[[VAR_7_3_]]#0, [[VAR_7_3_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_4_:%.+]] = memref.alloc() {{.*}}: memref<1xi32>
// CHECK-DAG:       [[LOOP_4_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_4_]]) with ([[LOOP_4_]] -> [[I_7_:%.+]] = 0 to 1){
// CHECK:             [[VAR_7_4_:%.+]] = krnl.get_induction_var_value([[LOOP_4_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_1_:%.+]] = krnl.load [[PARAM_3_]]{{.}}[[VAR_7_4_]]{{.}} : memref<1xui8>
// CHECK:             [[VAR_9_3_:%.+]] = builtin.unrealized_conversion_cast [[LOAD_PARAM_0_MEM_1_1_]] : ui8 to i8
// CHECK:             [[VAR_10_4_:%.+]] = arith.extui [[VAR_9_3_]] : i8 to i32
// CHECK:             krnl.store [[VAR_10_4_]], [[RES_4_]]{{.}}[[VAR_7_4_]]{{.}} : memref<1xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_5_:%.+]] = memref.alloc() {{.*}}: memref<32x64xi32>
// CHECK-DAG:       [[LOOP_5_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_5_]]#0, [[LOOP_5_]]#1) with ([[LOOP_5_]]#0 -> [[I_8_:%.+]] = 0 to 32, [[LOOP_5_]]#1 -> [[I_9_:%.+]] = 0 to 64){
// CHECK:             [[VAR_7_5_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_5_]]#0, [[LOOP_5_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_1_1_:%.+]] = krnl.load [[RES_3_]]{{.}}[[VAR_7_5_]]#0, [[VAR_7_5_]]#1] : memref<32x64xi32>
// CHECK-DAG:         [[VAR_9_3_:%.+]] = krnl.load [[RES_4_]]{{.}}[[CST_0_1_]]{{.}} : memref<1xi32>
// CHECK:             [[VAR_10_5_:%.+]] = arith.subi [[LOAD_PARAM_0_MEM_1_1_1_]], [[VAR_9_3_]] : i32
// CHECK:             krnl.store [[VAR_10_5_]], [[RES_5_]]{{.}}[[VAR_7_5_]]#0, [[VAR_7_5_]]#1] : memref<32x64xi32>
// CHECK:           }
// CHECK-DAG:       [[RES_6_:%.+]] = memref.alloc() {{.*}}: memref<16x64xi32>
// CHECK-DAG:       [[LOOP_6_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[RES_7_:%.+]] = memref.alloca() : memref<i32>
// CHECK:           krnl.iterate([[LOOP_6_]]#0, [[LOOP_6_]]#1) with ([[LOOP_6_]]#0 -> [[I_10_:%.+]] = 0 to 16, [[LOOP_6_]]#1 -> [[I_11_:%.+]] = 0 to 64, [[LOOP_6_]]#2 -> [[I_12_:%.+]] = 0 to 32){
// CHECK:             [[VAR_7_6_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_6_]]#0, [[LOOP_6_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_]], [[RES_7_]][] : memref<i32>
// CHECK:             krnl.iterate([[LOOP_6_]]#2) with (){
// CHECK:               [[VAR_9_4_:%.+]] = krnl.get_induction_var_value([[LOOP_6_]]#2) : (!krnl.loop) -> index
// CHECK-DAG:           [[VAR_10_5_:%.+]] = krnl.load [[RES_2_]]{{.}}[[VAR_7_6_]]#0, [[VAR_9_4_]]{{.}} : memref<16x32xi32>
// CHECK-DAG:           [[LOAD_RES_5_MEM_:%.+]] = krnl.load [[RES_5_]]{{.}}[[VAR_9_4_]], [[VAR_7_6_]]#1] : memref<32x64xi32>
// CHECK-DAG:           [[LOAD_RES_7_MEM_:%.+]] = krnl.load [[RES_7_]][] : memref<i32>
// CHECK:               [[VAR_13_:%.+]] = arith.muli [[VAR_10_5_]], [[LOAD_RES_5_MEM_]] : i32
// CHECK:               [[VAR_14_:%.+]] = arith.addi [[LOAD_RES_7_MEM_]], [[VAR_13_]] : i32
// CHECK:               krnl.store [[VAR_14_]], [[RES_7_]][] : memref<i32>
// CHECK:             }
// CHECK:             [[LOAD_RES_7_MEM_1_:%.+]] = krnl.load [[RES_7_]][] : memref<i32>
// CHECK:             krnl.store [[LOAD_RES_7_MEM_1_]], [[RES_6_]]{{.}}[[VAR_7_6_]]#0, [[VAR_7_6_]]#1] : memref<16x64xi32>
// CHECK:           }
// CHECK:           return [[RES_6_]] : memref<16x64xi32>
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

func.func @add_partial_splat(%arg0: tensor<2x3x4x5xf32>, %arg1: tensor<3x1x1xf32>) -> tensor<*xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<2x3x4x5xf32>, tensor<3x1x1xf32>) -> tensor<*xf32>
    return %0 : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @add_partial_splat
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x3x4x5xf32>, [[PARAM_1_:%.+]]: memref<3x1x1xf32>) -> memref<2x3x4x5xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
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
// CHECK:             [[VAR_3_:%.+]] = arith.cmpf oge, [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[LOAD_PARAM_0_MEM_]], [[CST_0_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_4_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
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
// CHECK:             [[VAR_4_:%.+]] = arith.cmpf ogt, [[VAR_3_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_5_:%.+]] = arith.select [[VAR_4_]], [[VAR_3_]], [[CST_0_dot_000000_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.cmpf olt, [[VAR_5_]], [[CST_1_dot_000000_]] : f32
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_6_]], [[VAR_5_]], [[CST_1_dot_000000_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1] : memref<?x10xf32>
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

// -----

func.func @test_trilu_lower(%arg0: tensor<4x5xi64>, %arg1: tensor<i64>) -> tensor<4x5xi64> {
  %0 = "onnx.Trilu"(%arg0, %arg1) {upper = 0 : si64} : (tensor<4x5xi64>, tensor<i64>) -> tensor<4x5xi64>
  return %0 : tensor<4x5xi64>

// CHECK-LABEL:  func.func @test_trilu_lower
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x5xi64>, [[PARAM_1_:%.+]]: memref<i64>) -> memref<4x5xi64> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<i64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x5xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 4, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_4_:%.+]] = arith.addi [[VAR_1_]], [[VAR_3_]]#0 : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi slt, [[VAR_4_]], [[VAR_3_]]#1 : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<4x5xi64>
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_5_]], [[CST_0_]], [[LOAD_PARAM_0_MEM_]] : i64
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<4x5xi64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x5xi64>
// CHECK:         }
}

// -----

func.func @test_trilu_upper(%arg0: tensor<4x5xi64>, %arg1: tensor<i64>) -> tensor<4x5xi64> {
  %0 = "onnx.Trilu"(%arg0, %arg1) {upper = 1 : si64} : (tensor<4x5xi64>, tensor<i64>) -> tensor<4x5xi64>
  return %0 : tensor<4x5xi64>

// CHECK-LABEL:  func.func @test_trilu_upper
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<4x5xi64>, [[PARAM_1_:%.+]]: memref<i64>) -> memref<4x5xi64> {
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : i64
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]][] : memref<i64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<4x5xi64>
// CHECK-DAG:       [[LOOP_0_:%.+]]:2 = krnl.define_loops 2
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 4, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 5){
// CHECK:             [[VAR_3_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             [[VAR_4_:%.+]] = arith.addi [[VAR_1_]], [[VAR_3_]]#0 : index
// CHECK-DAG:         [[VAR_5_:%.+]] = arith.cmpi sgt, [[VAR_4_]], [[VAR_3_]]#1 : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<4x5xi64>
// CHECK:             [[VAR_7_:%.+]] = arith.select [[VAR_5_]], [[CST_0_]], [[LOAD_PARAM_0_MEM_]] : i64
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_3_]]#0, [[VAR_3_]]#1] : memref<4x5xi64>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<4x5xi64>
// CHECK:         }
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
// CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<3x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_4_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.cmpf ogt, [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
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
// CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<3x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_4_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.cmpf olt, [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2] : memref<3x2xf32>
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
// CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<3x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_4_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2] : memref<3x2xf32>
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
// CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<3x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_4_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.addf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2] : memref<3x2xf32>
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
// CHECK:             [[VAR_3_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]]{{.}}[[VAR_3_1_]]#0, [[VAR_3_1_]]#1, [[VAR_3_1_]]#2] : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_2_]]#2 -> [[I_6_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_3_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_5_1_:%.+]] = arith.cmpi eq, [[LOAD_PARAM_1_MEM_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_6_1_:%.+]] = arith.select [[VAR_5_1_]], [[CST_0_1_]], [[VAR_3_2_]]#0 : index
// CHECK-DAG:         [[VAR_7_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_8_1_:%.+]] = arith.cmpi eq, [[VAR_7_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_9_:%.+]] = arith.select [[VAR_8_1_]], [[CST_0_1_]], [[VAR_3_2_]]#1 : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_11_:%.+]] = arith.cmpi eq, [[LOAD_RES_MEM_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_0_1_]], [[VAR_3_2_]]#2 : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_2_]]#0, [[VAR_3_2_]]#1, [[VAR_3_2_]]#2] : memref<3x2x2xf32>
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_6_1_]], [[VAR_9_]], [[VAR_12_]]{{.}} : memref<3x1x2xf32>
// CHECK:             [[VAR_15_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_15_]], [[RES_1_]]{{.}}[[VAR_6_1_]], [[VAR_9_]], [[VAR_12_]]{{.}} : memref<3x1x2xf32>
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
// CHECK:           [[VAR_0_:%.+]] = arith.cmpi eq, [[VAR_dim_]], [[CST_0_1_]] : index
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
// CHECK:             [[VAR_4_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK:             krnl.store [[CST_0_dot_000000_]], [[RES_1_]]{{.}}[[VAR_4_1_]]#0, [[VAR_4_1_]]#1, [[VAR_4_1_]]#2] : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_2_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) with ([[LOOP_2_]]#0 -> [[I_4_:%.+]] = 0 to 3, [[LOOP_2_]]#1 -> [[I_5_:%.+]] = 0 to 2, [[LOOP_2_]]#2 -> [[I_6_:%.+]] = 0 to 2){
// CHECK-DAG:         [[VAR_4_2_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_0_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_6_1_:%.+]] = arith.cmpi eq, [[LOAD_PARAM_1_MEM_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_7_1_:%.+]] = arith.select [[VAR_6_1_]], [[CST_0_1_]], [[VAR_4_2_]]#0 : index
// CHECK-DAG:         [[VAR_8_1_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_1_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_9_1_:%.+]] = arith.cmpi eq, [[VAR_8_1_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_10_:%.+]] = arith.select [[VAR_9_1_]], [[CST_0_1_]], [[VAR_4_2_]]#1 : index
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[CST_2_]]{{.}} : memref<3xi1>
// CHECK:             [[VAR_12_:%.+]] = arith.cmpi eq, [[LOAD_RES_MEM_]], [[VAR_true_]] : i1
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.select [[VAR_12_]], [[CST_0_1_]], [[VAR_4_2_]]#2 : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_4_2_]]#0, [[VAR_4_2_]]#1, [[VAR_4_2_]]#2] : memref<3x2x2xf32>
// CHECK:             [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_7_1_]], [[VAR_10_]], [[VAR_13_]]{{.}} : memref<3x1x2xf32>
// CHECK:             [[VAR_16_:%.+]] = arith.addf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_16_]], [[RES_1_]]{{.}}[[VAR_7_1_]], [[VAR_10_]], [[VAR_13_]]{{.}} : memref<3x1x2xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<3x1x2xf32>
// CHECK:         }
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
// CHECK:             [[VAR_2_:%.+]]:2 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1) : (!krnl.loop, !krnl.loop) -> (index, index)
// CHECK:             krnl.store [[CST_1_dot_000000_]], [[RES_]]{{.}}[[VAR_2_]]#0, [[VAR_2_]]#1] : memref<3x2xf32>
// CHECK:           }
// CHECK:           [[LOOP_1_:%.+]]:3 = krnl.define_loops 3
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) with ([[LOOP_1_]]#0 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_1_]]#1 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_1_]]#2 -> [[I_4_:%.+]] = 0 to 2){
// CHECK:             [[VAR_2_1_:%.+]]:3 = krnl.get_induction_var_value([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#1, [[VAR_2_1_]]#2] : memref<3x2x2xf32>
// CHECK-DAG:         [[LOAD_RES_MEM_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2] : memref<3x2xf32>
// CHECK:             [[VAR_5_:%.+]] = arith.mulf [[LOAD_RES_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:             krnl.store [[VAR_5_]], [[RES_]]{{.}}[[VAR_2_1_]]#0, [[VAR_2_1_]]#2] : memref<3x2xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<3x2xf32>
// CHECK:         }
}

