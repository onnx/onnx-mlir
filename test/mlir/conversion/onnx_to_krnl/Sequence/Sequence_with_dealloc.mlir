// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl --buffer-deallocation-pipeline  %s -split-input-file | FileCheck %s

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
// CHECK:           memref.dealloc [[VAR_2_]] : memref<?xmemref<?x4x5xf32>>
// CHECK:           memref.dealloc [[VAR_16_]] : memref<?x4x5xf32>
// CHECK:           return [[RES_]] : memref<3xi64>
// CHECK:         }
}
