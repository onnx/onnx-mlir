// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func @test_sequence_erase(%arg0: !onnx.Seq<tensor<?x4x5xf32>>) -> tensor<3xi64>  {
  %0 = onnx.Constant {value = dense<0> : tensor<i64>} : tensor<i64>
  %7 = "onnx.SequenceErase"(%arg0, %0) : (!onnx.Seq<tensor<?x4x5xf32>>, tensor<i64>) -> !onnx.Seq<tensor<?x4x5xf32>>
  %4 = "onnx.SequenceAt"(%7, %0) : (!onnx.Seq<tensor<?x4x5xf32>>, tensor<i64>) -> tensor<?x4x5xf32>
  %5 = "onnx.Shape"(%4) : (tensor<?x4x5xf32>) -> tensor<3xi64>
  return %5 : tensor<3xi64>
// This test case generates muliple maps, which differs from system to system.
// The check for #map is removed manually.
// CHECK-LABEL:  func.func @test_sequence_erase
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?xmemref<?x4x5xf32>>) -> memref<3xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<0> : tensor<i64>} : () -> memref<i64>
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_1_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?xmemref<?x4x5xf32>>
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_2_:%.+]] = affine.apply [[MAP0:#map.*]](){{.}}[[VAR_1_]]{{.}}
// CHECK-DAG:       [[VAR_3_:%.+]] = "krnl.seqalloc"([[VAR_2_]]) : (index) -> memref<?xmemref<?x4x5xf32>>
// CHECK-DAG:       [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<i64>
// CHECK:           [[VAR_5_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_]] : i64 to index
// CHECK-DAG:       [[VAR_6_:%.+]] = affine.apply [[MAP1:#map.*]](){{.}}[[VAR_1_]], [[VAR_5_]]{{.}}
// CHECK-DAG:       [[VAR_c0_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi slt, [[VAR_5_]], [[VAR_c0_0_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[VAR_6_]], [[VAR_5_]] : index
// CHECK-DAG:       [[VAR_c0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to [[VAR_8_]]){
// CHECK:             [[VAR_24_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_24_]]{{.}} : memref<?xmemref<?x4x5xf32>>
// CHECK:             "krnl.seqstore"([[LOAD_PARAM_0_MEM_]], [[VAR_3_]], [[VAR_8_]]) : (memref<?x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()
// CHECK:           }
// CHECK:           [[VAR_c1_2_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.addi [[VAR_8_]], [[VAR_c1_2_]] : index
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = [[VAR_10_]] to [[MAP2:#map.*]](){{.}}[[VAR_1_]], [[VAR_5_]]{{.}}){
// CHECK:             [[VAR_24_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_24_1_]]{{.}} : memref<?xmemref<?x4x5xf32>>
// CHECK-DAG:         [[VAR_c1_8_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_26_:%.+]] = arith.subi [[VAR_24_1_]], [[VAR_c1_8_]] : index
// CHECK:             "krnl.seqstore"([[LOAD_PARAM_0_MEM_1_]], [[VAR_3_]], [[VAR_26_]]) : (memref<?x4x5xf32>, memref<?xmemref<?x4x5xf32>>, index) -> ()
// CHECK:           }
// CHECK:           [[VAR_c0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_12_:%.+]] = memref.dim [[VAR_3_]], [[VAR_c0_3_]] : memref<?xmemref<?x4x5xf32>>
// CHECK-DAG:       [[LOAD_VAR_0_MEM_1_:%.+]] = krnl.load [[VAR_0_]][] : memref<i64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_14_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_1_]] : i64 to index
// CHECK-DAG:       [[VAR_c0_4_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.cmpi slt, [[VAR_14_]], [[VAR_c0_4_]] : index
// CHECK-DAG:       [[VAR_16_:%.+]] = affine.apply [[MAP3:#map.*]](){{.}}{{%.*}}, {{%.*}}{{.}}
// CHECK:           [[VAR_17_:%.+]] = arith.select [[VAR_15_]], [[VAR_16_]], [[VAR_14_]] : index
// CHECK-DAG:       [[VAR_18_:%.+]] = "krnl.seqextract"([[VAR_3_]], [[VAR_17_]]) {copy = 1 : ui1} : (memref<?xmemref<?x4x5xf32>>, index) -> memref<?x4x5xf32>
// CHECK-DAG:       [[VAR_c3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<3xi64>
// CHECK-DAG:       [[VAR_c0_5_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_20_:%.+]] = memref.dim [[VAR_18_]], [[VAR_c0_5_]] : memref<?x4x5xf32>
// CHECK-DAG:       [[VAR_c4_:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[VAR_c5_:%.+]] = arith.constant 5 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_21_:%.+]] = arith.index_cast [[VAR_20_]] : index to i64
// CHECK-DAG:       [[VAR_c0_6_:%.+]] = arith.constant 0 : index
// CHECK:           krnl.store [[VAR_21_]], [[RES_]]{{.}}[[VAR_c0_6_]]{{.}} : memref<3xi64>
// CHECK-DAG:       [[VAR_22_:%.+]] = arith.index_cast [[VAR_c4_]] : index to i64
// CHECK-DAG:       [[VAR_c1_7_:%.+]] = arith.constant 1 : index
// CHECK:           krnl.store [[VAR_22_]], [[RES_]]{{.}}[[VAR_c1_7_]]{{.}} : memref<3xi64>
// CHECK-DAG:       [[VAR_23_:%.+]] = arith.index_cast [[VAR_c5_]] : index to i64
// CHECK-DAG:       [[VAR_c2_:%.+]] = arith.constant 2 : index
// CHECK:           krnl.store [[VAR_23_]], [[RES_]]{{.}}[[VAR_c2_]]{{.}} : memref<3xi64>
// CHECK:           return [[RES_]] : memref<3xi64>
// CHECK:         }
}
