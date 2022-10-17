// RUN: onnx-mlir-opt -O3 --shape-inference --convert-onnx-to-krnl --buffer-deallocation  %s -split-input-file | FileCheck %s

func.func @test_sequence_ops2(%arg0: tensor<?xf32>) -> tensor<1xi64>  {
  %0 = "onnx.Constant"() {value = dense<0> : tensor<1xi64>} : () -> tensor<i64>
  %1 = "onnx.SequenceEmpty"() : () -> !onnx.Seq<tensor<?xf32>>
  %2 = "onnx.NoValue"() {value} : () -> none
  %3 = "onnx.Add"(%arg0, %arg0) : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
  %6 = "onnx.SequenceInsert"(%1, %3, %2) : (!onnx.Seq<tensor<?xf32>>, tensor<?xf32>, none) -> !onnx.Seq<tensor<?xf32>>
  %4 = "onnx.SequenceAt"(%6, %0) : (!onnx.Seq<tensor<?xf32>>, tensor<i64>) -> tensor<?xf32>
  %5 = "onnx.Shape"(%4) : (tensor<?xf32>) -> tensor<1xi64>
  return %5 : tensor<1xi64>
// CHECK-DAG: #map0 = affine_map<(d0, d1) -> (d0, d1)>
// CHECK-DAG: #map1 = affine_map<(d0) -> (d0)>
// CHECK-DAG: #map2 = affine_map<()[s0] -> (s0 + 1)>
// CHECK-LABEL:  func.func @test_sequence_ops2
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?xf32>) -> memref<1xi64> {
// CHECK-DAG:       [[VAR_0_:%.+]] = "krnl.global"() {name = {{.*}}, shape = [], value = dense<0> : tensor<1xi64>} : () -> memref<i64>
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<0xmemref<?xf32>>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.NoValue"() {value} : () -> none
// CHECK-DAG:       [[VAR_c1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_3_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_]] : memref<?xf32>
// CHECK-DAG:       [[VAR_c0_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_4_:%.+]] = memref.dim [[PARAM_0_]], [[VAR_c0_0_]] : memref<?xf32>
// CHECK:           [[VAR_5_:%.+]] = affine.max #map0([[VAR_3_]], [[VAR_4_]])
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc([[VAR_5_]]) {{.*}}: memref<?xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[VAR_c0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c0_2_:%.+]] = arith.constant 0 : index
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to #map1([[VAR_5_]])){
// CHECK-DAG:         [[VAR_18_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[VAR_c1_13_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_19_:%.+]] = arith.cmpi sgt, [[VAR_3_]], [[VAR_c1_13_]] : index
// CHECK-DAG:         [[VAR_c0_14_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_20_:%.+]] = arith.select [[VAR_19_]], [[VAR_18_]], [[VAR_c0_14_]] : index
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_20_]]{{.}} : memref<?xf32>
// CHECK-DAG:         [[VAR_c1_15_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_22_:%.+]] = arith.cmpi sgt, [[VAR_4_]], [[VAR_c1_15_]] : index
// CHECK-DAG:         [[VAR_c0_16_:%.+]] = arith.constant 0 : index
// CHECK:             [[VAR_23_:%.+]] = arith.select [[VAR_22_]], [[VAR_18_]], [[VAR_c0_16_]] : index
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_23_]]{{.}} : memref<?xf32>
// CHECK:             [[VAR_25_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_25_]], [[RES_1_]]{{.}}[[VAR_18_]]{{.}} : memref<?xf32>
// CHECK:           }
// CHECK-DAG:       [[VAR_c0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c0_5_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_8_:%.+]] = "krnl.seqinsert"([[RES_1_]], [[RES_]], [[VAR_c0_5_]]) : (memref<?xf32>, memref<0xmemref<?xf32>>, index) -> memref<1xmemref<?xf32>>
// This dealloc is the target to check
// CHECK:           memref.dealloc [[RES_1_]] : memref<?xf32>
// This dealloc is the target to check
// CHECK:           memref.dealloc [[RES_]] : memref<0xmemref<?xf32>>
// CHECK-DAG:       [[VAR_c0_6_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[VAR_c1_7_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[VAR_c1_8_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[LOAD_VAR_0_MEM_:%.+]] = krnl.load [[VAR_0_]][] : memref<i64>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_10_:%.+]] = arith.index_cast [[LOAD_VAR_0_MEM_]] : i64 to index
// CHECK-DAG:       [[VAR_c0_9_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_11_:%.+]] = arith.cmpi slt, [[VAR_10_]], [[VAR_c0_9_]] : index
// CHECK-DAG:       [[VAR_12_:%.+]] = affine.apply #map2(){{.}}[[VAR_10_]]{{.}}
// CHECK:           [[VAR_13_:%.+]] = arith.select [[VAR_11_]], [[VAR_12_]], [[VAR_10_]] : index
// CHECK-DAG:       [[VAR_14_:%.+]] = "krnl.seqextract"([[VAR_8_]], [[VAR_13_]]) {copy = 1 : ui1} : (memref<1xmemref<?xf32>>, index) -> memref<?xf32>
// CHECK-DAG:       [[VAR_c1_10_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1xi64>
// CHECK-DAG:       [[VAR_c0_11_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_16_:%.+]] = memref.dim [[VAR_14_]], [[VAR_c0_11_]] : memref<?xf32>
// This dealloc is the target to check
// CHECK:           memref.dealloc [[VAR_14_]] : memref<?xf32>
// CHECK-DAG:       [[VAR_17_:%.+]] = arith.index_cast [[VAR_16_]] : index to i64
// CHECK-DAG:       [[VAR_c0_12_:%.+]] = arith.constant 0 : index
// CHECK:           krnl.store [[VAR_17_]], [[RES_2_]]{{.}}[[VAR_c0_12_]]{{.}} : memref<1xi64>
// CHECK:           return [[RES_2_]] : memref<1xi64>
// CHECK:         }
}
