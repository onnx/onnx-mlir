// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

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
