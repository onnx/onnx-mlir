// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// -----


func.func private @test_transpose(%arg0 : tensor<10x20x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<10x20x30x40xf32>) -> tensor<*xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func.func private @test_transpose
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x20x30x40xf32>) -> memref<40x10x30x20xf32> {
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_30_:%.+]] = arith.constant 30 : index
// CHECK-DAG:       [[CST_20_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<40x30x20x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_40_1_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_30_1_:%.+]] = arith.constant 30 : index
// CHECK-DAG:       [[CST_20_1_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[CST_10_1_:%.+]] = arith.constant 10 : index
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#3 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.unroll [[BLOCK_IN__0_]] : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[BLOCK_TILE__0_]], [[BLOCK_IN__0_]]) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 40, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 30, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 20, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 10){
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_3_]], [[I_2_]], [[I_1_]], [[I_0_]]{{.}} : memref<10x20x30x40xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]], [[I_3_]]{{.}} : memref<40x30x20x10xf32>
// CHECK:           }
// CHECK-DAG:       [[CST_40_2_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_10_2_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_30_2_:%.+]] = arith.constant 30 : index
// CHECK-DAG:       [[CST_20_2_:%.+]] = arith.constant 20 : index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<40x10x30x20xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_40_3_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_10_3_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_30_3_:%.+]] = arith.constant 30 : index
// CHECK-DAG:       [[CST_20_3_:%.+]] = arith.constant 20 : index
// CHECK:           [[BLOCK_TILE__1_:%.+]], [[BLOCK_IN__1_:%.+]] = krnl.block [[LOOP_1_]]#3 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.unroll [[BLOCK_IN__1_]] : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[BLOCK_TILE__1_]], [[BLOCK_IN__1_]]) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 40, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 10, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to 30, [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 20){
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[I_4_]], [[I_6_]], [[I_7_]], [[I_5_]]{{.}} : memref<40x30x20x10xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_1_]], [[RES_1_]]{{.}}[[I_4_]], [[I_5_]], [[I_6_]], [[I_7_]]{{.}} : memref<40x10x30x20xf32>
// CHECK:           }
// CHECK:           return [[RES_1_]] : memref<40x10x30x20xf32>
// CHECK:         }
}

// -----

// COM: Test whether the lowering is correct in the presence of dynamic dimensions.

func.func private @test_transpose_dynamic_dims(%arg0 : tensor<10x?x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<10x?x30x40xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_transpose_dynamic_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x?x30x40xf32>) -> memref<10x40x?x30xf32> {
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<10x?x30x40xf32>
// CHECK-DAG:       [[CST_30_:%.+]] = arith.constant 30 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<10x40x?x30xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_10_1_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_40_1_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_30_1_:%.+]] = arith.constant 30 : index
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_]]#3 6 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.unroll [[BLOCK_IN__0_]] : !krnl.loop
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[BLOCK_TILE__0_]], [[BLOCK_IN__0_]]) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 40, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_]]), [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 30){
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_2_]], [[I_3_]], [[I_1_]]{{.}} : memref<10x?x30x40xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]], [[I_3_]]{{.}} : memref<10x40x?x30xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<10x40x?x30xf32>
// CHECK:         }
}

