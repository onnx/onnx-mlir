// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func private @test_transpose(%arg0 : tensor<10x20x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) : (tensor<10x20x30x40xf32>) -> tensor<*xf32>
  %1 = "onnx.Transpose"(%0) {perm = [0, 3, 1, 2]} : (tensor<*xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: test_transpose
  // CHECK:       [[RES1:%.+]] = memref.alloc() {{.*}}: memref<40x30x20x10xf32>
  // CHECK:       [[DEF_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK:       krnl.iterate([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) with ([[DEF_LOOPS]]#0 -> %arg1 = 0 to 10,
  // CHECK-SAME:               [[DEF_LOOPS]]#1 -> %arg2 = 0 to 20, [[DEF_LOOPS]]#2 -> %arg3 = 0 to 30, [[DEF_LOOPS]]#3 -> %arg4 = 0 to 40){
  // CHECK-NEXT:  [[IV:%.+]]:4 = krnl.get_induction_var_value([[DEF_LOOPS]]#0, [[DEF_LOOPS]]#1, [[DEF_LOOPS]]#2, [[DEF_LOOPS]]#3) :
  // CHECK-SAME:     (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
  // CHECK:       [[LOAD:%.+]] = krnl.load %arg0{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3{{.}} : memref<10x20x30x40xf32>
  // CHECK:       krnl.store [[LOAD]], [[RES1]]{{.}}[[IV]]#3, [[IV]]#2, [[IV]]#1, [[IV]]#0{{.}} : memref<40x30x20x10xf32>
  // CHECK:       [[RES0:%.+]] = memref.alloc() {{.*}}: memref<40x10x30x20xf32>
  // CHECK:       [[DEF_LOOPS1:%.+]]:4 = krnl.define_loops 4
  // CHECK:       krnl.iterate([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1, [[DEF_LOOPS1]]#2, [[DEF_LOOPS1]]#3) with ([[DEF_LOOPS1]]#0 -> %arg1 = 0 to 40,
  // CHECK-SAME:               [[DEF_LOOPS1]]#1 -> %arg2 = 0 to 30, [[DEF_LOOPS1]]#2 -> %arg3 = 0 to 20, [[DEF_LOOPS1]]#3 -> %arg4 = 0 to 10){
  // CHECK-NEXT:  [[IV1:%.+]]:4 = krnl.get_induction_var_value([[DEF_LOOPS1]]#0, [[DEF_LOOPS1]]#1, [[DEF_LOOPS1]]#2, [[DEF_LOOPS1]]#3) :
  // CHECK-SAME:     (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
  // CHECK:       [[LOAD:%.+]] = krnl.load [[RES1]]{{.}}[[IV1]]#0, [[IV1]]#1, [[IV1]]#2, [[IV1]]#3{{.}} : memref<40x30x20x10xf32>
  // CHECK:       krnl.store [[LOAD]], [[RES0]]{{.}}[[IV1]]#0, [[IV1]]#3, [[IV1]]#1, [[IV1]]#2{{.}} : memref<40x10x30x20xf32>
  // CHECK:       return [[RES0]] : memref<40x10x30x20xf32>
}

// -----

// COM: Test whether the lowering is correct in the presence of dynamic dimensions.

func.func private @test_transpose_dynamic_dims(%arg0 : tensor<10x?x30x40xf32>) -> tensor<*xf32> {
  %0 = "onnx.Transpose"(%arg0) {perm = [0, 3, 1, 2]} : (tensor<10x?x30x40xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0)>
// CHECK-LABEL:  func.func private @test_transpose_dynamic_dims
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<10x?x30x40xf32>) -> memref<40x30x?x10xf32> {
// CHECK-DAG:       [[CST_40_:%.+]] = arith.constant 40 : index
// CHECK-DAG:       [[CST_30_:%.+]] = arith.constant 30 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_]] : memref<10x?x30x40xf32>
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_dim_]]) {{.*}}: memref<40x30x?x10xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_10_1_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_1_]] : memref<10x?x30x40xf32>
// CHECK-DAG:       [[CST_30_1_:%.+]] = arith.constant 30 : index
// CHECK-DAG:       [[CST_40_1_:%.+]] = arith.constant 40 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 10, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[MAP_0_]]([[VAR_dim_2_]]), [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 30, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 40){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<10x?x30x40xf32>
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_1_]]#3, [[VAR_1_]]#2, [[VAR_1_]]#1, [[VAR_1_]]#0] : memref<40x30x?x10xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<40x30x?x10xf32>
// CHECK:         }
}

