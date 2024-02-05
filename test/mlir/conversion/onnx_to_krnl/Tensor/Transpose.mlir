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

  // CHECK:        [[MAP:#.+]] = affine_map<(d0) -> (d0)>
  // CHECK-LABEL:  func private @test_transpose_dynamic_dims
  // CHECK-SAME:   ([[PARAM_0:%.+]]: memref<10x?x30x40xf32>) -> memref<10x40x?x30xf32> {
  // CHECK:           [[CST_1:%.+]] = arith.constant 1 : index
  // CHECK:           [[DIM_0:%.+]] = memref.dim [[PARAM_0]], [[CST_1]] : memref<10x?x30x40xf32>
  // CHECK-DAG:       [[RES:%.+]] = memref.alloc([[DIM_0]]) {{.*}}: memref<10x40x?x30xf32>
  // CHECK-DAG:       [[LOOP_0:%.+]]:4 = krnl.define_loops 4
  // CHECK-DAG:       [[CST_1_1:%.+]] = arith.constant 1 : index
  // CHECK:           [[DIM_1:%.+]] = memref.dim [[PARAM_0]], [[CST_1_1]] : memref<10x?x30x40xf32>
  // CHECK:           krnl.iterate([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2, [[LOOP_0]]#3) with ([[LOOP_0]]#0 -> [[I_0:%.+]] = 0 to 10,
  // CHECK-SAME:        [[LOOP_0]]#1 -> [[I_1:%.+]] = 0 to [[MAP]]{{.}}[[DIM_1]]{{.}}, [[LOOP_0]]#2 -> [[I_2:%.+]] = 0 to 30,
  // CHECK-SAME:        [[LOOP_0]]#3 -> [[I_3:%.+]] = 0 to 40){
  // CHECK-NEXT:        [[IV:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0]]#0, [[LOOP_0]]#1, [[LOOP_0]]#2, [[LOOP_0]]#3) :
  // CHECK-SAME:          (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
  // CHECK:             [[LOAD_PARAM_0_MEM:%.+]] = krnl.load [[PARAM_0]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3{{.}} : memref<10x?x30x40xf32>
  // CHECK:             krnl.store [[LOAD_PARAM_0_MEM]], [[RES]]{{.}}[[IV]]#0, [[IV]]#3, [[IV]]#1, [[IV]]#2{{.}} : memref<10x40x?x30xf32>
  // CHECK:           }
  // CHECK:           return [[RES]] : memref<10x40x?x30xf32>
  // CHECK:         }
}
