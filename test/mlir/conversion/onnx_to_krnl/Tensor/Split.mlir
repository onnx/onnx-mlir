// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func private @test_split_equal(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %cst = "onnx.NoValue"() {value} : () -> none
  %0, %1 = "onnx.Split"(%arg0, %cst) { axis = 0 : si64} : (tensor<16x32x64xf32>, none) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 8)>
  // CHECK-LABEL: @test_split_equal
  // CHECK:     [[RES_0:%.+]] = memref.alloc() {{.*}}: memref<8x32x64xf32>
  // CHECK:     [[RES_1:%.+]] = memref.alloc() {{.*}}: memref<8x32x64xf32>
  // CHECK:     [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK:     krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 8,
  // CHECK-SAME:             [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64){
  // CHECK:       [[IV:%.+]]:3 = krnl.get_induction_var_value([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) :
  // CHECK-SAME:    (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:       [[LOAD_0:%.+]] = krnl.load %arg0{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:       krnl.store [[LOAD_0]], [[RES_0]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<8x32x64xf32>
  // CHECK:     }
  // CHECK:     [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK:     krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 8,
  // CHECK-SAME:             [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64){
  // CHECK:       [[IV:%.+]]:3 = krnl.get_induction_var_value([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) :
  // CHECK-SAME:    (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:       [[INDEX:%.+]] = affine.apply [[INDEX_MAP]]{{.}}[[IV]]#0{{.}}
  // CHECK:       [[LOAD_1:%.+]] = krnl.load %arg0{{.}}[[INDEX]], [[IV]]#1, [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:       krnl.store [[LOAD_1]], [[RES_1]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<8x32x64xf32>
  // CHECK:     }
  // CHECK:     return [[RES_0]], [[RES_1]] : memref<8x32x64xf32>, memref<8x32x64xf32>
}

// -----

func.func private @test_split_variable(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %split = onnx.Constant dense<[2, 30]> : tensor<2xi64>
  %0, %1 = "onnx.Split"(%arg0, %split) { axis = 1 : si64} : (tensor<16x32x64xf32>, tensor<2xi64>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 2)>
  // CHECK-LABEL: @test_split_variable

  // CHECK: [[RES_0:%.+]] = memref.alloc() {{.*}}: memref<16x2x64xf32>
  // CHECK: [[RES_1:%.+]] = memref.alloc() {{.*}}: memref<16x30x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 2, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[IV:%.+]]:3 = krnl.get_induction_var_value([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:   [[LOAD_0:%.+]] = krnl.load %arg0{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_0]], [[RES_0]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x2x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 30, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[IV:%.+]]:3 = krnl.get_induction_var_value([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:   [[INDEX:%.+]] = affine.apply [[INDEX_MAP]]{{.}}[[IV]]#1{{.}}
  // CHECK:   [[LOAD_1:%.+]] = krnl.load %arg0{{.}}[[IV]]#0, [[INDEX]], [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_1]], [[RES_1]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x30x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<16x2x64xf32>, memref<16x30x64xf32>
}

// -----

func.func private @test_splitv11_equal(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.SplitV11"(%arg0) { axis = 0 : si64} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 8)>
  // CHECK-LABEL: @test_splitv11_equal

  // CHECK: [[RES_0:%.+]] = memref.alloc() {{.*}}: memref<8x32x64xf32>
  // CHECK: [[RES_1:%.+]] = memref.alloc() {{.*}}: memref<8x32x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 8, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[IV:%.+]]:3 = krnl.get_induction_var_value([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:   [[LOAD_0:%.+]] = krnl.load %arg0{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_0]], [[RES_0]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<8x32x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 8, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 32, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[IV:%.+]]:3 = krnl.get_induction_var_value([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:   [[INDEX:%.+]] = affine.apply [[INDEX_MAP]]{{.}}[[IV]]#0{{.}}
  // CHECK:   [[LOAD_1:%.+]] = krnl.load %arg0{{.}}[[INDEX]], [[IV]]#1, [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_1]], [[RES_1]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<8x32x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<8x32x64xf32>, memref<8x32x64xf32>
}

// -----

func.func private @test_splitv11_variable(%arg0 : tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>) {
  %0, %1 = "onnx.SplitV11"(%arg0) { axis = 1 : si64, split = [2, 30]} : (tensor<16x32x64xf32>) -> (tensor<*xf32>, tensor<*xf32>)
  "func.return"(%0, %1) : (tensor<*xf32>, tensor<*xf32>) -> ()

  // CHECK: [[INDEX_MAP:#.+]] = affine_map<(d0) -> (d0 + 2)>
  // CHECK-LABEL: @test_splitv11_variable

  // CHECK: [[RES_0:%.+]] = memref.alloc() {{.*}}: memref<16x2x64xf32>
  // CHECK: [[RES_1:%.+]] = memref.alloc() {{.*}}: memref<16x30x64xf32>
  // CHECK: [[DEF_LOOP_0:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) with ([[DEF_LOOP_0]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_0]]#1 -> %arg2 = 0 to 2, [[DEF_LOOP_0]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[IV:%.+]]:3 = krnl.get_induction_var_value([[DEF_LOOP_0]]#0, [[DEF_LOOP_0]]#1, [[DEF_LOOP_0]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK:   [[LOAD_0:%.+]] = krnl.load %arg0{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_0]], [[RES_0]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x2x64xf32>
  // CHECK: }
  // CHECK: [[DEF_LOOP_1:%.+]]:3 = krnl.define_loops 3
  // CHECK: krnl.iterate([[DEF_LOOP_1]]#0, [[DEF_LOOP_1]]#1, [[DEF_LOOP_1]]#2) with ([[DEF_LOOP_1]]#0 -> %arg1 = 0 to 16, [[DEF_LOOP_1]]#1 -> %arg2 = 0 to 30, [[DEF_LOOP_1]]#2 -> %arg3 = 0 to 64){
  // CHECK:   [[INDEX:%.+]] = affine.apply [[INDEX_MAP]]{{.}}[[IV]]#1{{.}}
  // CHECK:   [[LOAD_1:%.+]] = krnl.load %arg0{{.}}[[IV]]#0, [[INDEX]], [[IV]]#2{{.}} : memref<16x32x64xf32>
  // CHECK:   krnl.store [[LOAD_1]], [[RES_1]]{{.}}[[IV]]#0, [[IV]]#1, [[IV]]#2{{.}} : memref<16x30x64xf32>
  // CHECK: }
  // CHECK: return [[RES_0]], [[RES_1]] : memref<16x2x64xf32>, memref<16x30x64xf32>
}

