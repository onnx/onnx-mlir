// RUN: onnx-mlir-opt --march=z17 --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s


// Note here: we added the march = z17 because not all machines have support for round even operation.
// Recent Z machines have it, same with recent macs.

// -----

// Test GridSample with 2D bilinear interpolation, align_corners=0
func.func @test_gridsample_2d_bilinear(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x2x3x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"} : (tensor<1x1x4x4xf32>, tensor<1x2x3x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_gridsample_2d_bilinear
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<1x1x4x4xf32>, [[GRID_:%.+]]: memref<1x2x3x2xf32>) -> memref<1x1x2x3xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x2x3xf32>
// CHECK-DAG:       [[INDICES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4xindex>
// CHECK-DAG:       [[WEIGHTS_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4xf32>
// CHECK-DAG:       [[MASK_:%.+]] = memref.alloc() {{.*}}: memref<2x3xi8>
// CHECK-DAG:       [[LOOP_N_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_N_]]) with ([[LOOP_N_]] -> [[I_N_:%.+]] = 0 to 1){
// CHECK:             [[LOOP_PLAN_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_PLAN_]]#0, [[LOOP_PLAN_]]#1) with ([[LOOP_PLAN_]]#0 -> [[I_H_:%.+]] = 0 to 2, [[LOOP_PLAN_]]#1 -> [[I_W_:%.+]] = 0 to 3){
// CHECK:               [[LOAD_GRID_X_:%.+]] = krnl.load [[GRID_]]
// CHECK:               [[LOAD_GRID_Y_:%.+]] = krnl.load [[GRID_]]
// CHECK-DAG:           krnl.store {{%.+}}, [[WEIGHTS_]]
// CHECK-DAG:           krnl.store {{%.+}}, [[INDICES_]]
// CHECK-DAG:           krnl.store {{%.+}}, [[MASK_]]
// CHECK:             }
// CHECK:             [[LOOP_APPLY_:%.+]]:3 = krnl.define_loops 3
// CHECK:             krnl.iterate([[LOOP_APPLY_]]#0, [[LOOP_APPLY_]]#1, [[LOOP_APPLY_]]#2) with ([[LOOP_APPLY_]]#0 -> [[I_C_:%.+]] = 0 to 1, [[LOOP_APPLY_]]#1 -> [[I_H2_:%.+]] = 0 to 2, [[LOOP_APPLY_]]#2 -> [[I_W2_:%.+]] = 0 to 3){
// CHECK:               krnl.load [[INDICES_]]
// CHECK:               krnl.load [[WEIGHTS_]]
// CHECK:               krnl.load [[INPUT_]]
// CHECK:               krnl.store {{%.+}}, [[RES_]]
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x1x2x3xf32>
}

// -----

// Test GridSample with 2D nearest interpolation, align_corners=1
func.func @test_gridsample_2d_nearest(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x2x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 1 : si64, mode = "nearest", padding_mode = "zeros"} : (tensor<1x1x3x3xf32>, tensor<1x2x2x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_gridsample_2d_nearest
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<1x1x3x3xf32>, [[GRID_:%.+]]: memref<1x2x2x2xf32>) -> memref<1x1x2x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x2x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3)
// CHECK:             [[LOAD_GRID_X_:%.+]] = krnl.load [[GRID_]]
// CHECK:             [[LOAD_GRID_Y_:%.+]] = krnl.load [[GRID_]]
// CHECK:             krnl.round_even
// CHECK:             krnl.store {{%.+}}, [[RES_]]
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x1x2x2xf32>
}

// -----

// Test GridSample with 2D bicubic interpolation
func.func @test_gridsample_2d_bicubic(%arg0: tensor<1x1x5x5xf32>, %arg1: tensor<1x3x3x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 0 : si64, mode = "cubic", padding_mode = "zeros"} : (tensor<1x1x5x5xf32>, tensor<1x3x3x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_gridsample_2d_bicubic
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<1x1x5x5xf32>, [[GRID_:%.+]]: memref<1x3x3x2xf32>) -> memref<1x1x3x3xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x3x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3)
// CHECK:             [[LOAD_GRID_X_:%.+]] = krnl.load [[GRID_]]
// CHECK:             [[LOAD_GRID_Y_:%.+]] = krnl.load [[GRID_]]
// CHECK:             math.floor
// CHECK:             krnl.store {{%.+}}, [[RES_]]
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x1x3x3xf32>
}

// -----

// Test GridSample with 3D trilinear interpolation
func.func @test_gridsample_3d_trilinear(%arg0: tensor<1x1x3x4x5xf32>, %arg1: tensor<1x2x3x4x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"} : (tensor<1x1x3x4x5xf32>, tensor<1x2x3x4x3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_gridsample_3d_trilinear
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<1x1x3x4x5xf32>, [[GRID_:%.+]]: memref<1x2x3x4x3xf32>) -> memref<1x1x2x3x4xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x2x3x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:5 = krnl.define_loops 5
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 3, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 4){
// CHECK:             [[IV:%.+]]:5 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4)
// CHECK:             [[LOAD_GRID_X_:%.+]] = krnl.load [[GRID_]]
// CHECK:             [[LOAD_GRID_Y_:%.+]] = krnl.load [[GRID_]]
// CHECK:             [[LOAD_GRID_Z_:%.+]] = krnl.load [[GRID_]]
// CHECK:             krnl.store {{%.+}}, [[RES_]]
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x1x2x3x4xf32>
}

// -----

// Test GridSample with 3D nearest interpolation
func.func @test_gridsample_3d_nearest(%arg0: tensor<1x2x2x3x4xf32>, %arg1: tensor<1x2x2x2x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 1 : si64, mode = "nearest", padding_mode = "zeros"} : (tensor<1x2x2x3x4xf32>, tensor<1x2x2x2x3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_gridsample_3d_nearest
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<1x2x2x3x4xf32>, [[GRID_:%.+]]: memref<1x2x2x2x3xf32>) -> memref<1x2x2x2x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x2x2x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:5 = krnl.define_loops 5
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:5 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4)
// CHECK:             [[LOAD_GRID_X_:%.+]] = krnl.load [[GRID_]]
// CHECK:             [[LOAD_GRID_Y_:%.+]] = krnl.load [[GRID_]]
// CHECK:             [[LOAD_GRID_Z_:%.+]] = krnl.load [[GRID_]]
// CHECK:             krnl.round_even
// CHECK:             krnl.store {{%.+}}, [[RES_]]
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2x2x2x2xf32>
}

// -----

// Test GridSample with 2D linear interpolation and border padding
func.func @test_gridsample_2d_linear_border(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x2x3x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 0 : si64, mode = "linear", padding_mode = "border"} : (tensor<1x1x4x4xf32>, tensor<1x2x3x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_gridsample_2d_linear_border
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<1x1x4x4xf32>, [[GRID_:%.+]]: memref<1x2x3x2xf32>) -> memref<1x1x2x3xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x2x3xf32>
// CHECK-DAG:       [[INDICES_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4xindex>
// CHECK-DAG:       [[WEIGHTS_:%.+]] = memref.alloc() {{.*}}: memref<2x3x4xf32>
// CHECK-NOT:       memref.alloc{{.*}}memref<2x3xi8>
// CHECK-DAG:       [[LOOP_N_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_N_]]) with ([[LOOP_N_]] -> [[I_N_:%.+]] = 0 to 1){
// CHECK:             [[LOOP_PLAN_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_PLAN_]]#0, [[LOOP_PLAN_]]#1) with ([[LOOP_PLAN_]]#0 -> [[I_H_:%.+]] = 0 to 2, [[LOOP_PLAN_]]#1 -> [[I_W_:%.+]] = 0 to 3){
// CHECK:               [[LOAD_GRID_X_:%.+]] = krnl.load [[GRID_]]
// CHECK:               [[LOAD_GRID_Y_:%.+]] = krnl.load [[GRID_]]
// CHECK:               math.floor
// CHECK-DAG:           krnl.store {{%.+}}, [[WEIGHTS_]]
// CHECK:               arith.maxnumf
// CHECK:               arith.minnumf
// CHECK-DAG:           krnl.store {{%.+}}, [[INDICES_]]
// CHECK:             }
// CHECK:             [[LOOP_APPLY_:%.+]]:3 = krnl.define_loops 3
// CHECK:             krnl.iterate([[LOOP_APPLY_]]#0, [[LOOP_APPLY_]]#1, [[LOOP_APPLY_]]#2) with ([[LOOP_APPLY_]]#0 -> [[I_C_:%.+]] = 0 to 1, [[LOOP_APPLY_]]#1 -> [[I_H2_:%.+]] = 0 to 2, [[LOOP_APPLY_]]#2 -> [[I_W2_:%.+]] = 0 to 3){
// CHECK:               krnl.load [[INDICES_]]
// CHECK:               krnl.load [[WEIGHTS_]]
// CHECK:               krnl.load [[INPUT_]]
// CHECK:               krnl.store {{%.+}}, [[RES_]]
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x1x2x3xf32>
}

// -----

// Test GridSample with 2D nearest interpolation and border padding
func.func @test_gridsample_2d_nearest_border(%arg0: tensor<1x1x3x3xf32>, %arg1: tensor<1x2x2x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 1 : si64, mode = "nearest", padding_mode = "border"} : (tensor<1x1x3x3xf32>, tensor<1x2x2x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_gridsample_2d_nearest_border
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<1x1x3x3xf32>, [[GRID_:%.+]]: memref<1x2x2x2xf32>) -> memref<1x1x2x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x2x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3)
// CHECK:             [[LOAD_GRID_X_:%.+]] = krnl.load [[GRID_]]
// CHECK:             [[LOAD_GRID_Y_:%.+]] = krnl.load [[GRID_]]
// CHECK:             krnl.round_even
// CHECK:             arith.maxnumf
// CHECK:             arith.minnumf
// CHECK:             krnl.store {{%.+}}, [[RES_]]
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x1x2x2xf32>
}

// -----

// Test GridSample with 2D cubic interpolation and border padding
func.func @test_gridsample_2d_cubic_border(%arg0: tensor<1x1x5x5xf32>, %arg1: tensor<1x3x3x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 0 : si64, mode = "cubic", padding_mode = "border"} : (tensor<1x1x5x5xf32>, tensor<1x3x3x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_gridsample_2d_cubic_border
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<1x1x5x5xf32>, [[GRID_:%.+]]: memref<1x3x3x2xf32>) -> memref<1x1x3x3xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x3x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 3, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3)
// CHECK:             [[LOAD_GRID_X_:%.+]] = krnl.load [[GRID_]]
// CHECK:             [[LOAD_GRID_Y_:%.+]] = krnl.load [[GRID_]]
// CHECK:             math.floor
// CHECK:             arith.maxnumf
// CHECK:             arith.minnumf
// CHECK:             krnl.store {{%.+}}, [[RES_]]
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x1x3x3xf32>
}

// -----

// Test GridSample with 3D linear interpolation and border padding
func.func @test_gridsample_3d_linear_border(%arg0: tensor<1x1x3x4x5xf32>, %arg1: tensor<1x2x3x4x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 0 : si64, mode = "linear", padding_mode = "border"} : (tensor<1x1x3x4x5xf32>, tensor<1x2x3x4x3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_gridsample_3d_linear_border
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<1x1x3x4x5xf32>, [[GRID_:%.+]]: memref<1x2x3x4x3xf32>) -> memref<1x1x2x3x4xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x2x3x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:5 = krnl.define_loops 5
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 3, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 4){
// CHECK:             [[IV:%.+]]:5 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4)
// CHECK:             [[LOAD_GRID_X_:%.+]] = krnl.load [[GRID_]]
// CHECK:             [[LOAD_GRID_Y_:%.+]] = krnl.load [[GRID_]]
// CHECK:             [[LOAD_GRID_Z_:%.+]] = krnl.load [[GRID_]]
// CHECK:             math.floor
// CHECK:             arith.maxnumf
// CHECK:             arith.minnumf
// CHECK:             krnl.store {{%.+}}, [[RES_]]
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x1x2x3x4xf32>
}

// -----

// Test GridSample with 3D nearest interpolation and border padding
func.func @test_gridsample_3d_nearest_border(%arg0: tensor<1x2x2x3x4xf32>, %arg1: tensor<1x2x2x2x3xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 1 : si64, mode = "nearest", padding_mode = "border"} : (tensor<1x2x2x3x4xf32>, tensor<1x2x2x2x3xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_gridsample_3d_nearest_border
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<1x2x2x3x4xf32>, [[GRID_:%.+]]: memref<1x2x2x2x3xf32>) -> memref<1x2x2x2x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x2x2x2x2xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:5 = krnl.define_loops 5
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 2, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 2, [[LOOP_0_]]#4 -> [[I_4_:%.+]] = 0 to 2){
// CHECK:             [[IV:%.+]]:5 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3, [[LOOP_0_]]#4)
// CHECK:             [[LOAD_GRID_X_:%.+]] = krnl.load [[GRID_]]
// CHECK:             [[LOAD_GRID_Y_:%.+]] = krnl.load [[GRID_]]
// CHECK:             [[LOAD_GRID_Z_:%.+]] = krnl.load [[GRID_]]
// CHECK:             krnl.round_even
// CHECK:             arith.maxnumf
// CHECK:             arith.minnumf
// CHECK:             krnl.store {{%.+}}, [[RES_]]
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2x2x2x2xf32>
}

// -----

// Test GridSample with mixed types: f32 input and f64 grid
func.func @test_gridsample_2d_mixed_types(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x2x2x2xf64>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"} : (tensor<1x1x4x4xf32>, tensor<1x2x2x2xf64>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_gridsample_2d_mixed_types
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<1x1x4x4xf32>, [[GRID_:%.+]]: memref<1x2x2x2xf64>) -> memref<1x1x2x2xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x2x2xf32>
// CHECK-DAG:       [[INDICES_:%.+]] = memref.alloc() {{.*}}: memref<2x2x4xindex>
// CHECK-DAG:       [[WEIGHTS_:%.+]] = memref.alloc() {{.*}}: memref<2x2x4xf32>
// CHECK-DAG:       [[MASK_:%.+]] = memref.alloc() {{.*}}: memref<2x2xi8>
// CHECK-DAG:       [[LOOP_N_:%.+]] = krnl.define_loops 1
// CHECK:           krnl.iterate([[LOOP_N_]]) with ([[LOOP_N_]] -> [[I_N_:%.+]] = 0 to 1){
// CHECK:             [[LOOP_PLAN_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_PLAN_]]#0, [[LOOP_PLAN_]]#1) with ([[LOOP_PLAN_]]#0 -> [[I_H_:%.+]] = 0 to 2, [[LOOP_PLAN_]]#1 -> [[I_W_:%.+]] = 0 to 2){
// CHECK:               [[LOAD_GRID_X_:%.+]] = krnl.load [[GRID_]]{{.*}} : memref<1x2x2x2xf64>
// CHECK:               [[LOAD_GRID_Y_:%.+]] = krnl.load [[GRID_]]{{.*}} : memref<1x2x2x2xf64>
// CHECK:               [[CAST_X_:%.+]] = arith.truncf [[LOAD_GRID_X_]] : f64 to f32
// CHECK:               [[CAST_Y_:%.+]] = arith.truncf [[LOAD_GRID_Y_]] : f64 to f32
// CHECK-DAG:           krnl.store {{%.+}}, [[WEIGHTS_]]
// CHECK-DAG:           krnl.store {{%.+}}, [[INDICES_]]
// CHECK-DAG:           krnl.store {{%.+}}, [[MASK_]]
// CHECK:             }
// CHECK:             [[LOOP_APPLY_:%.+]]:3 = krnl.define_loops 3
// CHECK:             krnl.iterate([[LOOP_APPLY_]]#0, [[LOOP_APPLY_]]#1, [[LOOP_APPLY_]]#2) with ([[LOOP_APPLY_]]#0 -> [[I_C_:%.+]] = 0 to 1, [[LOOP_APPLY_]]#1 -> [[I_H2_:%.+]] = 0 to 2, [[LOOP_APPLY_]]#2 -> [[I_W2_:%.+]] = 0 to 2){
// CHECK:               krnl.load [[INDICES_]]
// CHECK:               krnl.load [[WEIGHTS_]]
// CHECK:               krnl.load [[INPUT_]]
// CHECK:               krnl.store {{%.+}}, [[RES_]]
// CHECK:             }
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x1x2x2xf32>
}