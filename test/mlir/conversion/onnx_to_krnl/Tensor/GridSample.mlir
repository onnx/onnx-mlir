// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// -----

// Test GridSample with 2D bilinear interpolation, align_corners=0
func.func @test_gridsample_2d_bilinear(%arg0: tensor<1x1x4x4xf32>, %arg1: tensor<1x2x3x2xf32>) -> tensor<*xf32> {
  %0 = "onnx.GridSample"(%arg0, %arg1) {align_corners = 0 : si64, mode = "linear", padding_mode = "zeros"} : (tensor<1x1x4x4xf32>, tensor<1x2x3x2xf32>) -> tensor<*xf32>
  return %0 : tensor<*xf32>

// CHECK-LABEL:  func @test_gridsample_2d_bilinear
// CHECK-SAME:   ([[INPUT_:%.+]]: memref<1x1x4x4xf32>, [[GRID_:%.+]]: memref<1x2x3x2xf32>) -> memref<1x1x2x3xf32> {
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x1x2x3xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 1, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 2, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 3){
// CHECK:             [[IV:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3)
// CHECK:             [[LOAD_GRID_X_:%.+]] = krnl.load [[GRID_]]
// CHECK:             [[LOAD_GRID_Y_:%.+]] = krnl.load [[GRID_]]
// CHECK:             krnl.store {{%.+}}, [[RES_]]
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
// CHECK:             math.roundeven
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
// CHECK:             math.roundeven
// CHECK:             krnl.store {{%.+}}, [[RES_]]
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x2x2x2x2xf32>
}