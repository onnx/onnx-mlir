// RUN: onnx-mlir-opt -O3 --mtriple=s390x-ibm-loz --mcpu=z16 --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// use --mtriple=s390x-ibm-loz --mcpu=z16 to enable SIMD as we now need a machine
// can also use -march=x86-64 instead.

// -----

func.func @layernorm_4D_with_scale_bias(%arg0: tensor<2x64x32x8xf32>, %arg1: tensor<32x8xf32>, %arg2: tensor<32x8xf32>) -> tensor<*xf32> attributes {input_names = ["X", "S", "B"], output_names = ["LN"]} {
  %0 = "onnx.NoValue"() {value} : () -> none
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {axis = -2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x64x32x8xf32>, tensor<32x8xf32>, tensor<32x8xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  onnx.Return %Y : tensor<*xf32>

// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_4D_with_scale_bias
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x64x32x8xf32>, [[PARAM_1_:%.+]]: memref<32x8xf32>, [[PARAM_2_:%.+]]: memref<32x8xf32>) -> memref<2x64x32x8xf32> attributes {input_names = ["X", "S", "B"], output_names = ["LN"]} {
// CHECK:           [[BLOCK_TILE__0_:%.+]], [[BLOCK_IN__0_:%.+]] = krnl.block [[LOOP_0_:%.+]] 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
// CHECK:           krnl.iterate([[BLOCK_TILE__0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 128){
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 256 step 16 {
// CHECK:             affine.for [[I_2_:%.+]] = 0 to 256 step 16 {
// CHECK:           onnx.Return [[VAR_1_:%.+]] : tensor<2x64x32x8xf32>
}

// -----

func.func @layernorm_4D_with_scale_bias_no_SIMD(%arg0: tensor<2x64x31x3xf32>, %arg1: tensor<31x3xf32>, %arg2: tensor<31x3xf32>) -> tensor<*xf32> attributes {input_names = ["X", "S", "B"], output_names = ["LN"]} {
  %0 = "onnx.NoValue"() {value} : () -> none
  %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%arg0, %arg1, %arg2) {axis = -2 : si64, epsilon = 9.99999974E-6 : f32, stash_type = 1 : si64} : (tensor<2x64x31x3xf32>, tensor<31x3xf32>, tensor<31x3xf32>) -> (tensor<*xf32>, tensor<*xf32>, tensor<*xf32>)
  onnx.Return %Y : tensor<*xf32>

// mlir2FileCheck.py
// mlir2FileCheck.py
// CHECK-LABEL:  func.func @layernorm_4D_with_scale_bias_no_SIMD
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x64x31x3xf32>, [[PARAM_1_:%.+]]: memref<31x3xf32>, [[PARAM_2_:%.+]]: memref<31x3xf32>) -> memref<2x64x31x3xf32> attributes {input_names = ["X", "S", "B"], output_names = ["LN"]} {
// CHECK:           [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 64, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 1, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 1){
// CHECK:           [[LOOP_1_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1, [[LOOP_1_]]#2, [[LOOP_1_]]#3) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to 2, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to 64, [[LOOP_1_]]#2 -> [[I_6_:%.+]] = 0 to 31, [[LOOP_1_]]#3 -> [[I_7_:%.+]] = 0 to 3){
// CHECK:           [[LOOP_2_:%.+]]:4 = krnl.define_loops 4
// CHECK:           krnl.iterate([[LOOP_2_]]#0, [[LOOP_2_]]#1, [[LOOP_2_]]#2, [[LOOP_2_]]#3) with ([[LOOP_2_]]#0 -> [[I_8_:%.+]] = 0 to 2, [[LOOP_2_]]#1 -> [[I_9_:%.+]] = 0 to 64, [[LOOP_2_]]#2 -> [[I_10_:%.+]] = 0 to 1, [[LOOP_2_]]#3 -> [[I_11_:%.+]] = 0 to 1){
}

