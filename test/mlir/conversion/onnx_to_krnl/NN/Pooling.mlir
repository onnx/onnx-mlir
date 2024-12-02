// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func private @test_pool_general_computation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-DAG: #{{.*}} = affine_map<(d0) -> (0, d0)>
  // CHECK-DAG: #{{.*}} = affine_map<(d0) -> (32, d0 + 2)>
  // CHECK-DAG: #{{.*}} = affine_map<(d0, d1) -> (0, d1)>
  // CHECK-DAG: #{{.*}} = affine_map<(d0, d1) -> (32, d1 + 2)>
  // CHECK-DAG: #[[BOUND:.+]] = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>

  // CHECK-LABEL: @test_pool_general_computation

  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = arith.constant 0.000000e+00 : f32

  // CHECK: [[REDUCTION_VAL:%.+]] = memref.alloca() : memref<f32>
  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31){
  // CHECK:   [[IV:%.+]]:4 = krnl.get_induction_var_value([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)

  // CHECK:   krnl.store [[IDENTITY]], [[REDUCTION_VAL]][] : memref<f32>

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #[[BOUND]]([[IV]]#2)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #[[BOUND]]([[IV]]#3)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]){
  // CHECK:     {{.*}} = krnl.load %arg0[[[IV]]#0, [[IV]]#1, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     {{.*}} = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:     krnl.store {{.*}}, [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   }

  // CHECK:   [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   krnl.store [[LOAD_REDUCTION]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3] : memref<1x3x31x31xf32>
  // CHECK: }
}

// -----

func.func private @test_averagepool_identity_value(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_averagepool_identity_value
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK: [[REDUCTION_VAL:%.+]] = memref.alloca() : memref<f32>
  // CHECK: krnl.store [[IDENTITY]], [[REDUCTION_VAL]][] : memref<f32>
}

// -----

func.func private @test_maxpool_identity_value(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_maxpool_identity_value
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<1x3x31x31xf32>
  // CHECK: [[IDENTITY:%.+]] = arith.constant 0xFF800000 : f32
  // CHECK: [[REDUCTION_VAL:%.+]] = memref.alloca() : memref<f32>
  // CHECK: krnl.store [[IDENTITY]], [[REDUCTION_VAL]][] : memref<f32>
}

// -----

func.func private @test_averagepool_pooling_operation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.AveragePool"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

  // CHECK-LABEL: @test_averagepool_pooling_operation
  // CHECK: [[RES:%.+]] = memref.alloc() {{.*}}: memref<1x3x31x31xf32>

  // CHECK: [[REDUCTION_VAL:%.+]] = memref.alloca() : memref<f32>
  // CHECK: [[OUTPUT_LOOPS:%.+]]:4 = krnl.define_loops 4
  // CHECK: krnl.iterate([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) with ([[OUTPUT_LOOPS]]#0 -> %arg1 = 0 to 1, [[OUTPUT_LOOPS]]#1 -> %arg2 = 0 to 3, [[OUTPUT_LOOPS]]#2 -> %arg3 = 0 to 31, [[OUTPUT_LOOPS]]#3 -> %arg4 = 0 to 31){
  // CHECK:   [[IV:%.+]]:4 = krnl.get_induction_var_value([[OUTPUT_LOOPS]]#0, [[OUTPUT_LOOPS]]#1, [[OUTPUT_LOOPS]]#2, [[OUTPUT_LOOPS]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)

  // CHECK:   krnl.store {{.*}}, [[REDUCTION_VAL]][] : memref<f32>

  // CHECK:   [[POOL_LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK:   krnl.iterate([[POOL_LOOPS]]#0, [[POOL_LOOPS]]#1) with ([[POOL_LOOPS]]#0 -> %arg5 = 0 to min #{{.*}}([[IV]]#2)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}], [[POOL_LOOPS]]#1 -> %arg6 = 0 to min #{{.*}}([[IV]]#3)[{{.*}}, {{.*}}, {{.*}}, {{.*}}, {{.*}}]){

  // CHECK:     [[INPUT_LOAD:%.+]] = krnl.load %arg0[[[IV]]#0, [[IV]]#1, {{.*}}, {{.*}}] : memref<1x3x32x32xf32>
  // CHECK:     [[OUTPUT_LOAD:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:     [[SUM:%.+]] = arith.addf [[OUTPUT_LOAD]], [[INPUT_LOAD]] : f32
  // CHECK:     krnl.store [[SUM]], [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   }
  // CHECK:   [[LOAD_REDUCTION:%.+]] = krnl.load [[REDUCTION_VAL]][] : memref<f32>
  // CHECK:   krnl.store [[LOAD_REDUCTION]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3] : memref<1x3x31x31xf32>

  // CHECK:   [[NUMERATOR:%.+]] = krnl.load [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3] : memref<1x3x31x31xf32>
  // CHECK:   [[AVERAGE:%.+]] = arith.divf [[NUMERATOR]], {{.*}} : f32
  // CHECK:   krnl.store [[AVERAGE]], [[RES]][[[IV]]#0, [[IV]]#1, [[IV]]#2, [[IV]]#3] : memref<1x3x31x31xf32>
  // CHECK: }
}

// -----


func.func private @test_maxpool_pooling_operation(%arg0 : tensor<1x3x32x32xf32>) -> tensor<*xf32> {
  %0 = "onnx.MaxPoolSingleOut"(%arg0) {auto_pad = "NOTSET", kernel_shape = [2, 2]} : (tensor<1x3x32x32xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0) -> (d0 + 1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0) -> (d0 + 2)>
// CHECK-DAG:   [[MAP_2_:#.+]] = affine_map<(d0) -> (0, d0)>
// CHECK-DAG:   [[MAP_3_:#.+]] = affine_map<(d0) -> (32, d0 + 2)>
// CHECK-DAG:   [[MAP_4_:#.+]] = affine_map<(d0, d1) -> (d1 + 1)>
// CHECK-DAG:   [[MAP_5_:#.+]] = affine_map<(d0, d1) -> (d1 + 2)>
// CHECK-DAG:   [[MAP_6_:#.+]] = affine_map<(d0, d1) -> (0, d1)>
// CHECK-DAG:   [[MAP_7_:#.+]] = affine_map<(d0, d1) -> (32, d1 + 2)>
// CHECK-DAG:   [[MAP_8_:#.+]] = affine_map<(d0)[s0, s1, s2, s3, s4] -> (s0 - ((s2 ceildiv s4) * s4 - s2), -(d0 * s3 - s2) + s0, d0 * s3 + (s1 - 1) * s4 - s2 - ((s2 ceildiv s4) * s4 - s2) + 1, d0 * s3 + (s1 - 1) * s4 - s2 - (d0 * s3 - s2) + 1)>
// CHECK-DAG:   [[MAP_9_:#.+]] = affine_map<(d0, d1, d2) -> (d2)>
// CHECK-DAG:   [[MAP_10_:#.+]] = affine_map<(d0, d1, d2) -> (d0)>
// CHECK-DAG:   [[MAP_11_:#.+]] = affine_map<(d0, d1, d2) -> (d2 + d0)>
// CHECK-DAG:   [[MAP_12_:#.+]] = affine_map<(d0, d1, d2) -> (d2, d2 + d0)>
// CHECK-DAG:   [[MAP_13_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d3)>
// CHECK-DAG:   [[MAP_14_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d1)>
// CHECK-DAG:   [[MAP_15_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d3 + d1)>
// CHECK-DAG:   [[MAP_16_:#.+]] = affine_map<(d0, d1, d2, d3) -> (d3, d3 + d1)>
// CHECK-LABEL:  func.func private @test_maxpool_pooling_operation
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1x3x32x32xf32>) -> memref<1x3x31x31xf32> {
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_32_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_30_:%.+]] = arith.constant 30 : index
// CHECK-DAG:       [[CST_31_:%.+]] = arith.constant 31 : index
// CHECK-DAG:       [[CST_32_1_:%.+]] = arith.constant 32 : index
// CHECK-DAG:       [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_6_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1_7_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_2_3_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_30_1_:%.+]] = arith.constant 30 : index
// CHECK-DAG:       [[CST_31_1_:%.+]] = arith.constant 31 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1x3x31x31xf32>
// CHECK-DAG:       [[CST_0_5_:%.+]] = arith.constant 0xFF800000 : f32
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloca() : memref<f32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:4 = krnl.define_loops 4
// CHECK-DAG:       [[CST_0_6_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1_8_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_3_1_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_31_2_:%.+]] = arith.constant 31 : index
// CHECK-DAG:       [[CST_31_3_:%.+]] = arith.constant 31 : index
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 1, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to 3, [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 31, [[LOOP_0_]]#3 -> [[I_3_:%.+]] = 0 to 31){
// CHECK:             [[VAR_1_:%.+]]:4 = krnl.get_induction_var_value([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2, [[LOOP_0_]]#3) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
// CHECK:             krnl.store [[CST_0_5_]], [[RES_1_]][] : memref<f32>
// CHECK-DAG:         [[CST_32_2_:%.+]] = arith.constant 32 : index
// CHECK-DAG:         [[CST_2_4_:%.+]] = arith.constant 2 : index
// CHECK-DAG:         [[CST_0_7_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_9_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[CST_1_10_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[CST_32_3_:%.+]] = arith.constant 32 : index
// CHECK-DAG:         [[CST_2_5_:%.+]] = arith.constant 2 : index
// CHECK-DAG:         [[CST_0_8_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_1_11_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[CST_1_12_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[CST_1_13_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[CST_1_14_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_2_:%.+]] = affine.apply [[MAP_0_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[CST_1_15_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_3_:%.+]] = affine.apply [[MAP_1_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.max [[MAP_2_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[VAR_5_:%.+]] = affine.min [[MAP_3_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[CST_0_9_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[VAR_6_:%.+]] = affine.min [[MAP_2_]]([[VAR_1_]]#2)
// CHECK-DAG:         [[CST_1_16_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[CST_1_17_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_7_:%.+]] = affine.apply [[MAP_4_]]([[VAR_1_]]#2, [[VAR_1_]]#3)
// CHECK-DAG:         [[CST_1_18_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_8_:%.+]] = affine.apply [[MAP_5_]]([[VAR_1_]]#2, [[VAR_1_]]#3)
// CHECK-DAG:         [[VAR_9_:%.+]] = affine.max [[MAP_6_]]([[VAR_1_]]#2, [[VAR_1_]]#3)
// CHECK-DAG:         [[VAR_10_:%.+]] = affine.min [[MAP_7_]]([[VAR_1_]]#2, [[VAR_1_]]#3)
// CHECK-DAG:         [[CST_0_10_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[VAR_11_:%.+]] = affine.min [[MAP_6_]]([[VAR_1_]]#2, [[VAR_1_]]#3)
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[VAR_12_:%.+]] = arith.subi [[VAR_5_]], [[VAR_4_]] : index
// CHECK-DAG:         [[VAR_13_:%.+]] = arith.subi [[VAR_10_]], [[VAR_9_]] : index
// CHECK-DAG:         [[LOOP_1_:%.+]]:2 = krnl.define_loops 2
// CHECK:             krnl.iterate([[LOOP_1_]]#0, [[LOOP_1_]]#1) with ([[LOOP_1_]]#0 -> [[I_4_:%.+]] = 0 to min [[MAP_8_]]([[VAR_1_]]#2){{.}}[[CST_32_2_]], [[CST_2_4_]], [[CST_0_7_]], [[CST_1_9_]], [[CST_1_10_]]{{.}}, [[LOOP_1_]]#1 -> [[I_5_:%.+]] = 0 to min [[MAP_8_]]([[VAR_1_]]#3){{.}}[[CST_32_3_]], [[CST_2_5_]], [[CST_0_8_]], [[CST_1_11_]], [[CST_1_12_]]{{.}}){
// CHECK-DAG:           [[CST_0_11_:%.+]] = arith.constant 0 : index
// CHECK-DAG:           [[VAR_16_:%.+]] = affine.apply [[MAP_9_]]([[VAR_1_]]#2, [[VAR_1_]]#3, [[I_4_]])
// CHECK-DAG:           [[VAR_17_:%.+]] = affine.apply [[MAP_10_]]([[VAR_1_]]#2, [[VAR_1_]]#3, [[I_4_]])
// CHECK-DAG:           [[VAR_18_:%.+]] = affine.apply [[MAP_11_]]([[VAR_1_]]#2, [[VAR_1_]]#3, [[I_4_]])
// CHECK-DAG:           [[VAR_19_:%.+]] = affine.max [[MAP_12_]]([[VAR_1_]]#2, [[VAR_1_]]#3, [[I_4_]])
// CHECK-DAG:           [[CST_0_12_:%.+]] = arith.constant 0 : index
// CHECK-DAG:           [[VAR_20_:%.+]] = affine.apply [[MAP_13_]]([[VAR_1_]]#2, [[VAR_1_]]#3, [[I_4_]], [[I_5_]])
// CHECK-DAG:           [[VAR_21_:%.+]] = affine.apply [[MAP_14_]]([[VAR_1_]]#2, [[VAR_1_]]#3, [[I_4_]], [[I_5_]])
// CHECK-DAG:           [[VAR_22_:%.+]] = affine.apply [[MAP_15_]]([[VAR_1_]]#2, [[VAR_1_]]#3, [[I_4_]], [[I_5_]])
// CHECK-DAG:           [[VAR_23_:%.+]] = affine.max [[MAP_16_]]([[VAR_1_]]#2, [[VAR_1_]]#3, [[I_4_]], [[I_5_]])
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]9, [[VAR_23_]]{{.}} : memref<1x3x32x32xf32>
// CHECK-DAG:           [[LOAD_RES_1_MEM_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:               [[VAR_26_:%.+]] = arith.maxnumf [[LOAD_RES_1_MEM_]], [[LOAD_PARAM_0_MEM_]] : f32
// CHECK:               krnl.store [[VAR_26_]], [[RES_1_]][] : memref<f32>
// CHECK:             }
// CHECK:             [[LOAD_RES_1_MEM_1_:%.+]] = krnl.load [[RES_1_]][] : memref<f32>
// CHECK:             krnl.store [[LOAD_RES_1_MEM_1_]], [[RES_]]{{.}}[[VAR_1_]]#0, [[VAR_1_]]#1, [[VAR_1_]]#2, [[VAR_1_]]#3] : memref<1x3x31x31xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1x3x31x31xf32>
// CHECK:         }
}

