// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

// -----

// Fuse both Sqrt to Add
func.func @test_fuse_element3(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %1 = "onnx.Sqrt"(%0) : (tensor<1024xf32>) -> tensor<1024xf32>
    %2 = "onnx.Sqrt"(%1) : (tensor<1024xf32>) -> tensor<1024xf32>
    return %2 : tensor<1024xf32>
}
// CHECK-LABEL:  func.func @test_fuse_element3
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1024xf32>, [[PARAM_1_:%.+]]: memref<1024xf32>) -> memref<1024xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[CST_1024_1_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1024xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1024_2_:%.+]] = arith.constant 1024 : index
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 1024){
// CHECK-DAG:         [[VAR_1_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[CST_1024_3_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_1_]]{{.}} : memref<1024xf32>
// CHECK-DAG:         [[CST_1024_4_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_true_7_:%.+]] = arith.constant true
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_1_]]{{.}} : memref<1024xf32>
// CHECK:             [[VAR_4_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_5_:%.+]] = math.sqrt [[VAR_4_]] : f32
// CHECK:             [[VAR_6_:%.+]] = math.sqrt [[VAR_5_]] : f32
// CHECK:             krnl.store [[VAR_6_]], [[RES_]]{{.}}[[VAR_1_]]{{.}} : memref<1024xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<1024xf32>

// -----

// Stop fusion after the first Sqrt because it has more one user 
  func.func @test_fuse_element4(%arg0: tensor<1024xf32>, %arg1: tensor<1024xf32>) -> tensor<1024xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
    %0 = "onnx.Add"(%arg0, %arg1) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    %1 = "onnx.Sqrt"(%0) : (tensor<1024xf32>) -> tensor<1024xf32>
    %2 = "onnx.Sqrt"(%1) : (tensor<1024xf32>) -> tensor<1024xf32>
    %3 = "onnx.Add"(%1, %2) : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<1024xf32>
    return %3 : tensor<1024xf32>
  }
// CHECK-LABEL:  func.func @test_fuse_element4
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<1024xf32>, [[PARAM_1_:%.+]]: memref<1024xf32>) -> memref<1024xf32> attributes {input_names = ["x", "y"], output_names = ["sum"]} {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1024_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[CST_1024_1_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc() {{.*}}: memref<1024xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1024_2_:%.+]] = arith.constant 1024 : index
// CHECK:           krnl.iterate([[LOOP_0_]]) with ([[LOOP_0_]] -> [[I_0_:%.+]] = 0 to 1024){
// CHECK-DAG:         [[VAR_3_:%.+]] = krnl.get_induction_var_value([[LOOP_0_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[CST_1024_3_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:         [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_true_:%.+]] = arith.constant true
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[VAR_3_]]{{.}} : memref<1024xf32>
// CHECK-DAG:         [[CST_1024_4_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:         [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_true_17_:%.+]] = arith.constant true
// CHECK-DAG:         [[CST_0_2_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[VAR_3_]]{{.}} : memref<1024xf32>
// CHECK:             [[VAR_6_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:             [[VAR_7_:%.+]] = math.sqrt [[VAR_6_]] : f32
// CHECK:             krnl.store [[VAR_7_]], [[RES_]]{{.}}[[VAR_3_]]{{.}} : memref<1024xf32>
// CHECK:           }
// CHECK-DAG:       [[CST_1024_5_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[RES_1_:%.+]] = memref.alloc() {{.*}}: memref<1024xf32>
// CHECK-DAG:       [[LOOP_1_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[CST_0_3_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1024_6_:%.+]] = arith.constant 1024 : index
// CHECK:           krnl.iterate([[LOOP_1_]]) with ([[LOOP_1_]] -> [[I_1_:%.+]] = 0 to 1024){
// CHECK:             [[VAR_3_1_:%.+]] = krnl.get_induction_var_value([[LOOP_1_]]) : (!krnl.loop) -> index
// CHECK:             [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_3_1_]]{{.}} : memref<1024xf32>
// CHECK:             [[LOAD_PARAM_1_MEM_1_:%.+]] = math.sqrt [[LOAD_PARAM_0_MEM_1_]] : f32
// CHECK:             krnl.store [[LOAD_PARAM_1_MEM_1_]], [[RES_1_]]{{.}}[[VAR_3_1_]]{{.}} : memref<1024xf32>
// CHECK:           }
// CHECK-DAG:       [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_1024_7_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[CST_1024_8_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:       [[RES_2_:%.+]] = memref.alloc() {{.*}}: memref<1024xf32>
// CHECK-DAG:       [[LOOP_2_:%.+]] = krnl.define_loops 1
// CHECK-DAG:       [[CST_0_4_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_1024_9_:%.+]] = arith.constant 1024 : index
// CHECK:           krnl.iterate([[LOOP_2_]]) with ([[LOOP_2_]] -> [[I_2_:%.+]] = 0 to 1024){
// CHECK-DAG:         [[VAR_3_2_:%.+]] = krnl.get_induction_var_value([[LOOP_2_]]) : (!krnl.loop) -> index
// CHECK-DAG:         [[CST_1024_10_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:         [[CST_1_4_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_true_1_:%.+]] = arith.constant true
// CHECK-DAG:         [[CST_0_5_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_1_:%.+]] = krnl.load [[RES_]]{{.}}[[VAR_3_2_]]{{.}} : memref<1024xf32>
// CHECK-DAG:         [[CST_1024_11_:%.+]] = arith.constant 1024 : index
// CHECK-DAG:         [[CST_1_5_:%.+]] = arith.constant 1 : index
// CHECK-DAG:         [[VAR_true_17_1_:%.+]] = arith.constant true
// CHECK-DAG:         [[CST_0_6_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[RES_1_]]{{.}}[[VAR_3_2_]]{{.}} : memref<1024xf32>
// CHECK:             [[VAR_6_1_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_1_]], [[LOAD_PARAM_1_MEM_1_]] : f32
// CHECK:             krnl.store [[VAR_6_1_]], [[RES_2_]]{{.}}[[VAR_3_2_]]{{.}} : memref<1024xf32>
// CHECK:           }
// CHECK:           return [[RES_2_]] : memref<1024xf32>
// CHECK:         }
