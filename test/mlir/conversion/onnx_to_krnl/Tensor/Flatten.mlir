// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s

func.func private @test_flatten0(%arg0 : tensor<2x3x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Flatten"(%arg0) {axis = 0 : si64} : (tensor<2x3x4xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()
  // CHECK: [[MAP_FIRST:#.+]] = affine_map<() -> (0)>
  // CHECK: [[MAP_SECOND:#.+]] = affine_map<(d0, d1, d2)[s0, s1, s2] -> (d2 + d1 * s2 + d0 * (s1 * s2))>
  // CHECK-LABEL test_flatten0
  // CHECK:  [[ALLOC:%.+]] = memref.alloc() {{.*}}: memref<1x24xf32>
  // CHECK:  [[LOOP:%.+]]:3 = krnl.define_loops 3
  // CHECK:  krnl.iterate([[LOOP]]#0, [[LOOP]]#1, [[LOOP]]#2) with ([[LOOP]]#0 -> [[LOOPARG1:%.+]] = 0 to 2, [[LOOP]]#1 -> [[LOOPARG2:%.+]] = 0 to 3, [[LOOP]]#2 -> [[LOOPARG3:%.+]] = 0 to 4){
  // CHECK:    [[LOAD:%.+]] = krnl.load %arg0{{\[}}[[LOOPARG1]], [[LOOPARG2]], [[LOOPARG3]]{{\]}} : memref<2x3x4xf32>
  // CHECK:    [[FIRSTDIM:%.+]] = affine.apply [[MAP_FIRST]]()
  // CHECK:    [[C0:%.+]] = arith.constant 0 : index
  // CHECK:    [[R4:%.+]] = arith.constant 2 : index
  // CHECK:    [[C1:%.+]] = arith.constant 1 : index
  // CHECK:    [[R5:%.+]] = arith.constant 3 : index
  // CHECK:    [[C2:%.+]] = arith.constant 2 : index
  // CHECK:    [[R6:%.+]] = arith.constant 4 : index
  // CHECK:    [[SECONDDIM:%.+]] = affine.apply [[MAP_SECOND]]([[LOOPARG1]], [[LOOPARG2]], [[LOOPARG3]]){{\[}}[[R4]], [[R5]], [[R6]]{{\]}}
  // CHECK:    krnl.store [[LOAD]], [[ALLOC]]{{\[}}[[FIRSTDIM]], [[SECONDDIM]]{{\]}} : memref<1x24xf32>
}

// -----

// test partially known input shape
func.func private @test_flatten1(%arg0 : tensor<2x?x4xf32>) -> tensor<*xf32> {
  %1 = "onnx.Flatten"(%arg0) {axis = 2 : si64} : (tensor<2x?x4xf32>) -> tensor<*xf32>
  "func.return"(%1) : (tensor<*xf32>) -> ()

// mlir2FileCheck.py
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<(d0, d1)[s0, s1] -> (d1 + d0 * s1)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<(d0)[s0] -> (d0)>
// CHECK-LABEL:  func.func private @test_flatten1
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<2x?x4xf32>) -> memref<?x4xf32> {
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = arith.muli [[CST_1_]], [[CST_2_]] : index
// CHECK-DAG:       [[CST_1_1_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_1_]] : memref<2x?x4xf32>
// CHECK:           [[VAR_1_:%.+]] = arith.muli [[VAR_0_]], [[VAR_dim_]] : index
// CHECK-DAG:       [[RES_:%.+]] = memref.alloc([[VAR_1_]]) {{.*}}: memref<?x4xf32>
// CHECK-DAG:       [[LOOP_0_:%.+]]:3 = krnl.define_loops 3
// CHECK-DAG:       [[CST_1_2_:%.+]] = arith.constant 1 : index
// CHECK:           [[VAR_dim_2_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_2_]] : memref<2x?x4xf32>
// CHECK:           krnl.iterate([[LOOP_0_]]#0, [[LOOP_0_]]#1, [[LOOP_0_]]#2) with ([[LOOP_0_]]#0 -> [[I_0_:%.+]] = 0 to 2, [[LOOP_0_]]#1 -> [[I_1_:%.+]] = 0 to [[VAR_dim_2_]], [[LOOP_0_]]#2 -> [[I_2_:%.+]] = 0 to 4){
// CHECK-DAG:         [[LOAD_PARAM_0_MEM_:%.+]] = krnl.load [[PARAM_0_]]{{.}}[[I_0_]], [[I_1_]], [[I_2_]]{{.}} : memref<2x?x4xf32>
// CHECK-DAG:         [[CST_0_1_:%.+]] = arith.constant 0 : index
// CHECK-DAG:         [[CST_2_1_:%.+]] = arith.constant 2 : index
// CHECK-DAG:         [[CST_1_3_:%.+]] = arith.constant 1 : index
// CHECK:             [[VAR_dim_6_:%.+]] = memref.dim [[PARAM_0_]], [[CST_1_3_]] : memref<2x?x4xf32>
// CHECK-DAG:         [[VAR_4_:%.+]] = affine.apply [[MAP_0_]]([[I_0_]], [[I_1_]]){{.}}[[CST_2_1_]], [[VAR_dim_6_]]{{.}}
// CHECK-DAG:         [[CST_2_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:         [[CST_4_:%.+]] = arith.constant 4 : index
// CHECK:             [[VAR_5_:%.+]] = affine.apply [[MAP_1_]]([[I_2_]]){{.}}[[CST_4_]]{{.}}
// CHECK:             krnl.store [[LOAD_PARAM_0_MEM_]], [[RES_]]{{.}}[[VAR_4_]], [[VAR_5_]]{{.}} : memref<?x4xf32>
// CHECK:           }
// CHECK:           return [[RES_]] : memref<?x4xf32>
// CHECK:         }
}

