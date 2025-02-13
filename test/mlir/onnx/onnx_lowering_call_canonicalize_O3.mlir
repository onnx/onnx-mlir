// RUN: onnx-mlir-opt -O3 --mtriple=s390x-ibm-loz --march=z16 --shape-inference --convert-onnx-to-krnl='ops-for-call=Conv' --canonicalize %s -split-input-file | FileCheck %s

// use --mtriple=s390x-ibm-loz --march=z16 to enable SIMD as we now need a machine
// can also use --march=x86-64 instead.

// -----

func.func private @test_conv_unknown_dimensions(%arg0 : tensor<?x?x?x?xf32>, %arg1 : tensor<5x2x6x7xf32>, %arg2 : tensor<5xf32>) -> tensor<*xf32> {
  %0 = "onnx.Conv"(%arg0, %arg1, %arg2) {auto_pad = "NOTSET", group = 1 : si64} : (tensor<?x?x?x?xf32>, tensor<5x2x6x7xf32>, tensor<5xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> () 
// CHECK-DAG:   [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 - 5)>
// CHECK-DAG:   [[MAP_1_:#.+]] = affine_map<()[s0] -> (s0 - 6)>
// CHECK-LABEL:  func.func private @test_conv_unknown_dimensions
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?x?xf32>, [[PARAM_1_:%.+]]: memref<5x2x6x7xf32>, [[PARAM_2_:%.+]]: memref<5xf32>) -> memref<?x5x?x?xf32> {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x?x?x?xf32>
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_2_]] : memref<?x?x?x?xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_0_]]{{.}}
// CHECK-DAG:       [[VAR_dim_1_:%.+]] = memref.dim [[PARAM_0_]], [[CST_3_]] : memref<?x?x?x?xf32>
// CHECK:           [[VAR_1_:%.+]] = affine.apply [[MAP_1_]](){{.}}[[VAR_dim_1_]]{{.}}
// CHECK:           [[RES_:%.+]] = memref.alloc([[VAR_dim_]], [[VAR_0_]], [[VAR_1_]]) {{.*}}: memref<?x5x?x?xf32>
// CHECK:           "krnl.call"([[RES_]], [[PARAM_0_]], [[PARAM_1_]], [[PARAM_2_]]) {auto_pad = "NOTSET", funcName = "Conv", group = 1 : si64, numOfOutput = 1 : si64} : (memref<?x5x?x?xf32>, memref<?x?x?x?xf32>, memref<5x2x6x7xf32>, memref<5xf32>) -> ()
// CHECK:           return [[RES_]] : memref<?x5x?x?xf32>
// CHECK:         }
}
