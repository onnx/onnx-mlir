// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
func.func private @test_reshape(%arg0 : tensor<?x10xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<4xi64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func private @test_reshape
// CHECK:          ([[PARAM_0_:%.+]]: memref<?x10xf32>, [[PARAM_1_:%.+]]: memref<4xi64>) -> memref<?x?x?x?xf32> {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_0_]]{{.}}
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_3_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:       [[VAR_5_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[VAR_4_:%.+]] = arith.cmpi eq, [[VAR_3_]], [[CST_0_]] : index
// CHECK:           [[VAR_6_:%.+]] = arith.select [[VAR_4_]], [[VAR_5_]], [[VAR_3_]] : index
// CHECK:           [[VAR_7_:%.+]] = arith.cmpi eq, [[VAR_6_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_8_:%.+]] = arith.select [[VAR_7_]], [[CST_1_]], [[VAR_6_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_1_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_10_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_1_]] : i64 to index
// CHECK:           [[VAR_11_:%.+]] = arith.cmpi eq, [[VAR_10_]], [[CST_0_]] : index
// CHECK:           [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_10_]], [[VAR_10_]] : index
// CHECK:           [[VAR_13_:%.+]] = arith.cmpi eq, [[VAR_12_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_14_:%.+]] = arith.select [[VAR_13_]], [[CST_1_]], [[VAR_12_]] : index
// CHECK-DAG:       [[VAR_15_:%.+]] = arith.muli [[VAR_8_]], [[VAR_14_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_2_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_2_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_17_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_2_]] : i64 to index
// CHECK:           [[VAR_18_:%.+]] = arith.cmpi eq, [[VAR_17_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_19_:%.+]] = arith.select [[VAR_18_]], [[CST_1_]], [[VAR_17_]] : index
// CHECK-DAG:       [[VAR_20_:%.+]] = arith.muli [[VAR_15_]], [[VAR_19_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_3_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_3_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_22_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_3_]] : i64 to index
// CHECK:           [[VAR_23_:%.+]] = arith.cmpi eq, [[VAR_22_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_24_:%.+]] = arith.select [[VAR_23_]], [[CST_1_]], [[VAR_22_]] : index
// CHECK-DAG:       [[VAR_25_:%.+]] = arith.muli [[VAR_20_]], [[VAR_24_]] : index
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.cmpi eq, [[VAR_6_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_27_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_25_]] : index
// CHECK-DAG:       [[VAR_29_:%.+]] = arith.cmpi eq, [[VAR_12_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_30_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_25_]] : index
// CHECK-DAG:       [[VAR_28_:%.+]] = arith.select [[VAR_26_]], [[VAR_27_]], [[VAR_6_]] : index
// CHECK-DAG:       [[VAR_31_:%.+]] = arith.select [[VAR_29_]], [[VAR_30_]], [[VAR_12_]] : index
// CHECK-DAG:       [[VAR_32_:%.+]] = arith.cmpi eq, [[VAR_17_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_33_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_25_]] : index
// CHECK-DAG:       [[VAR_34_:%.+]] = arith.select [[VAR_32_]], [[VAR_33_]], [[VAR_17_]] : index
// CHECK-DAG:       [[VAR_36_:%.+]] = arith.floordivsi [[VAR_1_]], [[VAR_25_]] : index
// CHECK-DAG:       [[VAR_35_:%.+]] = arith.cmpi eq, [[VAR_22_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_37_:%.+]] = arith.select [[VAR_35_]], [[VAR_36_]], [[VAR_22_]] : index
// CHECK:           [[VAR_38_:%.+]] = arith.muli [[VAR_37_]], [[VAR_34_]] : index
// CHECK:           [[VAR_39_:%.+]] = arith.muli [[VAR_38_]], [[VAR_31_]] : index
// CHECK:           [[VAR_40_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: {{.}}[[VAR_28_]], [[VAR_31_]], [[VAR_34_]], [[VAR_37_]]{{.}}, strides: {{.}}[[VAR_39_]], [[VAR_38_]], [[VAR_37_]], 1] : memref<?x10xf32> to memref<?x?x?x?xf32>
// CHECK:           return [[VAR_40_]] : memref<?x?x?x?xf32>
// CHECK:         }
}

// -----

func.func @test_reshape_to_identity(%arg0 : tensor<?x?x?xf32>) -> tensor<*xf32> {
  %minus_one = onnx.Constant dense<-1> : tensor<1xi64>
  %dim1 = "onnx.Dim"(%arg0) {axis = 1 : si64} : (tensor<?x?x?xf32>) -> tensor<1xi64>
  %dim2 = "onnx.Dim"(%arg0) {axis = 2 : si64} : (tensor<?x?x?xf32>) -> tensor<1xi64>
  %shape = "onnx.Concat"(%minus_one, %dim1, %dim2) {axis = 0 : si64} : (tensor<1xi64>, tensor<1xi64>, tensor<1xi64>) -> tensor<3xi64>
  %0 = "onnx.Reshape"(%arg0, %shape) : (tensor<?x?x?xf32>, tensor<3xi64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func.func @test_reshape_to_identity
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x?x?xf32>) -> memref<?x?x?xf32> {
// CHECK:           return [[PARAM_0_]] : memref<?x?x?xf32>
// CHECK:         }
}
