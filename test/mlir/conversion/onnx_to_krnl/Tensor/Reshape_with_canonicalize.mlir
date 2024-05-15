// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// Adding canonicalize is important here as this is the only way to check the values of the map,
// which are otherwise before the function, and thus are hard to test.

// CHECK-DAG: [[MAP_0_:#.+]] = affine_map<()[s0] -> (s0 * 10)>
func.func private @test_reshape(%arg0 : tensor<?x10xf32>, %arg1 : tensor<4xi64>) -> tensor<*xf32> {
  %0 = "onnx.Reshape"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<4xi64>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()

// CHECK-LABEL:  func.func private @test_reshape
// CHECK-SAME:   ([[PARAM_0_:%.+]]: memref<?x10xf32>, [[PARAM_1_:%.+]]: memref<4xi64>) -> memref<?x?x?x?xf32> {
// CHECK-DAG:       [[CST_3_:%.+]] = arith.constant 3 : index
// CHECK-DAG:       [[CST_2_:%.+]] = arith.constant 2 : index
// CHECK-DAG:       [[CST_minus_1_:%.+]] = arith.constant -1 : index
// CHECK-DAG:       [[CST_10_:%.+]] = arith.constant 10 : index
// CHECK-DAG:       [[CST_1_:%.+]] = arith.constant 1 : index
// CHECK-DAG:       [[CST_0_:%.+]] = arith.constant 0 : index
// CHECK:           [[VAR_dim_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK-DAG:       [[VAR_0_:%.+]] = affine.apply [[MAP_0_]](){{.}}[[VAR_dim_]]{{.}}
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_2_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_]] : i64 to index
// CHECK-DAG:       [[VAR_3_:%.+]] = arith.cmpi eq, [[VAR_2_]], [[CST_0_]] : index
// CHECK-DAG:       [[VAR_dim_0_:%.+]] = memref.dim [[PARAM_0_]], [[CST_0_]] : memref<?x10xf32>
// CHECK:           [[VAR_4_:%.+]] = arith.select [[VAR_3_]], [[VAR_dim_0_]], [[VAR_2_]] : index
// CHECK:           [[VAR_5_:%.+]] = arith.cmpi eq, [[VAR_4_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_6_:%.+]] = arith.select [[VAR_5_]], [[CST_1_]], [[VAR_4_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_1_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_1_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_8_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_1_]] : i64 to index
// CHECK:           [[VAR_9_:%.+]] = arith.cmpi eq, [[VAR_8_]], [[CST_0_]] : index
// CHECK:           [[VAR_10_:%.+]] = arith.select [[VAR_9_]], [[CST_10_]], [[VAR_8_]] : index
// CHECK:           [[VAR_11_:%.+]] = arith.cmpi eq, [[VAR_10_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_12_:%.+]] = arith.select [[VAR_11_]], [[CST_1_]], [[VAR_10_]] : index
// CHECK-DAG:       [[VAR_13_:%.+]] = arith.muli [[VAR_6_]], [[VAR_12_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_2_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_2_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_15_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_2_]] : i64 to index
// CHECK:           [[VAR_16_:%.+]] = arith.cmpi eq, [[VAR_15_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_17_:%.+]] = arith.select [[VAR_16_]], [[CST_1_]], [[VAR_15_]] : index
// CHECK-DAG:       [[VAR_18_:%.+]] = arith.muli [[VAR_13_]], [[VAR_17_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_3_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_3_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_20_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_3_]] : i64 to index
// CHECK:           [[VAR_21_:%.+]] = arith.cmpi eq, [[VAR_20_]], [[CST_minus_1_]] : index
// CHECK:           [[VAR_22_:%.+]] = arith.select [[VAR_21_]], [[CST_1_]], [[VAR_20_]] : index
// CHECK-DAG:       [[VAR_23_:%.+]] = arith.muli [[VAR_18_]], [[VAR_22_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_4_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_0_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_25_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_4_]] : i64 to index
// CHECK-DAG:       [[VAR_26_:%.+]] = arith.cmpi eq, [[VAR_25_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_27_:%.+]] = arith.floordivsi [[VAR_0_]], [[VAR_23_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_28_:%.+]] = arith.select [[VAR_26_]], [[VAR_27_]], [[VAR_4_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_5_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_1_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_30_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_5_]] : i64 to index
// CHECK-DAG:       [[VAR_31_:%.+]] = arith.cmpi eq, [[VAR_30_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_32_:%.+]] = arith.floordivsi [[VAR_0_]], [[VAR_23_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_33_:%.+]] = arith.select [[VAR_31_]], [[VAR_32_]], [[VAR_10_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_6_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_2_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_35_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_6_]] : i64 to index
// CHECK-DAG:       [[VAR_36_:%.+]] = arith.cmpi eq, [[VAR_35_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_37_:%.+]] = arith.floordivsi [[VAR_0_]], [[VAR_23_]] : index
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_38_:%.+]] = arith.select [[VAR_36_]], [[VAR_37_]], [[VAR_15_]] : index
// CHECK-DAG:       [[LOAD_PARAM_1_MEM_7_:%.+]] = krnl.load [[PARAM_1_]]{{.}}[[CST_3_]]{{.}} : memref<4xi64>
// CHECK:           [[VAR_40_:%.+]] = arith.index_cast [[LOAD_PARAM_1_MEM_7_]] : i64 to index
// CHECK-DAG:       [[VAR_41_:%.+]] = arith.cmpi eq, [[VAR_40_]], [[CST_minus_1_]] : index
// CHECK-DAG:       [[VAR_42_:%.+]] = arith.floordivsi [[VAR_0_]], [[VAR_23_]] : index
// CHECK:           [[VAR_43_:%.+]] = arith.select [[VAR_41_]], [[VAR_42_]], [[VAR_20_]] : index
// CHECK:           [[VAR_44_:%.+]] = arith.muli [[VAR_43_]], [[VAR_38_]] : index
// CHECK:           [[VAR_45_:%.+]] = arith.muli [[VAR_44_]], [[VAR_33_]] : index
// CHECK:           [[VAR_reinterpret_cast_:%.+]] = memref.reinterpret_cast [[PARAM_0_]] to offset: [0], sizes: {{.}}[[VAR_28_]], [[VAR_33_]], [[VAR_38_]], [[VAR_43_]]{{.}}, strides: {{.}}[[VAR_45_]], [[VAR_44_]], [[VAR_43_]], 1] : memref<?x10xf32> to memref<?x?x?x?xf32>
// CHECK:           return [[VAR_reinterpret_cast_]] : memref<?x?x?x?xf32>
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
