// RUN: onnx-mlir-opt --convert-onnx-to-krnl --canonicalize %s -split-input-file | FileCheck %s

// -----

// COM: Test reshape lowering when its constant shape input is promoted to attribute
func private @test_reshape_attribute_promotion(%arg0 : tensor<?x10xf32>) -> tensor<?x5xf32> {
  %cst = constant unit
  %0 = "onnx.Reshape"(%arg0, %cst) {shape = dense<[-1, 5]> : tensor<2xi64>}: (tensor<?x10xf32>, none) -> tensor<?x5xf32>
  "std.return"(%0) : (tensor<?x5xf32>) -> ()

  // CHECK-LABEL: test_reshape_attribute_promotion
  // CHECK-DAG:   [[CST_0_:%.+]] = constant 0 : index
  // CHECK-DAG:   [[CST_10_:%.+]] = constant 10 : i64
  // CHECK-DAG:   [[FLOAT_SIZE:%.+]] = constant 4 : i64
  // CHECK-DAG:   [[OUTPUT_SIZE:%.+]] = constant 20 : i64

  // CHECK:       [[DIM_0_:%.+]] = dim %arg0, [[CST_0_]] : memref<?x10xf32>
  // CHECK:       [[DIM_0_i64:%.+]] = index_cast [[DIM_0_]] : index to i64
  // CHECK:       [[MUL:%.+]] = muli [[DIM_0_i64]], [[FLOAT_SIZE]] : i64
  // CHECK:       [[INPUT_SIZE:%.+]] = muli [[MUL]], [[CST_10_]] : i64

  // CHECK:       [[UNKNOWN_DIM_VALUE:%.+]] = divi_signed [[INPUT_SIZE]], [[OUTPUT_SIZE]] : i64
  // CHECK:       [[UNKNOWN_DIM_VALUE_i64:%.+]] = index_cast [[UNKNOWN_DIM_VALUE]] : i64 to index
  // CHECK:       [[RES_:%.+]] = alloc([[UNKNOWN_DIM_VALUE_i64]]) : memref<?x5xf32>
  // CHECK:       "krnl.memcpy"([[RES_]], %arg0, [[INPUT_SIZE]]) : (memref<?x5xf32>, memref<?x10xf32>, i64) -> ()
  // CHECK:       return [[RES_]] : memref<?x5xf32>
}
