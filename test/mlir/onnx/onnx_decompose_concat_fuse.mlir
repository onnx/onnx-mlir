// RUN: onnx-mlir-opt --shape-inference --decompose-onnx=enable-concat-fuse=true %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --shape-inference --decompose-onnx=enable-concat-fuse=false %s -split-input-file | FileCheck %s --check-prefix=DISABLED-CHECK --check-prefix=COMMON-CHECK

func.func @concat_fuse_0(%arg0: tensor<?x20xf32>, %arg1: tensor<?x30xf32>) -> (tensor<2xi64>, tensor<50x?xf32>)
{
    %1 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} : (tensor<?x20xf32>, tensor<?x30xf32>) -> tensor<?x50xf32>
    %2 = "onnx.Transpose"(%1) {perm = [1, 0]} : (tensor<?x50xf32>) -> tensor<50x?xf32>
    %3 = "onnx.Shape"(%1) : (tensor<?x50xf32>) -> tensor<2xi64>
    onnx.Return %3, %2 : tensor<2xi64>, tensor<50x?xf32>
// CHECK-LABEL:  func.func @concat_fuse_0
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x20xf32>, [[PARAM_1_:%.+]]: tensor<?x30xf32>) -> (tensor<2xi64>, tensor<50x?xf32>) {
// CHECK:           [[shape_:%.+]], [[VAR_transposed_:%.+]] = "onnx.ConcatShapeTranspose"([[PARAM_0_]], [[PARAM_1_]]) {axis = 1 : si64, perm = [1, 0], start = 0 : si64} : (tensor<?x20xf32>, tensor<?x30xf32>) -> (tensor<2xi64>, tensor<50x?xf32>)
// CHECK:           onnx.Return [[shape_]], [[VAR_transposed_]] : tensor<2xi64>, tensor<50x?xf32>
// CHECK:         }

// DISABLED-CHECK-LABEL:  func.func @concat_fuse_0
// DISABLED-CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x20xf32>, [[PARAM_1_:%.+]]: tensor<?x30xf32>) -> (tensor<2xi64>, tensor<50x?xf32>) {
// DISABLED-CHECK:           [[VAR_0_:%.+]] = "onnx.Concat"([[PARAM_0_]], [[PARAM_1_]]) {axis = 1 : si64} : (tensor<?x20xf32>, tensor<?x30xf32>) -> tensor<?x50xf32>
// DISABLED-CHECK:           [[VAR_1_:%.+]] = "onnx.Transpose"([[VAR_0_]]) {perm = [1, 0]} : (tensor<?x50xf32>) -> tensor<50x?xf32>
// DISABLED-CHECK:           [[VAR_2_:%.+]] = "onnx.Shape"([[VAR_0_]]) {start = 0 : si64} : (tensor<?x50xf32>) -> tensor<2xi64>
// DISABLED-CHECK:           onnx.Return [[VAR_2_]], [[VAR_1_]] : tensor<2xi64>, tensor<50x?xf32>
// DISABLED-CHECK:         }
}

// -----

func.func @test_concatfuse_1(%arg0: tensor<?x20xf32>, %arg1: tensor<?x30xf32>) -> (tensor<?x50xf32>, tensor<2xi64>, tensor<50x?xf32>)
{
    %1 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} : (tensor<?x20xf32>, tensor<?x30xf32>) -> tensor<?x50xf32>
    %2 = "onnx.Transpose"(%1) {perm = [1, 0]} : (tensor<?x50xf32>) -> tensor<50x?xf32>
    %3 = "onnx.Shape"(%1) : (tensor<?x50xf32>) -> tensor<2xi64>
    %4 = "onnx.Sin"(%1) : (tensor<?x50xf32>) -> tensor<?x50xf32>
    onnx.Return %4, %3, %2 : tensor<?x50xf32>, tensor<2xi64>, tensor<50x?xf32>
// COMMON-CHECK-LABEL:  func.func @test_concatfuse_1
// COMMON-CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x20xf32>, [[PARAM_1_:%.+]]: tensor<?x30xf32>) -> (tensor<?x50xf32>, tensor<2xi64>, tensor<50x?xf32>) {
// COMMON-CHECK:           [[VAR_0_:%.+]] = "onnx.Concat"([[PARAM_0_]], [[PARAM_1_]]) {axis = 1 : si64} : (tensor<?x20xf32>, tensor<?x30xf32>) -> tensor<?x50xf32>
// COMMON-CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Transpose"([[VAR_0_]]) {perm = [1, 0]} : (tensor<?x50xf32>) -> tensor<50x?xf32>
// COMMON-CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Shape"([[VAR_0_]]) {start = 0 : si64} : (tensor<?x50xf32>) -> tensor<2xi64>
// COMMON-CHECK-DAG:       [[VAR_3_:%.+]] = "onnx.Sin"([[VAR_0_]]) : (tensor<?x50xf32>) -> tensor<?x50xf32>
// COMMON-CHECK:           onnx.Return [[VAR_3_]], [[VAR_2_]], [[VAR_1_]] : tensor<?x50xf32>, tensor<2xi64>, tensor<50x?xf32>
}

// -----

func.func @test_concatfuse_2(%arg0: tensor<?x20xf32>, %arg1: tensor<?x30xf32>) -> (tensor<2xi64>, tensor<?x50xf32>)
{
    %1 = "onnx.Concat"(%arg0, %arg1) {axis = 1 : si64} : (tensor<?x20xf32>, tensor<?x30xf32>) -> tensor<?x50xf32>
    %3 = "onnx.Shape"(%1) : (tensor<?x50xf32>) -> tensor<2xi64>
    %4 = "onnx.Sin"(%1) : (tensor<?x50xf32>) -> tensor<?x50xf32>
    onnx.Return %3, %4 : tensor<2xi64>, tensor<?x50xf32>
// COMMON-CHECK-LABEL:  func.func @test_concatfuse_2
// COMMON-CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<?x20xf32>, [[PARAM_1_:%.+]]: tensor<?x30xf32>) -> (tensor<2xi64>, tensor<?x50xf32>) {
// COMMON-CHECK:           [[VAR_0_:%.+]] = "onnx.Concat"([[PARAM_0_]], [[PARAM_1_]]) {axis = 1 : si64} : (tensor<?x20xf32>, tensor<?x30xf32>) -> tensor<?x50xf32>
// COMMON-CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Shape"([[VAR_0_]]) {start = 0 : si64} : (tensor<?x50xf32>) -> tensor<2xi64>
// COMMON-CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Sin"([[VAR_0_]]) : (tensor<?x50xf32>) -> tensor<?x50xf32>
// COMMON-CHECK:           onnx.Return [[VAR_1_]], [[VAR_2_]] : tensor<2xi64>, tensor<?x50xf32>
// COMMON-CHECK:         }
}