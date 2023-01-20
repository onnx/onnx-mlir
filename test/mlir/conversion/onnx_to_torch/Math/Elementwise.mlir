// RUN: onnx-mlir-opt --convert-onnx-to-torch-pipeline --canonicalize %s -split-input-file | FileCheck %s

func.func @test_add(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<10x10xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<10x10xf32>
  "func.return"(%0) : (tensor<10x10xf32>) -> ()
// CHECK-LABEL:  func @test_add
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !torch.vtensor<[10,10],f32>, [[PARAM_1_:%.+]]: !torch.vtensor<[10,10],f32>) -> !torch.vtensor<[10,10],f32> {
// CHECK-NEXT:      [[INT_1_:%.+]] = torch.constant.int 1
// CHECK-NEXT:      [[VAR_0_:%.+]] = torch.aten.add.Tensor [[PARAM_0_]], [[PARAM_1_]], [[INT_1_]] : !torch.vtensor<[10,10],f32>, !torch.vtensor<[10,10],f32>, !torch.int -> !torch.vtensor<[10,10],f32>
// CHECK-NEXT:      return [[VAR_0_]] : !torch.vtensor<[10,10],f32>
// CHECK-NEXT:    }
}

func.func @test_add_dynamic(%arg0 : tensor<?x10xf32>, %arg1 : tensor<?x10xf32>) -> tensor<?x10xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<?x10xf32>, tensor<?x10xf32>) -> tensor<?x10xf32>
  "func.return"(%0) : (tensor<?x10xf32>) -> ()
// CHECK-LABEL:  func @test_add_dynamic
// CHECK-SAME:   ([[PARAM_0_:%.+]]: !torch.vtensor<[?,10],f32>, [[PARAM_1_:%.+]]: !torch.vtensor<[?,10],f32>) -> !torch.vtensor<[?,10],f32> {
// CHECK-NEXT:      [[INT_1_:%.+]] = torch.constant.int 1
// CHECK-NEXT:      [[VAR_0_:%.+]] = torch.aten.add.Tensor [[PARAM_0_]], [[PARAM_1_]], [[INT_1_]] : !torch.vtensor<[?,10],f32>, !torch.vtensor<[?,10],f32>, !torch.int -> !torch.vtensor<[?,10],f32>
// CHECK-NEXT:      return [[VAR_0_]] : !torch.vtensor<[?,10],f32>
// CHECK-NEXT:    }
}
