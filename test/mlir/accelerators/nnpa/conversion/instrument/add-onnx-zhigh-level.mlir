// RUN: onnx-mlir --march=z16 --maccel=NNPA --printIR --EmitZHighIR --profile-ir=ZHigh %s  | FileCheck %s

// -----

func.func @test_instrument_add_onnx_zhigh(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "onnx.Add1"} : (tensor<10x10xf32>, tensor<10x1xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%arg0, %0) {onnx_node_name = "onnx.Add2"} : (tensor<10x10xf32>, tensor<*xf32>) -> tensor<*xf32>  
  %2 = "onnx.Relu"(%1) {onnx_node_name = "onnx.Relu"} : (tensor<*xf32>) -> tensor<*xf32>
  "onnx.Return"(%2) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_instrument_add_onnx_zhigh
// CHECK:           "krnl.runtime_instrument"() {nodeName = "onnx.Add1", opName = "onnx.Add", tag = 21 : i64} : () -> ()
// CHECK:           "onnx.Add"
// CHECK:           "krnl.runtime_instrument"() {nodeName = "onnx.Add1", opName = "onnx.Add", tag = 6 : i64} : () -> ()
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-onnx-zhigh-level.mlir:7", opName = "zhigh.Stick", tag = 5 : i64} : () -> ()
// CHECK:           "zhigh.Stick"
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-onnx-zhigh-level.mlir:7", opName = "zhigh.Stick", tag = 6 : i64} : () -> ()
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-onnx-zhigh-level.mlir:7", opName = "zhigh.Stick", tag = 5 : i64} : () -> ()
// CHECK:           "zhigh.Stick"
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-onnx-zhigh-level.mlir:7", opName = "zhigh.Stick", tag = 6 : i64} : () -> ()
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-onnx-zhigh-level.mlir:7", opName = "zhigh.Add", tag = 5 : i64} : () -> ()
// CHECK:           "zhigh.Add"
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-onnx-zhigh-level.mlir:7", opName = "zhigh.Add", tag = 6 : i64} : () -> ()
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-onnx-zhigh-level.mlir:8", opName = "zhigh.Relu", tag = 5 : i64} : () -> ()
// CHECK:           "zhigh.Relu"
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-onnx-zhigh-level.mlir:8", opName = "zhigh.Relu", tag = 6 : i64} : () -> ()
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-onnx-zhigh-level.mlir:8", opName = "zhigh.Unstick", tag = 5 : i64} : () -> ()
// CHECK:           "zhigh.Unstick"
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-onnx-zhigh-level.mlir:8", opName = "zhigh.Unstick", tag = 6 : i64} : () -> ()
// CHECK:           return
// CHECK:         }

// -----
