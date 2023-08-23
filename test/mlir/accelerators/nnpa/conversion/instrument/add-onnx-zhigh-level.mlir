// RUN: onnx-mlir --maccel=NNPA --printIR --EmitZHighIR --instrument-stage=ZHigh --instrument-ops="onnx.*,zhigh.*" --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime -tag="test" %s  | FileCheck %s

func.func @test_instrument_add_onnx_zhigh(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x1xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x1xf32>) -> tensor<*xf32>
  %1 = "onnx.Add"(%arg0, %0) : (tensor<10x10xf32>, tensor<*xf32>) -> tensor<*xf32>  
  "onnx.Return"(%1) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_instrument_add_onnx_zhigh
// CHECK:           "krnl.runtime_instrument"() {opName = "onnx.Add", tag = 5 : i64} : () -> ()
// CHECK:           "onnx.Add"
// CHECK:           "krnl.runtime_instrument"() {opName = "onnx.Add", tag = 6 : i64} : () -> ()
// CHECK:           "krnl.runtime_instrument"() {opName = "zhigh.Stick", tag = 5 : i64} : () -> ()
// CHECK:           "zhigh.Stick"
// CHECK:           "krnl.runtime_instrument"() {opName = "zhigh.Stick", tag = 6 : i64} : () -> ()
// CHECK:           "krnl.runtime_instrument"() {opName = "zhigh.Stick", tag = 5 : i64} : () -> ()
// CHECK:           "zhigh.Stick"
// CHECK:           "krnl.runtime_instrument"() {opName = "zhigh.Stick", tag = 6 : i64} : () -> ()
// CHECK:           "krnl.runtime_instrument"() {opName = "zhigh.Add", tag = 5 : i64} : () -> ()
// CHECK:           "zhigh.Add"
// CHECK:           "krnl.runtime_instrument"() {opName = "zhigh.Add", tag = 6 : i64} : () -> ()
// CHECK:           "krnl.runtime_instrument"() {opName = "zhigh.Unstick", tag = 5 : i64} : () -> ()
// CHECK:           "zhigh.Unstick"
// CHECK:           "krnl.runtime_instrument"() {opName = "zhigh.Unstick", tag = 6 : i64} : () -> ()
// CHECK:           return
// CHECK:         }

// -----
