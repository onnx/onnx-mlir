// RUN: onnx-mlir --march=z16 --maccel=NNPA --printIR --EmitZLowIR --instrument-stage=ZHigh --instrument-ops=zhigh.* --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime %s  | FileCheck %s

// -----

func.func @test_instrument_add_zhigh(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_instrument_add_zhigh
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zhigh-level.mlir:6", opName = "zhigh.Stick", tag = 21 : i64} : () -> ()
// CHECK:           memref.alloc()
// CHECK:           "zlow.stick"
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zhigh-level.mlir:6", opName = "zhigh.Stick", tag = 6 : i64} : () -> ()
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zhigh-level.mlir:6", opName = "zhigh.Stick", tag = 5 : i64} : () -> ()
// CHECK:           memref.alloc()
// CHECK:           "zlow.stick"
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zhigh-level.mlir:6", opName = "zhigh.Stick", tag = 6 : i64} : () -> ()
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zhigh-level.mlir:6", opName = "zhigh.Add", tag = 5 : i64} : () -> ()
// CHECK-DAG:       memref.alloc()
// CHECK-DAG:       memref.alloc()
// CHECK:           krnl.store
// CHECK:           krnl.store
// CHECK:           "zlow.add"
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zhigh-level.mlir:6", opName = "zhigh.Add", tag = 6 : i64} : () -> ()
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zhigh-level.mlir:6", opName = "zhigh.Unstick", tag = 5 : i64} : () -> ()
// CHECK:           memref.alloc()
// CHECK:           "zlow.unstick"
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zhigh-level.mlir:6", opName = "zhigh.Unstick", tag = 6 : i64} : () -> ()
// CHECK:           return
// CHECK:         }

// -----
