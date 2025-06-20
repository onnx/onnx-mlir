// RUN:  onnx-mlir --march=z16 --maccel=NNPA --printIR --EmitZLowIR --instrument-stage=ZLow --instrument-ops=zlow.* --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime %s  | FileCheck %s

// -----

func.func @test_instrument_add_zlow(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_instrument_add_zlow
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zlow-level.mlir:6", opName = "zlow.stick", tag = 21 : i64} : () -> ()
// CHECK:           "zlow.stick"
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zlow-level.mlir:6", opName = "zlow.stick", tag = 6 : i64} : () -> ()
// CHECK:           memref.alloc()
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zlow-level.mlir:6", opName = "zlow.stick", tag = 5 : i64} : () -> ()
// CHECK:           "zlow.stick"
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zlow-level.mlir:6", opName = "zlow.stick", tag = 6 : i64} : () -> ()
// CHECK-DAG:       memref.alloc()
// CHECK-DAG:       memref.alloc()
// CHECK:           krnl.store
// CHECK:           krnl.store
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zlow-level.mlir:6", opName = "zlow.add", tag = 5 : i64} : () -> ()
// CHECK:           "zlow.add"
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zlow-level.mlir:6", opName = "zlow.add", tag = 6 : i64} : () -> ()
// CHECK:           memref.alloc()
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zlow-level.mlir:6", opName = "zlow.unstick", tag = 5 : i64} : () -> ()
// CHECK:           "zlow.unstick"
// CHECK:           "krnl.runtime_instrument"() {nodeName = "add-zlow-level.mlir:6", opName = "zlow.unstick", tag = 6 : i64} : () -> ()
// CHECK:           return
// CHECK:         }
