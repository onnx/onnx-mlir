// RUN: onnx-mlir --maccel=NNPA --printIR --EmitZLowIR --instrument-onnx-ops=ALL --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime --instrument-zhigh-ops=ALL --InstrumentBeforeZHighOp --InstrumentAfterZHighOp --InstrumentReportTimeZHigh %s  | FileCheck %s

func.func @test_instrument_add_zhigh(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "func.return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_instrument_add_zhigh
// CHECK:           "krnl.runtime_instrument"() {opID = [[ID_STICK_:.+]] : i64, opName = "zhigh.Stick", tag = 5 : i64} : () -> ()
// CHECK:           memref.alloc()
// CHECK:           "zlow.stick"
// CHECK:           "krnl.runtime_instrument"() {opID = [[ID_STICK_]] : i64, opName = "zhigh.Stick", tag = 6 : i64} : () -> ()
// CHECK:           "krnl.runtime_instrument"() {opID = [[ID_STICK_]] : i64, opName = "zhigh.Stick", tag = 5 : i64} : () -> ()
// CHECK:           memref.alloc()
// CHECK:           "zlow.stick"
// CHECK:           "krnl.runtime_instrument"() {opID = [[ID_STICK_]] : i64, opName = "zhigh.Stick", tag = 6 : i64} : () -> ()
// CHECK:           "krnl.runtime_instrument"() {opID = [[ID_ADD_:.+]] : i64, opName = "zhigh.Add", tag = 5 : i64} : () -> ()
// CHECK:           "zlow.add"
// CHECK:           "krnl.runtime_instrument"() {opID = [[ID_ADD_]] : i64, opName = "zhigh.Add", tag = 6 : i64} : () -> ()
// CHECK:           "krnl.runtime_instrument"() {opID = [[ID_UNSTICK_:.+]] : i64, opName = "zhigh.Unstick", tag = 5 : i64} : () -> ()
// CHECK:           memref.alloc()
// CHECK:           "zlow.unstick"
// CHECK:           "krnl.runtime_instrument"() {opID = [[ID_UNSTICK_]] : i64, opName = "zhigh.Unstick", tag = 6 : i64} : () -> ()
// CHECK:           return
// CHECK:         }

// -----
