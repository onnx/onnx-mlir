// RUN: onnx-mlir --printIR --EmitMLIR --instrument-ops=onnx.* --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime %s | FileCheck %s

// -----

func.func @test_instrument_add_onnx(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "model/add1"} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_instrument_add_onnx
// CHECK:           "krnl.runtime_instrument"() {nodeName = "model/add1", opName = "onnx.Add", tag = 21 : i64} : () -> ()
// CHECK:           [[RES_:%.+]] = memref.alloc()
// CHECK:           affine.for [[I_0_:%.+]] = 0 to 10 {
// CHECK:             affine.for [[I_1_:%.+]] = 0 to 10 {
// CHECK-DAG:           [[LOAD_PARAM_0_MEM_:%.+]] = affine.load
// CHECK-DAG:           [[LOAD_PARAM_1_MEM_:%.+]] = affine.load
// CHECK:               [[VAR_3_:%.+]] = arith.addf [[LOAD_PARAM_0_MEM_]], [[LOAD_PARAM_1_MEM_]] : f32
// CHECK:               affine.store [[VAR_3_]], [[RES_]]{{.}}[[I_0_]], [[I_1_]]{{.}} : memref<10x10xf32>
// CHECK:             }
// CHECK:           }
// CHECK:           "krnl.runtime_instrument"() {nodeName = "model/add1", opName = "onnx.Add", tag = 6 : i64} : () -> ()
// CHECK:           return
// CHECK:         }
