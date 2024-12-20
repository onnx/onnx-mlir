// RUN: onnx-mlir --march=z16 --maccel=NNPA  --printIR --EmitZHighIR -profile-ir=Onnx %s | FileCheck %s

// -----

func.func @test_instrument_add_onnx(%arg0 : tensor<10x10xf32>, %arg1 : tensor<10x10xf32>) -> tensor<*xf32> {
  %0 = "onnx.Add"(%arg0, %arg1) {onnx_node_name = "onnx.Add"} : (tensor<10x10xf32>, tensor<10x10xf32>) -> tensor<*xf32>
  "onnx.Return"(%0) : (tensor<*xf32>) -> ()
}

// CHECK-LABEL:  func.func @test_instrument_add_onnx
// CHECK:           "krnl.runtime_instrument"() {nodeName = "onnx.Add", opName = "onnx.Add", tag = 21 : i64} : () -> ()
// CHECK:           "zhigh.Stick"
// CHECK:           "zhigh.Stick"
// CHECK:           "zhigh.Add"
// CHECK:           "zhigh.Unstick"
// CHECK:           "krnl.runtime_instrument"() {nodeName = "onnx.Add", opName = "onnx.Add", tag = 6 : i64} : () -> ()
// CHECK:           return
// CHECK:         }

// -----
