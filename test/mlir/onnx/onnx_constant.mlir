// RUN: onnx-mlir-opt %s -mlir-print-op-generic | FileCheck -check-prefix=GENERIC %s
// RUN: onnx-mlir-opt %s | FileCheck %s

func.func @test() {
  %generic = "onnx.Constant"() {value = dense<-1> : tensor<1xi64> } : () -> tensor<1xi64>
  %pretty = onnx.Constant dense<-1> : tensor<1xi64>

  %generic_with_extra_attr = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>, extra_attr = 0 : i32 } : () -> tensor<1xi64>
  %pretty_with_extra_attr = onnx.Constant {extra_attr = 0 : i32} dense<-1> : tensor<1xi64>

  %generic_dynamic = "onnx.Constant"() {value = dense<-1> : tensor<1xi64> } : () -> tensor<*xi64>
  %pretty_dynamic = onnx.Constant {value = dense<-1> : tensor<1xi64>} : tensor<*xi64>

  %generic_dynamic_with_extra_attr = "onnx.Constant"() {value = dense<-1> : tensor<1xi64>, extra_attr = 0 : i32 } : () -> tensor<*xi64>
  %pretty_dynamic_with_extra_attr = onnx.Constant {extra_attr = 0 : i32, value = dense<-1> : tensor<1xi64>} : tensor<*xi64>
  func.return
}

// GENERIC: "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
// GENERIC: "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
// GENERIC: "onnx.Constant"() {extra_attr = 0 : i32, value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
// GENERIC: "onnx.Constant"() {extra_attr = 0 : i32, value = dense<-1> : tensor<1xi64>} : () -> tensor<1xi64>
// GENERIC: "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<*xi64>
// GENERIC: "onnx.Constant"() {value = dense<-1> : tensor<1xi64>} : () -> tensor<*xi64>
// GENERIC: "onnx.Constant"() {extra_attr = 0 : i32, value = dense<-1> : tensor<1xi64>} : () -> tensor<*xi64>
// GENERIC: "onnx.Constant"() {extra_attr = 0 : i32, value = dense<-1> : tensor<1xi64>} : () -> tensor<*xi64>
// CHECK: onnx.Constant dense<-1> : tensor<1xi64>
// CHECK: onnx.Constant dense<-1> : tensor<1xi64>
// CHECK: onnx.Constant {extra_attr = 0 : i32} dense<-1> : tensor<1xi64>
// CHECK: onnx.Constant {extra_attr = 0 : i32} dense<-1> : tensor<1xi64>
// CHECK: onnx.Constant {value = dense<-1> : tensor<1xi64>} : tensor<*xi64>
// CHECK: onnx.Constant {value = dense<-1> : tensor<1xi64>} : tensor<*xi64>
// CHECK: onnx.Constant {extra_attr = 0 : i32, value = dense<-1> : tensor<1xi64>} : tensor<*xi64>
// CHECK: onnx.Constant {extra_attr = 0 : i32, value = dense<-1> : tensor<1xi64>} : tensor<*xi64>
