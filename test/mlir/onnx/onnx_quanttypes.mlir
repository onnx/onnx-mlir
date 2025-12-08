// RUN: onnx-mlir-opt %s -quant-types | FileCheck %s

func.func @matmul_add(%arg0: tensor<1x128x768xf32>) -> tensor<1x128x768xf32> {
  %0 = onnx.Constant dense<35166> : tensor<ui16>
  %1 = onnx.Constant dense<2.13029256E-4> : tensor<f32>
  %2 = onnx.Constant dense<137> : tensor<ui8>
  %3 = onnx.Constant dense<0.00430401694> : tensor<f32>
  %4 = onnx.Constant dense_resource<__elided__> : tensor<768x768xui8>
  %5 = onnx.Constant dense<31929> : tensor<ui16>
  %6 = onnx.Constant dense<2.06352008E-4> : tensor<f32>
  %7 = onnx.Constant dense<41309> : tensor<ui16>
  %8 = onnx.Constant dense<6.79780783E-7> : tensor<f32>
  %9 = onnx.Constant dense_resource<__elided__> : tensor<768xui16>
  %10 = onnx.Constant dense<31907> : tensor<ui16>
  %11 = onnx.Constant dense<2.06478537E-4> : tensor<f32>
  %12 = "onnx.QuantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
  %13 = "onnx.DequantizeLinear"(%12, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
  %14 = "onnx.DequantizeLinear"(%4, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<768x768xui8>, tensor<f32>, tensor<ui8>) -> tensor<768x768xf32>
  %15 = "onnx.MatMul"(%13, %14) : (tensor<1x128x768xf32>, tensor<768x768xf32>) -> tensor<1x128x768xf32>
  %16 = "onnx.QuantizeLinear"(%15, %6, %5) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
  %17 = "onnx.DequantizeLinear"(%9, %8, %7) {axis = 1 : si64, block_size = 0 : si64} : (tensor<768xui16>, tensor<f32>, tensor<ui16>) -> tensor<768xf32>
  %18 = "onnx.DequantizeLinear"(%16, %6, %5) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
  %19 = "onnx.Add"(%17, %18) : (tensor<768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
  %20 = "onnx.QuantizeLinear"(%19, %11, %10) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
  %21 = "onnx.DequantizeLinear"(%20, %11, %10) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
  return %21 : tensor<1x128x768xf32>

// CHECK-LABEL: @matmul_add
// CHECK: onnx.QuantizeLinear
// CHECK-SAME: (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>)
// CHECK-SAME: -> tensor<1x128x768x!quant.uniform<u16:f32, 2.1302925597410649E-4:35166>>
// CHECK-NEXT: onnx.MatMul
// CHECK-SAME: (tensor<1x128x768x!quant.uniform<u16:f32, 2.1302925597410649E-4:35166>>, tensor<768x768x!quant.uniform<u8:f32, 0.0043040169402956963:137>>)
// CHECK-SAME: -> tensor<1x128x768x!quant.uniform<u16:f32, 2.0635200780816376E-4:31929>>
// CHECK-NEXT: onnx.Add
// CHECK-SAME: (tensor<768x!quant.uniform<u16:f32, 6.7978078277519671E-7:41309>>, tensor<1x128x768x!quant.uniform<u16:f32, 2.0635200780816376E-4:31929>>)
// CHECK-SAME: -> tensor<1x128x768x!quant.uniform<u16:f32, 2.0647853671107441E-4:31907>>
// CHECK-NEXT: onnx.DequantizeLinear
// CHECK-SAME: (tensor<1x128x768x!quant.uniform<u16:f32, 2.0647853671107441E-4:31907>>, tensor<f32>, tensor<ui16>)
// CHECK-SAME: -> tensor<1x128x768xf32>

}
