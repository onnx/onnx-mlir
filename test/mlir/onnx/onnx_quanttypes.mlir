// RUN: onnx-mlir-opt %s -quant-types | FileCheck %s

func.func @matmul_add(%arg0: tensor<1x128x768xf32>) -> tensor<1x128x768xf32> {
  %0 = onnx.Constant dense<35166> : tensor<i16>
  %1 = onnx.Constant dense<2.13029256E-4> : tensor<f32>
  %2 = onnx.Constant dense<137> : tensor<i8>
  %3 = onnx.Constant dense<0.00430401694> : tensor<f32>
  %4 = onnx.Constant dense_resource<__elided__> : tensor<768x768xi8>
  %5 = onnx.Constant dense<31929> : tensor<i16>
  %6 = onnx.Constant dense<2.06352008E-4> : tensor<f32>
  %7 = onnx.Constant dense<41309> : tensor<i16>
  %8 = onnx.Constant dense<6.79780783E-7> : tensor<f32>
  %9 = onnx.Constant dense_resource<__elided__> : tensor<768xi16>
  %10 = onnx.Constant dense<31907> : tensor<i16>
  %11 = onnx.Constant dense<2.06478537E-4> : tensor<f32>
  %12 = "onnx.QuantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<i16>) -> tensor<1x128x768xi16>
  %13 = "onnx.DequantizeLinear"(%12, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xi16>, tensor<f32>, tensor<i16>) -> tensor<1x128x768xf32>
  %14 = "onnx.DequantizeLinear"(%4, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<768x768xi8>, tensor<f32>, tensor<i8>) -> tensor<768x768xf32>
  %15 = "onnx.MatMul"(%13, %14) : (tensor<1x128x768xf32>, tensor<768x768xf32>) -> tensor<1x128x768xf32>
  %16 = "onnx.QuantizeLinear"(%15, %6, %5) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<i16>) -> tensor<1x128x768xi16>
  %17 = "onnx.DequantizeLinear"(%9, %8, %7) {axis = 1 : si64, block_size = 0 : si64} : (tensor<768xi16>, tensor<f32>, tensor<i16>) -> tensor<768xf32>
  %18 = "onnx.DequantizeLinear"(%16, %6, %5) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xi16>, tensor<f32>, tensor<i16>) -> tensor<1x128x768xf32>
  %19 = "onnx.Add"(%17, %18) : (tensor<768xf32>, tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
  %20 = "onnx.QuantizeLinear"(%19, %11, %10) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<i16>) -> tensor<1x128x768xi16>
  %21 = "onnx.DequantizeLinear"(%20, %11, %10) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xi16>, tensor<f32>, tensor<i16>) -> tensor<1x128x768xf32>
  return %21 : tensor<1x128x768xf32>
}

// CHECK-LABEL: @matmul_add
// CHECK: onnx.QuantizeLinear
// CHECK-SAME: (tensor<1x128x768xf32>, tensor<f32>, tensor<i16>) -> tensor<1x128x768xi16>
// CHECK-NEXT: quant.scast
// CHECK-SAME: tensor<1x128x768xi16> to tensor<1x128x768x!quant.uniform<i16:f32, 2.1302925597410649E-4:-30370>>
// CHECK-NEXT: onnx.MatMul
// CHECK-SAME: (tensor<1x128x768x!quant.uniform<i16:f32, 2.1302925597410649E-4:-30370>>, tensor<768x768x!quant.uniform<i8:f32, 0.0043040169402956963:-119>>) -> tensor<1x128x768x!quant.uniform<i16:f32, 2.0635200780816376E-4:31929>>
// CHECK-NEXT: onnx.Add
// CHECK-SAME: (tensor<768x!quant.uniform<i16:f32, 6.7978078277519671E-7:-24227>>, tensor<1x128x768x!quant.uniform<i16:f32, 2.0635200780816376E-4:31929>>) -> tensor<1x128x768x!quant.uniform<i16:f32, 2.0647853671107441E-4:31907>>
// CHECK-NEXT: quant.scast
// CHECK-SAME: tensor<1x128x768x!quant.uniform<i16:f32, 2.0647853671107441E-4:31907>> to tensor<1x128x768xi16>
// CHECK-NEXT: onnx.DequantizeLinear
// CHECK-SAME: (tensor<1x128x768xi16>, tensor<f32>, tensor<i16>) -> tensor<1x128x768xf32>


func.func @no_boundary_qdq(%arg0: tensor<1x128x768xi16>) -> tensor<1x128x768xi16> {
  %0 = onnx.Constant dense<35166> : tensor<i16>
  %1 = onnx.Constant dense<2.13029256E-4> : tensor<f32>
  %2 = onnx.Constant dense<137> : tensor<i8>
  %3 = onnx.Constant dense<0.00430401694> : tensor<f32>
  %4 = onnx.Constant dense_resource<__elided__> : tensor<768x768xi8>
  %5 = onnx.Constant dense<31929> : tensor<i16>
  %6 = onnx.Constant dense<2.06352008E-4> : tensor<f32>
  %7 = onnx.Constant dense<31907> : tensor<i16>
  %8 = onnx.Constant dense<2.06478537E-4> : tensor<f32>
  %9 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xi16>, tensor<f32>, tensor<i16>) -> tensor<1x128x768xf32>
  %10 = "onnx.DequantizeLinear"(%4, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<768x768xi8>, tensor<f32>, tensor<i8>) -> tensor<768x768xf32>
  %11 = "onnx.MatMul"(%9, %10) : (tensor<1x128x768xf32>, tensor<768x768xf32>) -> tensor<1x128x768xf32>
  %12 = "onnx.QuantizeLinear"(%11, %6, %5) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<i16>) -> tensor<1x128x768xi16>
  return %12 : tensor<1x128x768xi16>
}

// CHECK-LABEL: @no_boundary_qdq
// CHECK: onnx.Constant
// CHECK-SAME: tensor<768x768x!quant.uniform<i8:f32, 0.0043040169402956963:-119>>
// CHECK-NEXT: quant.scast
// CHECK-SAME: tensor<1x128x768xi16> to tensor<1x128x768x!quant.uniform<i16:f32, 2.1302925597410649E-4:-30370>>
// CHECK-NEXT: onnx.MatMul
// CHECK-SAME: (tensor<1x128x768x!quant.uniform<i16:f32, 2.1302925597410649E-4:-30370>>, tensor<768x768x!quant.uniform<i8:f32, 0.0043040169402956963:-119>>) -> tensor<1x128x768x!quant.uniform<i16:f32, 2.0635200780816376E-4:31929>>
// CHECK-NEXT: quant.scast
// CHECK-SAME: tensor<1x128x768x!quant.uniform<i16:f32, 2.0635200780816376E-4:31929>> to tensor<1x128x768xi16>


func.func @unequal_qdq(%arg0: tensor<1x128x768xi16>) -> tensor<1x128x768xi16> {
  %0 = onnx.Constant dense<32768> : tensor<i16>
  %1 = onnx.Constant dense<2.52349739E-4> : tensor<f32>
  %2 = onnx.Constant dense<1474> : tensor<i16>
  %3 = onnx.Constant dense<1.15280403E-4> : tensor<f32>
  %4 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xi16>, tensor<f32>, tensor<i16>) -> tensor<1x128x768xf32>
  %5 = "onnx.Gelu"(%4) {approximate = "none"} : (tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
  %6 = "onnx.QuantizeLinear"(%5, %3, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<i16>) -> tensor<1x128x768xi16>
  %7 = "onnx.DequantizeLinear"(%6, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xi16>, tensor<f32>, tensor<i16>) -> tensor<1x128x768xf32>
  %8 = "onnx.Gelu"(%7) {approximate = "none"} : (tensor<1x128x768xf32>) -> tensor<1x128x768xf32>
  %9 = "onnx.QuantizeLinear"(%8, %3, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<i16>) -> tensor<1x128x768xi16>
  return %9 : tensor<1x128x768xi16>
}

// CHECK-LABEL: @unequal_qdq
// CHECK: quant.scast
// CHECK-NEXT: onnx.Gelu
// CHECK-NEXT: quant.scast
// CHECK-NEXT: quant.scast
// CHECK-NEXT: onnx.Gelu
// CHECK-NEXT: quant.scast


func.func @multiuse_constant(%arg0: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
  %0 = onnx.Constant dense<35166> : tensor<i16>
  %1 = onnx.Constant dense<2.13029256E-4> : tensor<f32>
  %2 = onnx.Constant dense<137> : tensor<i8>
  %3 = onnx.Constant dense<0.00430401694> : tensor<f32>
  %4 = onnx.Constant dense_resource<__elided__> : tensor<128x128xi8>
  %5 = onnx.Constant dense<31929> : tensor<i16>
  %6 = onnx.Constant dense<2.06352008E-4> : tensor<f32>
  %7 = onnx.Constant dense<41309> : tensor<i16>
  %8 = onnx.Constant dense<6.79780783E-7> : tensor<f32>
  %9 = onnx.Constant dense<31907> : tensor<i16>
  %10 = onnx.Constant dense<2.06478537E-4> : tensor<f32>
  %11 = "onnx.QuantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x128xf32>, tensor<f32>, tensor<i16>) -> tensor<1x128x128xi16>
  %12 = "onnx.DequantizeLinear"(%11, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x128xi16>, tensor<f32>, tensor<i16>) -> tensor<1x128x128xf32>
  %13 = "onnx.DequantizeLinear"(%4, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<128x128xi8>, tensor<f32>, tensor<i8>) -> tensor<128x128xf32>
  %14 = "onnx.MatMul"(%12, %13) : (tensor<1x128x128xf32>, tensor<128x128xf32>) -> tensor<1x128x128xf32>
  %15 = "onnx.QuantizeLinear"(%14, %6, %5) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x128xf32>, tensor<f32>, tensor<i16>) -> tensor<1x128x128xi16>
  %16 = "onnx.DequantizeLinear"(%4, %8, %7) {axis = 1 : si64, block_size = 0 : si64} : (tensor<128x128xi8>, tensor<f32>, tensor<i16>) -> tensor<128x128xf32>
  %17 = "onnx.DequantizeLinear"(%15, %6, %5) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x128xi16>, tensor<f32>, tensor<i16>) -> tensor<1x128x128xf32>
  %18 = "onnx.Add"(%16, %17) : (tensor<128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
  %19 = "onnx.QuantizeLinear"(%18, %10, %9) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x128xf32>, tensor<f32>, tensor<i16>) -> tensor<1x128x128xi16>
  %20 = "onnx.DequantizeLinear"(%19, %10, %9) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x128xi16>, tensor<f32>, tensor<i16>) -> tensor<1x128x128xf32>
  return %20 : tensor<1x128x128xf32>
}

// CHECK-LABEL: @multiuse_constant
// CHECK: [[CONST:%[0-9]+]] = onnx.Constant dense_resource<__elided__> : tensor<128x128xi8>
// CHECK: quant.scast [[CONST]]
// CHECK-SAME: tensor<128x128x!quant.uniform<i8:f32, 0.0043040169402956963:-119>>
// CHECK: quant.scast [[CONST]]
// CHECK-SAME: tensor<128x128x!quant.uniform<i8:f32, 6.7978078277519671E-7:-24227>>


func.func @channelwise_quantization(%arg0: tensor<1x128x128xi16>) -> tensor<1x128x128xi16> {
  %0 = onnx.Constant dense<-30370> : tensor<128xi16>
  %1 = onnx.Constant dense<2.13029256E-4> : tensor<128xf32>
  %2 = onnx.Constant dense<137> : tensor<128xi16>
  %3 = onnx.Constant dense<0.00430401694> : tensor<128xf32>
  %4 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 2 : si64, block_size = 0 : si64} : (tensor<1x128x128xi16>, tensor<128xf32>, tensor<128xi16>) -> tensor<1x128x128xf32>
  %5 = "onnx.Gelu"(%4) {approximate = "none"} : (tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
  %6 = "onnx.QuantizeLinear"(%5, %3, %2) {axis = 2 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x128xf32>, tensor<128xf32>, tensor<128xi16>) -> tensor<1x128x128xi16>
  return %6 : tensor<1x128x128xi16>
}

// CHECK-LABEL: @channelwise_quantization
// CHECK: quant.scast
// CHECK-NEXT: onnx.Gelu
// CHECK-SAME: (tensor<1x128x128x!quant.uniform<i16:f32:2
// CHECK-SAME: -> tensor<1x128x128x!quant.uniform<i16:f32:2
// CHECK-NEXT: quant.scast
