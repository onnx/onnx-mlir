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
}

// CHECK-LABEL: @matmul_add
// CHECK: onnx.QuantizeLinear
// CHECK-SAME: (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
// CHECK-NEXT: quant.scast
// CHECK-SAME: tensor<1x128x768xui16> to tensor<1x128x768x!quant.uniform<u16:f32, 2.1302925597410649E-4:35166>>
// CHECK-NEXT: onnx.MatMul
// CHECK-SAME: (tensor<1x128x768x!quant.uniform<u16:f32, 2.1302925597410649E-4:35166>>, tensor<768x768x!quant.uniform<u8:f32, 0.0043040169402956963:137>>) -> tensor<1x128x768x!quant.uniform<u16:f32, 2.0635200780816376E-4:31929>>
// CHECK-NEXT: onnx.Add
// CHECK-SAME: (tensor<768x!quant.uniform<u16:f32, 6.7978078277519671E-7:41309>>, tensor<1x128x768x!quant.uniform<u16:f32, 2.0635200780816376E-4:31929>>) -> tensor<1x128x768x!quant.uniform<u16:f32, 2.0647853671107441E-4:31907>>
// CHECK-NEXT: quant.scast
// CHECK-SAME: tensor<1x128x768x!quant.uniform<u16:f32, 2.0647853671107441E-4:31907>> to tensor<1x128x768xui16>
// CHECK-NEXT: onnx.DequantizeLinear
// CHECK-SAME: (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>


func.func @no_boundary_qdq(%arg0: tensor<1x128x768xui16>) -> tensor<1x128x768xui16> {
  %0 = onnx.Constant dense<35166> : tensor<ui16>
  %1 = onnx.Constant dense<2.13029256E-4> : tensor<f32>
  %2 = onnx.Constant dense<137> : tensor<ui8>
  %3 = onnx.Constant dense<0.00430401694> : tensor<f32>
  %4 = onnx.Constant dense_resource<__elided__> : tensor<768x768xui8>
  %5 = onnx.Constant dense<31929> : tensor<ui16>
  %6 = onnx.Constant dense<2.06352008E-4> : tensor<f32>
  %7 = onnx.Constant dense<31907> : tensor<ui16>
  %8 = onnx.Constant dense<2.06478537E-4> : tensor<f32>
  %9 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x768xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xf32>
  %10 = "onnx.DequantizeLinear"(%4, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<768x768xui8>, tensor<f32>, tensor<ui8>) -> tensor<768x768xf32>
  %11 = "onnx.MatMul"(%9, %10) : (tensor<1x128x768xf32>, tensor<768x768xf32>) -> tensor<1x128x768xf32>
  %12 = "onnx.QuantizeLinear"(%11, %6, %5) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x768xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x768xui16>
  return %12 : tensor<1x128x768xui16>
}

// CHECK-LABEL: @no_boundary_qdq
// CHECK: onnx.Constant
// CHECK-SAME: tensor<768x768x!quant.uniform<u8:f32, 0.0043040169402956963:137>>
// CHECK-NEXT: quant.scast
// CHECK-SAME: tensor<1x128x768xui16> to tensor<1x128x768x!quant.uniform<u16:f32, 2.1302925597410649E-4:35166>>
// CHECK-NEXT: onnx.MatMul
// CHECK-SAME: (tensor<1x128x768x!quant.uniform<u16:f32, 2.1302925597410649E-4:35166>>, tensor<768x768x!quant.uniform<u8:f32, 0.0043040169402956963:137>>) -> tensor<1x128x768x!quant.uniform<u16:f32, 2.0635200780816376E-4:31929>>
// CHECK-NEXT: quant.scast
// CHECK-SAME: tensor<1x128x768x!quant.uniform<u16:f32, 2.0635200780816376E-4:31929>> to tensor<1x128x768xui16>


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


func.func @multiuse_small_constant(%arg0: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
  %0 = onnx.Constant dense<35166> : tensor<ui16>
  %1 = onnx.Constant dense<2.13029256E-4> : tensor<f32>
  %2 = onnx.Constant dense<137> : tensor<ui8>
  %3 = onnx.Constant dense<0.00430401694> : tensor<f32>
  %4 = onnx.Constant dense<255> : tensor<128x128xui8>
  %5 = onnx.Constant dense<31929> : tensor<ui16>
  %6 = onnx.Constant dense<2.06352008E-4> : tensor<f32>
  %7 = onnx.Constant dense<132> : tensor<ui8>
  %8 = onnx.Constant dense<6.79780783E-7> : tensor<f32>
  %9 = onnx.Constant dense<31907> : tensor<ui16>
  %10 = onnx.Constant dense<2.06478537E-4> : tensor<f32>
  %11 = "onnx.QuantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x128xui16>
  %12 = "onnx.DequantizeLinear"(%11, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x128xf32>
  %13 = "onnx.DequantizeLinear"(%4, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<128x128xui8>, tensor<f32>, tensor<ui8>) -> tensor<128x128xf32>
  %14 = "onnx.MatMul"(%12, %13) : (tensor<1x128x128xf32>, tensor<128x128xf32>) -> tensor<1x128x128xf32>
  %15 = "onnx.QuantizeLinear"(%14, %6, %5) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x128xui16>
  %16 = "onnx.DequantizeLinear"(%4, %8, %7) {axis = 1 : si64, block_size = 0 : si64} : (tensor<128x128xui8>, tensor<f32>, tensor<ui8>) -> tensor<128x128xf32>
  %17 = "onnx.DequantizeLinear"(%15, %6, %5) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x128xf32>
  %18 = "onnx.Add"(%16, %17) : (tensor<128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
  %19 = "onnx.QuantizeLinear"(%18, %10, %9) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x128xui16>
  %20 = "onnx.DequantizeLinear"(%19, %10, %9) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x128xf32>
  return %20 : tensor<1x128x128xf32>
}

// CHECK-LABEL: @multiuse_small_constant
// CHECK-COUNT-2: onnx.Constant {value = dense<255> : tensor<128x128xui8>} : tensor<128x128x!quant.uniform


func.func @multiuse_constant(%arg0: tensor<1x128x128xf32>) -> tensor<1x128x128xf32> {
  %0 = onnx.Constant dense<35166> : tensor<ui16>
  %1 = onnx.Constant dense<2.13029256E-4> : tensor<f32>
  %2 = onnx.Constant dense<137> : tensor<ui8>
  %3 = onnx.Constant dense<0.00430401694> : tensor<f32>
  %4 = onnx.Constant dense_resource<__elided__> : tensor<128x128xui8>
  %5 = onnx.Constant dense<31929> : tensor<ui16>
  %6 = onnx.Constant dense<2.06352008E-4> : tensor<f32>
  %7 = onnx.Constant dense<132> : tensor<ui8>
  %8 = onnx.Constant dense<6.79780783E-7> : tensor<f32>
  %9 = onnx.Constant dense<31907> : tensor<ui16>
  %10 = onnx.Constant dense<2.06478537E-4> : tensor<f32>
  %11 = "onnx.QuantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x128xui16>
  %12 = "onnx.DequantizeLinear"(%11, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x128xf32>
  %13 = "onnx.DequantizeLinear"(%4, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<128x128xui8>, tensor<f32>, tensor<ui8>) -> tensor<128x128xf32>
  %14 = "onnx.MatMul"(%12, %13) : (tensor<1x128x128xf32>, tensor<128x128xf32>) -> tensor<1x128x128xf32>
  %15 = "onnx.QuantizeLinear"(%14, %6, %5) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x128xui16>
  %16 = "onnx.DequantizeLinear"(%4, %8, %7) {axis = 1 : si64, block_size = 0 : si64} : (tensor<128x128xui8>, tensor<f32>, tensor<ui8>) -> tensor<128x128xf32>
  %17 = "onnx.DequantizeLinear"(%15, %6, %5) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x128xf32>
  %18 = "onnx.Add"(%16, %17) : (tensor<128x128xf32>, tensor<1x128x128xf32>) -> tensor<1x128x128xf32>
  %19 = "onnx.QuantizeLinear"(%18, %10, %9) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x128x128xui16>
  %20 = "onnx.DequantizeLinear"(%19, %10, %9) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x128x128xf32>
  return %20 : tensor<1x128x128xf32>
}

// CHECK-LABEL: @multiuse_constant
// CHECK: [[CONST:%[0-9]+]] = onnx.Constant dense_resource<__elided__> : tensor<128x128xui8>
// CHECK: quant.scast [[CONST]]
// CHECK-SAME: tensor<128x128x!quant.uniform<u8:f32, 0.0043040169402956963:137>>
// CHECK: quant.scast [[CONST]]
// CHECK-SAME: tensor<128x128x!quant.uniform<u8:f32, 6.7978078277519671E-7:132>>


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

func.func @i8_quants(%arg0: tensor<1x224x224x3xf32>) -> tensor<1x3x224x224xf32> {
  %0 = onnx.Constant dense<-128> : tensor<i8>
  %1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
  %2 = "onnx.QuantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x224x224x3xf32>, tensor<f32>, tensor<i8>) -> tensor<1x224x224x3xi8>
  %3 = "onnx.DequantizeLinear"(%2, %1, %0) { axis = 1 : si64, block_size = 0 : si64} : (tensor<1x224x224x3xi8>, tensor<f32>, tensor<i8>) -> tensor<1x224x224x3xf32>
  %4 = "onnx.Transpose"(%3) {perm = [0, 3, 1, 2]} : (tensor<1x224x224x3xf32>) -> tensor<1x3x224x224xf32>
  %5 = "onnx.QuantizeLinear"(%4, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x3x224x224xf32>, tensor<f32>, tensor<i8>) -> tensor<1x3x224x224xi8>
  %6 = "onnx.DequantizeLinear"(%5, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x3x224x224xi8>, tensor<f32>, tensor<i8>) -> tensor<1x3x224x224xf32>
  return %6 : tensor<1x3x224x224xf32>
}

// CHECK: onnx.Transpose
// CHECK-SAME: (tensor<1x224x224x3x!quant.uniform<i8:f32, 1.000000e+00:-128>>) -> tensor<1x3x224x224x!quant.uniform<i8:f32, 1.000000e+00:-128>>

func.func @cast_to_quant_types(%arg0: tensor<1x300x7xf32>) -> tensor<1x300x13xf32> {
  %0 = onnx.Constant dense<1> : tensor<1xi64>
  %1 = onnx.Constant dense<0> : tensor<ui16>
  %2 = onnx.Constant dense<1.49878533E-5> : tensor<f32>
  %3 = onnx.Constant dense<0> : tensor<ui16>
  %4 = onnx.Constant dense<1.49878533E-5> : tensor<f32>
  %5 = onnx.Constant dense<0> : tensor<ui16>
  %6 = onnx.Constant dense<9.15541313E-5> : tensor<f32>
  %7 = onnx.Constant dense<12015> : tensor<ui16>
  %8 = onnx.Constant dense<1.12106813E-4> : tensor<f32>
  %9 = "onnx.QuantizeLinear"(%arg0, %2, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x300x7xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x300x7xui16>
  %10 = "onnx.DequantizeLinear"(%9, %2, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x300x7xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x300x7xf32>
  %Values, %Indices = "onnx.TopK"(%10, %0) {axis = 2 : si64, largest = 1 : si64, sorted = 1 : si64} : (tensor<1x300x7xf32>, tensor<1xi64>) -> (tensor<1x300x1xf32>, tensor<1x300x1xi64>)
  %11 = "onnx.QuantizeLinear"(%Values, %4, %3) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x300x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x300x1xui16>
  %12 = "onnx.Cast"(%Indices) {saturate = 1 : si64, to = f32} : (tensor<1x300x1xi64>) -> tensor<1x300x1xf32>
  %13 = "onnx.DequantizeLinear"(%11, %4, %3) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x300x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x300x1xf32>
  %14 = "onnx.QuantizeLinear"(%12, %6, %5) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x300x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x300x1xui16>
  %15 = "onnx.DequantizeLinear"(%14, %6, %5) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x300x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x300x1xf32>
  %16 = "onnx.Concat"(%13, %15, %10) {axis = 2 : si64} : (tensor<1x300x1xf32>, tensor<1x300x1xf32>, tensor<1x300x7xf32>) -> tensor<1x300x13xf32>
  %17 = "onnx.QuantizeLinear"(%16, %8, %7) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x300x13xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x300x13xui16>
  %18 = "onnx.DequantizeLinear"(%17, %8, %7) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x300x13xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x300x13xf32>
  return %18 : tensor<1x300x13xf32>
}

// CHECK-LABEL: @cast_to_quant_types
// CHECK: onnx.TopK
// CHECK-SAME: (tensor<1x300x7x!quant.uniform<u16:f32, 1.498785331932595E-5>>, tensor<1xi64>) -> (tensor<1x300x1x!quant.uniform<u16:f32, 1.498785331932595E-5>>, tensor<1x300x1xi64>)
// CHECK-NEXT: onnx.Cast
// CHECK-SAME: (tensor<1x300x1xi64>) -> tensor<1x300x1x!quant.uniform<u16:f32, 9.1554131358861923E-5>>

func.func @q_dq_q_dq(%arg0: tensor<1x64x128x128xf32>, %arg1: tensor<1x64x128x128xf32>) -> tensor<1x128x64x64xf32> {
  %0 = onnx.Constant dense<128> : tensor<ui8>
  %1 = onnx.Constant dense<43> : tensor<ui8>
  %2 = onnx.Constant dense<127> : tensor<ui8>
  %3 = onnx.Constant dense<52> : tensor<ui8>
  %4 = onnx.Constant dense<0.0232218392> : tensor<f32>
  %5 = onnx.Constant dense<0> : tensor<i8>
  %6 = onnx.Constant dense_resource<__elided__> : tensor<128x64x3x3xi8>
  %7 = onnx.Constant dense_resource<__elided__> : tensor<128xi32>
  %8 = onnx.Constant dense<0> : tensor<i32>
  %9 = onnx.Constant dense<0.014117647> : tensor<f32>
  %10 = onnx.Constant dense<0.0392156877> : tensor<f32>
  %11 = onnx.Constant dense<0.0235294122> : tensor<f32>
  %12 = "onnx.DequantizeLinear"(%6, %4, %5) {axis = 0 : si64, block_size = 0 : si64} : (tensor<128x64x3x3xi8>, tensor<f32>, tensor<i8>) -> tensor<128x64x3x3xf32>
  %13 = "onnx.DequantizeLinear"(%7, %4, %8) {axis = 0 : si64, block_size = 0 : si64} : (tensor<128xi32>, tensor<f32>, tensor<i32>) -> tensor<128xf32>
  %14 = "onnx.QuantizeLinear"(%arg0, %9, %1) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x64x128x128xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x64x128x128xui8>
  %15 = "onnx.DequantizeLinear"(%14, %9, %1) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x64x128x128xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x64x128x128xf32>
  %16 = "onnx.Add"(%15, %arg1) : (tensor<1x64x128x128xf32>, tensor<1x64x128x128xf32>) -> tensor<1x64x128x128xf32>
  %17 = "onnx.QuantizeLinear"(%16, %10, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x64x128x128xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x64x128x128xui8>
  %18 = "onnx.DequantizeLinear"(%17, %10, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x64x128x128xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x64x128x128xf32>
  %19 = "onnx.QuantizeLinear"(%18, %4, %3) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x64x128x128xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x64x128x128xui8>
  %20 = "onnx.DequantizeLinear"(%19, %4, %3) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x64x128x128xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x64x128x128xf32>
  %21 = "onnx.Conv"(%20, %12, %13) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x64x128x128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>) -> tensor<1x128x64x64xf32>
  %22 = "onnx.QuantizeLinear"(%21, %11, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x64x64xf32>, tensor<f32>, tensor<ui8>) -> tensor<1x128x64x64xui8>
  %23 = "onnx.DequantizeLinear"(%22, %11, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x64x64xui8>, tensor<f32>, tensor<ui8>) -> tensor<1x128x64x64xf32>
  return %23 : tensor<1x128x64x64xf32>
}

// CHECK-LABEL: @q_dq_q_dq
// CHECK: onnx.Add
// CHECK-SAME: (tensor<1x64x128x128x!quant.uniform<u8:f32, 0.014117646962404251:43>>, tensor<1x64x128x128xf32>) -> tensor<1x64x128x128x!quant.uniform<u8:f32, 0.039215687662363052:127>>
// CHECK-NEXT: quant.scast
// CHECK-SAME: tensor<1x64x128x128x!quant.uniform<u8:f32, 0.039215687662363052:127>> to tensor<1x64x128x128xui8>
// CHECK-NEXT: quant.scast
// CHECK-SAME: tensor<1x64x128x128xui8> to tensor<1x64x128x128x!quant.uniform<u8:f32, 0.023221839219331741:52>>

// Test that RandomNormalLike with dtype=1 (f32) accepts a quantized result type
// when the expressed type matches the dtype.
func.func @randomnormallike_quant_types(%arg0: tensor<1x64x32xui16>) -> tensor<1x64x32xui16> {
  %zp = onnx.Constant dense<32031> : tensor<ui16>
  %sc = onnx.Constant dense<1.15724782E-4> : tensor<f32>
  %dq = "onnx.DequantizeLinear"(%arg0, %sc, %zp) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x64x32xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x64x32xf32>
  %rand = "onnx.RandomNormalLike"(%dq) {dtype = 1 : si64, mean = 0.0 : f32, scale = 1.0 : f32} : (tensor<1x64x32xf32>) -> tensor<1x64x32xf32>
  %q = "onnx.QuantizeLinear"(%rand, %sc, %zp) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x64x32xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x64x32xui16>
  return %q : tensor<1x64x32xui16>
}

// CHECK-LABEL: @randomnormallike_quant_types
// CHECK: "onnx.RandomNormalLike"
// CHECK-SAME: !quant.uniform<u16:f32,

// Test that RMSLayerNormalization accepts quantized input (X) and produces
// quantized output (Y) after -quant-types folds surrounding DQ/Q ops.
// Scale stays float and bias is none — only X and Y carry quant types.
func.func @rmslayernorm_quant_types(%arg0: tensor<1x128x2880xi16>) -> tensor<1x128x2880xi16> {
  %zp_x = onnx.Constant dense<8361> : tensor<i16>
  %sc_x = onnx.Constant dense<9.59339377E-4> : tensor<f32>
  %zp_y = onnx.Constant dense<8361> : tensor<i16>
  %sc_y = onnx.Constant dense<9.59339377E-4> : tensor<f32>
  %none = "onnx.NoValue"() {value} : () -> none
  %scale_const = onnx.Constant dense<1.0> : tensor<2880xf32>
  %dq_x = "onnx.DequantizeLinear"(%arg0, %sc_x, %zp_x) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x128x2880xi16>, tensor<f32>, tensor<i16>) -> tensor<1x128x2880xf32>
  %Y, %InvStdDev = "onnx.RMSLayerNormalization"(%dq_x, %scale_const, %none) {axis = -1 : si64, epsilon = 1.0E-5 : f32, stash_type = 1 : si64} : (tensor<1x128x2880xf32>, tensor<2880xf32>, none) -> (tensor<1x128x2880xf32>, none)
  %q_y = "onnx.QuantizeLinear"(%Y, %sc_y, %zp_y) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x128x2880xf32>, tensor<f32>, tensor<i16>) -> tensor<1x128x2880xi16>
  return %q_y : tensor<1x128x2880xi16>
}

// CHECK-LABEL: @rmslayernorm_quant_types
// CHECK: "onnx.RMSLayerNormalization"
// CHECK-SAME: (tensor<1x128x2880x!quant.uniform<i16:f32, 9.5933937700465322E-4:8361>>, tensor<2880xf32>, none)
// CHECK-SAME: -> (tensor<1x128x2880x!quant.uniform<i16:f32, 9.5933937700465322E-4:8361>>, none)
