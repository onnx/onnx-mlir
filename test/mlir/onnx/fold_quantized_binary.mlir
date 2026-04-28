// RUN: onnx-mlir-opt %s -fold-quantized-binary | FileCheck %s
// RUN: onnx-mlir-opt %s -canonicalize -fold-quantized-binary | FileCheck --check-prefix=CANON %s

func.func @dq_dq_mul_update_input(%arg0: tensor<10x1xf32>) -> tensor<10x1xf32> {
  %0 = onnx.Constant dense<0> : tensor<ui16>
  %1 = onnx.Constant dense<5.78499521E-6> : tensor<f32>
  %2 = onnx.Constant {value = dense<65535> : tensor<ui16>} : tensor<!quant.uniform<u16:f32, 0.0015259023057296872>>
  %3 = onnx.Constant dense<10> : tensor<ui16>
  %4 = onnx.Constant dense<1.000000e-01> : tensor<f32>
  %5 = "onnx.QuantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<10x1xf32>, tensor<f32>, tensor<ui16>) -> tensor<10x1xui16>
  %6 = quant.scast %5 : tensor<10x1xui16> to tensor<10x1x!quant.uniform<u16:f32, 5.7849952099786606E-6>>
  %7 = "onnx.Identity"(%6) : (tensor<10x1x!quant.uniform<u16:f32, 5.7849952099786606E-6>>) -> tensor<10x1x!quant.uniform<u16:f32, 5.7849952099786606E-6>>
  %8 = "onnx.Mul"(%7, %2) : (tensor<10x1x!quant.uniform<u16:f32, 5.7849952099786606E-6>>, tensor<!quant.uniform<u16:f32, 0.0015259023057296872>>) -> tensor<10x1x!quant.uniform<u16:f32, 0.10000000149011612:10>>
  %9 = "onnx.Identity"(%8) : (tensor<10x1x!quant.uniform<u16:f32, 0.10000000149011612:10>>) -> tensor<10x1x!quant.uniform<u16:f32, 0.10000000149011612:10>>
  %10 = quant.scast %9 : tensor<10x1x!quant.uniform<u16:f32, 0.10000000149011612:10>> to tensor<10x1xui16>
  %11 = "onnx.DequantizeLinear"(%10, %4, %3) {axis = 1 : si64, block_size = 0 : si64} : (tensor<10x1xui16>, tensor<f32>, tensor<ui16>) -> tensor<10x1xf32>
  return %11 : tensor<10x1xf32>
}

// CHECK-LABEL: @dq_dq_mul_update_input
// CHECK: onnx.Identity
// CHECK-NEXT: quant.scast
// CHECK-SAME: quant.uniform<u16:f32, 9.9999993884121538E-4:10>
// CHECK-NEXT: quant.scast
// CHECK-NEXT: onnx.Identity

func.func @dq_dq_add_update_input(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
  %0 = onnx.Constant {value = dense<10> : tensor<i8>} : tensor<!quant.uniform<i8:f32, 5.000000e+00>>
  %1 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i8>
  %3 = "onnx.QuantizeLinear"(%arg0, %1, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
  %4 = quant.scast %3 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %5 = "onnx.Identity"(%4) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %6 = "onnx.Add"(%5, %0) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<!quant.uniform<i8:f32, 5.000000e+00>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %7 = "onnx.Identity"(%6) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %8 = quant.scast %7 : tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>> to tensor<1x4xi8>
  return %8 : tensor<1x4xi8>
}

func.func @dq_const_add_update_input(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
  %0 = onnx.Constant dense<1.000000e+01> : tensor<f32>
  %1 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i8>
  %3 = "onnx.QuantizeLinear"(%arg0, %1, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
  %4 = quant.scast %3 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %5 = "onnx.Identity"(%4) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %6 = "onnx.Add"(%5, %0) : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %7 = "onnx.Identity"(%6) : (tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %8 = quant.scast %7 : tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>> to tensor<1x4xi8>
  return %8 : tensor<1x4xi8>
}

// CHECK-LABEL: @dq_const_add_update_input
// CHECK: onnx.Identity
// CHECK-NEXT: quant.scast
// CHECK-SAME: tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612:100>>
// CHECK-NEXT: quant.scast
// CHECK-NEXT: onnx.Identity

func.func @const_dq_add_update_input(%arg0: tensor<1x4xf32>) -> tensor<1x4xi8> {
  %0 = onnx.Constant dense<1.000000e+01> : tensor<f32>
  %1 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %2 = onnx.Constant dense<0> : tensor<i8>
  %3 = "onnx.QuantizeLinear"(%arg0, %1, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
  %4 = quant.scast %3 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>
  %5 = "onnx.Transpose"(%4) {perm = [1, 0]} : (tensor<1x4x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<4x1x!quant.uniform<i8:f32, 5.000000e-01>>
  %6 = "onnx.Add"(%0, %5) : (tensor<f32>, tensor<4x1x!quant.uniform<i8:f32, 5.000000e-01>>) -> tensor<4x1x!quant.uniform<i8:f32, 0.10000000149011612>>
  %7 = "onnx.Transpose"(%6) {perm = [1, 0]} : (tensor<4x1x!quant.uniform<i8:f32, 0.10000000149011612>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %8 = quant.scast %7 : tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>> to tensor<1x4xi8>
  return %8 : tensor<1x4xi8>
}

// CANON-LABEL: @const_dq_add_update_input
// CANON: onnx.Transpose
// CANON-NEXT: quant.scast
// CANON-SAME: tensor<4x1x!quant.uniform<i8:f32, 0.10000000149011612:100>>
// CANON-NEXT: quant.scast
// CANON-NEXT: onnx.Transpose

func.func @branchBefore_foldIntoDQ(%arg0: tensor<1x4xf32>) -> (tensor<1x4xf32>, tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>) {
  %0 = onnx.Constant dense<2.000000e-01> : tensor<f32>
  %1 = onnx.Constant dense<1.000000e+01> : tensor<f32>
  %2 = onnx.Constant dense<5.000000e-01> : tensor<f32>
  %3 = onnx.Constant dense<0> : tensor<i8>
  %4 = "onnx.QuantizeLinear"(%arg0, %2, %3) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x4xf32>, tensor<f32>, tensor<i8>) -> tensor<1x4xi8>
  %5 = quant.scast %4 : tensor<1x4xi8> to tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %6 = "onnx.Identity"(%5) : (tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  %7 = "onnx.Add"(%6, %1) : (tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>, tensor<f32>) -> tensor<1x4x!quant.uniform<i8:f32, 0.20000000298023224>>
  %8 = "onnx.Identity"(%7) : (tensor<1x4x!quant.uniform<i8:f32, 0.20000000298023224>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.20000000298023224>>
  %9 = quant.scast %7 : tensor<1x4x!quant.uniform<i8:f32, 0.20000000298023224>> to tensor<1x4xi8>
  %10 = "onnx.DequantizeLinear"(%9, %0, %3) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x4xi8>, tensor<f32>, tensor<i8>) -> tensor<1x4xf32>
  %11 = "onnx.Abs"(%6) : (tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>) -> tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
  return %10, %11 : tensor<1x4xf32>, tensor<1x4x!quant.uniform<i8:f32, 0.10000000149011612>>
}

