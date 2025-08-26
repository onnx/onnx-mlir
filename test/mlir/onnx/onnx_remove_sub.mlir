// RUN: onnx-mlir-opt --dq-binary-q-opt-onnx-to-onnx %s --split-input-file | FileCheck %s

// 1) dq1-dq2(const input)-sub-q-dq. remove->sub,q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern1a
// CHECK: %[[ZP:.*]] = onnx.Constant dense<65535> : tensor<ui16>
// CHECK-NOT: onnx.Sub
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: return
// CHECK-NOT: onnx.DequantizeLinear
func.func @test_removebinary_pattern1a(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%0 = onnx.Constant dense<0> : tensor<ui16>
%1 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%2 = onnx.Constant dense<65535> : tensor<ui16>
%3 = onnx.Constant dense<39664> : tensor<ui16>
%4 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
%5 = "onnx.DequantizeLinear"(%2, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>
%6 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
%7 = "onnx.Sub"(%6, %5) : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<1x1x1x128xf32>
%8 = "onnx.QuantizeLinear"(%7, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%9 = "onnx.DequantizeLinear"(%8, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
return %9 : tensor<1x1x1x128xf32>
}

// -----
// 2) dq1-dq2(const input)-sub-q-dq. remove->sub,q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern1b
// CHECK: %[[ZP:.*]] = onnx.Constant dense<65535> : tensor<ui16>
// CHECK-NOT: onnx.Sub
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: return
// CHECK-NOT: onnx.DequantizeLinear
func.func @test_removebinary_pattern1b(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%0 = onnx.Constant dense<0> : tensor<ui16>
%1 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%2 = onnx.Constant dense<65535> : tensor<ui16>
%3 = onnx.Constant dense<39664> : tensor<ui16>
%4 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
%5 = "onnx.DequantizeLinear"(%2, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>
%6 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
%7 = "onnx.Sub"(%5, %6) : (tensor<f32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
%8 = "onnx.QuantizeLinear"(%7, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%9 = "onnx.DequantizeLinear"(%8, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
return %9 : tensor<1x1x1x128xf32>
}

// -----
// 3) dq1-dq2(const input)-Sub-q-dq. remove->Sub,q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern1c
// CHECK-NOT: onnx.Sub
// CHECK-NOT: onnx.QuantizeLinear
func.func @test_removebinary_pattern1c(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%0 = onnx.Constant dense<0> : tensor<ui16>
%1 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%2 = onnx.Constant dense<65535> : tensor<ui16>
%3 = onnx.Constant dense<0.152590215> : tensor<f32>
%4 = onnx.Constant dense<0> : tensor<ui16>
%5 = "onnx.Identity"(%4) : (tensor<ui16>) -> tensor<ui16>
%6 = "onnx.DequantizeLinear"(%5, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>
%7 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
%8 = "onnx.Sub"(%7, %6) : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<1x1x1x128xf32>
%9 = "onnx.QuantizeLinear"(%8, %3, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%10 = "onnx.DequantizeLinear"(%9, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
return %10 : tensor<1x1x1x128xf32>
}

// -----
// 4) dq1-dq2(const input)-Sub-q-dq. remove->Sub,q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern1d
// CHECK-NOT: onnx.Sub
// CHECK-NOT: onnx.QuantizeLinear
func.func @test_removebinary_pattern1d(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%0 = onnx.Constant dense<0> : tensor<ui16>
%1 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%2 = onnx.Constant dense<65535> : tensor<ui16>
%3 = onnx.Constant dense<0.152590215> : tensor<f32>
%4 = onnx.Constant dense<0> : tensor<ui16>
%5 = "onnx.Identity"(%4) : (tensor<ui16>) -> tensor<ui16>
%6 = "onnx.DequantizeLinear"(%5, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>
%7 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
%8 = "onnx.Sub"(%6, %7) : (tensor<f32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
%9 = "onnx.QuantizeLinear"(%8, %3, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%10 = "onnx.DequantizeLinear"(%9, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
return %10 : tensor<1x1x1x128xf32>
}

//-----
// 5) dq1-const-add-q-dq. remove->add, q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern2a
// CHECK: %[[ZP:.*]] = onnx.Constant dense<102> : tensor<ui16>
// CHECK-NOT: onnx.Add
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: return
// CHECK-NOT: onnx.DequantizeLinear
func.func @test_removebinary_pattern2a(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%0 = onnx.Constant dense<101> : tensor<ui16>
%1 = onnx.Constant dense<1.000000e+00> : tensor<f32>
%2 = onnx.Constant dense<65535> : tensor<ui16>
%3 = onnx.Constant dense<0.152590215> : tensor<f32>
%4 = onnx.Constant dense<1.000000e+00> : tensor<f32>
%5 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
%6 = "onnx.Sub"(%5, %4) : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<1x1x1x128xf32>
%7 = "onnx.QuantizeLinear"(%6, %3, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%8 = "onnx.DequantizeLinear"(%7, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
return %8 : tensor<1x1x1x128xf32>
}
//-----
// 6) const-dq1-sub-q-dq. remove->sub,q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern2b
// CHECK-NOT: onnx.Sub
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: return
// CHECK-NOT: onnx.DequantizeLinear
func.func @test_removebinary_pattern2b(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%0 = onnx.Constant dense<0> : tensor<ui16>
%1 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%2 = onnx.Constant dense<65535> : tensor<ui16>
%3 = onnx.Constant dense<0.152590215> : tensor<f32>
%4 = onnx.Constant dense<-1.000000e+04> : tensor<f32>
%5 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
%6 = "onnx.Sub"(%4, %5) : (tensor<f32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
%7 = "onnx.QuantizeLinear"(%6, %3, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%8 = "onnx.DequantizeLinear"(%7, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
return %8 : tensor<1x1x1x128xf32>
}
//-----
// 7) const-dq1-sub-q-dq. kval=0. remove->sub,q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern3a
// CHECK-NOT: onnx.Sub
// CHECK-NOT: onnx.QuantizeLinear
// CHECK: return
// CHECK-NOT: onnx.DequantizeLinear
func.func @test_removebinary_pattern3a(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%0 = onnx.Constant dense<0> : tensor<ui16>
%1 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%2 = onnx.Constant dense<65535> : tensor<ui16>
%3 = onnx.Constant dense<0.152590215> : tensor<f32>
%4 = onnx.Constant dense<0.000000e+00> : tensor<f32>
%5 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
%6 = "onnx.Sub"(%4, %5) : (tensor<f32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
%7 = "onnx.QuantizeLinear"(%6, %3, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%8 = "onnx.DequantizeLinear"(%7, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
return %8 : tensor<1x1x1x128xf32>
}
//-----
// 8) const-dq1-Sub-q-dq. dst_scale=0. remove->Sub,q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern3b
// CHECK: onnx.Sub
// CHECK: onnx.QuantizeLinear
func.func @test_removebinary_pattern3b(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%0 = onnx.Constant dense<0> : tensor<ui16>
%1 = onnx.Constant dense<0.000000e+00> : tensor<f32>
%2 = onnx.Constant dense<65535> : tensor<ui16>
%3 = onnx.Constant dense<0.152590215> : tensor<f32>
%4 = onnx.Constant dense<-1.000000e+04> : tensor<f32>
%5 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
%6 = "onnx.Sub"(%4, %5) : (tensor<f32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
%7 = "onnx.QuantizeLinear"(%6, %3, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%8 = "onnx.DequantizeLinear"(%7, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
return %8 : tensor<1x1x1x128xf32>
}
//-----
// 9) dq1-dq2(const input)-sub-q-dq. remove->sub,q-dq.
// CHECK-LABEL: func.func @test_removebinary_pattern4
// CHECK-NOT: onnx.Sub
// CHECK: onnx.QuantizeLinear
func.func @test_removebinary_pattern4(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%0 = onnx.Constant dense<0> : tensor<ui16>
%1 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%2 = onnx.Constant dense<65535> : tensor<ui16>
%3 = onnx.Constant dense<39664> : tensor<ui16>
%4 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
%5 = "onnx.DequantizeLinear"(%2, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>
%6 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
%7 = "onnx.Sub"(%5, %6) : (tensor<f32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
%8 = "onnx.QuantizeLinear"(%7, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%9 = "onnx.DequantizeLinear"(%8, %4, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
return %9 : tensor<1x1x1x128xf32>
}
//-----
// 10) const-dq1-Sub-tanh. remove->none
// CHECK-LABEL: func.func @test_removebinary_pattern5
// CHECK: onnx.Sub
// CHECK: onnx.Tanh
func.func @test_removebinary_pattern5(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%0 = onnx.Constant dense<0> : tensor<ui16>
%1 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%2 = onnx.Constant dense<65535> : tensor<ui16>
%3 = onnx.Constant dense<39664> : tensor<ui16>
%4 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
%5 = "onnx.DequantizeLinear"(%2, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>
%6 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
%7 = "onnx.Sub"(%5, %6) : (tensor<f32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
%8 = "onnx.Tanh"(%7) : (tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
return %8 : tensor<1x1x1x128xf32>
}
//-----
// 11) dq1-dq2-sub-q-dq1-dq2-mul-Q-DQ. multi-use of scale and zp of dq-act before binary op. remove->mul, sub
// CHECK-LABEL: func.func @test_removebinary_pattern6
// CHECK-NOT: onnx.Sub
// CHECK-NOT: onnx.Sub
func.func @test_removebinary_pattern6(%arg0: tensor<1x1x1x128xui16>) -> tensor<1x1x1x128xf32> {
%0 = onnx.Constant dense<0> : tensor<ui16>
%1 = onnx.Constant dense<1.52590219E-5> : tensor<f32>
%2 = onnx.Constant dense<65535> : tensor<ui16>
%3 = onnx.Constant dense<0.152590215> : tensor<f32>
%4 = onnx.Constant dense<39664> : tensor<ui16>
%5 = onnx.Constant dense<2.57987776E-5> : tensor<f32>
%6 = "onnx.DequantizeLinear"(%2, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>
%7 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
%8 = "onnx.Mul"(%6, %7) : (tensor<f32>, tensor<1x1x1x128xf32>) -> tensor<1x1x1x128xf32>
%9 = "onnx.QuantizeLinear"(%8, %1, %0) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%10 = "onnx.DequantizeLinear"(%9, %1, %0) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
%11 = "onnx.DequantizeLinear"(%0, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<ui16>, tensor<f32>, tensor<ui16>) -> tensor<f32>
%12 = "onnx.Sub"(%10, %11) : (tensor<1x1x1x128xf32>, tensor<f32>) -> tensor<1x1x1x128xf32>
%13 = "onnx.QuantizeLinear"(%12, %3, %2) {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x1x1x128xf32>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xui16>
%14 = "onnx.DequantizeLinear"(%13, %3, %2) {axis = 1 : si64, block_size = 0 : si64} : (tensor<1x1x1x128xui16>, tensor<f32>, tensor<ui16>) -> tensor<1x1x1x128xf32>
return %14 : tensor<1x1x1x128xf32>
}
//-----
// 12) dq1-dq2(const input, per-axis length-2 on axis=0)-mul-q-dq.
// vectors wiht same values -> fusion
// CHECK-LABEL: func.func @test_removebinary_pattern7a
// CHECK-NOT: onnx.Sub
// CHECK-NOT: onnx.QuantizeLinear
func.func @test_removebinary_pattern7a(%arg0: tensor<2x1x1x128xui16>) -> tensor<2x1x1x128xf32> {
%0 = onnx.Constant dense<0> : tensor<2xui16>
%1 = onnx.Constant dense<1.52590219E-5> : tensor<2xf32>
%2 = onnx.Constant dense<65535> : tensor<2xui16>
%3 = onnx.Constant dense<0.152590215> : tensor<2xf32>
%4 = onnx.Constant dense<0> : tensor<2x1x1x1xui16>
%5 = "onnx.DequantizeLinear"(%4, %3, %2) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x1x1x1xui16>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x1xf32>
%6 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x1x1x128xui16>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x128xf32>
%7 = "onnx.Sub"(%6, %5) : (tensor<2x1x1x128xf32>, tensor<2x1x1x1xf32>) -> tensor<2x1x1x128xf32>
%8 = "onnx.QuantizeLinear"(%7, %3, %2) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x1x1x128xf32>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x128xui16>
%9 = "onnx.DequantizeLinear"(%8, %3, %2) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x1x1x128xui16>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x128xf32>
return %9 : tensor<2x1x1x128xf32>
}
//-----
//
// 13) dq1-dq2(const input, per-axis length-2 on axis=0)-mul-q-dq.
// vectors wiht different values -> no fusion
// CHECK-LABEL: func.func @test_removebinary_pattern7b
// CHECK: onnx.Sub
// CHECK: onnx.QuantizeLinear
func.func @test_removebinary_pattern7b(%arg0: tensor<2x1x1x128xui16>) -> tensor<2x1x1x128xf32> {
%0 = onnx.Constant dense<0> : tensor<2xui16>
%1 = onnx.Constant dense<1.52590219E-5> : tensor<2xf32>
%2 = onnx.Constant dense<[65535, 1]> : tensor<2xui16>
%3 = onnx.Constant dense<0.152590215> : tensor<2xf32>
%4 = onnx.Constant dense<0> : tensor<2x1x1x1xui16>
%5 = "onnx.DequantizeLinear"(%4, %3, %2) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x1x1x1xui16>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x1xf32>
%6 = "onnx.DequantizeLinear"(%arg0, %1, %0) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x1x1x128xui16>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x128xf32>
%7 = "onnx.Sub"(%6, %5) : (tensor<2x1x1x128xf32>, tensor<2x1x1x1xf32>) -> tensor<2x1x1x128xf32>
%8 = "onnx.QuantizeLinear"(%7, %3, %2) {axis = 0 : si64, block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<2x1x1x128xf32>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x128xui16>
%9 = "onnx.DequantizeLinear"(%8, %3, %2) {axis = 0 : si64, block_size = 0 : si64} : (tensor<2x1x1x128xui16>, tensor<2xf32>, tensor<2xui16>) -> tensor<2x1x1x128xf32>
return %9 : tensor<2x1x1x128xf32>
}
