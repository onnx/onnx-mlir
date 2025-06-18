// RUN: onnx-mlir-opt --shape-inference --decompose-onnx=enable-convtranspose-1d-phased %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --shape-inference --decompose-onnx %s -split-input-file | FileCheck %s --check-prefix=DISABLED

func.func @test_convtrans_stride_2_kernel_shape_4(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x4xf32>) -> tensor<1x24x400xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {dilations = [1], group = 1 : si64, kernel_shape = [4],  pads = [1,1], strides = [2]} : (tensor<1x64x200xf32>, tensor<64x24x4xf32>, tensor<24xf32>) -> tensor<1x24x400xf32>
  onnx.Return %1 : tensor<1x24x400xf32>
}
// CHECK-LABEL:   test_convtrans_stride_2_kernel_shape_4
// CHECK:           onnx.Conv
// CHECK:           onnx.Conv
// DISABLED-LABEL: test_convtrans_stride_2_kernel_shape_4
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_2_kernel_shape_4_b(%arg0: tensor<1x64x400xf32>, %arg1: tensor<64x24x4xf32>) -> tensor<1x24x800xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {dilations = [1], group = 1 : si64, kernel_shape = [4],  pads = [1,1], strides = [2]} : (tensor<1x64x400xf32>, tensor<64x24x4xf32>, tensor<24xf32>) -> tensor<1x24x800xf32>
  onnx.Return %1 : tensor<1x24x800xf32>
}
// CHECK-LABEL:   test_convtrans_stride_2_kernel_shape_4_b
// CHECK:           onnx.Conv
// CHECK:           onnx.Conv
// DISABLED-LABEL: test_convtrans_stride_2_kernel_shape_4_b
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_2_kernel_shape_8_unsupported(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x8xf32>) -> tensor<1x24x404xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {dilations = [1], group = 1 : si64, kernel_shape = [8],  pads = [1,1], strides = [2]} : (tensor<1x64x200xf32>, tensor<64x24x8xf32>, tensor<24xf32>) -> tensor<1x24x404xf32>
  onnx.Return %1 : tensor<1x24x404xf32>
}
// CHECK-LABEL:   test_convtrans_stride_2_kernel_shape_8_unsupported
// CHECK:           onnx.ConvTranspose
// DISABLED-LABEL: test_convtrans_stride_2_kernel_shape_8_unsupported
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_2_kernel_shape_4_nobias(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x4xf32>) -> tensor<1x24x400xf32> {      
  %0 = "onnx.NoValue"() { value} : () -> none
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { dilations = [1], group = 1 : si64, kernel_shape = [4],  pads = [1,1], strides = [2]} : (tensor<1x64x200xf32>, tensor<64x24x4xf32>, none) -> tensor<1x24x400xf32>
  onnx.Return %1 : tensor<1x24x400xf32>
}
// CHECK-LABEL:   test_convtrans_stride_2_kernel_shape_4_nobias
// CHECK:           onnx.Conv
// CHECK:           onnx.Conv
// DISABLED-LABEL: test_convtrans_stride_2_kernel_shape_4_nobias
// DISABLED: onnx.ConvTranspose
// -----

func.func @test_convtrans_stride_4(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { dilations = [1], group = 1 : si64, kernel_shape = [8],  pads = [2,2], strides = [4]} : (tensor<1x64x200xf32>, tensor<64x24x8xf32>, tensor<24xf32>) -> tensor<1x24x800xf32>
  onnx.Return %1 : tensor<1x24x800xf32>
}
// CHECK-LABEL:   test_convtrans_stride_4
// CHECK:           onnx.Conv
// CHECK:           onnx.Conv
// CHECK:           onnx.Conv
// CHECK:           onnx.Conv
// DISABLED-LABEL: test_convtrans_stride_4
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_5(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x10xf32>) -> tensor<1x24x1001xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { dilations = [1], group = 1 : si64, kernel_shape = [10],  pads = [2,2], strides = [5]} : (tensor<1x64x200xf32>, tensor<64x24x10xf32>, tensor<24xf32>) -> tensor<1x24x1001xf32>
  onnx.Return %1 : tensor<1x24x1001xf32>
}
// CHECK-LABEL:   test_convtrans_stride_5
// CHECK:           onnx.Conv
// CHECK:           onnx.Conv
// CHECK:           onnx.Conv
// CHECK:           onnx.Conv
// CHECK:           onnx.Conv
// DISABLED-LABEL: test_convtrans_stride_5
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_2_dilation2(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x4xf32>) -> tensor<1x24x403xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) {dilations = [2], group = 1 : si64, kernel_shape = [4],  pads = [1,1], strides = [2]} : (tensor<1x64x200xf32>, tensor<64x24x4xf32>, tensor<24xf32>) -> tensor<1x24x403xf32>
  onnx.Return %1 : tensor<1x24x403xf32>
}
// CHECK-LABEL:   test_convtrans_stride_2_dilation2
// CHECK:           onnx.ConvTranspose
// DISABLED-LABEL: test_convtrans_stride_2_dilation2
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_2_nodilation(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x4xf32>) -> tensor<1x24x400xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { group = 1 : si64, kernel_shape = [4],  pads = [1,1], strides = [2]} : (tensor<1x64x200xf32>, tensor<64x24x4xf32>, tensor<24xf32>) -> tensor<1x24x400xf32>
  onnx.Return %1 : tensor<1x24x400xf32>
}
// CHECK-LABEL:   test_convtrans_stride_2_nodilation
// CHECK:           onnx.Conv
// CHECK:           onnx.Conv
// DISABLED-LABEL: test_convtrans_stride_2_nodilation
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_4_lrelu(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { dilations = [1], group = 1 : si64, kernel_shape = [8],  pads = [2,2], strides = [4]} : (tensor<1x64x200xf32>, tensor<64x24x8xf32>, tensor<24xf32>) -> tensor<1x24x800xf32>
  %2 = "onnx.LeakyRelu"(%1) {alpha = 1.000000e-01 : f32} : (tensor<1x24x800xf32>) -> tensor<1x24x800xf32> 
  onnx.Return %2 : tensor<1x24x800xf32>
}
// CHECK-LABEL:   test_convtrans_stride_4_lrelu
// CHECK:           onnx.Conv
// CHECK:           onnx.LeakyRelu
// CHECK:           onnx.Conv
// CHECK:           onnx.LeakyRelu
// CHECK:           onnx.Conv
// CHECK:           onnx.LeakyRelu
// CHECK:           onnx.Conv
// CHECK:           onnx.LeakyRelu
// DISABLED-LABEL: test_convtrans_stride_4_lrelu
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_4_relu(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { dilations = [1], group = 1 : si64, kernel_shape = [8],  pads = [2,2], strides = [4]} : (tensor<1x64x200xf32>, tensor<64x24x8xf32>, tensor<24xf32>) -> tensor<1x24x800xf32>
  %2 = "onnx.Relu"(%1) : (tensor<1x24x800xf32>) -> tensor<1x24x800xf32> 
  onnx.Return %2 : tensor<1x24x800xf32>
}
// CHECK-LABEL:   test_convtrans_stride_4_relu
// CHECK:           onnx.Conv
// CHECK:           onnx.Relu
// CHECK:           onnx.Conv
// CHECK:           onnx.Relu
// CHECK:           onnx.Conv
// CHECK:           onnx.Relu
// CHECK:           onnx.Conv
// CHECK:           onnx.Relu
// DISABLED-LABEL: test_convtrans_stride_4_relu
// DISABLED: onnx.ConvTranspose

// -----

func.func @test_convtrans_stride_4_lrelu_default_value(%arg0: tensor<1x64x200xf32>, %arg1: tensor<64x24x8xf32>) -> tensor<1x24x800xf32> {    
  %0 = "onnx.Constant" () { value= dense<0.02> : tensor<24xf32>} : ()-> tensor<24xf32>
  %1 = "onnx.ConvTranspose"(%arg0, %arg1, %0) { dilations = [1], group = 1 : si64, kernel_shape = [8],  pads = [2,2], strides = [4]} : (tensor<1x64x200xf32>, tensor<64x24x8xf32>, tensor<24xf32>) -> tensor<1x24x800xf32>
  %2 = "onnx.LeakyRelu"(%1) : (tensor<1x24x800xf32>) -> tensor<1x24x800xf32> 
  onnx.Return %2 : tensor<1x24x800xf32>
}
// CHECK-LABEL:   test_convtrans_stride_4_lrelu
// CHECK:           onnx.Conv
// CHECK:           onnx.LeakyRelu
// CHECK:           onnx.Conv
// CHECK:           onnx.LeakyRelu
// CHECK:           onnx.Conv
// CHECK:           onnx.LeakyRelu
// CHECK:           onnx.Conv
// CHECK:           onnx.LeakyRelu
// DISABLED-LABEL: test_convtrans_stride_4_lrelu
// DISABLED: onnx.ConvTranspose
