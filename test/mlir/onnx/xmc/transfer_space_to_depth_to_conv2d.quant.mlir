// RUN: onnx-mlir-opt --split-input-file --transfer-space-to-depth-to-conv2d %s | FileCheck %s

  func.func @test_space_to_depth_qdq(%arg0: tensor<1x3x512x512xf32>) -> tensor<1x48x128x128xf32> {
    %0 = onnx.Constant dense<5.000000e-01> : tensor<f32>
    %1 = onnx.Constant dense<5> : tensor<i8>
    %2 = "onnx.QuantizeLinear"(%arg0, %0, %1) { block_size = 0 : si64, output_dtype = 0 : si64, saturate = 1 : si64} : (tensor<1x3x512x512xf32>, tensor<f32>, tensor<i8>) -> tensor<1x3x512x512x!quant.uniform<u8:f32, 5.000000e-01:5>>
    %3 = "onnx.SpaceToDepth"(%2) {blocksize = 4 : si64} : (tensor<1x3x512x512x!quant.uniform<u8:f32, 5.000000e-01:5>>) -> tensor<1x48x128x128x!quant.uniform<u8:f32, 5.000000e-01:5>>
    %4 = "onnx.DequantizeLinear"(%3, %0, %1) { block_size = 0 : si64} : (tensor<1x48x128x128x!quant.uniform<u8:f32, 5.000000e-01:5>>, tensor<f32>, tensor<i8>) -> tensor<1x48x128x128xf32>
    return %4 : tensor<1x48x128x128xf32>
  }
    // CHECK-DAG: %[[BIAS_CONST:.*]] = onnx.Constant {value = dense<0> : tensor<48xi8>} : tensor<48x!quant.uniform<i8:f32, 5.000000e-01>>
    // CHECK-DAG: %[[WEIGHT_CONST:.*]] = onnx.Constant {value = dense<{{.*}}> : tensor<48x3x4x4xi8>} : tensor<48x3x4x4x!quant.uniform<i8:f32, 1.000000e+00>>
    // CHECK-DAG: %[[INPUT_SCALE:.*]] = onnx.Constant dense<5.000000e-01> : tensor<f32>
    // CHECK-DAG: %[[INPUT_ZP:.*]] = onnx.Constant dense<5> : tensor<i8>

    // CHECK: %[[QUANT_INPUT:.*]] = "onnx.QuantizeLinear"(%arg0, %[[INPUT_SCALE]], %[[INPUT_ZP]])
    // CHECK-SAME: tensor<1x3x512x512x!quant.uniform<u8:f32, 5.000000e-01:5>>

    // CHECK: %[[CONV:.*]] = "onnx.Conv"(%[[QUANT_INPUT]], %[[WEIGHT_CONST]], %[[BIAS_CONST]])
    // CHECK-SAME: auto_pad = "NOTSET"
    // CHECK-SAME: kernel_shape = [4, 4]
    // CHECK-SAME: pads = [0, 0, 0, 0]
    // CHECK-SAME: strides = [4, 4]
    // CHECK-SAME: -> tensor<1x48x128x128x!quant.uniform<u8:f32, 5.000000e-01:5>>

    // CHECK: %[[DEQUANT_OUT:.*]] = "onnx.DequantizeLinear"(%[[CONV]], %[[INPUT_SCALE]], %[[INPUT_ZP]])
    // CHECK-SAME: tensor<1x48x128x128xf32>

    // CHECK: return %[[DEQUANT_OUT]]

    // CHECK-NOT: onnx.SpaceToDepth
