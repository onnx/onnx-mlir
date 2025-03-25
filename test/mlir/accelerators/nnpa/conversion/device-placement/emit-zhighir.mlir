
//&& RUN: onnx-mlir --EmitZHighIR --mcpu=z16 --maccel=NNPA --disable-constant-prop=true --printIR %s | FileCheck %s

// Note that, we intentionally add `device=cpu` into onnx.Gemm to force it run on CPU.
module { 
  func.func @mnist(%arg0: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
    %0 = onnx.Constant dense<[-0.0822488219, -0.108868778, -0.141039595, -0.204869166, -0.17913565, -0.215438381, -0.133805066, -0.195724562, -0.268250644, -0.258212209, -0.0761560649, 0.0132841459, -0.00444464432, -0.414740831, -0.17879115, -0.0386558883]> : tensor<16xf32>
    %1 = onnx.Constant dense<[-0.161539719, -0.433835655, 0.091641359, -0.0168522168, -0.0650264397, -0.131737873, 0.0204175506, -0.121110231]> : tensor<8xf32>
    %2 = onnx.Constant dense_resource<__elided__> : tensor<16x4x4x10xf32>
    %3 = onnx.Constant dense_resource<__elided__> : tensor<16x8x5x5xf32>
    %4 = onnx.Constant dense_resource<__elided__> : tensor<8x1x5x5xf32>
    %5 = onnx.Constant dense<[1, 256]> : tensor<2xi64>
    %6 = onnx.Constant dense<[256, 10]> : tensor<2xi64>
    %7 = onnx.Constant dense<[[-0.0448560268, 0.00779166119, 0.0681008175, 0.0299937408, -0.126409635, 0.14021875, -0.0552849025, -0.0493838154, 0.0843220502, -0.0545404144]]> : tensor<1x10xf32>
    %8 = "onnx.Reshape"(%2, %6) {allowzero = 0 : si64, onnx_node_name = "Times212_reshape1"} : (tensor<16x4x4x10xf32>, tensor<2xi64>) -> tensor<256x10xf32>
    %9 = "onnx.Conv"(%arg0, %4, %1) {auto_pad = "SAME_UPPER", device = "nnpa", dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], strides = [1, 1]} : (tensor<1x1x28x28xf32>, tensor<8x1x5x5xf32>, tensor<8xf32>) -> tensor<1x8x28x28xf32>
    %10 = "onnx.Relu"(%9) {device = "nnpa", onnx_node_name = "ReLU32"} : (tensor<1x8x28x28xf32>) -> tensor<1x8x28x28xf32>
    %11 = "onnx.MaxPoolSingleOut"(%10) {auto_pad = "NOTSET", ceil_mode = 0 : si64, device = "nnpa", kernel_shape = [2, 2], onnx_node_name = "Pooling66", pads = [0, 0, 0, 0], storage_order = 0 : si64, strides = [2, 2]} : (tensor<1x8x28x28xf32>) -> tensor<1x8x14x14xf32>
    %12 = "onnx.Conv"(%11, %3, %0) {auto_pad = "SAME_UPPER", device = "nnpa", dilations = [1, 1], group = 1 : si64, kernel_shape = [5, 5], strides = [1, 1]} : (tensor<1x8x14x14xf32>, tensor<16x8x5x5xf32>, tensor<16xf32>) -> tensor<1x16x14x14xf32>
    %13 = "onnx.Relu"(%12) {device = "nnpa", onnx_node_name = "ReLU114"} : (tensor<1x16x14x14xf32>) -> tensor<1x16x14x14xf32>
    %14 = "onnx.MaxPoolSingleOut"(%13) {auto_pad = "NOTSET", ceil_mode = 0 : si64, device = "nnpa", kernel_shape = [3, 3], onnx_node_name = "Pooling160", pads = [0, 0, 0, 0], storage_order = 0 : si64, strides = [3, 3]} : (tensor<1x16x14x14xf32>) -> tensor<1x16x4x4xf32>
    %15 = "onnx.Reshape"(%14, %5) {allowzero = 0 : si64, onnx_node_name = "Times212_reshape0"} : (tensor<1x16x4x4xf32>, tensor<2xi64>) -> tensor<1x256xf32>
    %16 = "onnx.Gemm"(%15, %8, %7) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, device = "cpu", onnx_node_name = "Times212_reshape0_onnx.Gemm_2", transA = 0 : si64, transB = 0 : si64} : (tensor<1x256xf32>, tensor<256x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
    return %16 : tensor<1x10xf32>
  }
  "onnx.EntryPoint"() {func = @mnist} : () -> ()

// CHECK-LABEL:  func.func @mnist
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<1x1x28x28xf32>) -> tensor<1x10xf32> {
// CHECK-DAG:       [[VAR_0_:%.+]] = onnx.Constant dense_resource<__elided__> : tensor<16x4x4x10xf32>
// CHECK-DAG:       [[VAR_1_:%.+]] = onnx.Constant dense_resource<__elided__> : tensor<16x8x5x5xf32>
// CHECK-DAG:       [[VAR_2_:%.+]] = onnx.Constant dense_resource<__elided__> : tensor<8x1x5x5xf32>
// CHECK-DAG:       [[VAR_3_:%.+]] = onnx.Constant dense<[1, 256]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_4_:%.+]] = onnx.Constant dense<[256, 10]> : tensor<2xi64>
// CHECK-DAG:       [[VAR_5_:%.+]] = onnx.Constant dense<{{.}}[-0.0448560268, 0.00779166119, 0.0681008175, 0.0299937408, -0.126409635, 0.14021875, -0.0552849025, -0.0493838154, 0.0843220502, -0.0545404144]{{.}}> : tensor<1x10xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Reshape"([[VAR_0_]], [[VAR_4_]]) {allowzero = 0 : si64, onnx_node_name = "Times212_reshape1"} : (tensor<16x4x4x10xf32>, tensor<2xi64>) -> tensor<256x10xf32>
// CHECK-DAG:       [[VAR_7_:%.+]] = "zhigh.Stick"([[PARAM_0_]]) {layout = "NHWC"} : (tensor<1x1x28x28xf32>) -> tensor<1x28x28x1xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_8_:%.+]] = "onnx.Transpose"([[VAR_2_]]) {perm = [2, 3, 1, 0]} : (tensor<8x1x5x5xf32>) -> tensor<5x5x1x8xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_9_:%.+]] = "zhigh.Stick"([[VAR_8_]]) {layout = "HWCK"} : (tensor<5x5x1x8xf32>) -> tensor<5x5x1x8xf16, #zhigh.layout<{dataLayout = "HWCK"}>>
// CHECK-DAG:       [[VAR_10_:%.+]] = "zhigh.StickifiedConstant"() {alignment = 4096 : i64, value = dense_resource<zhigh_1> : tensor<4096xi8>} : () -> tensor<8xf16, #zhigh.layout<{dataLayout = "1D"}>>
// CHECK:           [[VAR_11_:%.+]] = "zhigh.Conv2D"([[VAR_7_]], [[VAR_9_]], [[VAR_10_]]) {act_func = "ACT_RELU", kernel_shape = [5, 5], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x28x28x1xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<5x5x1x8xf16, #zhigh.layout<{dataLayout = "HWCK"}>>, tensor<8xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<1x28x28x8xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_12_:%.+]] = "zhigh.MaxPool2D"([[VAR_11_]]) {kernel_shape = [2, 2], padding_type = "VALID_PADDING", strides = [2, 2]} : (tensor<1x28x28x8xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x14x14x8xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK-DAG:       [[VAR_13_:%.+]] = "onnx.Transpose"([[VAR_1_]]) {perm = [2, 3, 1, 0]} : (tensor<16x8x5x5xf32>) -> tensor<5x5x8x16xf32>
// CHECK-NOT: separator of consecutive DAGs
// CHECK-DAG:       [[VAR_14_:%.+]] = "zhigh.Stick"([[VAR_13_]]) {layout = "HWCK"} : (tensor<5x5x8x16xf32>) -> tensor<5x5x8x16xf16, #zhigh.layout<{dataLayout = "HWCK"}>>
// CHECK-DAG:       [[VAR_15_:%.+]] = "zhigh.StickifiedConstant"() {alignment = 4096 : i64, value = dense_resource<zhigh> : tensor<4096xi8>} : () -> tensor<16xf16, #zhigh.layout<{dataLayout = "1D"}>>
// CHECK:           [[VAR_16_:%.+]] = "zhigh.Conv2D"([[VAR_12_]], [[VAR_14_]], [[VAR_15_]]) {act_func = "ACT_RELU", kernel_shape = [5, 5], padding_type = "SAME_PADDING", strides = [1, 1]} : (tensor<1x14x14x8xf16, #zhigh.layout<{dataLayout = "NHWC"}>>, tensor<5x5x8x16xf16, #zhigh.layout<{dataLayout = "HWCK"}>>, tensor<16xf16, #zhigh.layout<{dataLayout = "1D"}>>) -> tensor<1x14x14x16xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_17_:%.+]] = "zhigh.MaxPool2D"([[VAR_16_]]) {kernel_shape = [3, 3], padding_type = "VALID_PADDING", strides = [3, 3]} : (tensor<1x14x14x16xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x4x4x16xf16, #zhigh.layout<{dataLayout = "NHWC"}>>
// CHECK:           [[VAR_18_:%.+]] = "zhigh.Unstick"([[VAR_17_]]) : (tensor<1x4x4x16xf16, #zhigh.layout<{dataLayout = "NHWC"}>>) -> tensor<1x16x4x4xf32>
// CHECK:           [[VAR_19_:%.+]] = "onnx.Reshape"([[VAR_18_]], [[VAR_3_]]) {allowzero = 0 : si64, onnx_node_name = "Times212_reshape0"} : (tensor<1x16x4x4xf32>, tensor<2xi64>) -> tensor<1x256xf32>
// CHECK:           [[VAR_20_:%.+]] = "onnx.Gemm"([[VAR_19_]], [[VAR_6_]], [[VAR_5_]]) {alpha = 1.000000e+00 : f32, beta = 1.000000e+00 : f32, device = "cpu", onnx_node_name = "Times212_reshape0_onnx.Gemm_2", transA = 0 : si64, transB = 0 : si64} : (tensor<1x256xf32>, tensor<256x10xf32>, tensor<1x10xf32>) -> tensor<1x10xf32>
// CHECK:           return [[VAR_20_]] : tensor<1x10xf32>
// CHECK:         }
// CHECK:         "onnx.EntryPoint"() {func = @mnist} : () -> ()
}
