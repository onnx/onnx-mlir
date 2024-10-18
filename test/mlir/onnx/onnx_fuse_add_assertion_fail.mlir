// RUN: onnx-mlir-opt  --shape-inference --canonicalize="test-convergence=true" --shape-inference --cse %s | FileCheck %s

module attributes {llvm.data_layout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-f80:128-n8:16:32:64-S128", llvm.target_triple = "x86_64-unknown-linux-gnu", "onnx-mlir.symbol-postfix" = "onnxmodel"} {
  func.func @main_graph(%arg0: tensor<1x180x320x3xf32> {onnx.name = "src"}, %arg1: tensor<1x90x160x16xf32> {onnx.name = "r1i"}, %arg2: tensor<1x45x80x20xf32> {onnx.name = "r2i"}, %arg3: tensor<1x23x40x40xf32> {onnx.name = "r3i"}, %arg4: tensor<1x12x20x64xf32> {onnx.name = "r4i"}) -> (tensor<1x3x180x320xf32> {onnx.name = "fgr"}, tensor<1x3x180x320xf32> {onnx.name = "pha"}, tensor<1x3x180x320xf32> {onnx.name = "r1o"}, tensor<1x16x90x160xf32>{onnx.name = "r2o"}, tensor<1x16x90x160xf32> {onnx.name = "r3o"}, tensor<1x16x90x160xf32>{onnx.name = "r4o"}) {
    %0 = onnx.Constant dense<[[[-4.850000e-01]], [[-4.560000e-01]], [[-4.060000e-01]]]> : tensor<3x1x1xf32>
    %58 = onnx.Constant dense<3.000000e+00> : tensor<16x3x3x3xf32>
    %59 = onnx.Constant dense<[2.98861408, -1.22985208, 2.43826318, -3.98499513, 4.62797928, 2.54142761, 2.45345306, 2.64061832, 2.13576674, 2.30800247, -0.198341176, -0.427822977, -1.09159482, 4.85548782, 2.70597649, 2.6902504]> : tensor<16xf32>
    %164 = onnx.Constant dense<2> : tensor<1xi64>
    %165 = onnx.Constant dense<3.000000e+00> : tensor<f32>
    %166 = onnx.Constant dense<1.000000e+00> : tensor<f32>
    %167 = onnx.Constant dense<23> : tensor<1xi64>
    %168 = onnx.Constant dense<6.000000e+00> : tensor<f32>
    %169 = onnx.Constant dense<0.000000e+00> : tensor<f32>
    %170 = onnx.Constant dense<[[[2.290000e-01]], [[2.240000e-01]], [[2.250000e-01]]]> : tensor<3x1x1xf32>
    %171 = onnx.Constant dense<1> : tensor<1xi64>
    %172 = onnx.Constant dense<45> : tensor<1xi64>
    %173 = "onnx.Transpose"(%arg0) {onnx_node_name = "Transpose_0", perm = [0, 3, 1, 2]} : (tensor<1x180x320x3xf32>) -> tensor<1x3x180x320xf32>
    %174 = "onnx.Add"(%173, %0) {onnx_node_name = "Sub_6-Initializer_ortshared_1_3_3_1_token_113_24"} : (tensor<1x3x180x320xf32>, tensor<3x1x1xf32>) -> tensor<1x3x180x320xf32>
    %175 = "onnx.Div"(%174, %170) {onnx_node_name = "Div_8"} : (tensor<1x3x180x320xf32>, tensor<3x1x1xf32>) -> tensor<1x3x180x320xf32>
    %176 = "onnx.Conv"(%175, %58, %59) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_9", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x3x180x320xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x16x90x160xf32>
    %177 = "onnx.Add"(%176, %165) {onnx_node_name = "Add_11"} : (tensor<1x16x90x160xf32>, tensor<f32>) -> tensor<1x16x90x160xf32>
    %178 = "onnx.Clip"(%177, %169, %168) {onnx_node_name = "Clip_14_8"} : (tensor<1x16x90x160xf32>, tensor<f32>, tensor<f32>) -> tensor<1x16x90x160xf32>
    return %173, %175, %175, %176, %177, %178 : tensor<1x3x180x320xf32>, tensor<1x3x180x320xf32>, tensor<1x3x180x320xf32>, tensor<1x16x90x160xf32>, tensor<1x16x90x160xf32>, tensor<1x16x90x160xf32>
  }
  "onnx.EntryPoint"() {func = @main_graph} : () -> ()
}

//CHECK:  %{{[0-9]+}} = "onnx.Conv"(%{{.*}}, %{{.*}}, %{{.*}}) {auto_pad = "NOTSET", dilations = [1, 1], group = 1 : si64, kernel_shape = [3, 3], onnx_node_name = "Conv_9", pads = [1, 1, 1, 1], strides = [2, 2]} : (tensor<1x3x180x320xf32>, tensor<16x3x3x3xf32>, tensor<16xf32>) -> tensor<1x16x90x160xf32>
//CHECK-NEXT:  %{{[0-9]+}} = "onnx.Add"(%{{.*}}, %{{.*}}) {onnx_node_name = "Add_11"} : (tensor<1x16x90x160xf32>, tensor<f32>) -> tensor<1x16x90x160xf32>
