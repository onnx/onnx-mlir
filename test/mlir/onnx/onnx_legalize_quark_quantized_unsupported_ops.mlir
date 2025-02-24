// RUN: onnx-mlir-opt -legalize-quark-quantized-ops="from-type=f32 to-type=bf16" --split-input-file %s | FileCheck %s

func.func @test_one_f32_cast_one_bf16_one_f32(%arg0: tensor<1x1x50x768xbf16>, %arg1: tensor<?x?x?x?xbf16>) -> (tensor<?x?x?x?xbf16>, tensor<*xbf16>, tensor<*xbf16>) {
    %0 = onnx.Constant dense<0.000000e+00> : tensor<768xbf16>
    %1 = "onnx.NoValue"() {value} : () -> none
    %2 = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = f32} : (tensor<1x1x50x768xbf16>) -> tensor<1x1x50x768xf32>
    %3 = "onnx.ClipV12"(%2, %1, %1) : (tensor<1x1x50x768xf32>, none, none) -> tensor<?x?x?x?xf32>
    %4 = "onnx.Cast"(%3) {saturate = 1 : si64, to = bf16} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xbf16>
    %Y, %Mean, %InvStdDev = "onnx.LayerNormalization"(%4, %0, %0) {axis = -1 : si64, epsilon = 9.99999974E-6 : f32, onnx_node_name = "/vision_model/pre_layrnorm/LayerNormalization", stash_type = 1 : si64} : (tensor<?x?x?x?xbf16>, tensor<768xbf16>, tensor<768xbf16>) -> (tensor<?x?x?x?xbf16>, tensor<*xbf16>, tensor<*xbf16>)
    %5 = "onnx.Cast"(%Y) {saturate = 1 : si64, to = f32} : (tensor<?x?x?x?xbf16>) -> tensor<?x?x?x?xf32>
    %6 = "onnx.ClipV12"(%5, %1, %1) : (tensor<?x?x?x?xf32>, none, none) -> tensor<?x?x?x?xf32>
    %7 = "onnx.Cast"(%6) {saturate = 1 : si64, to = bf16} : (tensor<?x?x?x?xf32>) -> tensor<?x?x?x?xbf16>
    return %7, %Mean, %InvStdDev : tensor<?x?x?x?xbf16>, tensor<*xbf16>, tensor<*xbf16>
}

// -----

func.func @test_scan_simple_main_graph(%arg0: tensor<2xbf16>, %arg1: tensor<3x2xbf16>) -> (tensor<*xbf16>, tensor<*xbf16>) {
    %0 = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = f32} : (tensor<2xbf16>) -> tensor<2xf32>
    %1 = "onnx.Cast"(%arg1) {saturate = 1 : si64, to = f32} : (tensor<3x2xbf16>) -> tensor<3x2xf32>
    %2:2 = "onnx.Scan"(%0, %1) ({
    ^bb0(%arg2: tensor<*xf32>, %arg3: tensor<*xf32>):
      %5 = "onnx.Add"(%arg2, %arg3) : (tensor<*xf32>, tensor<*xf32>) -> tensor<*xf32>
      onnx.Yield %5, %5 : tensor<*xf32>, tensor<*xf32>
    }) {num_scan_inputs = 1 : si64} : (tensor<2xf32>, tensor<3x2xf32>) -> (tensor<*xf32>, tensor<*xf32>)
    %3 = "onnx.Cast"(%2#0) {saturate = 1 : si64, to = bf16} : (tensor<*xf32>) -> tensor<*xbf16>
    %4 = "onnx.Cast"(%2#1) {saturate = 1 : si64, to = bf16} : (tensor<*xf32>) -> tensor<*xbf16>
    onnx.Return %3, %4 : tensor<*xbf16>, tensor<*xbf16>
}

// -----

func.func @test_cast1(%arg0: tensor<2x3x4xbf16>) -> tensor<*xbf16> {
    %0 = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = bf16} : (tensor<2x3x4xbf16>) -> tensor<*xbf16>
    onnx.Return %0 : tensor<*xbf16>
}

// -----

func.func @test_cast2(%arg0: tensor<*xf32>) -> tensor<*xbf16> {
    %0 = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = bf16} : (tensor<*xf32>) -> tensor<*xbf16>
    onnx.Return %0 : tensor<*xbf16>
}

// -----

func.func @test_cast3(%arg0: tensor<?xf32>) -> tensor<?xbf16> {
    %0 = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = bf16} : (tensor<?xf32>) -> tensor<?xbf16>
    onnx.Return %0 : tensor<?xbf16>
}

// -----

func.func @test_cast4(%arg0: tensor<2x3x4xf32>) -> tensor<*xbf16> {
    %0 = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = bf16} : (tensor<2x3x4xf32>) -> tensor<*xbf16>
    onnx.Return %0 : tensor<*xbf16>
}