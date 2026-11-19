// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --constprop-zhigh %s | FileCheck --check-prefix=MAX %s
// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --nnpa-saturation=limit --constprop-zhigh %s | FileCheck --check-prefix=LIMIT %s
// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --nnpa-disable-saturation --constprop-zhigh %s | FileCheck --check-prefix=MAX %s

// COM: Saturation of a constant stickified at compile time: -FLT_MAX becomes
// COM: -DLF16_MAX (dlfloat16 0xFFFE) by default, and -DLF16_MAX/4 (0xFBFE)
// COM: with --nnpa-saturation=limit. Constants are always saturated, also
// COM: with --nnpa-disable-saturation.

func.func @test_constant_saturation_limit() -> tensor<1x64xf16, #zhigh.layout<{dataLayout = "2D"}>> {
  %c = onnx.Constant dense<-3.40282347E+38> : tensor<1x64xf32>
  %s = "zhigh.Stick"(%c) {layout = "2D"} : (tensor<1x64xf32>) -> tensor<1x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
  return %s : tensor<1x64xf16, #zhigh.layout<{dataLayout = "2D"}>>
}

// MAX:    zhigh: "0x{{[0-9A-F]*}}FFFEFFFEFFFEFFFE
// LIMIT:  zhigh: "0x{{[0-9A-F]*}}FBFEFBFEFBFEFBFE
