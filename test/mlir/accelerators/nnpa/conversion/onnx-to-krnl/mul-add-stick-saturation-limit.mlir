// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --convert-onnx-to-krnl --canonicalize %s | FileCheck --check-prefix=MAX %s
// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --nnpa-saturation=limit --convert-onnx-to-krnl --canonicalize %s | FileCheck --check-prefix=LIMIT %s
// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --nnpa-disable-saturation --convert-onnx-to-krnl --canonicalize %s | FileCheck --check-prefix=NOSAT %s

// COM: Saturation bounds of a stick fused into a compiler-generated loop
// COM: (the input of mul-add-stick.mlir): +/-DLF16_MAX by default,
// COM: +/-DLF16_MAX/4 with --nnpa-saturation=limit, none with
// COM: --nnpa-disable-saturation.

func.func @test_mul_add_stick_rotary(%arg0: tensor<?x12x?x32xf32>, %arg1: tensor<1x1x?x32xf32>, %arg2: tensor<?x12x?x32xf32>, %arg3: tensor<1x1x?x32xf32>, %arg4: tensor<3xi64>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>> {
  %0 = "onnx.Fused"(%arg0, %arg1, %arg2, %arg3, %arg4) <{kind = "zhigh.mul-add-stick"}> ({
  ^bb0(%a0: tensor<?x12x?x32xf32>, %a1: tensor<1x1x?x32xf32>, %b0: tensor<?x12x?x32xf32>, %b1: tensor<1x1x?x32xf32>, %shape: tensor<3xi64>):
    %k = onnx.Constant dense<0.420448214> : tensor<1xf32>
    %1 = "onnx.Mul"(%a0, %a1) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
    %2 = "onnx.Mul"(%b0, %b1) : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>) -> tensor<?x12x?x32xf32>
    %3 = "onnx.Add"(%1, %2) : (tensor<?x12x?x32xf32>, tensor<?x12x?x32xf32>) -> tensor<?x12x?x32xf32>
    %4 = "onnx.Mul"(%3, %k) : (tensor<?x12x?x32xf32>, tensor<1xf32>) -> tensor<?x12x?x32xf32>
    %5 = "onnx.Reshape"(%4, %shape) <{allowzero = 0 : si64}> : (tensor<?x12x?x32xf32>, tensor<3xi64>) -> tensor<?x?x32xf32>
    %6 = "zhigh.Stick"(%5) <{layout = "3DS"}> : (tensor<?x?x32xf32>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
    onnx.Yield %6 : tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  }) {isSub = false, mulScalar = 0.420448214 : f32, reshapeCollapsedCount = 2 : i64, reshapeFirstCollapsedDim = 0 : i64, stickFormat = "3DS"} : (tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>, tensor<?x12x?x32xf32>, tensor<1x1x?x32xf32>, tensor<3xi64>) -> tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>
  return %0 : tensor<?x?x32xf16, #zhigh.layout<{dataLayout = "3DS"}>>

// MAX-LABEL:  func.func @test_mul_add_stick_rotary
// MAX-DAG:      arith.constant dense<-8.57315738E+9> : vector<4xf32>
// MAX-DAG:      arith.constant dense<8.57315738E+9> : vector<4xf32>

// LIMIT-LABEL:  func.func @test_mul_add_stick_rotary
// LIMIT-DAG:      arith.constant dense<-2.14328934E+9> : vector<4xf32>
// LIMIT-DAG:      arith.constant dense<2.14328934E+9> : vector<4xf32>
// LIMIT-NOT:      8.57315738E+9

// NOSAT-LABEL:  func.func @test_mul_add_stick_rotary
// NOSAT-NOT:      E+9 : vector<4xf32>
}
