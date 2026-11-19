// RUN: onnx-mlir --march=z17 --maccel=NNPA --EmitZHighIR --printIR -opt-report=NNPASaturation %s 2>&1 | FileCheck --check-prefix=MASK %s
// RUN: onnx-mlir --march=z17 --maccel=NNPA --EmitZHighIR --printIR -opt-report=NNPASaturation --nnpa-saturation=limit %s 2>&1 | FileCheck --check-prefix=LIMIT %s
// RUN: onnx-mlir --march=z17 --maccel=NNPA --EmitZHighIR --printIR -opt-report=NNPASaturation --nnpa-disable-saturation %s 2>&1 | FileCheck --check-prefix=DISABLED %s
// RUN: onnx-mlir --march=z17 --maccel=NNPA --EmitZHighIR --printIR %s 2>&1 | FileCheck --check-prefix=NOREPORT %s

// COM: -opt-report=NNPASaturation prints one line with the constant values
// COM: saturated at compile time. %c has 2 values beyond the dlfloat16 range
// COM: (+/-1e10) and 2 within it but beyond the default limit of
// COM: --nnpa-saturation=limit (DLF16_MAX/4, ~2.1e9): +/-3e9. %m has 5
// COM: -FLT_MAX mask values feeding a Softmax, raised to -1e9 by the default
// COM: mode (mask) only. With --nnpa-disable-saturation, the mode is
// COM: "disabled": no mask handling, but constants are still saturated.

func.func @main_graph(%x: tensor<4x8xf32>) -> (tensor<4x8xf32>, tensor<4x8xf32>) {
  %c = onnx.Constant dense<[[1.0e10, 3.0e9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [-1.0e10, -3.0e9, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0]]> : tensor<4x8xf32>
  %a = "onnx.Add"(%x, %c) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %m = onnx.Constant dense<[[0.0, 0.0, 0.0, 0.0, -3.40282347e38, -3.40282347e38, -3.40282347e38, -3.40282347e38], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -3.40282347e38], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]> : tensor<4x8xf32>
  %s0 = "onnx.Add"(%x, %m) : (tensor<4x8xf32>, tensor<4x8xf32>) -> tensor<4x8xf32>
  %s = "onnx.Softmax"(%s0) {axis = -1 : si64} : (tensor<4x8xf32>) -> tensor<4x8xf32>
  onnx.Return %a, %s : tensor<4x8xf32>, tensor<4x8xf32>
}

// MASK: ==SATURATION-REPORT==, dlf16, mode, mask, masked, 5, outside-range, 2, outside-limit, 8.57316e+09, 0, total, {{[0-9]+}}

// LIMIT: ==SATURATION-REPORT==, dlf16, mode, limit, masked, 0, outside-range, 2, outside-limit, 2.14329e+09, 2, total, {{[0-9]+}}

// DISABLED: ==SATURATION-REPORT==, dlf16, mode, disabled, masked, 0, outside-range, 2, outside-limit, 8.57316e+09, 0, total, {{[0-9]+}}

// NOREPORT-NOT: ==SATURATION-REPORT==
