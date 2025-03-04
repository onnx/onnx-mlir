// RUN: onnx-mlir-opt -legalize-quark-quantized-ops --mlir-print-debuginfo --split-input-file %s | FileCheck %s

func.func @onnx_add_static(%arg0: tensor<1x256x128xbf16>, %arg1: tensor<1xbf16>) -> tensor<1x256x128xbf16> {
    %0 = "onnx.Cast"(%arg0) { saturate = 1 : si64, to = f32} : (tensor<1x256x128xbf16>) -> tensor<1x256x128xf32> loc("Cast_Inpu1")
    %1 = "onnx.Cast"(%arg1) { saturate = 1 : si64, to = f32} : (tensor<1xbf16>) -> tensor<1xf32> loc("Cast_Inpu2")
    %2 = "onnx.Add"(%0, %1) : (tensor<1x256x128xf32>, tensor<1xf32>) -> tensor<1x256x128xf32> loc("Add")
    %3 = "onnx.Cast"(%2) { saturate = 1 : si64, to = bf16} : (tensor<1x256x128xf32>) -> tensor<1x256x128xbf16> loc("Cast_Output")
    return %3 : tensor<1x256x128xbf16> loc("Return")
}
// CHECK-LABEL:   func.func @onnx_add_static
// CHECK:           %[[VAL_2:.*]] = "onnx.Add"(%arg0, %arg1) : (tensor<1x256x128xbf16>, tensor<1xbf16>) -> tensor<1x256x128xbf16> loc(#[[FUSED:.*]])
// CHECK:           return %[[VAL_2]] : tensor<1x256x128xbf16>
// CHECK:         }

// CHECK-DAG: #[[CAST1:.*]] = loc("Cast_Inpu1")
// CHECK-DAG: #[[CAST2:.*]] = loc("Cast_Inpu2")
// CHECK-DAG: #[[OP:.*]] = loc("Add")
// CHECK-DAG: #[[CASTOUT:.*]] = loc("Cast_Output")
// CHECK-DAG: [[FUSED]] = loc(fused[#[[OP]], #[[CASTOUT:.*]], #[[CAST1]], #[[CAST2]]])
