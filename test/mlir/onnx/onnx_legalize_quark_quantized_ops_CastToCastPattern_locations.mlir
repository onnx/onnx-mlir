// RUN: onnx-mlir-opt -legalize-quark-quantized-ops --mlir-print-debuginfo --split-input-file %s | FileCheck %s

func.func @test_cast_cast_canonicalization(%arg0: tensor<1x1x256x256xbf16>, %arg1: tensor<bf16>, %arg2: tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xbf16> {
    %0 = "onnx.Cast"(%arg0) {
      onnx_node_name = "/bert/Sub_onnx.Cast_39",
      saturate = 1 : si64,
      to = f32} : (tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xf32> loc("cast1")
    %1 = "onnx.Cast"(%0) {
      onnx_node_name = "/bert/Cast_1",
      saturate = 1 : si64,
      to = i1} : (tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xi1> loc("cast2")
    %2 = "onnx.Where"(%1, %arg1, %arg2) {onnx_node_name = "/bert/Where_1"} : (tensor<1x1x256x256xi1>, tensor<bf16>, tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xbf16>  loc("Where")
    "onnx.Return"(%2) : (tensor<1x1x256x256xbf16>) -> ()
}
// CHECK-LABEL:   func.func @test_cast_cast_canonicalization
// CHECK:           %[[VAL_3:.*]] = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = i1} : (tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xi1> loc(#[[FUSE_CAST1_CAST2:.*]])
// CHECK:           %[[VAL_4:.*]] = "onnx.Where"(%[[VAL_3]], %arg1, %arg2) {onnx_node_name = "/bert/Where_1"} : (tensor<1x1x256x256xi1>, tensor<bf16>, tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xbf16> loc(#[[WHERE:.*]])
// CHECK:           onnx.Return %[[VAL_4]] : tensor<1x1x256x256xbf16>
// CHECK:         }

// CHECK-DAG: #[[CAST1:.*]] = loc("cast1")
// CHECK-DAG: #[[CAST2:.*]] = loc("cast2")
// CHECK-DAG: #[[WHERE]] = loc("Where")
// CHECK-DAG: [[FUSE_CAST1_CAST2]] = loc(fused[#[[CAST1]], #[[CAST2]]])

// -----

func.func @test_cast_cast_canonicalization_multi_uses(%arg0: tensor<1x1x256x256xbf16>, %arg1: tensor<bf16>, %arg2: tensor<1x1x256x256xbf16>) -> (tensor<1x1x256x256xf32>, tensor<1x1x256x256xbf16>) {
    %0 = "onnx.Cast"(%arg0) {
      onnx_node_name = "/bert/Sub_onnx.Cast_39",
      saturate = 1 : si64,
      to = f32} : (tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xf32> loc("cast1")
    %1 = "onnx.Cast"(%0) {
      onnx_node_name = "/bert/Cast_1",
      saturate = 1 : si64,
      to = i1} : (tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xi1> loc("cast2")
    %2 = "onnx.Where"(%1, %arg1, %arg2) {onnx_node_name = "/bert/Where_1"} : (tensor<1x1x256x256xi1>, tensor<bf16>, tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xbf16>  loc("Where")
    "onnx.Return"(%0, %2) : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xbf16>) -> ()
}
// CHECK-LABEL:   func.func @test_cast_cast_canonicalization_multi_uses(
// CHECK:           %[[VAL_3:.*]] = "onnx.Cast"(%arg0) {onnx_node_name = "/bert/Sub_onnx.Cast_39", saturate = 1 : si64, to = f32} : (tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xf32>
// CHECK:           %[[VAL_4:.*]] = "onnx.Cast"(%[[VAL_3]]) {onnx_node_name = "/bert/Cast_1", saturate = 1 : si64, to = i1} : (tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xi1>
// CHECK:           %[[VAL_5:.*]] = "onnx.Where"(%[[VAL_4]], %arg1, %arg2) {onnx_node_name = "/bert/Where_1"} : (tensor<1x1x256x256xi1>, tensor<bf16>, tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xbf16>
// CHECK:           onnx.Return %[[VAL_3]], %[[VAL_5]] : tensor<1x1x256x256xf32>, tensor<1x1x256x256xbf16>
// CHECK:         }

// -----

func.func @test_no_cast_cast_canonicalization(%arg0: tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xbf16> {
    %0 = "onnx.Cast"(%arg0) {
      onnx_node_name = "/bert/Sub_onnx.Cast_39",
      saturate = 1 : si64,
      to = f32} : (tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xf32> loc("cast2")
    %1 = "onnx.Cast"(%0) {
      onnx_node_name = "/bert/Cast_1",
      saturate = 1 : si64,
      to = bf16} : (tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xbf16> loc("cast2")
    %2 = "onnx.Add"(%1, %arg0) {onnx_node_name = "/bert/Where_1"} : (tensor<1x1x256x256xbf16>, tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xbf16> loc("add")
    "onnx.Return"(%2) : (tensor<1x1x256x256xbf16>) -> ()
}
// CHECK-LABEL:   func.func @test_no_cast_cast_canonicalization
// CHECK:           %[[VAL_1:.*]] = "onnx.Cast"(%arg0) {saturate = 1 : si64, to = bf16} : (tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xbf16>
// CHECK:           %[[VAL_2:.*]] = "onnx.Add"(%[[VAL_1]], %arg0) {onnx_node_name = "/bert/Where_1"} : (tensor<1x1x256x256xbf16>, tensor<1x1x256x256xbf16>) -> tensor<1x1x256x256xbf16>
// CHECK:           onnx.Return %[[VAL_2]] : tensor<1x1x256x256xbf16>
// CHECK:         }

