// RUN: cfg_file=$(dirname %s)/save-cfg.json && onnx-mlir-opt --device-placement=save-config-file=$cfg_file --nnpa-placement-heuristic=QualifyingOps  --march=z16 --maccel=NNPA --split-input-file %s && cat $cfg_file | FileCheck %s && rm $cfg_file

// -----

func.func @test_save_config_file(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "Relu_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "onnx.Relu"(%0) {device="cpu", onnx_node_name = "Relu_1"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "onnx.Relu"(%1) {onnx_node_name = "Relu_2"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "onnx.Sigmoid"(%2) {device="nnpa", onnx_node_name = "Sigmoid_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  onnx.Return %3 : tensor<?x?x?xf32>

// CHECK-LABEL test_save_config_file
// CHECK: {
// CHECK:   "device_placement": [
// CHECK:     {
// CHECK:       "device": "nnpa",
// CHECK:       "node_type": "onnx.Relu",
// CHECK:       "onnx_node_name": "Relu_0"
// CHECK:     },
// CHECK:     {
// CHECK:       "device": "cpu",
// CHECK:       "node_type": "onnx.Relu",
// CHECK:       "onnx_node_name": "Relu_1"
// CHECK:     },
// CHECK:     {
// CHECK:       "device": "nnpa",
// CHECK:       "node_type": "onnx.Relu",
// CHECK:       "onnx_node_name": "Relu_2"
// CHECK:     },
// CHECK:     {
// CHECK:       "device": "nnpa",
// CHECK:       "node_type": "onnx.Sigmoid",
// CHECK:       "onnx_node_name": "Sigmoid_0"
// CHECK:     }
// CHECK:   ]
// CHECK: }
}

