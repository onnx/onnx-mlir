// RUN: save_cfg_file=$(dirname %s)/save-cfg.json && load_cfg_file=$(dirname %s)/cfg.json && onnx-mlir-opt --nnpa-quant-ops-selection=save-config-file="$save_cfg_file load-config-file=$load_cfg_file" --march=z17 --maccel=NNPA --split-input-file %s && cat $save_cfg_file | FileCheck %s && rm $save_cfg_file

func.func @test_save_config_file(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg0) {onnx_node_name = "MatMul_0"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "onnx.MatMul"(%arg0, %0) {onnx_node_name = "MatMul_1"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "onnx.MatMul"(%0, %1) {onnx_node_name = "MatMul_2"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "onnx.Sigmoid"(%2) {onnx_node_name = "Sigmoid_0"} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %3 : tensor<?x?xf32>

// CHECK-LABEL test_save_config_file
// CHECK: {
// CHECK:   "nnpa_ops_config": [
// CHECK:     {
// CHECK:       "pattern": {
// CHECK:         "match": {
// CHECK:           "node_type": "onnx.MatMul",
// CHECK:           "onnx_node_name": "MatMul_0"
// CHECK:         },
// CHECK:         "rewrite": {
// CHECK:           "quantize": false
// CHECK:         }
// CHECK:       }
// CHECK:     },
// CHECK:     {
// CHECK:       "pattern": {
// CHECK:         "match": {
// CHECK:           "node_type": "onnx.MatMul",
// CHECK:           "onnx_node_name": "MatMul_1"
// CHECK:         },
// CHECK:         "rewrite": {
// CHECK:           "quantize": true
// CHECK:         }
// CHECK:       }
// CHECK:     },
// CHECK:     {
// CHECK:       "pattern": {
// CHECK:         "match": {
// CHECK:           "node_type": "onnx.MatMul",
// CHECK:           "onnx_node_name": "MatMul_2"
// CHECK:         },
// CHECK:         "rewrite": {
// CHECK:           "quantize": false
// CHECK:         }
// CHECK:       }
// CHECK:     }
// CHECK:   ]
// CHECK: }
}

