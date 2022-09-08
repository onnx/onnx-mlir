// RUN: sed 's/^[ \t]*\/\/.*//' %s | onnx-mlir --EmitONNXBasic --useOnnxModelTypes --printIR | FileCheck %s

// --useOnnxModelTypes is needed, otherwise onnx-mlir hits
// llvm_unreachable("unexpected type") in FrontendGenImpl::ImportType()
// in FrontendDialectTransformer

// json is generated with:
// utils/onnx2json.py third_party/onnx/onnx/backend/test/data/node/test_range_float_type_positive_delta_expanded/model.onnx
{
  "irVersion": "6",
  "producerName": "backend-test",
  "graph": {
    "node": [
      {
        "input": [
          "limit",
          "start"
        ],
        "output": [
          "Range_test_range_float_type_positive_delta_expanded_functionsub_result"
        ],
        "opType": "Sub"
      },
      {
        "input": [
          "Range_test_range_float_type_positive_delta_expanded_functionsub_result"
        ],
        "output": [
          "Range_test_range_float_type_positive_delta_expanded_functionsub_result_casted"
        ],
        "opType": "Cast",
        "attribute": [
          {
            "name": "to",
            "i": "1",
            "type": "INT"
          }
        ]
      },
      {
        "input": [
          "delta"
        ],
        "output": [
          "Range_test_range_float_type_positive_delta_expanded_functiondelta_casted"
        ],
        "opType": "Cast",
        "attribute": [
          {
            "name": "to",
            "i": "1",
            "type": "INT"
          }
        ]
      },
      {
        "input": [
          "Range_test_range_float_type_positive_delta_expanded_functionsub_result_casted",
          "Range_test_range_float_type_positive_delta_expanded_functiondelta_casted"
        ],
        "output": [
          "Range_test_range_float_type_positive_delta_expanded_functiondiv_result"
        ],
        "opType": "Div"
      },
      {
        "input": [
          "Range_test_range_float_type_positive_delta_expanded_functiondiv_result"
        ],
        "output": [
          "Range_test_range_float_type_positive_delta_expanded_functionceil_result"
        ],
        "opType": "Ceil"
      },
      {
        "input": [
          "Range_test_range_float_type_positive_delta_expanded_functionceil_result"
        ],
        "output": [
          "Range_test_range_float_type_positive_delta_expanded_functionceil_result_relu"
        ],
        "opType": "Relu"
      },
      {
        "input": [
          "Range_test_range_float_type_positive_delta_expanded_functionceil_result_relu"
        ],
        "output": [
          "Range_test_range_float_type_positive_delta_expanded_functionceil_result_relu_int"
        ],
        "opType": "Cast",
        "attribute": [
          {
            "name": "to",
            "i": "7",
            "type": "INT"
          }
        ]
      },
      {
        "input": [
          "Range_test_range_float_type_positive_delta_expanded_functionceil_result_relu"
        ],
        "output": [
          "Range_test_range_float_type_positive_delta_expanded_functionceil_result_relu_bool"
        ],
        "opType": "Cast",
        "attribute": [
          {
            "name": "to",
            "i": "9",
            "type": "INT"
          }
        ]
      },
      {
        "input": [
          "Range_test_range_float_type_positive_delta_expanded_functionceil_result_relu_int",
          "Range_test_range_float_type_positive_delta_expanded_functionceil_result_relu_bool",
          "start"
        ],
        "output": [
          "Range_test_range_float_type_positive_delta_expanded_functionvariadic_output",
          "output"
        ],
        "opType": "Loop",
        "attribute": [
          {
            "name": "body",
            "g": {
              "node": [
                {
                  "input": [
                    "cond"
                  ],
                  "output": [
                    "cond_out"
                  ],
                  "opType": "Identity"
                },
                {
                  "input": [
                    "prev",
                    "delta"
                  ],
                  "output": [
                    "current"
                  ],
                  "opType": "Add"
                },
                {
                  "input": [
                    "prev"
                  ],
                  "output": [
                    "range"
                  ],
                  "opType": "Identity"
                }
              ],
              "name": "loop_body_attribute",
              "input": [
                {
                  "name": "i",
                  "type": {
                    "tensorType": {
                      "elemType": 7,
                      "shape": {}
                    }
                  }
                },
                {
                  "name": "cond",
                  "type": {
                    "tensorType": {
                      "elemType": 9,
                      "shape": {}
                    }
                  }
                },
                {
                  "name": "prev"
                }
              ],
              "output": [
                {
                  "name": "cond_out"
                },
                {
                  "name": "current"
                },
                {
                  "name": "range"
                }
              ]
            },
            "type": "GRAPH"
          }
        ]
      }
    ],
    "name": "test_range_float_type_positive_delta_expanded",
    "input": [
      {
        "name": "start",
        "type": {
          "tensorType": {
            "elemType": 1,
            "shape": {}
          }
        }
      },
      {
        "name": "limit",
        "type": {
          "tensorType": {
            "elemType": 1,
            "shape": {}
          }
        }
      },
      {
        "name": "delta",
        "type": {
          "tensorType": {
            "elemType": 1,
            "shape": {}
          }
        }
      }
    ],
    "output": [
      {
        "name": "output",
        "type": {
          "tensorType": {
            "elemType": 1,
            "shape": {
              "dim": [
                {
                  "dimValue": "2"
                }
              ]
            }
          }
        }
      }
    ]
  },
  "opsetImport": [
    {
      "version": "11"
    }
  ]
}
// CHECK-LABEL:  func.func @main_graph
// CHECK-SAME:   ([[PARAM_0_:%.+]]: tensor<f32>, [[PARAM_1_:%.+]]: tensor<f32>, [[PARAM_2_:%.+]]: tensor<f32>) -> tensor<2xf32> {{.*}} {
// CHECK:           [[VAR_0_:%.+]] = "onnx.Sub"([[PARAM_1_]], [[PARAM_0_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_1_:%.+]] = "onnx.Cast"([[VAR_0_]]) {to = f32} : (tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_2_:%.+]] = "onnx.Cast"([[PARAM_2_]]) {to = f32} : (tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_3_:%.+]] = "onnx.Div"([[VAR_1_]], [[VAR_2_]]) : (tensor<f32>, tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_4_:%.+]] = "onnx.Ceil"([[VAR_3_]]) : (tensor<f32>) -> tensor<f32>
// CHECK:           [[VAR_5_:%.+]] = "onnx.Relu"([[VAR_4_]]) : (tensor<f32>) -> tensor<f32>
// CHECK-DAG:       [[VAR_6_:%.+]] = "onnx.Cast"([[VAR_5_]]) {to = i64} : (tensor<f32>) -> tensor<i64>
// CHECK-DAG:       [[VAR_7_:%.+]] = "onnx.Cast"([[VAR_5_]]) {to = i1} : (tensor<f32>) -> tensor<i1>
// CHECK:           [[VAR_8_:%.+]]:2 = "onnx.Loop"([[VAR_6_]], [[VAR_7_]], [[PARAM_0_]]) ({
// CHECK:           ^bb0([[PARAM_3_:%.+]]: tensor<i64>, [[PARAM_4_:%.+]]: tensor<i1>, [[PARAM_5_:%.+]]: tensor<*xf32>):
// CHECK-DAG:         [[VAR_9_:%.+]] = "onnx.Identity"([[PARAM_4_]]) : (tensor<i1>) -> tensor<i1>
// CHECK-DAG:         [[VAR_10_:%.+]] = "onnx.Add"([[PARAM_5_]], [[PARAM_2_]]) : (tensor<*xf32>, tensor<f32>) -> tensor<*xf32>
// CHECK-DAG:         [[VAR_11_:%.+]] = "onnx.Identity"([[PARAM_5_]]) : (tensor<*xf32>) -> tensor<*xf32>
// CHECK:             onnx.Return [[VAR_9_]], [[VAR_10_]], [[VAR_11_]] : tensor<i1>, tensor<*xf32>, tensor<*xf32>
// CHECK:           }) {{.*}} : (tensor<i64>, tensor<i1>, tensor<f32>) -> (tensor<i1>, tensor<2xf32>)
// CHECK:           return [[VAR_8_]]#1 : tensor<2xf32>
// CHECK:         }
