// RUN: sed 's/^[ \t]*\/\/.*//' %s | (onnx-mlir --EmitONNXBasic --printIR 2>&1 || true) | FileCheck --check-prefix=FAILED %s

// json is generated with:
//
// utils/onnx2json.py zipmap.onnx
//
// where zipmap.onnx is generated with the following python program:
//
// import onnx
// from onnx import helper
//
// def make_map_type_proto(key_type, value_type):
//     map_type_proto = onnx.TypeProto()
//     map_type_proto.map_type.key_type = key_type
//     map_type_proto.map_type.value_type.tensor_type.CopyFrom(value_type.tensor_type)
//     return map_type_proto
//
// def make_map_sequence_value_info(name, key_type, value_type):
//     map_type_proto = make_map_type_proto(key_type, value_type)
//     seq_type_proto = helper.make_sequence_type_proto(map_type_proto)
//     value_info_proto = onnx.ValueInfoProto()
//     value_info_proto.name = name
//     value_info_proto.type.sequence_type.CopyFrom(seq_type_proto.sequence_type)
//     return value_info_proto
//
// input_info = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, (3,))
// node = helper.make_node("ZipMap", ["input"], ["output"], classlabels_int64s=[10, 20, 30])
// value_type = helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ())
// output_info = make_map_sequence_value_info("output", onnx.TensorProto.INT64, value_type)
// graph = helper.make_graph([node], "zipmapper", [input_info], [output_info])
// model = helper.make_model(graph)
// onnx.save(model, "zipmap.onnx")
{
  "irVersion": "8",
  "graph": {
    "node": [
      {
        "input": [
          "input"
        ],
        "output": [
          "output"
        ],
        "opType": "ZipMap",
        "attribute": [
          {
            "name": "classlabels_int64s",
            "ints": [
              "10",
              "20",
              "30"
            ],
            "type": "INTS"
          }
        ]
      }
    ],
    "name": "zipmapper",
    "input": [
      {
        "name": "input",
        "type": {
          "tensorType": {
            "elemType": 1,
            "shape": {
              "dim": [
                {
                  "dimValue": "3"
                }
              ]
            }
          }
        }
      }
    ],
    "output": [
      {
        "name": "output",
        "type": {
          "sequenceType": {
            "elemType": {
              "mapType": {
                "keyType": 7,
                "valueType": {
                  "tensorType": {
                    "elemType": 1,
                    "shape": {}
                  }
                }
              }
            }
          }
        }
      }
    ]
  },
  "opsetImport": [
    {
      "version": "16"
    }
  ]
}
// Sequences of non-tensors are not supported by the ONNX parser in
// FrontendGenImpl::ImportSequenceType in FrontendDialectTransformer.cpp

// FAILED: "expect tensor inside sequence type"
