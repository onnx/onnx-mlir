// RUN: sed 's/^[ \t]*\/\/.*//' %s | (onnx-mlir --EmitONNXBasic --printIR 2>&1 || true) | FileCheck --check-prefix=FAILED %s

// json is generated with:
// utils/onnx2json.py third_party/onnx/onnx/backend/test/data/node/test_gridsample/model.onnx
{
  "irVersion": "8",
  "producerName": "backend-test",
  "graph": {
    "node": [
      {
        "input": [
          "X",
          "Grid"
        ],
        "output": [
          "Y"
        ],
        "opType": "GridSample",
        "attribute": [
          {
            "name": "align_corners",
            "i": "0",
            "type": "INT"
          },
          {
            "name": "mode",
            "s": "YmlsaW5lYXI=",
            "type": "STRING"
          },
          {
            "name": "padding_mode",
            "s": "emVyb3M=",
            "type": "STRING"
          }
        ]
      }
    ],
    "name": "test_gridsample",
    "input": [
      {
        "name": "X",
        "type": {
          "tensorType": {
            "elemType": 1,
            "shape": {
              "dim": [
                {
                  "dimValue": "1"
                },
                {
                  "dimValue": "1"
                },
                {
                  "dimValue": "4"
                },
                {
                  "dimValue": "4"
                }
              ]
            }
          }
        }
      },
      {
        "name": "Grid",
        "type": {
          "tensorType": {
            "elemType": 1,
            "shape": {
              "dim": [
                {
                  "dimValue": "1"
                },
                {
                  "dimValue": "6"
                },
                {
                  "dimValue": "6"
                },
                {
                  "dimValue": "2"
                }
              ]
            }
          }
        }
      }
    ],
    "output": [
      {
        "name": "Y",
        "type": {
          "tensorType": {
            "elemType": 1,
            "shape": {
              "dim": [
                {
                  "dimValue": "1"
                },
                {
                  "dimValue": "1"
                },
                {
                  "dimValue": "6"
                },
                {
                  "dimValue": "6"
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
      "version": "16"
    }
  ]
}
// GridSample was added in ONNX v17 and not yet added to onnx-mlir

// FAILED: GridSample this Op is not supported by onnx-mlir's onnx dialect
