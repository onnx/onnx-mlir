<!--- SPDX-License-Identifier: Apache-2.0 -->

The compiler provides two options to allow users control how ONNX operators run on NNPA.
- `--config-file` (or `-config-file`) to load configuration settings from a JSON file.
- `--save-config-file` to save configuration settings to a JSON file.

It is common to use ONNX node names to match ONNX operators. If you're unsure about the node names in your model, you can use the `--save-config-file` option to generate a configuration file as a starting point.

Alternatively, you can open the ONNX model using a visualizer like Netron to inspect the node names. Note that the actual node names used by the compiler may differ slightly from those shown in Netron due to compiler optimizations, though they are usually the same.

That said, using `--save-config-file` is the recommended approach.

By using a JSON file, users can currently control two following features:
- device placement: to decide which ONNX operators run on CPU or NNPA.
- quantization: to decide which ONNX operators are quantized to utilize i8 computation on NNPA.

# JSON schema description
## Top-level keys

| Key              | Type                       | Description                                                             |
| ---------------- | -------------------------- | ----------------------------------------------------------------------- |
| compile_options  | array of string (optional) | List of compiler command-line options.                                  |
| nnpa_ops_config  | array of object (optional) | List of operation configurations for device placement and quantization. |

## compile_options key
- See [JSON Config File](JsonConfigFile.md)

## nnpa_ops_config[] object fields

Each object in the `nnpa_ops_config` array has the following structure:

| Field   | Type   | Description                                    |
| ------- | ------ | ---------------------------------------------- |
| pattern | object | Contains `match` and `rewrite` sub-objects.    |

### pattern.match fields

| Field          | Type              | Description                                                             |
| -------------- | ----------------- | ----------------------------------------------------------------------- |
| node_type      | string            | ONNX operator type (e.g., "onnx.Relu", "onnx.*").                       |
| onnx_node_name | string (optional) | Specific ONNX node name (via a regex) to match.                         |

### pattern.rewrite fields

| Field    | Type              | Description                                                             |
| -------- | ----------------- | ----------------------------------------------------------------------- |
| device   | string (optional) | Target device for execution: "cpu", "nnpa", or ""                       |
| quantize | boolean (optional)| Whether to apply quantization (true or false).                          |

- Fields have the same names as ONNX operator's attributes.
- Strings for `node _type` and `onnx_node_name` can be any [ECMAScript regular expressions](https://cplusplus.com/reference/regex/ECMAScript/).

- Strings for `node_type` and `onnx_node_name` can be any [ECMAScript regular expressions](https://cplusplus.com/reference/regex/ECMAScript/).

### Semantics

- Each object in the `nnpa_ops_config` array specifies configuration for matching ONNX operators:
  - **Device placement** (`"device"` in rewrite):
    - `"device": "cpu"`: the matched ONNX operators run on CPU. 
    - `"device": "nnpa"`: the matched ONNX operators **may** run on NNPA. The compiler will check again if these operators are really suitable for NNPA or not. 
    - `"device": ""`: The compiler will decide on which device the matched ONNX operators will run.
  - **Quantization** (`"quantize"` in rewrite):
    - `"quantize": false`: the matched ONNX operators are not quantized. 
    - `"quantize": true`: the matched ONNX operators **may** be quantized. The compiler will check again if these operators are really suitable for quantization or not.
- An ONNX operator is matched if both `node_type` AND `onnx_node_name` (if specified) in the `match` section are matched. Once matched, the attributes in the `rewrite` section are applied.
- The list is evaluated in sequence, with earlier items having precedence. If an ONNX operator matches a pattern, it does not match against the remaining patterns in the list. 

# Examples
- Let's use the following input model as an example:
```mlir
func.func @matmul(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
  %0 = "onnx.MatMul"(%arg0, %arg0) {onnx_node_name = "MatMul_0"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %1 = "onnx.MatMul"(%arg0, %0) {onnx_node_name = "MatMul_1"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %2 = "onnx.MatMul"(%0, %1) {onnx_node_name = "MatMul_2"} : (tensor<?x?xf32>, tensor<?x?xf32>) -> tensor<?x?xf32>
  %3 = "onnx.Sigmoid"(%2) {onnx_node_name = "Sigmoid_0"} : (tensor<?x?xf32>) -> tensor<?x?xf32>
  onnx.Return %3 : tensor<?x?xf32>
  }
```

Below are JSON files for different situations.

1. Schedule all operators to run on CPU
```json
{
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.*",
          "onnx_node_name": ".*"
        },
        "rewrite": {
          "device": "cpu"
        }
      }
    }
  ]
}
```

2. Schedule all MatMul operators to run on CPU:
```json
{
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul",
          "onnx_node_name": ".*"
        },
        "rewrite": {
          "device": "cpu"
        }
      }
    }
  ]
}
```

3.  Schedule operators using `onnx_node_name`: here we use regex to chose only `MatMul_1` and `MatMul_2` operators, exact match is used for `onnx.Sigmoid`.
```json
{
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul",
          "onnx_node_name": "MatMul_(1|2)"
        },
        "rewrite": {
          "device": "cpu"
        }
      }
    },
    {
      "pattern": {
        "match": {
          "node_type": "onnx.Sigmoid",
          "onnx_node_name": "Sigmoid_0"
        },
        "rewrite": {
          "device": "nnpa"
        }
      }
    }
  ]
}
```

4. `onnx.MatMul` does not match because there is no operator with `node_type = MatMul`, so only `onnx.Sigmoid` is set device.
```json
{
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "MatMul",
          "onnx_node_name": "MatMul_(1|2)"
        },
        "rewrite": {
          "device": "cpu"
        }
      }
    },
    {
      "pattern": {
        "match": {
          "node_type": "onnx.Sigmoid",
          "onnx_node_name": "Sigmoid_0"
        },
        "rewrite": {
          "device": "cpu"
        }
      }
    }
  ]
}
```

5. We have two overlapping patterns both matching on `onnx.MatMul`. In this case, only the first matched pattern will apply. Thus, `MatMul_0` and `MatMul_1` have device "cpu" by matching the first pattern, `MatMul_2` operator has device "cpu" by matching the third pattern.
```json
{
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul",
          "onnx_node_name": "MatMul_(0|1)"
        },
        "rewrite": {
          "device": "cpu"
        }
      }
    },
    {
      "pattern": {
        "match": {
          "node_type": "onnx.Sigmoid",
          "onnx_node_name": "Sigmoid_0"
        },
        "rewrite": {
          "device": "nnpa"
        }
      }
    },
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul",
          "onnx_node_name": "MatMul_(1|2)"
        },
        "rewrite": {
          "device": "cpu"
        }
      }
    }
  ]
}
```

6. We want to quantize `MatMul_1` only. We set `quantize` to true for `MatMul_1` and explicitly set `quantize` to false for all remaining MatMul ops, which is to ensure that the compiler does not quantize for the remaining MatMul ops.
```json
{
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul",
          "onnx_node_name": "MatMul_1"
        },
        "rewrite": {
          "quantize": true
        }
      }
    },
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul"
        },
        "rewrite": {
          "quantize": false
        }
      }
    }
  ]
}
```

7. We want `MatMul_0` to run on CPU and to quantize `MatMul_1` only. We set `quantize` to true for `MatMul_1` and explicitly set `quantize` to false for all remaining MatMul ops, which is to ensure that the compiler does not quantize for the remaining MatMul ops.
```json
{
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul",
          "onnx_node_name": "MatMul_0"
        },
        "rewrite": {
          "device": "cpu"
        }
      }
    },
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul",
          "onnx_node_name": "MatMul_1"
        },
        "rewrite": {
          "quantize": true
        }
      }
    },
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul"
        },
        "rewrite": {
          "quantize": false
        }
      }
    }
  ]
}
```

## A example JSON file for transformers models

This JSON configuration file is to quantize four MatMul operators in the self-attention layer, two MatMul operators in the linear layer and the MatMul operators in the LM head.
```json
{
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul",
          "onnx_node_name": "^/model/layers\\.[0-9]+/self_attn/(q|k|v)_proj/MatMul"
        },
        "rewrite": {
          "quantize": true
        }
      }
    },
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul",
          "onnx_node_name": "^/model/layers\\.[0-9]+(.*)/self_attn/o_proj/MatMul.*"
        },
        "rewrite": {
          "quantize": true
        }
      }
    },
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul",
          "onnx_node_name": "^/model/layers\\.[0-9]+/mlp/((gate)|(up))_proj/MatMul"
        },
        "rewrite": {
          "quantize": true
        }
      }
    },
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul",
          "onnx_node_name": ".*/lm_head/MatMul.*"
        },
        "rewrite": {
          "quantize": true
        }
      }
    },
    {
      "_comment": "do not quantize the remaining matmuls",
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul"
        },
        "rewrite": {
          "quantize": false
        }
      }
    }
  ]
}
```

# Related Documentation
- [JSON Config File](JsonConfigFile.md)

