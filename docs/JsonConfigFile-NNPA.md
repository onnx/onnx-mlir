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

| Key              | Type                       | Description                                                                                       |
| ---------------- | -------------------------- | ------------------------------------------------------------------------------------------------- |
| compile_options  | array of string (optional) | A list of compiler command-line options that are prepended to the existing command-line arguments. |
| nnpa_ops_config  | array of object (optional) | List of operation configurations for device placement and quantization.                           |

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
| node_type      | string (required) | ONNX operator type (e.g., "onnx.Relu", "onnx.*").                       |
| onnx_node_name | string (optional) | Specific ONNX node name (via a regex) to match.                         |
| inputs         | object (optional) | Tensor information constraints for input operands.                       |
| outputs        | object (optional) | Tensor information constraints for output results.                       |

### pattern.rewrite fields

| Field    | Type              | Description                                                             |
| -------- | ----------------- | ----------------------------------------------------------------------- |
| device   | string (optional) | Target device for execution: "cpu", "nnpa", or ""                       |
| quantize | boolean (optional)| Whether to apply quantization (true or false).                          |

- Strings for `node_type` and `onnx_node_name` can be any [ECMAScript regular expressions](https://cplusplus.com/reference/regex/ECMAScript/).

### Tensor Information Matching (inputs/outputs)

The `inputs` and `outputs` fields in the `match` section allow matching operations based on their tensor properties. Each field is an object where keys are tensor indices and values are tensor constraint objects.

**Tensor Indexing:**
- Positive indices: 0-based from the start (0 is first tensor, 1 is second, etc.)
- Negative indices: Count from the end (-1 is last tensor, -2 is second-to-last, -3 is third-to-last, etc.)

#### Tensor constraint object fields

| Field | Type              | Description                                                             |
| ----- | ----------------- | ----------------------------------------------------------------------- |
| rank  | string (optional) | Constraint on tensor rank (e.g., "4", ">2", ">=3").                     |
| type  | string (optional) | Element type (e.g., "f32", "i64").                                      |
| dims  | object (optional) | Dimension constraints where keys are dimension indices (0-based, negative indices count from end: -1 is last dimension, -2 is second-to-last, etc.) and values are constraints. |

#### Constraint Pattern Syntax

Constraint patterns support the following operators:

**Comparison Operators:**
- `"3"` - Exact match (implicit equality): value must equal 3
- `">3"` - Greater than: value must be > 3
- `">=3"` - Greater than or equal: value must be >= 3
- `"<3"` - Less than: value must be < 3
- `"<=3"` - Less than or equal: value must be <= 3
- `"==3"` - Explicit equality: value must equal 3
- `"!=3"` - Not equal: value must not equal 3

**Modulo Operations (for divisibility/alignment checks):**
- `"%32==0"` - Modulo constraint: (value % 32) must equal 0
- `"%64==0"` - Divisibility by 64: (value % 64) must equal 0
- `"%N==R"` - General form: (value % N) must equal R

**Special Values:**
- `"-1"` - Matches dynamic dimensions

**Dimension Indexing:**
- Positive indices: 0-based from the start (0 is first dimension, 1 is second, etc.)
- Negative indices: Count from the end (-1 is last dimension, -2 is second-to-last, -3 is third-to-last, etc.)

**Examples:**
```json
{
  "inputs": {
    "0": {
      "rank": "4",
      "type": "f32",
      "dims": {
        "0": ">=2",
        "1": "3",
        "2": "%32==0",
        "-1": "%64==0"
      }
    }
  }
}
```

This matches operations where:
- The first input has rank 4
- Element type is f32
- Dimension 0 (first dimension) is >= 2
- Dimension 1 (second dimension) equals 3
- Dimension 2 (third dimension) is divisible by 32
- Dimension -1 (last dimension, equivalent to dimension 3 for rank 4) is divisible by 64

### Semantics

- Each object in the `nnpa_ops_config` array specifies configuration for matching ONNX operators:
  - **Device placement** (`"device"` in rewrite):
    - `"device": "cpu"`: the matched ONNX operators run on CPU. 
    - `"device": "nnpa"`: the matched ONNX operators **may** run on NNPA. The compiler will check again if these operators are really suitable for NNPA or not. 
    - `"device": ""`: The compiler will decide on which device the matched ONNX operators will run.
  - **Quantization** (`"quantize"` in rewrite):
    - `"quantize": false`: the matched ONNX operators are not quantized. 
    - `"quantize": true`: the matched ONNX operators **may** be quantized. The compiler will check again if these operators are really suitable for quantization or not.
- An ONNX operator is matched if ALL specified criteria in the `match` section are satisfied:
  - `node_type` must match (required)
  - `onnx_node_name` must match (if specified)
  - `inputs` tensor constraints must match (if specified)
  - `outputs` tensor constraints must match (if specified)
- Once matched, the attributes in the `rewrite` section are applied.
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

## Examples with Tensor Information Matching

8. Match MatMul operations where the first input has rank 2 and dimensions are divisible by 32:
```json
{
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul",
          "inputs": {
            "0": {
              "rank": "2",
              "dims": {
                "0": "%32==0",
                "1": "%32==0"
              }
            }
          }
        },
        "rewrite": {
          "quantize": true
        }
      }
    }
  ]
}
```

9. Match Conv operations with specific input tensor properties (4D tensor with batch size >= 1 and channels divisible by 16):
```json
{
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.Conv",
          "inputs": {
            "0": {
              "rank": "4",
              "type": "f32",
              "dims": {
                "0": ">=1",
                "1": "%16==0"
              }
            }
          }
        },
        "rewrite": {
          "device": "nnpa"
        }
      }
    }
  ]
}
```

10. Match operations with dynamic dimensions in specific positions:
```json
{
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul",
          "inputs": {
            "0": {
              "dims": {
                "0": "-1"
              }
            }
          }
        },
        "rewrite": {
          "device": "cpu"
        }
      }
    }
  ]
}
```

11. Match Conv operations where rank is not 4 (e.g., to handle 3D or 5D convolutions differently):
```json
{
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.Conv",
          "inputs": {
            "0": {
              "rank": "!=4"
            }
          }
        },
        "rewrite": {
          "device": "cpu"
        }
      }
    }
  ]
}
```

12. Match MatMul operations where the last dimension is not 768 (to exclude specific embedding sizes):
```json
{
  "nnpa_ops_config": [
    {
      "pattern": {
        "match": {
          "node_type": "onnx.MatMul",
          "inputs": {
            "0": {
              "dims": {
                "-1": "!=768"
              }
            }
          }
        },
        "rewrite": {
          "quantize": true
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

