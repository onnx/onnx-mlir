<!--- SPDX-License-Identifier: Apache-2.0 -->

The compiler provides two options to allow users control how ONNX operators run on NNPA.
- `--nnpa-load-config-file` to load configuration settings for NNPA from a JSON file.
- `--nnpa-save-config-file` to save configuration settings for NNPA to a JSON file.

It is common to use ONNX node names to match ONNX operators. If you're unsure about the node names in your model, you can use the `--nnpa-save-config-file` option to generate a configuration file as a starting point.

Alternatively, you can open the ONNX model using a visualizer like Netron to inspect the node names. Note that the actual node names used by the compiler may differ slightly from those shown in Netron due to compiler optimizations, though they are usually the same.

That said, using `--nnpa-save-config-file` is the recommended approach.

By using a JSON file, users can currenlty control two following features:
- device placement: to decide which ONNX operators run on CPU or NNPA.
- quantization: to decide which ONNX operators are quantized to utilize i8 computation on NNPA.

# JSON schema description
## Top-level keys

| Key              | Type            | Description                                              |
| ---------------- | --------------- | -------------------------------------------------------- |
| device_placement | array of object | List of device assignments for specific ONNX node types. |
| quantization     | array of object | List of quantization settings for ONNX node types.       |

## device_placement[] object fields

| Field          | Type              | Description                                                             |
| -------------- | ----------------- | ---------------------------------------------------------               |
| device         | string            | Target device for execution: "cpu", "nnpa", or ""                       |
| node_type      | string            | ONNX operator type (e.g., "onnx.Relu", "onnx.*").                       |
| onnx_node_name | string (optional) | Specific ONNX node name (via a regex) to apply the device placement to. |

- Fields have the same names as ONNX operator's attributes.
- Strings for `node _type` and `onnx_node_name` can be any [ECMAScript regular expressions](https://cplusplus.com/reference/regex/ECMAScript/).

### Semantics

- Each object in this `device_placement` list specifies where (on which device) specific ONNX operators should be executed:
  - `"device": "cpu"`: the matched ONNX operators run on CPU. 
  - `"device": "nnpa"`: the matched ONNX operators **may** run on NNPA. The compiler will check again if these operators are really suitable for NNPA or not. 
  - `"device": ""`: The compiler will decide on which device the matched ONNX operators will run. 
- An ONNX operator is matched if both `node_type` AND `onnx_node_name` are matched. Once an ONNX operator is matched, its attribute `device` is updated using the JSON value.
- The list is evaluated in sequence, with earlier items having precedence. If an ONNX operator matches an object, it does not matches against the remaining objects in the list. 

## quantization[] object fields

| Field          | Type              | Description                                                     |
| -------------  | -------------     | --------------------------------------------------------------  |
| quantize       | boolean           | Whether to apply quantization (true or false).                  |
| node_type      | string            | ONNX operator type (e.g., "onnx.Relu", "onnx.*").               |
| onnx_node_name | string (optional) | Specific ONNX node name (via a regex) to apply quantization to. |

- Fields have the same names as ONNX operator's attributes.
- Strings for `node _type` and `onnx_node_name` can be any [ECMAScript regular expressions](https://cplusplus.com/reference/regex/ECMAScript/).
 
### Semantics

- Each object in this `quantization` list specifies whether ONNX operators should be quantized or not:
  - `"quantize": false`: the matched ONNX operators are not quantized. 
  - `"quantize": true`: the matched ONNX operators **may** be quantized. The compiler will check again if these operators are really suitable for quantization or not. 
- An ONNX operator is matched if both `node_type` AND `onnx_node_name` are matched. Once an ONNX operator is matched, its attribute `quantize` is updated using the JSON value.
- The list is evaluated in sequence, with earlier items having precedence. If an ONNX operator matches an object, it does not matches against the remaining objects in the list. 

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
  "device_placement": [
    {
      "device": "cpu",
      "node_type": "onnx.*",
      "onnx_node_name": ".*"
    }
  ]
}
```

2. Schedule all MatMul operators to run on CPU:
```json
{
  "device_placement": [
    {
      "device": "cpu",
      "node_type": "onnx.MatMul",
      "onnx_node_name": ".*"
    }
  ]
}
```

3.  Schedule operators using `onnx_node_name`: here we use regex to chose only `MatMul_1` and `MatMul_2` operators, exact match is used for `onnx.Sigmoid`.
```json
{
  "device_placement": [
    {
      "device": "cpu",
      "node_type": "onnx.MatMul",
      "onnx_node_name": "MatMul_(1|2)"
    },
    {
      "device": "nnpa",
      "node_type": "onnx.Sigmoid",
      "onnx_node_name": "Sigmoid_0"
    }
  ]
}
```

4. `onnx.MatMul` does not match because there is no operator with `node_type = MatMul`, so only `onnx.Sigmoid` is set device.
```json
{
  "device_placement": [
    {
      "device": "cpu",
      "node_type": "MatMul",
      "onnx_node_name": "MatMul_(1|2)"
    },
    {
      "device": "cpu",
      "node_type": "onnx.Sigmoid",
      "onnx_node_name": "Sigmoid_0"
    }
  ]
}
```

5. We have two overlapping objects both matching on `onnx.MatMul`. In this case, only the first matched object will set device. Thus, `MatMul_0` and `MatMul_1` have device "cpu" by matching the first object, `MatMul_2` operator has device "cpu" by matching the third object.
```json
{
  "device_placement": [
    {
      "device": "cpu",
      "node_type": "onnx.MatMul",
      "onnx_node_name": "MatMul_(0|1)"
    },
    {
      "device": "nnpa",
      "node_type": "onnx.Sigmoid",
      "onnx_node_name": "Sigmoid_0"
    },
    {
      "device": "cpu",
      "node_type": "onnx.MatMul",
      "onnx_node_name": "MatMul_(1|2)"
    }
  ]
}
```

6. We want to quantize `MatMul_1` only. We set `quantize` to true for `MatMul_1` and explicitly set `quantize` to false for all remaining MatMul ops, which is to ensure that the compiler does not quantize for the remaining MatMul ops.
```json
{
  "quantization": [
    {
      "quantize": true,
      "node_type": "onnx.MatMul",
      "onnx_node_name": "MatMul_1"
    },
    {
      "quantize": false,
      "node_type": "onnx.MatMul"
    }
  ]
}
```

7. We want `MatMul_0` to run on CPU and to quantize `MatMul_1` only. We set `quantize` to true for `MatMul_1` and explicitly set `quantize` to false for all remaining MatMul ops, which is to ensure that the compiler does not quantize for the remaining MatMul ops.
```json
{
  "device_placement": [
    {
      "device": "cpu",
      "node_type": "onnx.MatMul",
      "onnx_node_name": "MatMul_0"
    }
  ],
  "quantization": [
    {
      "quantize": true,
      "node_type": "onnx.MatMul",
      "onnx_node_name": "MatMul_1"
    },
    {
      "quantize": false,
      "node_type": "onnx.MatMul"
    }
  ]
}
```

## A example JSON file for transformers models

This JSON configuration file is to quantize four MatMul operators in the self-attention layer, two MatMul operators in the linear layer and the MatMul operators in the LM head.
```json
{
  "quantization": [
    {
      "quantize": true,
      "node_type": "onnx.MatMul",
      "onnx_node_name": "^/model/layers\\.[0-9]+/self_attn/(q|k|v)_proj/MatMul"
    },
    {
      "quantize": true,
      "node_type": "onnx.MatMul",
      "onnx_node_name": "^/model/layers\\.[0-9]+(.*)/self_attn/o_proj/MatMul.*"
    },
    {
      "quantize": true,
      "node_type": "onnx.MatMul",
      "onnx_node_name": "^/model/layers\\.[0-9]+/mlp/((gate)|(up))_proj/MatMul"
    },
    {
      "quantize": true,
      "node_type": "onnx.MatMul",
      "onnx_node_name": ".*/lm_head/MatMul.*"
    },
    {
      "_comment":"do not quantize the remaining matmuls",
      "node_type": "onnx.MatMul",
      "quantize": false
    }
  ]
}
```
