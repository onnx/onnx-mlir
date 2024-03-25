<!--- SPDX-License-Identifier: Apache-2.0 -->

# Device placement

Device placement is how the compiler place one operation on CPU or NNPA.

## Query device placement configuration

There are two ways to know which device an operation is placed on:
- Using `onnx-mlir --EmitONNXIR --maccel=NNPA model.onnx`, or
- Using `onnx-mlir --nnpa-save-device-placement-file=cfg.json model.onnx`.
 
1. Using `--EmitONNXIR --maccel=NNPA`

When using `--EmitONNXIR --maccel=NNPA` options, each operation in the generated IR is annotated with an attribute `device` to show which device the operation is placed on. There are three posible values for `device`:
- "": the operation may be on CPU or NNPA depending on optimizations in the compiler. 
- "nnpa": the operation is on NNPA.
- "cpu": the operation is on CPU.

Below is an example of the output of `--EmitONNXIR --maccel=NNPA`:
```mlir
%0 = "onnx.Relu"(%arg0) {onnx_node_name = "Relu_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
%1 = "onnx.Relu"(%0) {device="cpu", onnx_node_name = "Relu_1"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
%2 = "onnx.Relu"(%1) {onnx_node_name = "Relu_2"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
%3 = "onnx.Sigmoid"(%2) {device="nnpa", onnx_node_name = "Sigmoid_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
```

2. Using `--nnpa-save-device-placement-file=cfg.json`

The option is to save the device placement configuration into a JSON file. This option is convenient when users don't want to interrupt the compilation.

The JSON file will contains a list of operation records. Each record includes three key-value pairs where keys are: 
- "device": similar to `device` attribute in the operation.
- "node_type": ONNX node type, e.g. `onnx.Conv`, `onnx.MatMul`.
- "onnx_node_name": a string to denote ONNX node names.

Below is one example of a JSON file:
```json
{
  "device_placement": [
    {
      "device":"nnpa",
      "node_type":"onnx.Relu",
      "onnx_node_name":"Relu_0"
    },
    {
      "device":"cpu",
      "node_type":"onnx.Relu",
      "onnx_node_name":"Relu_1"},
    {
      "device":"nnpa",
      "node_type":"onnx.Relu",
      "onnx_node_name":"Relu_2"
    },
    {
      "device":"nnpa",
      "node_type":"onnx.Sigmoid",
      "onnx_node_name":"Sigmoid_0"
    }
  ]
}
```

## Set device placement manually.

We allow users to force one operation to run on a specific device. However, at this moment, only placing on CPU is guaranted to be successful done. It means that even when `device=NNPA` is specified, it is not guaranted that the operation will run on NNPA. 

There are two ways to change device of an operation:
- by editing the output of `--EmitONNXIR --maccel=NNPA` directly and compile again.
- by passing a JSON file for device placement to the compiler by using `--nnpa-load-device-placement-file=json`.

For the former option, it is straighforward, just changing the value of the `device` attribute of an operation, for example, changing `device=nnpa` to `device=cpu`.

For the later option, users can obtain a template file from `--nnpa-save-device-placement-file`, and use it as the starting point of modification.
We use C++ std::regex_match function to match operations based on `node_type` and `onnx_node_name`. Both `node_type` and `onnx_node_name` must be satisfied.
The JSON file will contain a list of records for each operation matching. The order of the records does matter. If one operation matches a record and is set device, it will not be set device again even when it matches the later records in the list. If one operation does not match a record but matches a later record, the operation is still set device by the later record. In other words, the device of an operation is set by the first matched record.

Below are some examples for the later option. Given an input program:
```mlir
func.func @test_load_config_file_all_on_cpu(%arg0: tensor<?x?x?xf32>) -> tensor<?x?x?xf32> {
  %0 = "onnx.Relu"(%arg0) {onnx_node_name = "Relu_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %1 = "onnx.Relu"(%0) {onnx_node_name = "Relu_1"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %2 = "onnx.Relu"(%1) {onnx_node_name = "Relu_2"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  %3 = "onnx.Sigmoid"(%2) {onnx_node_name = "Sigmoid_0"} : (tensor<?x?x?xf32>) -> tensor<?x?x?xf32>
  onnx.Return %3 : tensor<?x?x?xf32>
```

1. Schedule all operations to run on CPU
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

2. Schedule all Relu operations to run on CPU:
```json
{
  "device_placement": [
    {
      "device": "cpu",
      "node_type": "onnx.Relu",
      "onnx_node_name": ".*"
    }
  ]
}
```
3.  Schedule operations using onnx_node_name: here we use regex to chose only Relu_1 and Relu_2 operations, exact match is used for onnx.Sigmoid.
```json
{
  "device_placement": [
    {
      "device": "cpu",
      "node_type": "onnx.Relu",
      "onnx_node_name": "Relu_(1|2)"
    },
    {
      "device": "nnpa",
      "node_type": "onnx.Sigmoid",
      "onnx_node_name": "Sigmoid_0"
    }
  ]
}
```

4. `onnx.Relu` does not match because there is no operation with `node_type = Relu`, so only `onnx.Sigmoid` is set device.
```json
{
  "device_placement": [
    {
      "device": "cpu",
      "node_type": "Relu",
      "onnx_node_name": "Relu_(1|2)"
    },
    {
      "device": "cpu",
      "node_type": "onnx.Sigmoid",
      "onnx_node_name": "Sigmoid_0"
    }
  ]
}
```

5. We have two overlapping records both matching on `onnx.Relu`. In this case, only the first matched record will set device. Thus, `Relu_0` and `Relu_1` have device "cpu" by matching the first record, `Relu_2` operation has device "cpu" by matching the third record.
```json
{
  "device_placement": [
    {
      "device": "cpu",
      "node_type": "onnx.Relu",
      "onnx_node_name": "Relu_(0|1)"
    },
    {
      "device": "nnpa",
      "node_type": "onnx.Sigmoid",
      "onnx_node_name": "Sigmoid_0"
    },
    {
      "device": "cpu",
      "node_type": "onnx.Relu",
      "onnx_node_name": "Relu_(1|2)"
    }
  ]
}
```
