#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

################################################################################
#
# Generates a model with a custom op and a relu, with shape information
# in the model
#
################################################################################

import onnx
from onnx import helper
from onnx import TensorProto
from google.protobuf.json_format import MessageToJson


X = helper.make_tensor_value_info("X", TensorProto.FLOAT, [4, 5])


Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [5, 6])

# Create one output (ValueInfoProto)
Z = helper.make_tensor_value_info("Z", TensorProto.FLOAT, [5, 6])

# Create a node (NodeProto)
custom_def = helper.make_node(
    "testOp",
    ["X"],
    ["Y"],
    domain="test",
)

relu_def = helper.make_node(
    "Relu",
    ["Y"],
    ["Z"],
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [custom_def, relu_def],
    "test-custom",
    [X],
    [Z],
    value_info=[Y],
)

# Create the model (ModelProto)
model_def = helper.make_model(
    graph_def,
    producer_name="onnx-mlir",
    opset_imports=[
        helper.make_opsetid("", 22),
        helper.make_opsetid("test", 1),
    ],
)

onnx.checker.check_model(model_def)
print(MessageToJson(model_def))
