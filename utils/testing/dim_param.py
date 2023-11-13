#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

################################# dim_param.py #################################
#
# Generates dim_param.json in test/mlir/onnx/parse/
#
# The model has a single Add operations whose operands and outputs use
# dim_param to represent unknown dimensions.
#
################################################################################

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto
from google.protobuf.json_format import MessageToJson


# The protobuf definition can be found here:
# https://github.com/onnx/onnx/blob/main/onnx/onnx.proto


# Create one input (ValueInfoProto)
X = helper.make_tensor_value_info(
    "X", TensorProto.FLOAT, ["batch_size", "sequence_len"]
)

# Create second input (ValueInfoProto)
Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1, "sequence_len"])

# Create one output (ValueInfoProto)
Z = helper.make_tensor_value_info(
    "Z", TensorProto.FLOAT, ["batch_size", "sequence_len"]
)

# Create a node (NodeProto)
node_def = helper.make_node(
    "Add",  # node name
    ["X", "Y"],  # inputs
    ["Z"],  # outputs
)

# Create the graph (GraphProto)
graph_def = helper.make_graph(
    [node_def],
    "test-dim-param",
    [X, Y],
    [Z],
)

# Create the model (ModelProto)
model_def = helper.make_model(graph_def, producer_name="onnx-mlir")

onnx.checker.check_model(model_def)
print(MessageToJson(model_def))
