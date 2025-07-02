#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

######################### add_missing_output_types.py ##########################
#
# Generates add_missing_output_types.json in test/mlir/onnx/parse/
#
# The model has two add operations, one with a custom domain and one with the
# default ONNX domain.
# Both of the outputs have missing types, which is not allowed in ONNX.
# The model is expected to be invalid, and is used to test that the output types
# are inferred by ONNX-MLIR.
#
################################################################################
import onnx
from onnx import ValueInfoProto, helper, TensorProto
from google.protobuf.json_format import MessageToJson

input_a = helper.make_tensor_value_info("input_a", TensorProto.FLOAT, [3, 1])
input_b = helper.make_tensor_value_info("input_b", TensorProto.FLOAT, [1, 3])


# outputs are missing the required type
output_c = ValueInfoProto()
output_c.name = "output_c"


output_d = ValueInfoProto()
output_d.name = "output_d"


add_node = helper.make_node(
    "test.Add",
    ["input_a", "input_b"],
    ["output_c"],
    name="add_node_custom",
    domain="test",
)

add_node_2 = helper.make_node(
    "Add",
    ["input_a", "input_b"],
    ["output_d"],
    name="add_node",
)


graph_def = helper.make_graph(
    [add_node, add_node_2], "add_graph", [input_a, input_b], [output_c, output_d]
)


model_def = helper.make_model(
    graph_def,
    producer_name="onnx-example",
    opset_imports=[helper.make_opsetid("test", 1)],
)

try:
    onnx.checker.check_model(model_def)
    assert False, "The model should be invalid due to missing output types"
except onnx.checker.ValidationError as e:
    pass

print(MessageToJson(model_def))
