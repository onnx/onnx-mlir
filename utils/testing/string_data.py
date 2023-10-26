#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

################################ string_data.py ################################
#
# Generates a string model in test/mlir/onnx/parse/
#
# The model outputs a constant tensor with string data type.
#
################################################################################

import argparse
import onnx
from onnx import helper
from google.protobuf.json_format import MessageToJson
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--save_dot_onnx", action="store_true", help="Save model as .onnx")
args = parser.parse_args()


def main():
    nptensor = np.array([b"hello", b"world"])
    nodes = [
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["output"],
            value=helper.make_tensor(
                "tensor",
                data_type=onnx.TensorProto.STRING,
                dims=nptensor.shape,
                vals=nptensor,
            ),
        )
    ]
    outputs = [
        helper.make_tensor_value_info("output", onnx.TensorProto.STRING, nptensor.shape)
    ]
    inputs = []
    graph = helper.make_graph(nodes, "string_data", inputs, outputs)
    model = helper.make_model(graph)
    onnx.checker.check_model(model)
    if args.save_dot_onnx:
        onnx.save_model(model, f"string_data.onnx")
    print(MessageToJson(model))


if __name__ == "__main__":
    main()
