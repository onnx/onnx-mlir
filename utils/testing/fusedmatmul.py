#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

######################### fusedmatmul.py #######################################
#
# Generates fusedmatmul model in test/mlir/onnx/parse/fusedmatmul.json
#
# The model has a single FusedMatMul node.
#
################################################################################

import argparse
import onnx
from onnx import helper
from google.protobuf.json_format import MessageToJson

parser = argparse.ArgumentParser()
parser.add_argument('--save_dot_onnx', action='store_true', help="Save model as fusedmatmul.onnx")
args = parser.parse_args()

def main():
    lhs_info = helper.make_tensor_value_info("lhs", onnx.TensorProto.FLOAT, (2,3))
    rhs_info = helper.make_tensor_value_info("rhs", onnx.TensorProto.FLOAT, (4,3))
    node = helper.make_node("FusedMatMul", ["lhs", "rhs"], ["output"], domain="com.microsoft", alpha=0.125, transA=0, transB=1)
    value_type = helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ())
    output_info = helper.make_tensor_value_info("output", onnx.TensorProto.FLOAT, (2,4))
    graph = helper.make_graph([node], "fusedmatmuller", [lhs_info, rhs_info], [output_info])
    model = helper.make_model(graph)
    # check_model(model) fails with "ValidationError: No opset import for domain 'com.microsoft'
    # onnx.checker.check_model(model)
    if args.save_dot_onnx:
        onnx.save_model(model, "fusedmatmul.onnx")
    print(MessageToJson(model))

if __name__ == '__main__':
    main()
