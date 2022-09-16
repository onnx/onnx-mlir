#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

######################### zipmap.py #########################################
#
# Generates zipmap model in test/mlir/onnx/parse/zipmap.json
#
# The model has a single ZipMap node.
#
################################################################################

import argparse
import onnx
from onnx import helper
from google.protobuf.json_format import MessageToJson

parser = argparse.ArgumentParser()
parser.add_argument('--save_dot_onnx', action='store_true', help="Save model as zipmap.onnx")
args = parser.parse_args()

def make_map_type_proto(key_type, value_type):
    map_type_proto = onnx.TypeProto()
    map_type_proto.map_type.key_type = key_type
    map_type_proto.map_type.value_type.tensor_type.CopyFrom(value_type.tensor_type)
    return map_type_proto

def make_map_sequence_value_info(name, key_type, value_type):
    map_type_proto = make_map_type_proto(key_type, value_type)
    seq_type_proto = helper.make_sequence_type_proto(map_type_proto)
    value_info_proto = onnx.ValueInfoProto()
    value_info_proto.name = name
    value_info_proto.type.sequence_type.CopyFrom(seq_type_proto.sequence_type)
    return value_info_proto

def main():
    input_info = helper.make_tensor_value_info("input", onnx.TensorProto.FLOAT, (3,))
    node = helper.make_node("ZipMap", ["input"], ["output"], classlabels_int64s=[10, 20, 30])
    value_type = helper.make_tensor_type_proto(onnx.TensorProto.FLOAT, ())
    output_info = make_map_sequence_value_info("output", onnx.TensorProto.INT64, value_type)
    graph = helper.make_graph([node], "zipmapper", [input_info], [output_info])
    model = helper.make_model(graph)
    # check_model(model) fails with "ValidationError: No Op registered for ZipMap with domain_version of 16"
    # onnx.checker.check_model(model)
    if args.save_dot_onnx:
        onnx.save_model(model, "zipmap.onnx")
    print(MessageToJson(model))

if __name__ == '__main__':
    main()
