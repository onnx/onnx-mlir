#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

######################### zipmap.py #########################################
#
# Generates raw/external/nonraw_data models in test/mlir/onnx/parse/
#
################################################################################

import argparse
import onnx
from onnx import helper
from google.protobuf.json_format import MessageToJson
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('data_format', choices=['raw', 'external', 'nonraw'], help="Save model as .onnx")
parser.add_argument('--save_dot_onnx', action='store_true', help="Save model as .onnx")
args = parser.parse_args()

def little_endian_bytes(nparray):
    return nparray.newbyteorder('<').tobytes()

nptypes = [
    np.dtype('float32'),
    np.dtype('uint8'),
    np.dtype('int8'),
    np.dtype('uint16'),
    np.dtype('int16'),
    np.dtype('int32'),
    np.dtype('int64'),
    np.dtype('bool'),
    # np.dtype('float16'),
    np.dtype('float64'),
    np.dtype('uint32'),
    np.dtype('uint64'),
]
assert all(ty == onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[ty]] for ty in nptypes)

def main():
    nonraw = args.data_format == 'nonraw'
    # array of values that make sense for every data type, including bool, ints, floats
    nptensor = np.array([1, 0, 1])
    shape = nptensor.shape
    nodes = [
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=[f"output{i}"],
            value=helper.make_tensor(
                f"tensor{i}",
                data_type=onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[ty],
                dims=shape,
                vals=nptensor.astype(ty) if nonraw else little_endian_bytes(nptensor.astype(ty)),
                raw=not nonraw,
            ),
        )
        for i, ty in enumerate(nptypes)
    ]
    outputs = [
        helper.make_tensor_value_info(f"output{i}", onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[ty], shape)
        for i, ty in enumerate(nptypes)
    ]
    inputs = []
    graph = helper.make_graph(nodes, "rawdata", inputs, outputs)
    model = helper.make_model(graph)
    onnx.checker.check_model(model)
    if args.data_format == "external":
        onnx.save_model(model, f"{args.data_format}_data.onnx",
            save_as_external_data=True,
            location=f"{args.data_format}_data.external",
            size_threshold=0,
            convert_attribute=True
        )
        onnx.checker.check_model(model)
    elif args.save_dot_onnx:
        onnx.save_model(model, f"{args.data_format}_data.onnx")
    print(MessageToJson(model))

if __name__ == '__main__':
    main()
