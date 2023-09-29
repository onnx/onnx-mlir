#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

################################## x_data.py ###################################
#
# Generates raw/external/nonraw_data models in test/mlir/onnx/parse/
#
# The model outputs 11 constant tensors with different data types.
# The data_format command line argument controls whether the tensor data
# is represented "nonraw" (normally), "raw" (as raw_data byte arrays), or
# "external" (as external data in a separate file).
#
################################################################################

import argparse
import onnx
from onnx import helper
from google.protobuf.json_format import MessageToJson
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_format",
    choices=["raw", "external", "nonraw"],
    help="Tensor data representation",
)
parser.add_argument("--save_dot_onnx", action="store_true", help="Save model as .onnx")
args = parser.parse_args()

nptypes = [
    np.dtype("float32"),
    np.dtype("uint8"),
    np.dtype("int8"),
    np.dtype("uint16"),
    np.dtype("int16"),
    np.dtype("int32"),
    np.dtype("int64"),
    np.dtype("bool"),
    np.dtype("float16"),
    np.dtype("float64"),
    np.dtype("uint32"),
    np.dtype("uint64"),
]
assert all(
    ty == onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[ty]]
    for ty in nptypes
)


# formats nptensor for make_tensor vals arg: bytes if raw else np.ndarray
def tensor_vals(nptensor, ty, raw):
    nptensor = nptensor.astype(ty)
    if raw:
        # onnx proto spec for raw_data requires "fixed-width, little-endian order"
        return nptensor.astype(ty.newbyteorder("<")).tobytes()
    else:
        # NOTE: onnx proto spec requires "float16 values must be bit-wise
        # converted to an uint16_t" but we shouldn't do that here because
        # make_tensor takes care of that
        return nptensor


def main():
    raw = args.data_format != "nonraw"  # True if data format is 'raw' or 'external'
    nptensor = np.array([1, 0, 1])  # 0, 1 make sense for all bool/int/float data types
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
                vals=tensor_vals(nptensor, ty, raw),
                raw=raw,
            ),
        )
        for i, ty in enumerate(nptypes)
    ]
    outputs = [
        helper.make_tensor_value_info(
            f"output{i}", onnx.mapping.NP_TYPE_TO_TENSOR_TYPE[ty], shape
        )
        for i, ty in enumerate(nptypes)
    ]
    inputs = []
    name = f"{args.data_format}_data"
    graph = helper.make_graph(nodes, name, inputs, outputs)
    model = helper.make_model(graph)
    onnx.checker.check_model(model)
    if args.data_format == "external":
        onnx.save_model(
            model,
            f"{name}.onnx",
            save_as_external_data=True,
            location=f"{name}.external",
            size_threshold=0,
            convert_attribute=True,
        )
        onnx.checker.check_model(model)
    elif args.save_dot_onnx:
        onnx.save_model(model, f"{name}.onnx")
    print(MessageToJson(model))


if __name__ == "__main__":
    main()
