#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

################################# int4_data.py #################################
#
# Generates raw/nonraw_data (u)int4 models in test/mlir/onnx/parse/
#
# The model outputs 2 constant tensors with int4 and uint4 data types.
# The data_format command line argument controls whether the tensor data
# is represented "nonraw" (normally) or "raw" (as raw_data byte arrays).
#
################################################################################

from typing import Any, Sequence
import argparse
import onnx
from onnx import helper
from google.protobuf.json_format import MessageToJson
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument(
    "data_format", choices=["raw", "nonraw"], help="Tensor data representation"
)
parser.add_argument("--save_dot_onnx", action="store_true", help="Save model as .onnx")
args = parser.parse_args()


def make_i4_tensor(
    name: str, data_type: int, dims: Sequence[int], vals: Any, raw: bool, dtype: Any
) -> onnx.TensorProto:
    tensor = onnx.TensorProto()
    tensor.data_type = data_type
    tensor.name = name
    # From onnx.proto3:
    # - Each pair of uint4, int4, and float4 values MUST be packed as two 4-bit elements into a single byte.
    #   The first element is stored in the 4 least significant bits (LSB),
    #   and the second element is stored in the 4 most significant bits (MSB).
    flattend = vals.flatten().tolist()
    num_elements = len(flattend)
    packed = []
    for i in range(0, num_elements, 2):
        if i + 1 < num_elements:
            packed_value = (flattend[i] & 0x0F) | ((flattend[i + 1] & 0x0F) << 4)
        else:
            packed_value = flattend[i] & 0x0F
        packed.append(packed_value)
    if raw:
        # From onnx.proto3:
        # uint4 and int4 values must be packed to 4bitx2, the first element is stored in the 4 LSB and the second element is stored in the 4 MSB
        # As its only a single byte, we do not need to care about endianness.
        packed_array = np.array(packed, dtype=dtype)
        tensor.raw_data = packed_array.tobytes()
    else:
        tensor.int32_data.extend(packed)
    tensor.dims.extend(dims)
    return tensor


def main():
    raw = args.data_format == "raw"

    dtype = np.dtype("int8")  # Numpy does not support int4, so we use int8

    int4_nptensor = np.array([0, 1, -1, 2, -2]).astype(dtype)
    uint4_nptensor = np.array([0, 1, 15]).astype(dtype)

    nodes = [
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["output_int4"],
            value=make_i4_tensor(
                f"tensor_int4",
                data_type=onnx.TensorProto.INT4,
                dims=int4_nptensor.shape,
                vals=int4_nptensor,
                raw=raw,
                dtype=dtype,
            ),
        ),
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["output_uint4"],
            value=make_i4_tensor(
                f"tensor_uint4",
                data_type=onnx.TensorProto.UINT4,
                dims=uint4_nptensor.shape,
                vals=uint4_nptensor,
                raw=raw,
                dtype=dtype,
            ),
        ),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "output_int4", onnx.TensorProto.INT4, int4_nptensor.shape
        ),
        helper.make_tensor_value_info(
            "output_uint4", onnx.TensorProto.UINT4, uint4_nptensor.shape
        ),
    ]
    inputs = []
    name = f"int4_{args.data_format}_data"
    graph = helper.make_graph(nodes, name, inputs, outputs)
    model = helper.make_model(graph)
    onnx.checker.check_model(model)
    if args.save_dot_onnx:
        onnx.save_model(model, f"{name}.onnx")
    print(MessageToJson(model))


if __name__ == "__main__":
    main()
