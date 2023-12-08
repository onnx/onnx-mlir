#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

################################# fp16_data.py #################################
#
# Generates raw/nonraw_data (b)float16 models in test/mlir/onnx/parse/
#
# The model outputs 2 constant tensors with float16 and bfloat16 data types.
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


# formats nptensor for make_tensor vals arg: bytes if raw else np.ndarray
def tensor_vals(nptensor, raw):
    if raw:
        # onnx proto spec for raw_data requires "fixed-width, little-endian order"
        return nptensor.astype(np.dtype("uint16").newbyteorder("<")).tobytes()
    else:
        return nptensor


# Variant of onnx.helper.make_tensor that works for both float16 and bfloat16
# with uint16 np.ndarray vals when raw == False.
def make_fp16_tensor(
    name: str, data_type: int, dims: Sequence[int], vals: Any, raw: bool = False
) -> onnx.TensorProto:
    tensor = onnx.TensorProto()
    tensor.data_type = data_type
    tensor.name = name
    if raw:
        tensor.raw_data = vals
    else:
        tensor.int32_data.extend(vals.flatten().tolist())
    tensor.dims.extend(dims)
    return tensor


def main():
    raw = args.data_format == "raw"

    f16_minus_one = 48128  # FLOAT16    -1 is represented by UINT16 48128
    f16_9984 = 28896  # FLOAT16  9984 is represented by UINT16 28896
    bf16_minus_one = 49024  # BFLOAT16   -1 is represented by UINT16 49024
    bf16_9984 = 17948  # BFLOAT16 9984 is represented by UINT16 17948
    dtype = np.dtype("uint16")

    f16_nptensor = np.array([f16_minus_one, f16_9984]).astype(dtype)
    bf16_nptensor = np.array([bf16_minus_one, bf16_9984]).astype(dtype)

    nodes = [
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["output_f16"],
            value=make_fp16_tensor(
                f"tensor_f16",
                data_type=onnx.TensorProto.FLOAT16,
                dims=f16_nptensor.shape,
                vals=tensor_vals(f16_nptensor, raw),
                raw=raw,
            ),
        ),
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["output_bf16"],
            value=make_fp16_tensor(
                f"tensor_bf16",
                data_type=onnx.TensorProto.BFLOAT16,
                dims=bf16_nptensor.shape,
                vals=tensor_vals(bf16_nptensor, raw),
                raw=raw,
            ),
        ),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "output_f16", onnx.TensorProto.FLOAT16, f16_nptensor.shape
        ),
        helper.make_tensor_value_info(
            "output_bf16", onnx.TensorProto.BFLOAT16, bf16_nptensor.shape
        ),
    ]
    inputs = []
    name = f"fp16_{args.data_format}_data"
    graph = helper.make_graph(nodes, name, inputs, outputs)
    model = helper.make_model(graph)
    onnx.checker.check_model(model)
    if args.save_dot_onnx:
        onnx.save_model(model, f"{name}.onnx")
    print(MessageToJson(model))


if __name__ == "__main__":
    main()
