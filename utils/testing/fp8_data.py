#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

################################# fp8_data.py ##################################
#
# Generates raw/nonraw_data float8 models in test/mlir/onnx/parse/
#
# The model outputs 4 constant tensors with the 4 different float8 data types.
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
        return nptensor.astype(np.dtype("uint8")).tobytes()
    else:
        return nptensor


# Variant of onnx.helper.make_tensor that works for all float8 data types
# with uint8 np.ndarray vals when raw == False.
def make_fp8_tensor(
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

    f8e4m3fn_minus1 = 184  # UINT8 184 represents F8E4M3FN -1
    f8e4m3fn_plus192 = 116  # UINT8 116 represents F8E4M3FN 192
    f8e4m3fnuz_minus1 = 192  # UINT8 192 represents F8E4M3FNUZ -1
    f8e4m3fnuz_plus192 = 124  # UINT8 124 represents F8E4M3FNUZ 192
    f8e5m2_minus1 = 188  # UINT8 188 represents F8E5M2 -1
    f8e5m2_plus192 = 90  # UINT8 90 represents F8E5M2 192
    f8e5m2fnuz_minus1 = 192  # UINT8 192 represents F8E5M2FNUZ -1
    f8e5m2fnuz_plus192 = 94  # UINT8 94 represents F8E5M2FNUZ 192
    dtype = np.dtype("uint8")

    f8e4m3fn_nptensor = np.array([f8e4m3fn_minus1, f8e4m3fn_plus192]).astype(dtype)
    f8e4m3fnuz_nptensor = np.array([f8e4m3fnuz_minus1, f8e4m3fnuz_plus192]).astype(
        dtype
    )
    f8e5m2_nptensor = np.array([f8e5m2_minus1, f8e5m2_plus192]).astype(dtype)
    f8e5m2fnuz_nptensor = np.array([f8e5m2fnuz_minus1, f8e5m2fnuz_plus192]).astype(
        dtype
    )

    nodes = [
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["output_f8e4m3fn"],
            value=make_fp8_tensor(
                f"tensor_f8e4m3fn",
                data_type=onnx.TensorProto.FLOAT8E4M3FN,
                dims=f8e4m3fn_nptensor.shape,
                vals=tensor_vals(f8e4m3fn_nptensor, raw),
                raw=raw,
            ),
        ),
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["output_f8e4m3fnuz"],
            value=make_fp8_tensor(
                f"tensor_f8e4m3fnuz",
                data_type=onnx.TensorProto.FLOAT8E4M3FNUZ,
                dims=f8e4m3fnuz_nptensor.shape,
                vals=tensor_vals(f8e4m3fnuz_nptensor, raw),
                raw=raw,
            ),
        ),
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["output_f8e5m2"],
            value=make_fp8_tensor(
                f"tensor_f8e5m2",
                data_type=onnx.TensorProto.FLOAT8E5M2,
                dims=f8e5m2_nptensor.shape,
                vals=tensor_vals(f8e5m2_nptensor, raw),
                raw=raw,
            ),
        ),
        helper.make_node(
            "Constant",
            inputs=[],
            outputs=["output_f8e5m2fnuz"],
            value=make_fp8_tensor(
                f"tensor_f8e5m2fnuz",
                data_type=onnx.TensorProto.FLOAT8E5M2FNUZ,
                dims=f8e5m2fnuz_nptensor.shape,
                vals=tensor_vals(f8e5m2fnuz_nptensor, raw),
                raw=raw,
            ),
        ),
    ]
    outputs = [
        helper.make_tensor_value_info(
            "output_f8e4m3fn", onnx.TensorProto.FLOAT8E4M3FN, f8e4m3fn_nptensor.shape
        ),
        helper.make_tensor_value_info(
            "output_f8e4m3fnuz",
            onnx.TensorProto.FLOAT8E4M3FNUZ,
            f8e4m3fnuz_nptensor.shape,
        ),
        helper.make_tensor_value_info(
            "output_f8e5m2", onnx.TensorProto.FLOAT8E5M2, f8e5m2_nptensor.shape
        ),
        helper.make_tensor_value_info(
            "output_f8e5m2fnuz",
            onnx.TensorProto.FLOAT8E5M2FNUZ,
            f8e5m2fnuz_nptensor.shape,
        ),
    ]
    inputs = []
    name = f"fp8_{args.data_format}_data"
    graph = helper.make_graph(nodes, name, inputs, outputs)
    model = helper.make_model(graph)
    onnx.checker.check_model(model)
    if args.save_dot_onnx:
        onnx.save_model(model, f"{name}.onnx")
    print(MessageToJson(model))


if __name__ == "__main__":
    main()
