#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

######################### onnxExternalizeData.py ###############################
#
# Converts the data in tensors in an onnx model to external data.
# Useful tool for constructing external data examples for testing.
#
# Call with --make_raw to convert non-raw tensors to raw_data to make them
# eligible to become external data, otherwise
# onnx.save_model(model, path, save_as_external_data=True)
# doesn't convert them to external data.
# For example, given the 249MB arcfaceresnet100-8.onnx file from the model zoo
#
#   utils/onnxExternalizeData.py arcfaceresnet100-8.onnx --make_raw
#
# creates a 249MB external data file arcfaceresnet100-8.onnx.ext and shrinks
# arcfaceresnet100-8.onnx to 189KB.
#
################################################################################

import argparse
import onnx
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str, help="Path to the ONNX model")
parser.add_argument(
    "--size_threshold",
    type=int,
    default=0,
    help="Only convert tensors with byte size no smaller than this",
)
parser.add_argument(
    "--no_all_tensors_to_one_file",
    action="store_true",
    help="Save tensors to multiple files",
)
parser.add_argument(
    "--no_convert_attribute",
    action="store_true",
    help="Only convert initializer tensors to external data",
)
parser.add_argument(
    "--make_raw", action="store_true", help="Convert non-raw tensors to raw_data"
)
args = parser.parse_args()


def get_tensors(onnx_model_proto):
    # HACK: Use these convenient private onnx.external_data_helper methods.
    #       Will need to be updated/reimplemented if onnx.external_data_helper changes.
    if args.no_convert_attribute:
        return onnx.external_data_helper._get_initializer_tensors(onnx_model_proto)
    else:
        return onnx.external_data_helper._get_all_tensors(onnx_model_proto)


def main():
    filepath = args.model_path
    basename = os.path.basename(filepath)
    model = onnx.load_model(filepath)
    if args.make_raw:
        tensors = get_tensors(model)
        for tensor in tensors:
            if (
                not tensor.HasField("raw_data")
                and tensor.data_type != onnx.TensorProto.STRING
            ):
                arr = onnx.numpy_helper.to_array(tensor)
                # TODO: If this is too slow, calculate bytes size without converting to bytes
                #       to avoid conversion when bytes size is below threshold.
                bytes = arr.tobytes()
                if sys.getsizeof(bytes) >= args.size_threshold:
                    storage_field = onnx.helper.tensor_dtype_to_field(tensor.data_type)
                    tensor.ClearField(storage_field)
                    tensor.raw_data = bytes
                    if sys.byteorder == "big":
                        # Convert endian from big to little
                        onnx.numpy_helper.convert_endian(tensor)
    onnx.save_model(
        model,
        args.model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=not args.no_all_tensors_to_one_file,
        location=f"{basename}.ext",
        size_threshold=args.size_threshold,
        convert_attribute=not args.no_convert_attribute,
    )


if __name__ == "__main__":
    main()
