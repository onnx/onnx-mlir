#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

######################### onnx2json.py #########################################
#
# Converts a binary onnx model to json.
# Useful tool for constructing test/mlir/onnx/parse lit tests from onnx files
# (see comments in the parse lit tests).
#
################################################################################

import argparse
import onnx
from google.protobuf.json_format import MessageToJson

parser = argparse.ArgumentParser()
parser.add_argument("model_path", type=str, help="Path to the ONNX model")
parser.add_argument(
    "--load_external_data",
    action="store_true",
    help="Load external data under the same directory of the model",
)
args = parser.parse_args()


def main():
    model = onnx.load_model(args.model_path, load_external_data=args.load_external_data)
    print(MessageToJson(model))


if __name__ == "__main__":
    main()
