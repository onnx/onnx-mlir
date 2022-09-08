#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

##################### RunONNXModel.py #########################################
#
# Converts a binary onnx model to json.
# Useful for constructing test/mlir/onnx/parse lit tests.
#
################################################################################

import argparse
import onnx
from google.protobuf.json_format import MessageToJson

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str, help="Path to the ONNX model")
args = parser.parse_args()

def main():
    f = open(args.model_path, "rb")
    model = onnx.ModelProto()
    model.ParseFromString(f.read())
    json=MessageToJson(model)
    print(json)

if __name__ == '__main__':
    main()
