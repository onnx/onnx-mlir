#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

######################### onnxExternalizeData.py ###############################
#
# Converts the data in tensors in an onnx model to external data.
# Useful tool for constructing external data examples for testing.
#
################################################################################

import argparse
import os
import onnx

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str, help="Path to the ONNX model")
parser.add_argument('--no_all_tensors_to_one_file', action='store_true', help="Save tensors to multiple files")
parser.add_argument('--no_convert_attribute', action='store_true', help="Only convert initializer tensors to external data")
args = parser.parse_args()

def main():
    filepath = args.model_path
    basename = os.path.basename(filepath)
    model = onnx.load_model(filepath)
    onnx.save_model(model, args.model_path,
        save_as_external_data=True,
        all_tensors_to_one_file=not args.no_all_tensors_to_one_file,
        location=f"{basename}.ext",
        size_threshold=0,
        convert_attribute=not args.no_convert_attribute
    )

if __name__ == '__main__':
    main()
