"""
This script is intended to display, verify or convert an onnx model.
With option -v or --VERBOSE, the model will be displayed on output
with option --save, the converted model will be saved to file too.
onnx package is required
Example of usage:
To convert a model, add.onnx, to the opset currently supported by 
onnx-mlir (e.g. 13) , use command:
 python pre-onnx-mlir add.onnx --save
The converted model will be saved into file add-opset-13.onnx
To display a model, add.onnx, use command:
  python pre-onnx-mlir add.onnx -v --no_convert
"""
import onnx
import argparse
from onnx import version_converter, helper

parser = argparse.ArgumentParser()
parser.add_argument("model",
        help="onnx model")
parser.add_argument("--save",
        help="save the converted model",
        action="store_true")
parser.add_argument("-v", "--VERBOSE",
        help="turn on verbosity",
        action="store_true")
parser.add_argument("--no_convert",
        help="turn off converter",
        action="store_true")
args = parser.parse_args()
original_model = onnx.load(args.model)
try:
    onnx.checker.check_model(original_model)
except onnx.checker.ValidationError as e:
    print('The model is invalid: %s' % e)
else:
    print('The model is valid!')

if args.VERBOSE :
    print('The model before conversion:\n{}'.format(original_model))

if args.no_convert :
    quit()

# Opset version supported by current onnx-mlir
# Should be consistent with gen_onnx_mlir.py
current_onnx_mlir_support_version=16

converted_model = version_converter.convert_version(
        original_model, current_onnx_mlir_support_version)

if args.VERBOSE :
    print('The model after conversion:\n{}'.format(converted_model))

if args.save :
    inputFile = args.model
    if inputFile.endswith(
            '-opset'+str(current_onnx_mlir_support_version)+'.onnx') :
        printf('Converted model is not saved due to name conflict')
    else :
        outFile = inputFile[:inputFile.rfind(".onnx")]+'-opset-'+str(current_onnx_mlir_support_version)+'.onnx'
        onnx.save(converted_model, outFile)
        print('The converted model is aved to '+outFile)
