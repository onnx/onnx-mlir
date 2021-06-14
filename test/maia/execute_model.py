import argparse
import pathlib

import onnxruntime
import numpy as np

parser = argparse.ArgumentParser(description='Execute an ONNX test case on CPU')
parser.add_argument('--input', dest='graph_path', action='store', default="",
                    help='path-to-input-graph (ONNX file format)')
parser.add_argument('--datapath', dest='data_path', action='store', default="",
                    help='directory containing inputs and outputs directories')
parser.add_argument('--show-output', dest='show_output', action='store_true', default="",
                    help='display results to stdout')
args = parser.parse_args()

graph_path = args.graph_path
data_path = args.data_path
show_output = args.show_output

session = onnxruntime.InferenceSession(graph_path)
session.get_modelmeta()

model_dir = pathlib.Path(data_path) if (data_path != '') else pathlib.Path(graph_path).parent
input_path = model_dir / 'inputs'
output_path = model_dir / 'outputs'

model_inputs = {}

for input in session.get_inputs():
  raw_input = np.fromfile(input_path / input.name, np.float32)
  model_inputs[input.name] = np.reshape(raw_input, input.shape)

results = session.run([], model_inputs)

if not output_path.exists():
  output_path.mkdir()

for i, output in enumerate(session.get_outputs()):
  (output_path / output.name).write_bytes(results[i])
  if show_output:
    print(f'--- {output.name} ---')
    round_trip = np.fromfile(output_path / output.name, np.float32)
    for val in round_trip:
      print(val)

