import argparse
from pathlib import Path
from sys import path
from typing import Callable
import onnxruntime
import numpy as np

# TODO: this should be configurable
np.random.seed(0)

def get_size(shape : "list[int]"):
  prod = 1
  for dim in shape:
    prod *= dim
  return prod

def generate_random(shape : "list[int]"):
  return np.random.ranf(shape).astype(np.float32)

def generate_sequential(shape : "list[int]", start, stop):
  iterable = (x%stop+start for x in range(get_size(shape)))
  return np.fromiter(iterable, np.float32).reshape(shape)

def generate_fixed(shape : "list[int]", value):
  iterable = (value for _ in range(get_size(shape)))
  return np.fromiter(iterable, np.float32).reshape(shape)


def generate_input(output_dir : Path, data_name : str, shape, generator : Callable[["list[int]"], "list[float]"]):
  input_data = generator(shape)
  (output_dir / data_name).write_bytes(input_data)


if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Generate inputs for an ONNX test case')
  parser.add_argument('--input', dest='graph_path', action='store', default="",
                      help='path-to-input-graph (ONNX file format)')
  parser.add_argument('--output', dest='output_dir', action='store', default="",
                      help='output location (defaults to current directory)')
  args = parser.parse_args()

  graph_path = args.graph_path
  output_dir = Path(args.output_dir if args.output_dir != "" else '.')

  session = onnxruntime.InferenceSession(graph_path)
  session.get_modelmeta()

  inputs_dir = output_dir / 'inputs'
  if not inputs_dir.exists():
    inputs_dir.mkdir()

  for input in session.get_inputs():
    print(f'Generating: {input.name}')
    generate_input(output_dir / 'inputs', input.name, input.shape, generate_random)
