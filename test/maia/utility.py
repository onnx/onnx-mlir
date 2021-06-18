from pathlib import Path
from typing import Callable
from enum import Flag
from collections import namedtuple
from model_generators.generate_inputs import generate_input, generate_random
from model_generators.generate_host import generate_host

import onnx
from onnx import TensorProto, ModelProto, TypeProto

list_only = False
out_dir = ""
test_list = []
test_index = 0


class TestType(Flag):
  COMPILE_ONLY = 0
  DIFF = 1
  EXECUTE = 2
  def __str__(self):
    return self.name

in_out = namedtuple('in_out', ['name', 'shape'])

def get_shape(type : TypeProto) -> "list[int]":
  return list(map(lambda x: x.dim_value, type.tensor_type.shape.dim))

def get_inputs(model : ModelProto) -> "list[int]":
  return list(map(lambda x: in_out(x.name, get_shape(x.type)), model.graph.input))

def get_outputs(model : ModelProto) -> "list[int]":
  return list(map(lambda x: in_out(x.name, get_shape(x.type)), model.graph.output))


def add_test(gen_model: Callable[[TensorProto], ModelProto], test_name: str, test_type: TestType = TestType.COMPILE_ONLY, epsilon: float = 0.0, **kwargs):
  global test_index
  
  # check if we're just listing available tests
  if list_only:
    print(f'{test_name}\t{test_type}')
    return

  # if the users has specificed specific tests, gen those only
  if len(test_list) > 0 and not test_name in test_list:
    return

  print(f"  Generating {test_name}...")

  test_dir = Path(out_dir) / test_name

  if not test_dir.exists():
    test_dir.mkdir(parents=True)

  model_bf16 = gen_model(TensorProto.BFLOAT16)
  model_f32 = gen_model(TensorProto.FLOAT)
  
  # write out the models to disk
  onnx.save(model_bf16, str(test_dir / test_name) + '.onnx')
  onnx.save(model_f32, str(test_dir / test_name) + '.f32.onnx')

  if test_type & TestType.EXECUTE:
    inputs_dir = test_dir / 'inputs'
    if not inputs_dir.exists():
      inputs_dir.mkdir()

    # generate model inputs
    for input in model_bf16.graph.input:
      generator = kwargs[input.name] if kwargs.get(input.name) else generate_random
      generate_input(inputs_dir, input.name, get_shape(input.type), generator)

    # generate the host code
    generate_host(test_dir / 'host.cpp', "", "Sku_ArrayFloat32", get_inputs(model_bf16), get_outputs(model_bf16), 9000 + (test_index * 5))
    
    # increment the test index for the next test
    test_index += 1

