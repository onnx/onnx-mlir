from model_generators.mnist import create_mnist_model
from utility import *
from model_generators.generate_inputs import generate_fixed

from typing import Tuple, Callable

import onnx
from onnx import TensorProto, ModelProto

# generate model and name given the parameters
def generate_mnist_model(layer_dims : list[int]) -> Tuple[Callable[[TensorProto], ModelProto], str]:
  model_name = f'mnist_padded_static_noconsts_2d_bias'
  def make_model(float_mode: TensorProto) -> ModelProto:
    return create_mnist_model(layer_dims, float_mode)
  return make_model, model_name

def all_ones(shape: list[int]):
  return generate_fixed(shape, 1.0)

##
## Actual tests go here
##

def generate_tests():
  model_gen, name = generate_mnist_model([256, 256])
  add_test(model_gen, name, test_type = TestType.EXECUTE, epsilon = 0.0,
     X = all_ones, weight_0 = all_ones, weight_1 = all_ones, bias_0 = all_ones, bias_1 = all_ones)
