from model_generators.assembler import ModelAssembler
from utility import *

from typing import Tuple, Callable
from enum import Enum

import onnx
from onnx import TensorProto, ModelProto


##
## Helper functions for building models
##

# create a model with a transpose of a 2d matrix with the provided dims
def matrix_transpose_model(dims: "list[int]") -> Tuple[Callable[[TensorProto], ModelProto], str]:
  model_name = f"matrix_transpose"
  for i in dims:
    model_name += f"_{i}"

  def make_model(float_mode: TensorProto) -> ModelProto:
    assembler = ModelAssembler(model_name)
    p0 = assembler.add_input("p0", float_mode, dims)
    p0t = assembler.transpose("p0_transposed", p0, [1, 0])
    assembler.add_output(p0t)
    return assembler.model()

  return make_model, model_name


# create a model with a shuffle (N-d matrix transpose)
def complex_matrix_transpose_model(dims: "list[int]", shuffle: "list[int]") -> Tuple[Callable[[TensorProto], ModelProto], str]:
  model_name = f"complex_transpose"
  for i in dims:
    model_name += f"_{i}"
  model_name += "_w"
  for i in shuffle:
    model_name += f"_{i}"

  def make_model(float_mode: TensorProto) -> ModelProto:
    assembler = ModelAssembler(model_name)
    p0 = assembler.add_input("p0", float_mode, dims)
    assert all(map(lambda e: e[0] == e[1], enumerate(sorted(shuffle)))), "Invalid shuffle provided"
    p0t = assembler.transpose("p0_transposed", p0, shuffle)
    assembler.add_output(p0t)
    return assembler.model()

  return make_model, model_name



class BertTransposeKind(Enum):
  QV = 0
  KT = 1
  OUT = 2
  def __str__(self):
    return self.name

# create a model with a transpose of a 2d matrix with the provided dims
def bert_reshape_transpose_model(batch: int, seqLen: int, heads: int, hidden: int, model_mode : BertTransposeKind) -> Tuple[Callable[[TensorProto], ModelProto], str]:
  model_name = f"bert_reshape_transpose_{batch}x{seqLen}x{heads}x{hidden}_{model_mode}"

  def make_model_QV(float_mode: TensorProto) -> ModelProto:
    assembler = ModelAssembler(model_name)
    p0 = assembler.add_input("p0", float_mode, [batch, seqLen, heads * hidden])
    p0r = assembler.reshape("p0_reshape", p0, [batch, seqLen, heads, hidden])
    p0t = assembler.transpose("p0_transposed", p0r, [0, 2, 1, 3])
    assembler.add_output(p0t)
    return assembler.model()
    
  def make_model_KT(float_mode: TensorProto) -> ModelProto:
    assembler = ModelAssembler(model_name)
    p0 = assembler.add_input("p0", float_mode, [batch, seqLen, heads * hidden])
    p0r = assembler.reshape("p0_reshape", p0, [batch, seqLen, heads, hidden])
    p0t = assembler.transpose("p0_transposed", p0r, [0, 2, 3, 1])
    assembler.add_output(p0t)
    return assembler.model()
    
  def make_model_OUT(float_mode: TensorProto) -> ModelProto:
    assembler = ModelAssembler(model_name)
    p0 = assembler.add_input("p0", float_mode, [batch, heads, seqLen, hidden])
    p0t = assembler.transpose("p0_transposed", p0, [0, 2, 1, 3])
    p0r = assembler.reshape("p0_reshape", p0t, [batch, seqLen, heads * hidden])
    assembler.add_output(p0r)
    return assembler.model()

  if model_mode == BertTransposeKind.QV:
    return make_model_QV, model_name
  if model_mode == BertTransposeKind.KT:
    return make_model_KT, model_name
  if model_mode == BertTransposeKind.OUT:
    return make_model_OUT, model_name
  raise "unexpected model"

##
## Actual tests go here
##

def generate_tests():

  # basic 2d

  # model_gen, name = matrix_transpose_model([128, 128])
  # add_test(model_gen, name)

  # model_gen, name = matrix_transpose_model([128, 256])
  # add_test(model_gen, name)

  # model_gen, name = matrix_transpose_model([256, 128])
  # add_test(model_gen, name)

  # model_gen, name = matrix_transpose_model([1024, 256])
  # add_test(model_gen, name)


  # complex n-d

  # model_gen, name = complex_matrix_transpose_model([2, 128, 128], [0, 2, 1])
  # add_test(model_gen, name)

  # model_gen, name = complex_matrix_transpose_model([128, 128, 128], [2, 1, 0])
  # add_test(model_gen, name)

  # model_gen, name = complex_matrix_transpose_model([128, 128, 128], [2, 0, 1])
  # add_test(model_gen, name)

  # model_gen, name = complex_matrix_transpose_model([128, 128, 128], [1, 0, 2])
  # add_test(model_gen, name)


 # Bert Subgraph tests

  model_gen, name = bert_reshape_transpose_model(1, 512, 12, 256, BertTransposeKind.QV)
  add_test(model_gen, name, test_type=TestType.EXECUTE)
  
  model_gen, name = bert_reshape_transpose_model(1, 512, 12, 256, BertTransposeKind.KT)
  add_test(model_gen, name, test_type=TestType.EXECUTE)
  
  model_gen, name = bert_reshape_transpose_model(1, 512, 12, 256, BertTransposeKind.OUT)
  add_test(model_gen, name, test_type=TestType.EXECUTE)
