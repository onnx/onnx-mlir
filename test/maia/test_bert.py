

from model_generators.assembler import TensorDataSource
from model_generators.bert import create_bert
from utility import *

from typing import Tuple, Callable

import onnx
from onnx import TensorProto, ModelProto


##
## Helper functions for building models
##

# generate model and name given the parameters
def generate_bert_model(batch_size: int, layers: int, seqlen: int, heads: int, head_size: int, use_layernorm_op: bool) -> Tuple[Callable[[TensorProto], ModelProto], str]:
  
  model_name = f"bert.layers_{layers}_batch_{batch_size}_seqlen_{seqlen}_heads_{heads}_headsize_{head_size}{'' if use_layernorm_op else '_layernorm_off'}"

  def make_model(float_mode: TensorProto) -> ModelProto:
    return create_bert(batch_size=batch_size, seqlen=seqlen, layers=layers, heads=heads, head_size=head_size, use_layernorm_op=use_layernorm_op, tensor_data_source=TensorDataSource.MODEL_INPUT, float_mode=float_mode)

  return make_model, model_name

##
## Actual tests go here
##

def generate_tests():

  # w/ layernorm op
  model_gen, name = generate_bert_model(batch_size=1, layers=1, seqlen=512, heads=12, head_size=256, use_layernorm_op=True)
  add_test(model_gen, name)
  model_gen, name = generate_bert_model(batch_size=1, layers=1, seqlen=32, heads=12, head_size=64, use_layernorm_op=True)
  add_test(model_gen, name)

  # w/o layernorm op
  model_gen, name = generate_bert_model(batch_size=1, layers=1, seqlen=512, heads=12, head_size=256, use_layernorm_op=False)
  add_test(model_gen, name)
  model_gen, name = generate_bert_model(batch_size=1, layers=1, seqlen=32, heads=12, head_size=64, use_layernorm_op=False)
  add_test(model_gen, name)
