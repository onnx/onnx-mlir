#
# Utitilities for building BERT models and portions of BERT models
#

from onnx.onnx_ml_pb2 import ModelProto
from model_generators.assembler import ModelAssembler, TensorDataSource

import onnx
from onnx import TensorProto, ValueInfoProto


def create_bert_multihead_attention(assembler: ModelAssembler, input: ValueInfoProto, mask: ValueInfoProto, split_shape: "list[int]", tensor_data_source: TensorDataSource, constValue: float = None, altNodeOrder=False, resInput=None, float_mode=TensorProto.FLOAT) -> ValueInfoProto:

  # COMMENT FROM ORIGINAL BRAINWAVE TEST:
  # if altNodeOrder is specified we create the left and right branches of the heads in a different order
  # which allows the integration tests to match the models that we see in the unit tests
  # (somehow converting the model to ONNX and back changes the order)
  # keeping the legacy test order as the default to avoid having to update all of the tests
  # Since compiler behavior is dependent on node order, tests with both orders allow us to catch additional bugs

  size = input.type.tensor_type.shape.dim[-1].dim_value
  matmul_shape = [size, size]
  bias_shape = [size]
  merged_shape = [0, 0, size]

  def create_v_branch():
    matmul0_const = assembler.make_weights("m0", matmul_shape, tensor_data_source, float_mode)
    bias0_const = assembler.make_weights("b0", bias_shape, tensor_data_source, float_mode)
    matmul0 = assembler.matmul("MatMul0", input, matmul0_const)
    bias0 = assembler.add("bias0", matmul0, bias0_const)
    reshape0 = assembler.reshape("r0", bias0, split_shape)
    return assembler.transpose("t0", reshape0, [0, 2, 1, 3])

  # left branch
  transpose0 = None
  if (not altNodeOrder):
    transpose0 = create_v_branch()

  # middle branch
  matmul1_const = assembler.make_weights("m1", matmul_shape, tensor_data_source, float_mode)
  bias1_const = assembler.make_weights("b1", bias_shape, tensor_data_source, float_mode)
  matmul1 = assembler.matmul("MatMul1", input, matmul1_const)
  bias1 = assembler.add("bias1", matmul1, bias1_const)
  reshape1 = assembler.reshape("r1", bias1, split_shape)
  transpose1 = assembler.transpose("t1", reshape1, [0, 2, 1, 3])

  # right branch
  matmul2_const = assembler.make_weights("m2", matmul_shape, tensor_data_source, float_mode)
  bias2_const = assembler.make_weights("b2", bias_shape, tensor_data_source, float_mode)
  matmul2 = assembler.matmul("MatMul2", input, matmul2_const)
  bias2 = assembler.add("bias2", matmul2, bias2_const)
  reshape2 = assembler.reshape("r2", bias2, split_shape)
  transpose2 = assembler.transpose("t2", reshape2, [0, 2, 3, 1])

  matmul3 = assembler.matmul("matmul3", transpose1, transpose2)
  div = assembler.div("div", matmul3, assembler.const("divc", 8.0, float_mode))
  masked = assembler.add("masked", div, mask)

  sm = assembler.softmax("sm", masked, 3)

  if altNodeOrder:
    transpose0 = create_v_branch()

  # note the order of the matmul operands - these are often shown incorrectly in NetScope
  matmul4 = assembler.matmul("matmul4", sm, transpose0)

  # This is the end of the head
  matmul5_const = assembler.make_weights("m5", matmul_shape, tensor_data_source, float_mode)
  bias3_const = assembler.make_weights("b3", bias_shape, tensor_data_source, float_mode)
  transpose3 = assembler.transpose("t3", matmul4, [0, 2, 1, 3])
  reshape3 = assembler.reshape("r3", transpose3, merged_shape)
  matmul5 = assembler.matmul("matmul5", reshape3, matmul5_const)
  bias3 = assembler.add("bias3", matmul5, bias3_const)

  residual = assembler.add("res", resInput if resInput != None else input, bias3)
  return residual


def create_bert_layernorm(assembler: ModelAssembler, input: ValueInfoProto, suffix: int, constValue: float = None, use_layernorm_op=False, float_mode=TensorProto.FLOAT) -> ValueInfoProto:
  eps = 0.000009999999747378752
  gamma = 0.0005 if constValue == None else constValue
  bias = 0.0006 if constValue == None else constValue

  if use_layernorm_op:
    return assembler.layernorm(f"layernorm_{suffix}", input, eps, gamma, bias)

  layernorm_prefix = f"layernorm_{suffix}"

  mean = assembler.reduce_mean(f"{layernorm_prefix}_mean", input, True, 2)
  sub = assembler.sub(f"{layernorm_prefix}_sub", input, mean)

  if (float_mode == TensorProto.BFLOAT16):
    # Pow support for bf16 is limited in onnx-mlir
    pow = assembler.mul(f"{layernorm_prefix}_pow", sub, sub)
  else: 
    pow_const = assembler.const(f"{layernorm_prefix}_pow_const", 2, float_mode)
    pow = assembler.pow(f"{layernorm_prefix}_pow", sub, pow_const)

  powmean = assembler.reduce_mean(f"{layernorm_prefix}_pow_mean", pow, True, 2)
  poweps = assembler.add(f"{layernorm_prefix}_pow_mean_eps", powmean, assembler.const(f"{layernorm_prefix}_pow_mean_eps_const", eps, float_mode))
  sqrt = assembler.sqrt(f"{layernorm_prefix}_sqrt", poweps)
  div = assembler.div(f"{layernorm_prefix}_div", sub, sqrt)

  _, output_dims = ModelAssembler.compute_unary_op_type(input.type.tensor_type)
  bias_shape = [output_dims[-1]]

  mul_const = assembler.const(f"{layernorm_prefix}_mul_const", gamma, float_mode, bias_shape)
  mul = assembler.mul(f"{layernorm_prefix}_mul", div, mul_const)

  bias_const = assembler.const(f"{layernorm_prefix}_bias_const", bias, float_mode, bias_shape)
  bias = assembler.add(f"{layernorm_prefix}_bias", mul, bias_const)

  return bias


# FF layer
# MLP -> GELU -> MLP
def create_bert_feedforward(assembler: ModelAssembler, input: ValueInfoProto, ffratio: int, tensor_data_source: TensorDataSource, constValueP: float = None, resInput=None, float_mode=TensorProto.FLOAT) -> ValueInfoProto:

  constValue = constValueP if constValueP != None else 0.0001
  elem_type, dims = ModelAssembler.compute_unary_op_type(input.type.tensor_type)
  assert elem_type == float_mode

  size = dims[-1]
  largeSize = size * ffratio

  elements = [constValue * i for i in range(0, size * largeSize)]
  mlp0_matmul_const = assembler.make_weights("ff_mlp0_matmul_const", [size, largeSize], tensor_data_source, float_mode, elements)
  mlp0_matmul = assembler.matmul("ff_mlp0_matmul", input, mlp0_matmul_const)

  mlp0_bias_const = assembler.make_weights("ff_mlp0_bias_const", [largeSize], tensor_data_source, float_mode, elements[-largeSize])
  mlp0_bias = assembler.add("ff_mlp0_bias", mlp0_matmul, mlp0_bias_const)

  gelu = assembler.gelu("ff_gelu", mlp0_bias)

  mlp1_matmul_const = assembler.make_weights("ff_mlp1_matmul_const", [largeSize, size], tensor_data_source, float_mode, elements)
  mlp1_matmul = assembler.matmul("ff_mlp1_matmul", gelu, mlp1_matmul_const)

  mlp1_bias_const = assembler.make_weights("ff_mlp1_bias_const", [size], tensor_data_source, float_mode, elements[-size])

  mlp1_bias = assembler.add("ff_mlp1_bias", mlp1_matmul, mlp1_bias_const)
  last_add = assembler.add("ff_mlp1_bias_last", mlp1_bias, resInput if resInput != None else input)

  return last_add


def create_bert_layer(assembler: ModelAssembler, input: ValueInfoProto, mask: ValueInfoProto, split_shape: "list[int]", ffratio: int, tensor_data_source: TensorDataSource, constValue: float = None, altNodeOrder=False, use_layernorm_op=False, float_mode=TensorProto.FLOAT) -> ValueInfoProto:
  attention = create_bert_multihead_attention(assembler, input, mask, split_shape, tensor_data_source, constValue, altNodeOrder, float_mode=float_mode)
  normed = create_bert_layernorm(assembler, attention, 0, constValue, use_layernorm_op, float_mode=float_mode)
  ff = create_bert_feedforward(assembler, normed, ffratio, tensor_data_source, constValue, float_mode=float_mode)
  normed_1 = create_bert_layernorm(assembler, ff, 1, constValue, use_layernorm_op, float_mode=float_mode)
  return normed_1


def create_bert(batch_size: int, seqlen: int, layers: int, heads: int, head_size: int, ffratio=4, tensor_data_source=TensorDataSource.MODEL_INPUT, sameConstValue=None, altNodeOrder=False, use_layernorm_op=False, float_mode=TensorProto.FLOAT) -> ModelProto:
  assembler = ModelAssembler(f"bert-model-{layers}_layers")

  inputsize = heads * head_size
  in1Shape = [batch_size, seqlen, inputsize]
  in2Shape = [batch_size, seqlen]
  split_shape = [0, 0, heads, head_size]

  p0 = assembler.add_input('P0', float_mode, in1Shape)

  ###
  #
  # disable int->fp conversion due to Apollo limitations
  #
  # ORIG_CODE
  #  p1 = assembler.add_input("P1", TensorProto.INT, in2Shape)
  #  u0 = assembler.unsqueeze("u0", 1, [p1]);
  #  u1 = assembler.unsqueeze("u1", 2, [u0]);
  #  cast = assembler.cast("cast0", u1, floatType);
  ###
  p1 = assembler.add_input("P1", float_mode, in2Shape)
  u0 = assembler.unsqueeze("u0", 1, p1)
  u1 = assembler.unsqueeze("u1", 2, u0)
  cast = u1

  sub1 = assembler.sub("sub0", assembler.const("sub_c0", 1.0, float_mode), cast)
  mul1 = assembler.mul("mul0", sub1, assembler.const("mul_c0", -10000.0, float_mode))

  cur_output = p0
  for i in range(0, layers):
    assembler.name_prefix = f"bert_{i}"
    cur_output = create_bert_layer(assembler, cur_output, mul1, split_shape, ffratio, tensor_data_source, sameConstValue, altNodeOrder, use_layernorm_op, float_mode)

  assembler.add_output(cur_output)

  return assembler.model()
