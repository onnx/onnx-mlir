#
# Wrapper around ONNX API to match the model generator code from Brainwave
#

from functools import reduce
from operator import mul
import struct
from enum import Enum

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, OperatorSetIdProto, ValueInfoProto, NodeProto, TypeProto
import numpy as np
from onnx import numpy_helper
from onnx.onnx_ml_pb2 import ModelProto


class TensorDataSource(Enum):
  MODEL_INPUT = 0
  RANDOM = 1
  PROVIDED = 2


class ModelAssembler:
  def add_opset(self, name: str, ver: int):
      opset = OperatorSetIdProto()
      opset.domain = name
      opset.version = ver
      self.required_opsets.append(opset)

  def __init__(self, name: str, op_name_prefix: str = ""):
    self.name = name
    self.graph_nodes: list = []
    self.inputs: list = []
    self.outputs: list = []
    self.graph_initializers: list = []
    self.op_name_prefix = op_name_prefix
    self.required_opsets = []
    self.add_opset('', 13) #default opset

  def format_name(self, name):
    return name if self.op_name_prefix == "" else f"{self.op_name_prefix}_{name}"

  @staticmethod
  def compute_unary_op_type(input_type):
    return input_type.elem_type, [x.dim_value for x in input_type.shape.dim]

  @staticmethod
  def compute_binary_op_type(left_type, right_type):
    if left_type == right_type:
      return ModelAssembler.compute_unary_op_type(left_type)

    assert left_type.elem_type == right_type.elem_type

    left_dims = [x.dim_value for x in left_type.shape.dim]
    right_dims = [x.dim_value for x in right_type.shape.dim]

    largeInput, otherInput = left_dims, right_dims
    if len(left_dims) < len(right_dims):
      largeInput, otherInput = right_dims, left_dims

    def unify(x: int, y: int):
      if x == y or y == 1:
        return x
      assert x == 1, f"Expected broadcast for non-matching sizes (got {x} for {left_dims} * {right_dims})"
      return y

    # Prepend appropraite number of 1s to front of smaller input and unify.
    diff = len(largeInput) - len(otherInput)
    dims = [unify(z[0], z[1]) for z in zip(largeInput, [1 for _ in range(diff)] + otherInput)]
    return left_type.elem_type, dims

  @staticmethod
  def compute_matmul_type(leftType, rightType):
    assert leftType.elem_type == rightType.elem_type

    leftDims = [x.dim_value for x in (leftType.shape.dim)]
    rightDims = [x.dim_value for x in (rightType.shape.dim)]

    if len(leftDims) == 1:
      leftDims.insert(0, 1)

    if len(rightDims) == 1:
      rightDims.append(1)

    assert len(leftDims) >= 2 and len(rightDims) >= 2 and leftDims[-1] == rightDims[-2], f"Invalid input {leftDims} , {rightDims}"

    # magic one-liner to extract the batch dimensions
    output_dims: "list[int]" = sorted([x[:-2] for x in [leftDims, rightDims]], key=lambda x: len(list(filter(lambda d: d != 1, x))))[0]

    if len(leftDims) >= 2:
      output_dims.append(leftDims[-2])

    if len(rightDims) >= 2:
      output_dims.append(rightDims[-1])

    return leftType.elem_type, output_dims

  def add_input(self, name: str, floatType: TensorProto, shape: "list[int]") -> ValueInfoProto:
    input = helper.make_tensor_value_info(name, floatType, shape)
    self.inputs.append(input)
    return input

  def add_output(self, output: ValueInfoProto):
    self.outputs.append(output)

  # generate a weight tensor in one of three ways
  #   using random data
  #   using the provided data
  #   as an input to the model
  def make_weights(self, name: str, shape: "list[int]", tensor_data_source: TensorDataSource, float_mode: TensorProto, value: "list[float]" = None) -> ValueInfoProto:
    if tensor_data_source == TensorDataSource.MODEL_INPUT:
      return self.add_input(name, float_mode, shape)
    if tensor_data_source == TensorDataSource.PROVIDED:
      return self.const(name, value, float_mode, shape)
    if tensor_data_source == TensorDataSource.RANDOM:
      raise "NYI: random"
    raise f"NYI: datasource {tensor_data_source}"

  # Create a constant node
  def const(self, name: str, value, elem_type: TensorProto, dims: "list[int]" = None) -> ValueInfoProto:
    if dims is None:
      if type(value) is list:
        dims = [len(value)]
      else:
        dims = [1]
        value = [value]
    elif type(value) is not list:
      # broadcast the constant if the tensorType is null
      value = [value for _ in range(reduce(mul, dims))]

    if elem_type == TensorProto.BFLOAT16:
      dims = []
      value = [int.from_bytes(struct.pack("f", value[0]), "little") >> 16]

    initializer = helper.make_tensor(name, elem_type, dims, value)
    self.graph_initializers.append(initializer)
    return helper.make_tensor_value_info(name, elem_type, dims)

  def unsqueeze(self, name: str, axes: "list[int]|int", input: ValueInfoProto) -> ValueInfoProto:
    if type(axes) is int:
      axes = [axes]
    assert len(axes) == 1
    assert axes[0] >= 0 and axes[0] < len(input.type.tensor_type.shape.dim)

    output_elem_type, input_dims = ModelAssembler.compute_unary_op_type(input.type.tensor_type)

    output_dims = []
    for i in range(0, len(input_dims)):
      if i == axes[0]:
        output_dims.append(1)
      output_dims.append(input_dims[i])

    axis_input = self.const(f'{name}_axis', axes, TensorProto.INT64, [len(axes)])

    name = self.format_name(name)
    output = helper.make_tensor_value_info(f'{name}_output', output_elem_type, output_dims)
    node = helper.make_node('Unsqueeze', [input.name, axis_input.name], [output.name], name=name)
    self.graph_nodes.append(node)
    return output

  def add(self, name: str, input1: ValueInfoProto, input2: ValueInfoProto) -> ValueInfoProto:
    name = self.format_name(name)
    output_elem_type, output_dims = ModelAssembler.compute_binary_op_type(input1.type.tensor_type, input2.type.tensor_type)
    output = helper.make_tensor_value_info(f'{name}_output', output_elem_type, output_dims)
    node = helper.make_node('Add', [input1.name, input2.name], [output.name], name=name)
    self.graph_nodes.append(node)
    return output

  def sub(self, name: str, input1: ValueInfoProto, input2: ValueInfoProto) -> ValueInfoProto:
    name = self.format_name(name)
    output_elem_type, output_dims = ModelAssembler.compute_binary_op_type(input1.type.tensor_type, input2.type.tensor_type)
    output = helper.make_tensor_value_info(f'{name}_output', output_elem_type, output_dims)
    node = helper.make_node('Sub', [input1.name, input2.name], [output.name], name=name)
    self.graph_nodes.append(node)
    return output

  def mul(self, name: str, input1: ValueInfoProto, input2: ValueInfoProto) -> ValueInfoProto:
    name = self.format_name(name)
    output_elem_type, output_dims = ModelAssembler.compute_binary_op_type(input1.type.tensor_type, input2.type.tensor_type)
    output = helper.make_tensor_value_info(f'{name}_output', output_elem_type, output_dims)
    node = helper.make_node('Mul', [input1.name, input2.name], [output.name], name=name)
    self.graph_nodes.append(node)
    return output

  def div(self, name: str, input1: ValueInfoProto, input2: ValueInfoProto) -> ValueInfoProto:
    name = self.format_name(name)
    output_elem_type, output_dims = ModelAssembler.compute_binary_op_type(input1.type.tensor_type, input2.type.tensor_type)
    output = helper.make_tensor_value_info(f'{name}_output', output_elem_type, output_dims)
    node = helper.make_node('Div', [input1.name, input2.name], [output.name], name=name)
    self.graph_nodes.append(node)
    return output

  def matmul(self, name: str, input1: ValueInfoProto, input2: ValueInfoProto) -> ValueInfoProto:
    name = self.format_name(name)
    output_elem_type, output_dims = ModelAssembler.compute_matmul_type(input1.type.tensor_type, input2.type.tensor_type)
    output = helper.make_tensor_value_info(f'{name}_output', output_elem_type, output_dims)
    node = helper.make_node('MatMul', [input1.name, input2.name], [output.name], name=name)
    self.graph_nodes.append(node)
    return output

  # Example
  #   %0 = "onnx.Constant"() {value = dense<[5, 5, 16, 2]> : tensor<4xi64> } : () -> tensor<4xi64>
  #   %1 = "onnx.Reshape"(%arg0, %0) : (tensor<5x5x1x32xf32>, tensor<4xi64>) -> tensor<*xf32>
  def reshape(self, name: str, input: ValueInfoProto, shape: "list[int]") -> ValueInfoProto:
    shape_const = self.const(f'{name}_shape', shape, TensorProto.INT64)
    name = self.format_name(name)
    output_elem_type, input_dims = ModelAssembler.compute_unary_op_type(input.type.tensor_type)
    output_dims = [shape[i] if shape[i] != 0 else input_dims[i] for i in range(0, len(shape))]
    output = helper.make_tensor_value_info(f'{name}_output', output_elem_type, output_dims)
    node = helper.make_node('Reshape', [input.name, shape_const.name], [output.name], name=name)
    self.graph_nodes.append(node)
    return output

  # Example:
  #   %27 = "onnx.Transpose"(%26) {onnx_node_name = "Transpose_66", perm = [0, 2, 3, 1]} : (tensor<?x?x2x64xf32>) -> tensor<?x2x64x?xf32>
  def transpose(self, name: str, input: ValueInfoProto, perm: "list[int]") -> ValueInfoProto:
    name = self.format_name(name)
    output_elem_type, dims = ModelAssembler.compute_unary_op_type(input.type.tensor_type)
    output_dims = [dims[i] for i in perm]
    output = helper.make_tensor_value_info(f'{name}_output', output_elem_type, output_dims)
    node = helper.make_node('Transpose', [input.name], [output.name], name=name, perm=perm)
    self.graph_nodes.append(node)
    return output

  # Example:
  #   %45 = "onnx.Softmax"(%44) {axis = 3 : si64, onnx_node_name = "Softmax_86"} : (tensor<?x2x?x?xf32>) -> tensor<?x2x?x?xf32>

  def softmax(self, name: str, input: ValueInfoProto, axis: int) -> ValueInfoProto:
    name = self.format_name(name)
    output_elem_type, output_dims = ModelAssembler.compute_unary_op_type(input.type.tensor_type)
    output = helper.make_tensor_value_info(f'{name}_output', output_elem_type, output_dims)
    node = helper.make_node('Softmax', [input.name], [output.name], name=name, axis=axis)
    self.graph_nodes.append(node)
    return output

  # Example:
  #   %out_28, %saved_mean_29, %saved_inv_std_var_30 = "onnx.LayerNormalization"(%157, %158, %159) {axis = -1 : i64, epsilon = 9.99999996E-13 : f32, onnx_node_name = "LayerNormalization_12"} : (tensor<?x?x128xf32>, tensor<128xf32>, tensor<128xf32>) -> (tensor<?x?x128xf32>, tensor<?x?x128xf32>, tensor<?x128xf32>)

  def layernorm(self, name: str, input: ValueInfoProto, epsilon: float, gamma: float, bias: float) -> ValueInfoProto:

    output_elem_type, output_dims = ModelAssembler.compute_unary_op_type(input.type.tensor_type)
    other_dims = [output_dims[-1]]

    axis = -1
    gammaTensor = self.const(f"{name}_gamma", gamma, output_elem_type, other_dims)
    biasTensor = self.const(f"{name}_bias", bias, output_elem_type, other_dims)

    name = self.format_name(name)
    output = helper.make_tensor_value_info(f'{name}_output', output_elem_type, output_dims)
    output2 = helper.make_tensor_value_info(f'{name}_saved_mean', output_elem_type, output_dims)
    output3 = helper.make_tensor_value_info(f'{name}_saved_inv_std_var', output_elem_type, other_dims)
    node = helper.make_node('LayerNormalization', [input.name, gammaTensor.name, biasTensor.name], [output.name, output2.name, output3.name], name=name, axis=axis, epsilon=epsilon)
    self.graph_nodes.append(node)
    return output

  def reduce_mean(self, name: str, input: ValueInfoProto, keep_dims: bool, axes: "list[int]|int") -> ValueInfoProto:

    if type(axes) is int:
      axes = [axes]

    output_elem_type, input_dims = ModelAssembler.compute_unary_op_type(input.type.tensor_type)

    output_dims = []
    for i in range(0, len(input_dims)):
      if i in axes:
        if keep_dims:
          output_dims.append(1)
      else:
        output_dims.append(input_dims[i])

    if len(output_dims) == 0:
      output_dims = [1]

    name = self.format_name(name)
    output = helper.make_tensor_value_info(f'{name}_output', output_elem_type, output_dims)
    node = helper.make_node('ReduceMean', [input.name], [output.name], name=name, keepdims=keep_dims, axes=axes)
    self.graph_nodes.append(node)
    return output

  def pow(self, name: str, input: ValueInfoProto, exponent: ValueInfoProto) -> ValueInfoProto:
    assert type(input) == type(exponent)
    output_elem_type, output_dims = ModelAssembler.compute_unary_op_type(input.type.tensor_type)
    name = self.format_name(name)
    output = helper.make_tensor_value_info(f'{name}_output', output_elem_type, output_dims)
    node = helper.make_node('Pow', [input.name, exponent.name], [output.name], name=name)
    self.graph_nodes.append(node)
    return output

  def sqrt(self, name: str, input: ValueInfoProto) -> ValueInfoProto:
    output_elem_type, output_dims = ModelAssembler.compute_unary_op_type(input.type.tensor_type)
    name = self.format_name(name)
    output = helper.make_tensor_value_info(f'{name}_output', output_elem_type, output_dims)
    node = helper.make_node('Sqrt', [input.name], [output.name], name=name)
    self.graph_nodes.append(node)
    return output

  # Example:
  #   %164 = "onnx.Gelu"(%163) {onnx_node_name = "Gelu_2"} : (tensor<?x?x3072xf32>) -> tensor<?x?x3072xf32>

  def gelu(self, name: str, input: ValueInfoProto) -> ValueInfoProto:
    name = self.format_name(name)
    output_elem_type, output_dims = ModelAssembler.compute_unary_op_type(input.type.tensor_type)
    output = helper.make_tensor_value_info(f'{name}_output', output_elem_type, output_dims)
    node = helper.make_node('Gelu', [input.name], [output.name], name=name, domain="com.microsoft")
    self.add_opset('com.microsoft', 1)
    self.graph_nodes.append(node)
    return output

  def model(self) -> ModelProto:

    # build graph and model
    graph = helper.make_graph(self.graph_nodes, self.name, self.inputs, self.outputs, initializer=self.graph_initializers)
    model = helper.make_model(graph, opset_imports=self.required_opsets)

    ## LayerNormalization and Gelu not supported in ONNX checker
    # onnx.checker.check_model(model)

    return onnx.shape_inference.infer_shapes(model)
