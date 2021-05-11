#
# This script builds test bfloat16 MNIST-like model padded for Apollo architecture.
# It is based on example provided by Cheng Tang.
#
# - All dimension sizes are a multiple of 256 (to satisfy Apollo ISA limitations).
# - All dimensions are static (to eliminate need for padding at runtime).
# - All floating point data is in bfloat16 format (since Apollo ISA does not support float32 math).
# - All tensor constants became graph input arguments (no constant tensors inside model).
#

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, OperatorSetIdProto
import numpy as np
from onnx import numpy_helper

import sys

def create_mnist_layer(id, input_tensor, output_dim, create_relu = False):
    input_dims = input_tensor.type.tensor_type.shape.dim
    batch_dim = input_dims[0].dim_value
    input_dim = input_dims[1].dim_value
    # matmul node
    matmul_output = helper.make_tensor_value_info('matmul_%d_output' % id,
                                                  input_tensor.type.tensor_type.elem_type,
                                                  [batch_dim, output_dim])
    weight = helper.make_tensor_value_info('weight_%d' % id, TensorProto.BFLOAT16, [input_dim, output_dim])
    matmul_node = helper.make_node('MatMul', [input_tensor.name, weight.name], [matmul_output.name])
    #add node
    add_output = helper.make_tensor_value_info('add_%d_output' % id,
                                               input_tensor.type.tensor_type.elem_type,
                                               [batch_dim,output_dim])

    bias = helper.make_tensor_value_info('bias_%d' % id, TensorProto.BFLOAT16, [output_dim, output_dim])
    add_node = helper.make_node('Add', [matmul_output.name, bias.name], [add_output.name])

    if create_relu:
        relu_output = helper.make_tensor_value_info('relu_%d_output' % id,
                                                  input_tensor.type.tensor_type.elem_type,
                                                  [batch_dim, output_dim])
        relu_node = helper.make_node('Relu', [add_output.name], [relu_output.name])
        # nodes, values, initializers, outputs
        return [matmul_node, add_node, relu_node], [matmul_output, add_output], [weight, bias], relu_output
    else:
        # nodes, values, initializers, outputs
        return [matmul_node, add_node], [matmul_output], [weight, bias], add_output

def create_mnist_model(layer_dims, use_bfloat16=True):
    assert(len(layer_dims) > 0)
    #create input:
    X = helper.make_tensor_value_info('X', TensorProto.BFLOAT16, [256, 256])
    i = 0
    graph_nodes = []
    value_infos = []
    graph_initializers = []
    outputs = []
    inputs = [X]
    temp_input = X
    #create mul
    mul_output = helper.make_tensor_value_info('mul_output',
                                               temp_input.type.tensor_type.elem_type,
                                               [256, 256])
    #this constant will become a scalar constant, so keep it                                           
    mul_value_np = np.array([1/255], dtype=np.float32).reshape(())
    mul_value = numpy_helper.from_array(mul_value_np)
    mul_value.data_type = TensorProto.BFLOAT16
    mul_value.name = 'mul_operand'
    mul_node = helper.make_node('Mul', [temp_input.name, mul_value.name], [mul_output.name])
    graph_nodes.append(mul_node)
    graph_initializers.append(mul_value)
    value_infos.append((mul_output))
    temp_input = mul_output

    for dim in layer_dims:
        nodes, value, initializers, output = create_mnist_layer(i, temp_input, dim, i < (len(layer_dims) - 1))
        graph_nodes.extend(nodes)
        value_infos.extend(value)
        inputs.extend(initializers)
        if i == (len(layer_dims) - 1):
            outputs.append(output)
        else:
            value_infos.append(output)
        temp_input = output
        i += 1

    #build graph and model
    graph = helper.make_graph(graph_nodes, 'mnist-model', inputs, outputs, initializer=graph_initializers, value_info=value_infos)
    opset_id_proto = OperatorSetIdProto()
    opset_id_proto.domain = 'ai.onnx'
    opset_id_proto.version = 13
    model = helper.make_model(graph, opset_imports=[opset_id_proto])
    return model

model = create_mnist_model([256, 256], use_bfloat16=True)
onnx.save(model, 'mnist_padded_static_noconsts_2d_bias.onnx')