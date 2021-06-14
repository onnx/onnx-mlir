#
# This script builds test bfloat16/float32 MNIST-like model padded for Apollo architecture.
# It is based on example provided by Cheng Tang.
# 
# The script drops the output ONNX files into the local directory:
#  .\xxx.ONNX
#  .\xxx.f32.ONNX
#
# - All dimension sizes are a multiple of 256 (to satisfy Apollo ISA limitations).
# - All dimensions are static (to eliminate need for padding at runtime).
# - All tensor constants became graph input arguments (no constant tensors inside model).
# - Both f32 and bfloat16 graphs are created (bfloat16 for Apollo, f32 for OnnxRuntime)

import onnx
from onnx import helper
from onnx import AttributeProto, TensorProto, GraphProto, OperatorSetIdProto
import numpy as np
from onnx import numpy_helper

import sys


def create_mnist_layer(id, input_tensor, output_dim, create_relu=False, float_type=TensorProto.BFLOAT16):
    input_dims = input_tensor.type.tensor_type.shape.dim
    batch_dim = input_dims[0].dim_value
    input_dim = input_dims[1].dim_value
    # matmul node
    matmul_output = helper.make_tensor_value_info('matmul_%d_output' % id,
                                                  input_tensor.type.tensor_type.elem_type,
                                                  [batch_dim, output_dim])
    weight = helper.make_tensor_value_info('weight_%d' % id, float_type, [input_dim, output_dim])
    matmul_node = helper.make_node('MatMul', [input_tensor.name, weight.name], [matmul_output.name])
    #add node
    add_output = helper.make_tensor_value_info('add_%d_output' % id,
                                               input_tensor.type.tensor_type.elem_type,
                                               [batch_dim,output_dim])

    bias = helper.make_tensor_value_info('bias_%d' % id, float_type, [output_dim, output_dim])
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


def create_mnist_model(layer_dims, float_type=TensorProto.BFLOAT16):
    assert(len(layer_dims) > 0)
    #create input:
    X = helper.make_tensor_value_info('X', float_type, [256, 256])
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
    constant_value = 0.00390625 # 1/255 rounded to fit in bfloat16
    mul_value_np = np.array([constant_value], dtype=np.float32).reshape(())
    mul_value = numpy_helper.from_array(mul_value_np)
    mul_value.data_type = float_type
    mul_value.name = 'mul_operand'
    mul_node = helper.make_node('Mul', [temp_input.name, mul_value.name], [mul_output.name])
    graph_nodes.append(mul_node)
    graph_initializers.append(mul_value)
    value_infos.append((mul_output))
    temp_input = mul_output

    for dim in layer_dims:
        nodes, value, initializers, output = create_mnist_layer(i, temp_input, dim, i < (len(layer_dims) - 1), float_type=float_type)
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

if __name__ == "__main__":
    model = create_mnist_model([256, 256], TensorProto.BFLOAT16)
    onnx.save(model, 'mnist_padded_static_noconsts_2d_bias.onnx')

    model = create_mnist_model([256, 256], TensorProto.FLOAT)
    onnx.save(model, 'mnist_padded_static_noconsts_2d_bias.f32.onnx')
