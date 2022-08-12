#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict, OrderedDict
from io import StringIO
import io
import os
import sys
import datetime
import argparse

import numpy as np  # type: ignore

from onnx import defs, FunctionProto, helper, OperatorStatus
from onnx.defs import OpSchema, ONNX_DOMAIN, ONNX_ML_DOMAIN
from onnx.backend.test.case import collect_snippets
from onnx.backend.sample.ops import collect_sample_implementations
from typing import Any, Text, Sequence, Dict, List, Type, Set, Tuple

import pprint
import onnx

# change this variable only when upgrading the ONNX support within ONNX-MLIR
current_onnx_version = "1.9.0"

parser = argparse.ArgumentParser()
parser.add_argument("--dry-run-onnx-ops",
                    help="Output ONNXOps.td.inc content to stdout.",
                    action="store_true",
                    default=False)
parser.add_argument("--dry-run-op-build-table",
                    help="Output OpBuildTable.inc content to stdout.",
                    action="store_true",
                    default=False)
parser.add_argument("--check-operation-version",
                    help="check whether the imported onnx package has new operation or "
                         " newer version of operation compared with version stored in  version_dicts",
                    action="store_true",
                    default=False)
parser.add_argument("--list-operation-version",
                    help="list the version stored in  version_dicts without performing checks",
                    action="store_true",
                    default=False)

args = parser.parse_args()

check_operation_version = args.check_operation_version
list_operation_version = args.list_operation_version
current_onnx_version = "1.11.0"
# check the version of onnx package being used
if (not check_operation_version and not list_operation_version) and current_onnx_version != onnx.__version__ :
    print("version of expected onnx is {}, ".format(current_onnx_version)+
          "while onnx package being used is {}".format(onnx.__version__))
    quit()

# Record the version of each operation that is treated as the current version.
# To check whether the onnx package being used has newer version operation,
# run this script with --check-operation-version flag.
# Update this dictionary when a newer version is implemented
# TODO: how to keep the old version
 
version_dict = {
 'Abs': [13],
 'Acos': [7],
 'Acosh': [9],
 'Adagrad': [1],
 'Adam': [1],
 'Add': [14],
 'And': [7],
 'ArgMax': [13],
 'ArgMin': [13],
 'ArrayFeatureExtractor': [1],
 'Asin': [7],
 'Asinh': [9],
 'Atan': [7],
 'Atanh': [9],
 'AveragePool': [11],
 'BatchNormalization': [14],
 'Binarizer': [1],
 'BitShift': [11],
 'Cast': [13],
 'CastMap': [1],
 'CategoryMapper': [1],
 'Ceil': [13],
 'Celu': [12],
 'Clip': [13, 12, 11, 6],
 'Compress': [11],
 'Concat': [13],
 'ConcatFromSequence': [11],
 'Constant': [13],
 'ConstantOfShape': [9],
 'Conv': [11],
 'ConvInteger': [10],
 'ConvTranspose': [11],
 'Cos': [7],
 'Cosh': [9],
 'CumSum': [14],
 'DepthToSpace': [13],
 'DequantizeLinear': [13],
 'Det': [11],
 'DictVectorizer': [1],
 'Div': [14],
 'Dropout': [13],
 'DynamicQuantizeLinear': [11],
 'Einsum': [12],
 'Elu': [6],
 'Equal': [13],
 'Erf': [13],
 'Exp': [13],
 'Expand': [13],
 'EyeLike': [9],
 'FeatureVectorizer': [1],
 'Flatten': [13],
 'Floor': [13],
 'GRU': [14],
 'Gather': [13],
 'GatherElements': [13],
 'GatherND': [13],
 'Gemm': [13],
 'GlobalAveragePool': [1],
 'GlobalLpPool': [2],
 'GlobalMaxPool': [1],
 'Gradient': [1],
 'Greater': [13],
 'GreaterOrEqual': [16],
 'HardSigmoid': [6],
 'Hardmax': [13],
 'HardSwish': [14],
 'Identity': [16],
 'If': [16],
 'Imputer': [1],
 'InstanceNormalization': [6],
 'IsInf': [10],
 'IsNaN': [13],
 'LRN': [13],
 'LSTM': [14],
 'LabelEncoder': [2],
 'LeakyRelu': [16],
 'Less': [13],
 'LessOrEqual': [16],
 'LinearClassifier': [1],
 'LinearRegressor': [1],
 'Log': [13],
 'LogSoftmax': [13],
 'Loop': [16],
 'LpNormalization': [1],
 'LpPool': [11],
 'MatMul': [13],
 'MatMulInteger': [10],
 'Max': [13],
 'MaxPool': [12],
 'MaxRoiPool': [1],
 'MaxUnpool': [11],
 'Mean': [13],
 'MeanVarianceNormalization': [13],
 'Min': [13],
 'Mod': [13],
 'Momentum': [1],
 'Mul': [14],
 'Multinomial': [7],
 'Neg': [13],
 'NegativeLogLikelihoodLoss': [13],
 'NonMaxSuppression': [11],
 'NonZero': [13],
 'Normalizer': [1],
 'Not': [1],
 'OneHot': [11],
 'OneHotEncoder': [1],
 'Optional' : [15],
 'OptionalGetElement' : [15],
 'OptionalHasElement' : [15],
 'Or': [7],
 'PRelu': [16],
 'Pad': [13, 11, 2],
 'Pow': [15],
 'QLinearConv': [10],
 'QLinearMatMul': [10],
 'QuantizeLinear': [13],
 'RNN': [14],
 'RandomNormal': [1],
 'RandomNormalLike': [1],
 'RandomUniform': [1],
 'RandomUniformLike': [1],
 'Range': [11],
 'Reciprocal': [13],
 'ReduceL1': [13],
 'ReduceL2': [13],
 'ReduceLogSum': [13],
 'ReduceLogSumExp': [13],
 'ReduceMax': [13],
 'ReduceMean': [13],
 'ReduceMin': [13],
 'ReduceProd': [13],
 'ReduceSum': [13, 11],
 'ReduceSumSquare': [13],
 'Relu': [14],
 'Reshape': [14],
 'Resize': [13, 11, 10],
 'ReverseSequence': [10],
 'RoiAlign': [10],
 'Round': [11],
 'SVMClassifier': [1],
 'SVMRegressor': [1],
 'Scaler': [1],
 'Scan': [16],
 'Scatter': [11],
 'ScatterElements': [13],
 'ScatterND': [16],
 'Selu': [6],
 'SequenceAt': [11],
 'SequenceConstruct': [11],
 'SequenceEmpty': [11],
 'SequenceErase': [11],
 'SequenceInsert': [11],
 'SequenceLength': [11],
 'Shape': [13], # When going to 15, rewrite rules must also be changed for start/end
 'Shrink': [9],
 'Sigmoid': [13],
 'Sign': [13],
 'Sin': [7],
 'Sinh': [9],
 'Size': [13],
 'Slice': [13],
 'Softmax': [13],
 'SoftmaxCrossEntropyLoss': [13],
 'Softplus': [1],
 'Softsign': [1],
 'SpaceToDepth': [13],
 'Split': [13, 11],
 'SplitToSequence': [11],
 'Sqrt': [13],
 'Squeeze': [13, 11],
 'StringNormalizer': [10],
 'Sub': [14],
 'Sum': [13],
 'Tan': [7],
 'Tanh': [13],
 'TfIdfVectorizer': [9],
 'ThresholdedRelu': [10],
 'Tile': [13],
 'TopK': [11],
 'Transpose': [13],
 'Trilu': [14],
 'TreeEnsembleClassifier': [1],
 'TreeEnsembleRegressor': [1],
 'Unique': [11],
 'Unsqueeze': [13, 11],
 'Upsample': [10, 9, 7],
 'Where': [16],
 'Xor': [7],
 'ZipMap': [1]}

# Manual specification of attribute type.
special_attr_types = dict([("Cast.to", 'type')])

# Special operation importing handlers.
special_op_handler = dict([
    ("BatchNormalization", "ImportNodeBatchNormalization"),
    ("CategoryMapper", "ImportCategoryMapper"),
    ("Dropout", "ImportNodeDropout"),
    ("Cast", "ImportNodeCast"),
    ("MaxPool", "ImportNodeMaxPool"),
    ("Pad", "ImportNodePad"),
    ("Slice", "ImportNodeSlice"),
    ("Softmax", "ImportNodeSoftmax"),
    #("Transpose", "ImportNodeTranspose")
])

# Operations supporting canonicalization (alphabetical order).
OpsWithCanonicalizer = [
    'Add',
    'Cast',
    'Constant',
    'DepthToSpace',
    'Dropout',
    'GlobalAveragePool',
    'GlobalMaxPool',
    'Identity',
    'Less',
    'Loop',
    'Mul',
    'Reshape',
    'Shape',
    'Size',
    'SpaceToDepth',
    'Squeeze',
    'SqueezeV11',
    'Transpose',
    'Unsqueeze',
    'UnsqueezeV11',
]

# Operations with custom verifiers (alphabetical order).
OpsWithVerifier = [
    'AveragePool',
    'ArgMax',
    'ArgMin',
    'CategoryMapper',    
    'Compress',
    'Concat',
    'ConcatFromSequence',
    'ConstantOfShape',
    'Conv',
    'DepthToSpace',
    'DequantizeLinear',
    'Einsum',
    'Expand',
    'Flatten',
    'Gather',
    'GatherElements',
    'GatherND',        
    'Hardmax',
    'InstanceNormalization',
    'LogSoftmax',
    'Mod',
    'NonMaxSuppression',
    'OneHot',
    'OneHotEncoder',
    'Optional',
    'OptionalGetElement',
    'OptionalHasElement',
    'Pow',
    'RandomNormalLike',
    'ReverseSequence',
    "RoiAlign",
    "ScatterElements",
    'ScatterND',
    'SequenceEmpty',
    'SequenceInsert',
    'SpaceToDepth',
    'Split',
    'SplitToSequence',
    'TopK',
    'Unique'
]

OpsWithHelpers = {
  "Loop": """
    mlir::Operation::result_range v_final();
    mlir::Operation::result_range scan_outputs();
  """,
  "Scan": """
    mlir::Operation::operand_range v_initial();
    mlir::Operation::result_range v_final();
    mlir::Operation::operand_range scan_inputs();
    mlir::Operation::result_range scan_outputs();
  """
}
# Interface for special handling of type inference
# The common code are put into get_type_inference_func
OpsWithResultTypeInference = {
  "Constant":
  '''if (auto attr = valueAttr()) {
        resultTypes.push_back(attr.getType());
      } else if (auto attr = sparse_valueAttr()) {
        resultTypes.push_back(attr.getType());
      }''',
  "Cast":
    '''// ae auto builder = mlir::OpBuilder(getContext());
      resultTypes.push_back(mlir::UnrankedTensorType::get(to()));''',
  "ConstantOfShape":
  '''if (auto attr = valueAttr()) {
        resultTypes.push_back(mlir::UnrankedTensorType::get(
          attr.getType().cast<ShapedType>().getElementType()));
      } else {
        resultTypes.push_back(mlir::UnrankedTensorType::get(
          FloatType::getF32(getContext())));
      }'''
}

# Add an Op in this list if the Op needs result type deduction which is required
# when writing declarative rewriting rules. Deduced type is always
# an UnrankedTensorType whose element type is the same as the first operand's
# element type.
#
# Currently, there are only two build methods generated:
#  - one with operands and attributes having a separate parameter, and
#  - one with operands and attributes having aggregated parameters.
custom_builder_unranked_ops_list = [
    'Abs',
    'Exp',
    'Identity',
    'Neg',
    'Pad',
    'ReduceLogSum',
    'ReduceMax',
    'ReduceSum',
    'ReduceSumSquare',
    'ReduceSumV11',
    'Softmax',
    'Split',
    'Sqrt',
    'SqueezeV11',
    'UnsqueezeV11',
]
# Custom builder op list for operations with broadcast; we can deduce the right
# output type, no need to leave it undef as in the above list.
# Ops must have two operands, not one, not three... And there shall be two.
# TODO: handle variadic ops omitted here: Max, Min, Min, Sum.
custom_builder_broadcast_to_same_type_ops_list = [
    'Add',
    'And',
    'Div',
    'Mul',
    'Or',
    'Pow',
    'Sub',
    'Xor',
]
custom_builder_broadcast_to_bool_ops_list = [
    'Equal',
    'Greater',
    'Less',
]
custom_builder_broadcast_ops_list = custom_builder_broadcast_to_same_type_ops_list + \
    custom_builder_broadcast_to_bool_ops_list
# union of both
custom_builder_ops_list = custom_builder_unranked_ops_list + custom_builder_broadcast_ops_list

#a dictionary to add any special definition for an operation
custom_definition_misc = dict([ ('Constant',
 '''  let builders = [
  OpBuilder<(ins "Attribute":$sparse_value, "Attribute":$value), [{
   if (value) {
    auto tensorType = value.getType();
    build($_builder, $_state, tensorType, sparse_value, value,
      FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(), ArrayAttr());
   } else {
    auto tensorType = sparse_value.getType();
    build($_builder, $_state, tensorType, sparse_value, value,
      FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(), ArrayAttr());
   }
  }]>
  ];'''),
  ('Cast',
 '''   let builders = [
  OpBuilder<(ins "Value":$input, "TypeAttr":$to), [{
   auto resultType = mlir::UnrankedTensorType::get(to.getValue());
   build($_builder, $_state, resultType, input, to);
  }] >
  ];'''
 )])

onnx_types = (
    'bool', 'int8', 'int16', 'int32', 'int64', 'unkown', 'float16',
    'float', 'double', 'complex64', 'complex128', 'string'
)
tblgen_types = ('AnyI1', 'AnyI8', 'AnyI16', 'AnyI32', 'AnyI64',
    'BF16', 'F16', 'F32', 'F64', 'Complex<F32>', 'Complex<F64>',
    'StringType'
)

MAX_NUM_TYPES=20

def should_render_domain(domain):  # type: (Text) -> bool
    return True


def display_attr_type(v):  # type: (OpSchema.AttrType) -> Text
    assert isinstance(v, OpSchema.AttrType)
    s = Text(v)
    s = s[s.rfind('.') + 1:].lower()
    if s[-1] == 's':
        s = 'list of ' + s
    return s


def get_unique_output_name(schema, name):
    for input in schema.inputs:
        if input.name == name:
            return 'out_' + name
    return name


def onnx_attr_type_to_mlir_attr_type(t):
    onnx_attr_type = Text(t)
    onnx_attr_type = onnx_attr_type[onnx_attr_type.rfind('.') + 1:].lower()

    if onnx_attr_type == 'int':
        mlir_attr_type = 'SI64Attr'
    elif onnx_attr_type == 'float':
        mlir_attr_type = 'F32Attr'
    elif onnx_attr_type == 'ints':
        mlir_attr_type = 'I64ArrayAttr'
    elif onnx_attr_type == 'floats':
        mlir_attr_type = 'F32ArrayAttr'
    elif onnx_attr_type == "string":
        mlir_attr_type = 'StrAttr'
    elif onnx_attr_type == "strings":
        mlir_attr_type = 'StrArrayAttr'
    elif onnx_attr_type in {'type', 'type_proto'}:
        # 'type' is the attribute type used in special_attr_types,
        # 'type_proto' is Optional op's type attribute's type
        mlir_attr_type = 'TypeAttr'
    else:
        mlir_attr_type = 'AnyAttr'
    #TODO: tensor and sparse tensor
    return mlir_attr_type


#TODO: any better way to do this.
def tblgen_attr_type_to_cpp_type(t):
    if 'I64Attr' in t:
        cpp_type = 'IntegerAttr'
    elif 'F32Attr' in t:
        cpp_type = 'FloatAttr'
    elif 'I64ArrayAttr' in t or 'F32ArrayAttr' in t:
        cpp_type = 'ArrayAttr'
    elif 'StrAttr' in t:
        cpp_type = 'StringAttr'
    elif 'strings' in t:
        cpp_type = 'ArrayAttr'
    else:
        cpp_type = 'Attribute'
    return cpp_type


def tblgen_operand_type_to_cpp_type(op_type):
    if op_type.startswith('Variadic'):
        mytype = 'ValueRange'
    else:
        mytype = 'Value'
    return mytype


def np_type_to_tblgen_attr_type(tstr):
    index = -1
    for i in range(len(onnx_types)):
        if onnx_types[i] in tstr:
            index = i
            break
    if index == -1:
        return None
    else:
        return tblgen_types[i]

def get_tblgen_type_index(type_str):
    return tblgen_types.index(type_str)

#the possible data structures are tensor, map and seq(tensor())
def get_data_structure_element(allowed_type_str):
    structure_list = ['tensor', 'seq', 'map']
    for structure in structure_list:
        if allowed_type_str.startswith(structure) :
            element = allowed_type_str.replace(
                structure+'(', '', 1).replace(')', '', 1)
            return (structure, element)
    return (None, None)

def get_allowed_elem_types(schema, input):
    #allowed_types_str = None
    # return allowed_types_str
    # TODO: enable type constraints.
    if input.typeStr :
        tstr = input.typeStr
        structure, element = get_data_structure_element(tstr);
        # In case the type is directly specified
        if structure and element :
            t = np_type_to_tblgen_attr_type(element)
            if t == None :
                return allowed_structure, None
            else :
                return structure, [t]
    else :
        return None
    if schema.type_constraints:
        for type_constraint in schema.type_constraints:
            if type_constraint.type_param_str != tstr :
                continue
            allowed_type_list=[]
            allowedTypes = type_constraint.allowed_type_strs
            allowed_structure = None
            for allowedType in allowedTypes:
                structure, element = get_data_structure_element(allowedType);
                if structure == None or element == None:
                    return None, None

                if allowed_structure != None and allowed_structure != structure :
                    return None, None
                allowed_structure = structure
                t = np_type_to_tblgen_attr_type(element)
                if t == None :
                    return allowed_structure, None
                if  not t in allowed_type_list :
                    allowed_tyoe_list = allowed_type_list.append(t)

            return allowed_structure,allowed_type_list

    return None, None


def inc_indent(indent=None):
    return "" if indent is None else indent + ' ' * 2


def dec_indent(indent):
    return indent[:-2]


def join_args(args):
    return ", ".join(args)

def get_operands_or_results(schema, type_str_dict,  is_input):
    value_list = schema.inputs if is_input else schema.outputs
    if not value_list:
        return OrderedDict()

    def any_type_of(types):
        assert isinstance(types, list)
        if len(types) == 1:
            return types[0]
        else:
            return "AnyTypeOf<[{}]>".format(", ".join(types))

    name_to_types = OrderedDict()
    for i, value in enumerate(value_list):
        str_types = get_onnx_mlir_types(schema, type_str_dict,  value)

        # In case the type string is used more than once
        types = str_types.copy()

        # No need to add AnyMemRef type. Keep the code in case.
        # types.append("AnyMemRef")

        if OpSchema.FormalParameterOption.Optional == value.option:
            types.append("NoneType")

        if OpSchema.FormalParameterOption.Variadic == value.option:
            if value.isHomogeneous:
                types = ["Variadic<{}>".format(any_type_of(types))]
            else:
                #TODO handle(variadic, heterogeneous) "
                types = ["Variadic<{}>".format(any_type_of(types))]
                sys.stderr.write("warning: (variadic, heterogeneous) for " + schema.name +
                      ' ' + value.name + "\n")

        # Since output name can coincide with that of an input, we explicitly
        # append a suffix "_out" to such names for disambiguation.
        if is_input:
            value_name = value.name
        else:
            value_name = get_unique_output_name(schema, value.name)

        name_to_types[value_name] = any_type_of(types)
    return name_to_types


def get_attrs(schema):
    def get_attr_type_optional(attr_type):
        return 'OptionalAttr<{}>'.format(
            onnx_attr_type_to_mlir_attr_type(attr_type))

    def get_attr_type_with_default(attr_type, attr_default):
        if attr_type == OpSchema.AttrType.STRING:
            return 'DefaultValuedStrAttr<{}, "{}">'.format(
                onnx_attr_type_to_mlir_attr_type(attr_type), attr_default)
        else:
            return 'DefaultValuedAttr<{}, "{}">'.format(
                onnx_attr_type_to_mlir_attr_type(attr_type), attr_default)

    if not schema.attributes:
        return OrderedDict()

    name_to_type = OrderedDict()
    for _, attr in sorted(schema.attributes.items()):
        if attr.type == OpSchema.AttrType.GRAPH:
          continue

        qualified_attr_name = "{}.{}".format(schema.name, attr.name)
        if qualified_attr_name in special_attr_types:
            name_to_type[attr.name] = onnx_attr_type_to_mlir_attr_type(
                special_attr_types[qualified_attr_name])
        # option holds either required or default value
        elif attr.required:
            name_to_type[attr.name] = onnx_attr_type_to_mlir_attr_type(
                attr.type)
        elif attr.default_value.name:

            def format_value(value):  # type: (Any) -> Text
                if isinstance(value, float):
                    formatted = str(np.round(value, 5))
                    # use default formatting, unless too long.
                    if (len(formatted) > 10):
                        formatted = str("({:e})".format(value))
                    return formatted
                elif isinstance(
                        value,
                    (bytes, bytearray)) and sys.version_info[0] == 3:
                    return str(value.decode('utf-8'))
                return str(value)

            default_value = helper.get_attribute_value(attr.default_value)
            if isinstance(default_value, list):
                default_value = [format_value(val) for val in default_value]
                default_value_str = '{}'.format(default_value)
                default_value_str = default_value_str.replace('[', '{', 1)
                default_value_str = default_value_str.replace(']', '}', 1)
                if Text(attr.type) == "AttrType.STRINGS":
                    default_value_str = default_value_str.replace("'", '\\"')
                else:
                    default_value_str = default_value_str.replace("'", '')
            else:
                default_value = format_value(default_value)
                default_value_str = default_value

            name_to_type[attr.name] = get_attr_type_with_default(
                attr.type, default_value_str)
        else:
            name_to_type[attr.name] = get_attr_type_optional(attr.type)
    return name_to_type

def get_numberof_list(mylist):
    expected_num = len(mylist)
    for element in mylist :
        if OpSchema.FormalParameterOption.Variadic == element.option:
            expected_num = -1
    return expected_num

def get_output_type_mapping(schema):
    mapping=[]
    for output in schema.outputs :
        #if only one type is allowed, just set that
        structure, allowed_elem_types = get_allowed_elem_types(schema, output)
        if allowed_elem_types != None and len(allowed_elem_types) == 1 :
            mapping.append(str(get_tblgen_type_index(allowed_elem_types[0])))
            continue

        #map the type string
        if output.typeStr :
            tstr = output.typeStr
            found = False
            for i, input in enumerate(schema.inputs):
                if input.typeStr and input.typeStr == tstr:
                    mapping.append(str(i+MAX_NUM_TYPES))
                    found = True
                    break
            if found:
                continue

        #unknown output type
        mapping.append(str(-1))

    return mapping

def get_numberof_inout(s, indent, schema):
    expected_num_operands = get_numberof_list(schema.inputs)
    indent = inc_indent(indent)
    s += indent + "static int getNumberOfOperands() {\n"
    indent = inc_indent(indent)
    s += indent + "return {};\n".format(expected_num_operands)
    indent = dec_indent(indent)
    s += indent + "}\n"

    expected_num_results = get_numberof_list(schema.outputs)
    s += indent + "static int getNumberOfResults() {\n"
    indent = inc_indent(indent)
    s += indent + "return {};\n".format(expected_num_results)
    indent = dec_indent(indent)
    s += indent + "}\n"

    s += indent + "static std::vector<int> getTypeMap() {\n"
    mapping = get_output_type_mapping(schema)
    indent = inc_indent(indent)
    s += indent + "return {" + ",".join(mapping) + "};\n"
    indent = dec_indent(indent)
    s += indent + "}\n"

    return s


def get_promotable_const_operands_func(s, indent, const_operands_name_to_idx):
    cpp_name_to_idx_literal = "{" + ", ".join([
        "{{\"{}\", {}}}".format(*name_to_idx)
        for name_to_idx in const_operands_name_to_idx
    ]) + "}"

    #s += indent + "let extraClassDeclaration = [{\n"
    indent = inc_indent(indent)
    s += indent + "std::map<std::string, size_t> promotableConstOperands() {\n"
    indent = inc_indent(indent)
    s += indent + "return {};\n".format(cpp_name_to_idx_literal)
    indent = dec_indent(indent)
    s += indent + "}\n"
    #indent = dec_indent(indent)
    #s += indent + "}];\n"

    return s

def get_type_inference_func(s, indent, type_inference_code):
    indent = inc_indent(indent)

    s += indent + "std::vector<mlir::Type> resultTypeInference() {" + "\n"
    indent = inc_indent(indent)
    s += indent + "std::vector<mlir::Type> resultTypes;" + "\n"

    s += indent + type_inference_code + '\n'

    s += indent + "return resultTypes;" + "\n"
    indent = dec_indent(indent)
    s += indent + "}" + "\n"

    indent = dec_indent(indent)
    return s

def parse_type_str(allowedType):
    # AnyI may be used for uint because the onnx_mlir is not generating uint output
    # This will be fixed later and UI will be replace AnyI
    onnx_to_mlir_type_dict = { '(': '<[',
        ')': ']>',
        'tensor' : 'TensorOf',
        'seq' : 'SeqOf',
        'map' : 'TupleOf',
        'bool': 'I1',
        #'uint8' : 'AnyI8',
        #uint16' : 'AnyI16',
        #uint32' : 'AnyI32',
        #uint64' : 'AnyI64',
        'uint8' : 'UI8',
        'uint16' : 'UI16',
        'uint32' : 'UI32',
        'uint64' : 'UI64',
        'int8' : 'I8',
        'int16' : 'I16',
        'int32' : 'I32',
        'int64' : 'I64',
        'float16' : 'F16',
        'bfloat16' : 'BF16',
        'float' : 'F32',
        'double' : 'F64',
        'unkown' : 'BF16',
        'complex64' : 'Complex<F32>',
        'complex128' : 'Complex<F64>',
        'string' : 'StringType'}

    # optional(...) always appears outermost
    if allowedType.find("optional") == 0 :
      allowedType = allowedType.replace("optional(", "OptOf<", 1);
      allowedType = allowedType[:-1] + '>'

    # Apply substitutions in decreasing order of key-length, so that float16 is replaced
    # before float, and uint16 is replaced before int16, etc.
    mapping = list(onnx_to_mlir_type_dict.items())
    mapping.sort(key=lambda pair:len(pair[0]), reverse=True)
    for key, item in mapping:
        allowedType = allowedType.replace(key, item)
    return allowedType

def parse_a_type_constraint(constraint):
    allowedTypes = constraint.allowed_type_strs
    mlirTypes = []
    for allowedType in allowedTypes:
        mlirType = parse_type_str(allowedType)
        mlirTypes.append(mlirType)

    # Remove redundant and sort.
    # However onnx keeps a consitently meaningful order
    # There is no redundancy as long as each onnx type is mapped uniquely
    # mlirTypes = sorted(list(set(mlirTypes)))

    return mlirTypes

def parse_type_constraints(schema):
    type_str_dict = dict()
    for type_constraint in schema.type_constraints:
        type_str_dict[type_constraint.type_param_str]  = parse_a_type_constraint(type_constraint)
    return type_str_dict

def get_onnx_mlir_types(schema, type_str_dict, input):
    if input.typeStr :
         if not input.typeStr in type_str_dict :
             # some arguments use type description directly
             # instead of constraint
             type_str = parse_type_str(input.typeStr)
             return [type_str]
         else :
             return type_str_dict[input.typeStr]
    else :
        print('No typeStr ', schema.name)
        return []

def gen_op_def(schema, with_version = False):
    indent = inc_indent()
    if with_version :
        opName = schema.name+"V"+str(schema.since_version)
    else :
        opName = schema.name
    s = 'def ONNX{0}Op:ONNX_Op<"{0}",\n'.format(opName)

    regions = OrderedDict()
    for _, attr in sorted(schema.attributes.items()):
      if attr.type == OpSchema.AttrType.GRAPH:
        if attr.required:
          regions[attr.name] = "SizedRegion<1>"
        else:
          regions[attr.name] = "AnyRegion"

    # Generate decl for op traits.
    traits = ["NoSideEffect"]
    # OpsWithShapeInference:
    # Now the ShapeInference traits are added to all operation
    # Dummy implementations are added to ONNXOps.cpp
    # Error will be report if these operations are encountered at runtime
    traits.append("DeclareOpInterfaceMethods<ShapeInferenceOpInterface>")
    if opName in OpsWithResultTypeInference.keys():
        traits.append("OpInterface<\"ResultTypeInferenceOpInterface\">")
    if len(regions):
        traits.append("OpInterface<\"HasOnnxSubgraphOpInterface\">")
    s += inc_indent(indent) + '[{}]> {{\n'.format(join_args(traits))

    # Generate decl for canonicalizer.
    indent = inc_indent(indent)
    if opName in OpsWithCanonicalizer:
        s += indent + 'let hasCanonicalizer = 1;\n'

    # Generate decl for summary.
    s += indent + 'let summary = "ONNX {} operation";\n'.format(schema.name)

    # Generate description.
    s += indent + 'let description = [{\n'
    if schema.doc:
        lines = schema.doc.lstrip().splitlines()
        for line in lines:
            escaped_line = line.replace('"', '\\"')\
                               .replace('}]', '\\}\\]')
            s += indent + '"{}"\n'.format(escaped_line)
    s += indent + '}];\n'

    # handle the type constraint for input and output
    # parse type constraint into onnx-mlir type string list
    type_str_dict =  parse_type_constraints(schema)

    # Generate ins (consisting of operands and attributes).
    ins = get_operands_or_results(schema, type_str_dict, is_input=True)
    ins.update(get_attrs(schema))

    ins_strs = ["{1}:${0}".format(*i) for i in ins.items()]
    s += indent + 'let arguments = (ins {});\n'.format(
        (',\n' + inc_indent(indent)).join(ins_strs))

    # Generate outs (operation results).
    outs = get_operands_or_results(schema, type_str_dict, is_input=False)
    outs_strs = ["{1}:${0}".format(*i) for i in outs.items()]
    s += indent + 'let results = (outs {});\n'.format(
        (',\n' + inc_indent(indent)).join(outs_strs))

    regions_strs = ["{1}:${0}".format(*i) for i in regions.items()]

    if len(regions):
        s += indent + 'let regions = (region {});\n'.format(
            (',\n' + inc_indent(indent)).join(regions_strs))

    # custom_builder_broadcast_ops_list

    # add custom builders
    # use element type of the first operand to construct an UnrankedTensorType for the output.
    if opName in custom_builder_ops_list:
        if len(ins) == 0:
            raise RuntimeWarning(
                "warning: not generate custom build methods for " +
                schema.name + " since it does not have operands.")

        r = '' # r is the resultType, use it with r.format(*operands, indent=indent)
        if opName in custom_builder_broadcast_ops_list:
            numOperands = 2
            r += '{indent}auto lhsTy = {0}.getType();\n'
            r += '{indent}auto rhsTy = {1}.getType();\n'
            if opName in custom_builder_broadcast_to_bool_ops_list:
                r += '{indent}auto elTy = $_builder.getI1Type();\n'
                elTy = 'elTy'
            else:
                elTy = ''
            r += '{indent}auto resultType = getBroadcastedRankedType(lhsTy, rhsTy' + \
                (', ' + elTy if elTy else '') + ');\n'
            r += '{indent}auto shapedType = resultType.dyn_cast_or_null<ShapedType>();\n'
            r += '{indent}if (!shapedType || !shapedType.hasStaticShape())\n'
            r += '{indent}  resultType = UnrankedTensorType::get(' + \
                (elTy if elTy else 'lhsTy.cast<ShapedType>().getElementType()') + ');\n'
        else:
            numOperands = 1
            r += '{indent}auto resultType = UnrankedTensorType::get(' + \
                '{0}.getType().cast<ShapedType>().getElementType());\n'
        resultType = r

        s += indent + 'let builders = [\n'
        # Custom builders with operands and attributes having a separate parameter.
        # E.g. OpBuilder<(ins "Value":$X, "Value":$Y, "Attribute":$A), [{}]>
        indent = inc_indent(indent)
        s += indent + 'OpBuilder<(ins '
        operands_dict = get_operands_or_results(schema, type_str_dict, is_input=True)
        attrs_dict = get_attrs(schema)
        s += ', '.join('"{}":${}'.format(tblgen_operand_type_to_cpp_type(ty),
                                    name) for name, ty in operands_dict.items())
        if operands_dict and attrs_dict:
            s += ', '
        s += ', '.join('"{}":${}'.format(tblgen_attr_type_to_cpp_type(ty),
                                    name) for name, ty in attrs_dict.items())
        s += '), [{\n'
        indent = inc_indent(indent)
        # Get output type from first operand's type.
        operands = operands_dict.keys()
        s += resultType.format(*operands, indent=indent)
        s += indent + 'build($_builder, $_state, resultType'
        for name, _ in ins.items():
            s += ', ' + name
        s += ');\n'
        indent = dec_indent(indent)
        s += indent + '}]>,\n'

        # Custom builders with all operands and attributes having aggregate parameters.
        # E.g. OpBuilder<(ins "ValueRange operands,
        #    ArrayRef<NamedAttribute> attributes", [{}]>'
        s += indent + 'OpBuilder<(ins ' + \
            '"ValueRange":$operands, "ArrayRef<NamedAttribute>":$attributes), [{\n'
        indent = inc_indent(indent)
        operands = (f'operands[{i}]' for i in range(numOperands))
        s += resultType.format(*operands, indent=indent)
        s += indent + 'build($_builder, $_state, {resultType}, operands, attributes);\n'
        indent = dec_indent(indent)
        s += indent + '}]>'

        s += '\n' + indent + '];\n'

    # Generate extraClassDeclaration.
    s += indent + "let extraClassDeclaration = [{\n"
    #indent = inc_indent(indent)

    # Generate input/output number.
    s = get_numberof_inout(s, indent, schema)

    if opName in OpsWithResultTypeInference:
        s = get_type_inference_func(
            s, indent, OpsWithResultTypeInference[opName])

    if opName in OpsWithHelpers:
        s += OpsWithHelpers[opName]

    if len(regions):
        s += indent + "int64_t getSubgraphRegionIdx(const std::string& name) {\n"
        indent = inc_indent(indent)
        for idx, region_name in enumerate(regions.keys()):
          s += indent + "if (name == \"{}\") return {};\n".format(region_name, idx)
        s += indent + "llvm_unreachable(\"region with the specified name does not exist\");\n"
        indent = dec_indent(indent)
        s += indent + "}\n"

    s += indent + '}];\n'

    if ( opName in custom_definition_misc) :
        s += custom_definition_misc[opName] + '\n'

    # Generate decl for verifier.
    if opName in OpsWithVerifier:
        s += indent + 'let hasVerifier = 1;\n'

    s += '}\n\n'
    return s


def gen_op_versions(file) :
    indent = inc_indent()
    s = ""
    for key, item in version_dict.items() :
        s += indent + 'op_dialect_version_map_["' + key +'"] = '
        s += "{" +  "{}".format(", ".join(str(x) for x in item)) + "};\n"
    file.write(s)

# create the top opset value of each op for current onnx

def gen_op_new_version(file, new_version_dict) :
    indent = inc_indent()
    s = ""
    for key, item in new_version_dict.items() :
        s += indent + 'op_dialect_top_version_map_["' + key +'"] = '
        s +=  "{}".format(", ".join(str(x) for x in item)) + ";\n"
    file.write(s)

"""
special cases:
* Split: attr split default value: sizeof(output1) namely 1
* Conv: attr dilations default value is {num_dim of first input - 2, 1}
* Conv: attr kernel_shape type is ints
* Transpose: attr perm default value is {} empty int list
"""


def gen_op_importer(schema, file, with_version=False):
    indent = inc_indent()
    if with_version :
        opName = schema.name + "V"+str(schema.since_version)
    else :
        opName = schema.name
    s = indent + 'import_handler_map_["' + opName +'"] = \n '

    expected_num_operands = len(schema.inputs)
    expected_num_results = len(schema.outputs)
    for input in schema.inputs:
        if OpSchema.FormalParameterOption.Variadic == input.option:
            expected_num_operands = -1
    for output in schema.outputs:
        if OpSchema.FormalParameterOption.Variadic == output.option:
            expected_num_results = -1

    # Only support special op handler for the op without version.
    if with_version:
        handler_func = "buildOperation<mlir::ONNX{}Op>".format(opName)
    else:
        handler_func = special_op_handler.get(
            schema.name, "buildOperation<mlir::ONNX{}Op>".format(opName))

    # Special handlers currently require expected num operands/results to be specified.
    # TODO: remove special handlers.
    args = ["node"]
    """
    if expected_num_operands != -1 or expected_num_results != -1 or "buildOperation" not in handler_func:
        args.append(
            "/* expected_num_operands = */ {}".format(expected_num_operands))
        args.append(
            '/* expected_num_results = */ {}'.format(expected_num_results))
    """
    s += inc_indent(indent) + '&onnx_mlir::detail::FrontendGenImpl::'
    s += handler_func+';\n'

    file.write(s)


def build_operator_schemas():
    # domain -> support level -> name -> [schema]
    index = defaultdict(lambda: defaultdict(lambda: defaultdict(
        list)))  # type: Dict[Text, Dict[int, Dict[Text, List[OpSchema]]]]
    for schema in defs.get_all_schemas_with_history():
        index[schema.domain][int(
            schema.support_level)][schema.name].append(schema)

    # Preprocess the Operator Schemas
    # [(domain, [(support_level, [(schema name, current schema, all versions schemas)])])]
    operator_schemas = list(
    )  # type: List[Tuple[Text, List[Tuple[int, List[Tuple[Text, OpSchema, List[OpSchema]]]]]]]
    exsting_ops = set()  # type: Set[Text]
    for domain, _supportmap in sorted(index.items()):
        if not should_render_domain(domain):
            continue
        processed_supportmap = list()
        for _support, _namemap in sorted(_supportmap.items()):
            processed_namemap = list()
            for n, unsorted_versions in sorted(_namemap.items()):
                versions = sorted(unsorted_versions,
                                  key=lambda s: s.since_version)
                schema = versions[-1]
                if schema.name in exsting_ops:
                    continue

                if check_operation_version:
                    # Generate operation of the latest version of your onnx.
                    exsting_ops.add(schema.name)
                    processed_namemap.append((n, schema, versions))

                    # Add checks against version_dict
                    if schema.name not in version_dict :
                        print("Check-operation-version: Operation {} is new  with version {}"
                            .format(schema.name, schema.since_version))
                    elif schema.since_version >  version_dict[schema.name][0]:
                        print("Check-operation-version: Operation {}"
                            .format(schema.name)+
                            " has a newer version {} over old version {}"
                            .format(schema.since_version, version_dict[schema.name][0]))
                else:
                    # Generate operation according to the version in version_dict.
                    if schema.name not in version_dict :
                        continue
                    found = False
                    vcounter = 0
                    for schema in reversed(versions):
                        # Check the version number against the version_dict
                        specified_version = version_dict[schema.name][vcounter]
                        if schema.since_version == specified_version:
                            exsting_ops.add(schema.name)
                            processed_namemap.append((n, schema, versions))
                            found = True
                            vcounter += 1
                            if len(version_dict[schema.name]) == vcounter :
                                break
                    if not found:
                        print("Your onnx installation may be too old. "
                           "The desired version for operation {} is not found.".format(
                            schema.name))
                        sys.exit()
            processed_supportmap.append((_support, processed_namemap))
        operator_schemas.append((domain, processed_supportmap))
    return operator_schemas


def main(args):  # type: (Type[Args]) -> None
    if list_operation_version:
        pprint.pprint(version_dict)
        return

    curr_utc_time = datetime.datetime.now(
        datetime.timezone.utc).strftime("%m/%d/%Y, %H:%M:%S")
    autogen_warning = (
        '//********************************************************\n'
        '//   Do not modify this file directly.\n'
        '//   This file is automatically generated via script.\n'
        '//   Details can be found in docs/ImportONNXDefs.md .\n'
        '//********************************************************\n\n')
    autogen_warning = autogen_warning.format(curr_utc_time)

    op_def = args.op_def
    op_def.write(autogen_warning)

    op_importer = args.op_importer
    op_importer.write(autogen_warning)
    gen_op_versions(op_importer)

    new_version_dict = dict()
    for domain, supportmap in build_operator_schemas():
        for _, namemap in supportmap:
            # Generate Op with version number if not the latest version
            previous_name = ""
            for op_type, schema, versions in namemap:
                new_version_dict[schema.name] = [schema.since_version]
                if not check_operation_version :
                    with_version = previous_name == schema.name
                    gen_op_importer(schema, op_importer, with_version)
                    r = gen_op_def(schema, with_version)
                    op_def.write(r)
                    previous_name = schema.name

    gen_op_new_version(op_importer, new_version_dict)
    if check_operation_version :
        for key in version_dict :
            if not key in new_version_dict :
                print("op {} is not in the version".format(key))
            # Assume the top version will be upgraded to the latest version
            # The existing extra version (from index 1) will be kept
            for x in version_dict[key][1:] :
                new_version_dict[key].append(x)
        pprint.pprint(new_version_dict)

if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    class Args(object):
        dry_run = args.dry_run_onnx_ops or args.dry_run_op_build_table

        # If either dry_run_onnx_ops or dry_run_op_build_table is true, then treat both of them
        # as true. Otherwise, one of them runs as a dry-run and one of them runs as a real run
        # creating unnecessary artifacts in the wrong locations in the build tree.
        if dry_run:
            op_def = StringIO()
            op_importer = StringIO()
        else:
            op_def_file_path = os.path.join(curr_dir, 'ONNXOps.td.inc')
            op_def = io.open(op_def_file_path, 'w', newline='')
            op_importer_file_path = os.path.join(curr_dir, 'OpBuildTable.inc')
            op_importer = io.open(op_importer_file_path, 'w', newline='')
    main(Args)

    # This is based on diff.py from llvm-project (llvm\utils\lit\lit\builtin_commands\diff.py).
    # On Windows, by default, stdout uses \r\n for newlines, however, all the files we compare against
    # use \n. This piece of code forces the windows stdout to use \n for newlines.
    if sys.platform == "win32":
        if hasattr(sys.stdout, 'buffer'):
            # python 3
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, newline='\n')
        else:
            # python 2.7
            import msvcrt
            msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

    # Only output the generated values for the specifically requested dry run.
    if args.dry_run_onnx_ops:
        sys.stdout.write(Args.op_def.getvalue())
    if args.dry_run_op_build_table:
        sys.stdout.write(Args.op_importer.getvalue())
