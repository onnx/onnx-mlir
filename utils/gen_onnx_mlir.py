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
parser.add_argument("--domain", 
                    help="specify domain, ONNX or ONNX_ML",
                    default = "ONNX")

args = parser.parse_args()

check_operation_version = args.check_operation_version


# Record the version of each operation that is treated as the current version.
# To check whether the onnx package being used has newer version operation,
# run this script with --check-operation-version flag.
# Update this dictionary when a newer version is implemented
# TODO: how to keep the old version
onnx_version_dict = {'Abs': 6,
 'Acos': 7,
 'Acosh': 9,
 'Add': 7,
 'And': 7,
 'ArgMax': 11,
 'ArgMin': 11,
 'Asin': 7,
 'Asinh': 9,
 'Atan': 7,
 'Atanh': 9,
 'AveragePool': 11,
 'BatchNormalization': 9,
 'BitShift': 11,
 'Cast': 9,
 'Ceil': 6,
 'Clip': 11,
 'Compress': 11,
 'Concat': 11,
 'ConcatFromSequence': 11,
 'Constant': 11,
 'ConstantOfShape': 9,
 'Conv': 11,
 'ConvInteger': 10,
 'ConvTranspose': 11,
 'Cos': 7,
 'Cosh': 9,
 'CumSum': 11,
 'DepthToSpace': 11,
 'DequantizeLinear': 10,
 'Det': 11,
 'Div': 7,
 'Dropout': 10,
 'DynamicQuantizeLinear': 11,
 'Elu': 6,
 'Equal': 11,
 'Erf': 9,
 'Exp': 6,
 'Expand': 8,
 'EyeLike': 9,
 'Flatten': 11,
 'Floor': 6,
 'GRU': 7,
 'Gather': 11,
 'GatherElements': 11,
 'GatherND': 11,
 'Gemm': 11,
 'GlobalAveragePool': 1,
 'GlobalLpPool': 2,
 'GlobalMaxPool': 1,
 'Greater': 9,
 'HardSigmoid': 6,
 'Hardmax': 11,
 'Identity': 1,
 'If': 11,
 'InstanceNormalization': 6,
 'IsInf': 10,
 'IsNaN': 9,
 'LRN': 1,
 'LSTM': 7,
 'LeakyRelu': 6,
 'Less': 9,
 'Log': 6,
 'LogSoftmax': 11,
 'Loop': 11,
 'LpNormalization': 1,
 'LpPool': 11,
 'MatMul': 9,
 'MatMulInteger': 10,
 'Max': 8,
 'MaxPool': 11,
 'MaxRoiPool': 1,
 'MaxUnpool': 11,
 'Mean': 8,
 'MeanVarianceNormalization': 9,
 'Min': 8,
 'Mod': 10,
 'Mul': 7,
 'Multinomial': 7,
 'Neg': 6,
 'NonMaxSuppression': 11,
 'NonZero': 9,
 'Not': 1,
 'OneHot': 11,
 'Or': 7,
 'PRelu': 9,
 'Pad': 11,
 'Pow': 7,
 'QLinearConv': 10,
 'QLinearMatMul': 10,
 'QuantizeLinear': 10,
 'RNN': 7,
 'RandomNormal': 1,
 'RandomNormalLike': 1,
 'RandomUniform': 1,
 'RandomUniformLike': 1,
 'Range': 11,
 'Reciprocal': 6,
 'ReduceL1': 11,
 'ReduceL2': 11,
 'ReduceLogSum': 11,
 'ReduceLogSumExp': 11,
 'ReduceMax': 11,
 'ReduceMean': 11,
 'ReduceMin': 11,
 'ReduceProd': 11,
 'ReduceSum': 11,
 'ReduceSumSquare': 11,
 'Relu': 6,
 'Reshape': 5,
 'Resize': 11,
 'ReverseSequence': 10,
 'RoiAlign': 10,
 'Round': 11,
 'Scan': 11,
 'Scatter': 11,
 'ScatterElements': 11,
 'ScatterND': 11,
 'Selu': 6,
 'SequenceAt': 11,
 'SequenceConstruct': 11,
 'SequenceEmpty': 11,
 'SequenceErase': 11,
 'SequenceInsert': 11,
 'SequenceLength': 11,
 'Shape': 1,
 'Shrink': 9,
 'Sigmoid': 6,
 'Sign': 9,
 'Sin': 7,
 'Sinh': 9,
 'Size': 1,
 'Slice': 11,
 'Softmax': 11,
 'Softplus': 1,
 'Softsign': 1,
 'SpaceToDepth': 1,
 'Split': 11,
 'SplitToSequence': 11,
 'Sqrt': 6,
 'Squeeze': 11,
 'StringNormalizer': 10,
 'Sub': 7,
 'Sum': 8,
 'Tan': 7,
 'Tanh': 6,
 'TfIdfVectorizer': 9,
 'ThresholdedRelu': 10,
 'Tile': 6,
 'TopK': 11,
 'Transpose': 1,
 'Unique': 11,
 'Unsqueeze': 11,
 'Upsample': 10,
 'Where': 9,
 'Xor': 7}

onnx_ml_version_dict = {'ArrayFeatureExtractor': 1,
 'Binarizer': 1,
 'CastMap': 1,
 'CategoryMapper': 1,
 'DictVectorizer': 1,
 'FeatureVectorizer': 1,
 'Imputer': 1,
 'LabelEncoder': 2,
 'LinearClassifier': 1,
 'LinearRegressor': 1,
 'Normalizer': 1,
 'OneHotEncoder': 1,
 'SVMClassifier': 1,
 'SVMRegressor': 1,
 'Scaler': 1,
 'TreeEnsembleClassifier': 1,
 'TreeEnsembleRegressor': 1,
 'ZipMap': 1}

# Manual specification of attribute defaults.
special_attr_defaults = dict([
    # ("AveragePool.kernel_shape", ('ints', '{}')),
    # ("MaxPool.kernel_shape", ('ints', '{}')),
    # ("Cast.to", ('int', '0')),
    # ("Concat.axis", ('int', '0')),
    # ("Conv.group", ('int', '1')),
    # ("Unsqueeze.axes", ('ints', '{}')),
    # ("RNN.activation_alpha", ('floats', '{}')),
    # ("RNN.activation_beta", ('floats', '{}')),
])

# Special operation importing handlers.
special_op_handler = dict([
    ("MaxPool", "ImportNodeMaxPool"),
    ("BatchNormalization", "ImportNodeBatchNormalization"),
    ("Pad", "ImportNodePad"),
    ("Reshape", "ImportNodeReshape"),
    #("Transpose", "ImportNodeTranspose")
])

# Operations supporting shape inference.
OpsWithShapeInference = [
    'Exp', 'Tanh', 'Sinh', 'Cosh', 'Sigmoid', 'Relu', 'Add', 'Mul', 'Div',
    'Sub', 'And', 'Or', 'Xor', 'Sum', 'Max', 'Min', 'MatMul', 'Gemm',
    'LeakyRelu', 'Elu', 'Selu', 'HardSigmoid', 'Reshape', 'Reciprocal',
    'Identity', 'Cos', 'Log', 'Transpose', 'Softmax', 'ReduceMax', 'ReduceMin',
    'ReduceProd', 'ReduceSum', 'Softplus', 'Softsign', 'Sqrt', 'Unsqueeze',
    'Sign', 'Constant', 'AveragePool', 'Abs', 'Conv', 'Concat', 'Neg', 'RNN',
    'LSTM', 'GRU', 'Split', 'Pad'
]

# Operations supporting canonicalization.
OpsWithCanonicalizer = ['Add', 'Identity', 'Gemm', 'Conv']

# Operations who have operands that, if produced by constant operations, should
# be promoted to become an attribute (via attribute promotion).
#
# For each operation, a key/value pair is used to specify how attribute promotion
# should proceed. The key is the operation's name and the value is a list of
# tuples, whose first item is the attribute/operand name, and the second item is
# the index at which such operand occurs in the list of the operation's inputs.
OpsWithPromotableConstOperands = {"Reshape": [("shape", 1)],
                                  "Pad": [("pads", 1), ("constant_value", 2)]}

OpsWithTypeInference = {"Constant":
  '''static std::vector<mlir::Type> typeInferenceFunc(
        std::vector<mlir::Value> inputs,  
        std::vector<NamedAttribute> attributes) {
      std::vector<mlir::Type> resultTypes;
      for (auto pair : attributes) {
        auto attr = pair.second;
        if (attr) {
          auto tensorType = attr.getType();
          resultTypes.push_back(tensorType);
        }
      } 
      return resultTypes;
   }'''
  }

# Add an Op in this list if the Op needs result type deduction which is required
# when writing declarative rewriting rules. Deduced type is always
# an UnrankedTensorType whose element type is the same as the first operand's
# element type.
#
# Currenlty, there are only two build methods generated:
#  - one with operands and attributes having a separate parameter, and
#  - one with operands and attributes having aggregated parameters.
custom_builder_ops_list = ['Abs', 'Mul', 'Exp', 'ReduceSum', 'ReduceSumSquare', 'Pad']


#a dictionary to add any special definition for an operation
custom_definition_misc = dict([ ('Constant', 
  '''    let builders = [
    OpBuilder<"OpBuilder &builder, OperationState &state, Attribute sparse_value, Attribute value", [{
      if (value) {
        auto tensorType = value.getType();
        build(builder, state, tensorType, sparse_value, value);
      } else {
        auto tensorType = sparse_value.getType();
        build(builder, state, tensorType, sparse_value, value);
      }
    }]>
    ];'''
  )])


onnx_types = (
    'bool', 'int8', 'int16', 'int32', 'int64', 'unkown', 'float16',
    'float', 'double', 'complex64', 'complex128'
)
tblgen_types = ('I1', 'I8', 'I16', 'I32', 'I64', 'BF16', 'F16', 'F32', 'F64', 
    'Complex<F32>', 'Complex<F64>'
)

MAX_NUM_TYPES=20

SNIPPETS = collect_snippets()
SAMPLE_IMPLEMENTATIONS = collect_sample_implementations()
ONNX_ML = bool(args.domain == "ONNX_ML")

sys.stderr.write("ONNX_ML {}\n".format(ONNX_ML))

def should_render_domain(domain):  # type: (Text) -> bool
    if domain == ONNX_ML_DOMAIN and not ONNX_ML:
        return False
    elif ONNX_ML and domain != ONNX_ML_DOMAIN:
        return False
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
        mlir_attr_type = 'I64Attr'
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
#TOFIX: currently, only tensor structure is supported
def get_data_structure_element(allowed_type_str): 
    if allowed_type_str.startswith('tensor') :
        element = allowed_type_str.replace('tensor(', '', 1).replace(')', '', 1)
        return ('tensor', element)
    else :
        return (None, None)

def get_allowed_elem_types(schema, input):
    #allowed_types_str = None
    # return allowed_types_str
    # TODO: enable type constraints.
    if input.typeStr :
         tstr = input.typeStr
    else :
        return None
    if schema.type_constraints:
        for type_constraint in schema.type_constraints:
            if type_constraint.type_param_str != tstr :
                continue
            allowed_type_list=[]
            allowedTypes = type_constraint.allowed_type_strs
            for allowedType in allowedTypes:
                structure, element = get_data_structure_element(allowedType);
                if structure == None or element == None:
                    return None
                t = np_type_to_tblgen_attr_type(element)
                if t == None :
                    return None
                if  not t in allowed_type_list :
                    allowed_tyoe_list = allowed_type_list.append(t)
    
            return allowed_type_list
    
    return None


def inc_indent(indent=None):
    return "" if indent is None else indent + ' ' * 2


def dec_indent(indent):
    return indent[:-2]


def join_args(args):
    return ", ".join(args)

def get_operands_or_results(schema, is_input):
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
        elem_types = get_allowed_elem_types(schema, value)

        if elem_types is None:
            types = ["AnyMemRef", "AnyTensor"]
        else:
            elem_types_str = ','.join(elem_types)
            types = ["TensorOf<[{}]>", "MemRefOf<[{}]>"]
            types = list(map(lambda x: x.format(elem_types_str), types))

        # If operand is promotable to an attribute, then it must be
        # nullable in case it migrates to be an attribute.
        if schema.name in OpsWithPromotableConstOperands:
            idxs = dict(OpsWithPromotableConstOperands[schema.name]).values()
            if i in idxs and not OpSchema.FormalParameterOption.Optional == value.option:
                types.append("NoneType")

        if OpSchema.FormalParameterOption.Optional == value.option:
            types.append("NoneType")
        elif OpSchema.FormalParameterOption.Variadic == value.option:
            if value.isHomogeneous:
                types = ["Variadic<{}>".format(any_type_of(types))]
            else:
                #TODO handle(variadic, heterogeneous) "
                sys.stderr.write("warning: (variadic, heterogeneous) for" + schema.name +
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
        return 'DefaultValuedAttr<{}, "{}">'.format(
            onnx_attr_type_to_mlir_attr_type(attr_type), attr_default)

    if not schema.attributes:
        return OrderedDict()

    name_to_type = OrderedDict()
    for _, attr in sorted(schema.attributes.items()):
        qualified_attr_name = "{}.{}".format(schema.name, attr.name)
        if qualified_attr_name in special_attr_defaults:
            name_to_type[attr.name] = get_attr_type_with_default(
                *special_attr_defaults[qualified_attr_name])

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
        allowed_elem_types = get_allowed_elem_types(schema, output)
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
    s += indent+ type_inference_code + '\n'
    indent = dec_indent(indent)

    return s
  
  

def gen_op_def(schema):
    indent = inc_indent()
    if (ONNX_ML) :
        s = 'def MLONNX{0}Op:MLONNX_Op<"{0}",\n'.format(schema.name)
    else :
        s = 'def ONNX{0}Op:ONNX_Op<"{0}",\n'.format(schema.name)

    # Generate decl for op traits.
    traits = ["NoSideEffect"]
    if schema.name in OpsWithShapeInference:
        traits.append("DeclareOpInterfaceMethods<ShapeInferenceOpInterface>")
    if schema.name in OpsWithPromotableConstOperands.keys():
        traits.append("OpInterface<\"PromotableConstOperandsOpInterface\">")
    s += inc_indent(indent) + '[{}]> {{\n'.format(join_args(traits))

    # Generate decl for canonicalizer.
    indent = inc_indent(indent)
    if schema.name in OpsWithCanonicalizer:
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

    # Generate ins (consisting of operands and attributes).
    ins = get_operands_or_results(schema, is_input=True)
    ins.update(get_attrs(schema))
    ins_strs = ["{1}:${0}".format(*i) for i in ins.items()]
    s += indent + 'let arguments = (ins {});\n'.format(
        (',\n' + inc_indent(indent)).join(ins_strs))

    # Generate outs (operation results).
    outs = get_operands_or_results(schema, is_input=False)
    outs_strs = ["{1}:${0}".format(*i) for i in outs.items()]
    s += indent + 'let results = (outs {});\n'.format(
        (',\n' + inc_indent(indent)).join(outs_strs))

    # add custom builders
    # use element type of the first operand to construct an UnrankedTensorType for the output.
    if schema.name in custom_builder_ops_list:
        if len(ins) == 0:
            raise RuntimeWarning(
                "warning: not generate custom build methods for " +
                schema.name + " since it does not have operands.")
        else:
            s += indent + 'let builders = [\n'
            # Custom builders with operands and attributes having a seperate parameter.
            # E.g. OpBuilder<"OpBuilder &builder, OperationState &state, Value X, Value, Y, Attribute A", [{}]>
            indent = inc_indent(indent)
            s += indent + 'OpBuilder<"OpBuilder &builder, OperationState &state'
            operands_dict = get_operands_or_results(schema, is_input=True)
            for name, ty in operands_dict.items():
                s += ', {} {}'.format(tblgen_operand_type_to_cpp_type(ty),
                                      name)
            for name, ty in get_attrs(schema).items():
                s += ', {} {}'.format(tblgen_attr_type_to_cpp_type(ty), name)
            s += '", [{\n'
            indent = inc_indent(indent)

            # Get output type from first operand's type.
            first_operand_name = list(ins.items())[0][0]
            s += indent + 'auto elementType = {}.getType().cast<TensorType>().getElementType();\n'.format(
                first_operand_name)
            s += indent + 'build(builder, state, UnrankedTensorType::get(elementType)'
            for name, _ in ins.items():
                s += ', ' + name
            s += ');\n'
            indent = dec_indent(indent)
            s += indent + '}]>,\n'

            # Custom builders with all operands and attributes having aggregate parameters.
            # E.g. OpBuilder<"OpBuilder &builder, OperationState &state, ValueRange operands, ArrayRef<NamedAttribute> attributes", [{}]>'
            s += indent + 'OpBuilder<"OpBuilder &builder, OperationState &state, ValueRange operands, ArrayRef<NamedAttribute> attributes", [{\n'
            indent = inc_indent(indent)
            s += indent + 'auto elementType = operands[0].getType().cast<TensorType>().getElementType();\n'
            s += indent + 'std::vector<mlir::Type> outputTypes;\n'
            s += indent + 'outputTypes.emplace_back(UnrankedTensorType::get(elementType));\n'
            s += indent + 'build(builder, state, outputTypes, operands, attributes);\n'
            indent = dec_indent(indent)
            s += indent + '}]>'

            s += '\n' + indent + '];\n'

    # generate extracClassDeclaration
    s += indent + "let extraClassDeclaration = [{\n"
    #indent = inc_indent(indent)

    # generate input/output number
    s = get_numberof_inout(s, indent, schema)

    # generate ProtableConst 
    if schema.name in OpsWithPromotableConstOperands:
        s = get_promotable_const_operands_func(
            s, indent, OpsWithPromotableConstOperands[schema.name])

    if schema.name in OpsWithTypeInference:
        s = get_type_inference_func(
            s, indent, OpsWithTypeInference[schema.name])

    s += indent + '}];\n'

    if ( schema.name in custom_definition_misc) :
        s += custom_definition_misc[schema.name]

    s += '}\n\n'
    return s


"""
special cases:
* Split: attr split default value: sizeof(output1) namely 1
* Conv: attr dilations default value is {num_dim of first input - 2, 1}
* Conv: attr kernel_shape type is ints
* Transpose: attr perm default value is {} empty int list
"""


def gen_op_importer(schema, file):
    indent = inc_indent()
    s = indent + 'if (opName == "' + schema.name + '")\n'

    expected_num_operands = len(schema.inputs)
    expected_num_results = len(schema.outputs)
    for input in schema.inputs:
        if OpSchema.FormalParameterOption.Variadic == input.option:
            expected_num_operands = -1
    for output in schema.outputs:
        if OpSchema.FormalParameterOption.Variadic == output.option:
            expected_num_results = -1

    if ONNX_ML:
        handler_func = special_op_handler.get(
            schema.name, "buildOperation<mlir::MLONNX{}Op>".format(schema.name))
    else:
        handler_func = special_op_handler.get(
            schema.name, "buildOperation<mlir::ONNX{}Op>".format(schema.name))

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
    s += inc_indent(indent) + " {}({});\n".format(
        handler_func, ", ".join(args))

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
        if domain == ONNX_ML_DOMAIN:
            version_dict = onnx_ml_version_dict
        else:
            version_dict = onnx_version_dict
        processed_supportmap = list()
        for _support, _namemap in sorted(_supportmap.items()):
            processed_namemap = list()
            for n, unsorted_versions in sorted(_namemap.items()):
                versions = sorted(unsorted_versions,
                                  key=lambda s: s.since_version)
                schema = versions[-1]
                if schema.name in exsting_ops:
                    continue

                if check_operation_version :
                    # Generate operation of the latest version of your onnx.
                    exsting_ops.add(schema.name)
                    processed_namemap.append((n, schema, versions))

                    # Add checks against version_dict
                    if schema.name not in version_dict :
                        print("Check-operation-version: Operation {} with version is new".format(
                            schema.since_version, schema.name))
                    elif schema.since_version >  version_dict[schema.name]:
                        print("Check-operation-version: Operation {} has a newer version {}"+
                            "(old version {})".format( schema.name, 
                            schema.since_version, version_dict[schema.name]))
                else:
                    # Generate operation according to the version in version_dict.
                    if schema.name not in version_dict :
                        continue
                    found = False
                    for schema in reversed(versions):
                        # Check the version number against the version_dict
                        if schema.since_version == version_dict[schema.name]:
                            exsting_ops.add(schema.name)
                            processed_namemap.append((n, schema, versions))
                            found = True
                            break
                    if not found:
                        print("Your onnx may be too old."
                           "right version for opertion {} not found".format(
                            schema.name))
            processed_supportmap.append((_support, processed_namemap))
        operator_schemas.append((domain, processed_supportmap))
    return operator_schemas


def main(args):  # type: (Type[Args]) -> None
    curr_utc_time = datetime.datetime.now(
        datetime.timezone.utc).strftime("%m/%d/%Y, %H:%M:%S")
    autogen_warning = (
        '//********************************************************\n'
        '//   Do not modify this file directly.\n'
        '//   This file is automatically generated via script.\n'
        '//   Details can be found in docs/readonnxdefs.md .\n'
        '//********************************************************\n\n')
    autogen_warning = autogen_warning.format(curr_utc_time)

    op_def = args.op_def
    op_def.write(autogen_warning)

    op_importer = args.op_importer
    op_importer.write(autogen_warning)

    version_dict = dict()
    for domain, supportmap in build_operator_schemas():
        for _, namemap in supportmap:
            for op_type, schema, versions in namemap:
                if check_operation_version:
                    version_dict[schema.name] = schema.since_version
                else:
                    gen_op_importer(schema, op_importer)
                    r = gen_op_def(schema)
                    op_def.write(r)
    if check_operation_version :
        pprint.pprint(version_dict)

if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    class Args(object):
        if args.dry_run_onnx_ops:
            op_def = StringIO()
        else:
            if args.domain == 'ONNX_ML':
                op_def_file_path = os.path.join(curr_dir, 'MLONNXOps.td.inc')
            else:
                op_def_file_path = os.path.join(curr_dir, 'ONNXOps.td.inc')
            op_def = io.open(op_def_file_path, 'w', newline='')

        if args.dry_run_op_build_table:
            op_importer = StringIO()
        else:
            if args.domain == 'ONNX_ML':
                op_importer_file_path = os.path.join(curr_dir, 'MLOpBuildTable.inc')
            else :
                op_importer_file_path = os.path.join(curr_dir, 'OpBuildTable.inc')
            op_importer = io.open(op_importer_file_path, 'w', newline='')
    main(Args)

    if args.dry_run_onnx_ops:
        sys.stdout.write(Args.op_def.getvalue())
    if args.dry_run_op_build_table:
        sys.stdout.write(Args.op_importer.getvalue())

