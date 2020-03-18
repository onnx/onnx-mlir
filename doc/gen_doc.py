#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict, OrderedDict
import io
import os
import sys
import datetime

import numpy as np  # type: ignore

from onnx import defs, FunctionProto, helper, OperatorStatus
from onnx.defs import OpSchema, ONNX_DOMAIN, ONNX_ML_DOMAIN
from onnx.backend.test.case import collect_snippets
from onnx.backend.sample.ops import collect_sample_implementations
from typing import Any, Text, Sequence, Dict, List, Type, Set, Tuple

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
    ("Conv", "ImportNodeConv"),
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
    'Sign', 'Constant', 'ONNXAveragePoolOp', 'Abs'
]

# Operations supporting canonicalization.
OpsWithCanonicalizer = [
    'Add', 'Identity', 'Gemm'
]

# Add an Op in this list if the Op needs result type deduction which is required
# when writing declarative rewriting rules. Deduced type is always
# an UnrankedTensorType whose element type is the same as the first operand's
# element type.
#
# Currenlty, there are only two build methods generated:
#  - one with operands and attributes having a separate parameter, and
#  - one with operands and attributes having aggregated parameters.
custom_builder_ops_list = ['Abs', 'Mul', 'Exp', 'ReduceSum', 'ReduceSumSquare']

SNIPPETS = collect_snippets()
SAMPLE_IMPLEMENTATIONS = collect_sample_implementations()
ONNX_ML = not bool(os.getenv('ONNX_ML') == '0')

ONNX_ML = False
print("ONNX_ML", ONNX_ML)

if ONNX_ML:
    ext = '-ml.md'
else:
    ext = '.md'


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
    tfrom = np.array([
        'bool', 'int8', 'int16', 'int32', 'int64', 'unkown', 'float16',
        'float', 'double'
    ])
    tto = np.array(
        ['I1', 'I8', 'I16', 'I32', 'I64', 'BF16', 'F16', 'F32', 'F64'])
    index = -1
    for i in range(len(tfrom)):
        if tfrom[i] in tstr:
            index = i
            break
    if index == -1:
        print("error", tstr)
        return ''
    else:
        return tto[i]


def get_allowed_elem_types(schema, input):
    allowed_types_str = None
    return allowed_types_str
    # TODO: enable type constraints.
    # if input.typeStr :
    #     tstr = input.typeStr
    # else :
    #     return allwedTypeStr
    # if schema.type_constraints:
    #     for type_constraint in schema.type_constraints:
    #         if type_constraint.type_param_str != tstr :
    #             continue
    #         allowedTypes = type_constraint.allowed_type_strs
    #         allowedTypeStr=''
    #         if (len(allowedTypes) > 0):
    #             t = convert_type(allowedTypes[0])
    #             if t == '' :
    #                 return ''
    #             allowedTypeStr += t
    #         for allowedType in allowedTypes[1:]:
    #             t = convert_type(allowedType)
    #             if t == '' :
    #                 return ''
    #             if  not t in allowedTypeStr :
    #                 allowedTypeStr += ', '+t
    #
    #         return allowedTypeStr
    #
    # return allowedTypeStr


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
    for value in value_list:
        elem_types = get_allowed_elem_types(schema, value)

        if elem_types is None:
            types = ["AnyMemRef", "AnyTensor"]
        else:
            types = ["TensorOf<[{}]>", "MemRefOf<[{}]>"]
            types = list(map(lambda x: x.format(elem_types), types))

        if OpSchema.FormalParameterOption.Optional == value.option:
            types.append("NoneType")
        elif OpSchema.FormalParameterOption.Variadic == value.option:
            if value.isHomogeneous:
                types = ["Variadic<{}>".format(any_type_of(types))]
            else:
                #TODO handle(variadic, heterogeneous) "
                print("warning: (variadic, heterogeneous) for" + schema.name +
                      ' ' + value.name)

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


def gen_op_def(schema):
    indent = inc_indent()
    s = 'def ONNX{0}Op:ONNX_Op<"{0}",\n'.format(schema.name)

    # Generate decl for op traits.
    traits = ["NoSideEffect"]
    if schema.name in OpsWithShapeInference:
        traits.append("DeclareOpInterfaceMethods<ShapeInferenceOpInterface>")
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
            # E.g. OpBuilder<"Builder *builder, OperationState &state, Value X, Value, Y, Attribute A", [{}]>
            indent = inc_indent(indent)
            s += indent + 'OpBuilder<"Builder *builder, OperationState &state'
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
            # E.g. OpBuilder<"Builder *builder, OperationState &state, ValueRange operands, ArrayRef<NamedAttribute> attributes", [{}]>'
            s += indent + 'OpBuilder<"Builder *builder, OperationState &state, ValueRange operands, ArrayRef<NamedAttribute> attributes", [{\n'
            indent = inc_indent(indent)
            s += indent + 'auto elementType = operands[0].getType().cast<TensorType>().getElementType();\n'
            s += indent + 'std::vector<mlir::Type> outputTypes;\n'
            s += indent + 'outputTypes.emplace_back(UnrankedTensorType::get(elementType));\n'
            s += indent + 'build(builder, state, outputTypes, operands, attributes);\n'
            indent = dec_indent(indent)
            s += indent + '}]>'

            s += '\n' + indent + '];\n'

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

    handler_func = special_op_handler.get(
        schema.name, "buildOperation<mlir::ONNX{}Op>".format(schema.name))

    # Special handlers currently require expected num operands/results to be specified.
    # TODO: remove special handlers.
    args = ["node"]
    if expected_num_operands != -1 or expected_num_results != -1 or "buildOperation" not in handler_func:
        args.append(
            "/* expected_num_operands = */ {}".format(expected_num_operands))
        args.append(
            '/* expected_num_results = */ {}'.format(expected_num_results))
    s += inc_indent(indent) + "return {}({});\n".format(
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

        processed_supportmap = list()
        for _support, _namemap in sorted(_supportmap.items()):
            processed_namemap = list()
            for n, unsorted_versions in sorted(_namemap.items()):
                versions = sorted(unsorted_versions,
                                  key=lambda s: s.since_version)
                schema = versions[-1]
                if schema.name in exsting_ops:
                    continue
                exsting_ops.add(schema.name)
                processed_namemap.append((n, schema, versions))
            processed_supportmap.append((_support, processed_namemap))
        operator_schemas.append((domain, processed_supportmap))
    return operator_schemas


def main(args):  # type: (Type[Args]) -> None
    curr_utc_time = datetime.datetime.now(
        datetime.timezone.utc).strftime("%m/%d/%Y, %H:%M:%S")
    autogen_warning = (
        '//********************************************************\n'
        '//   This file is generated on UTC-{}.\n'
        '//   Do not modify this file directly.\n'
        '//   This file is automatically generated via script.\n'
        '//   Details can be found in doc/readonnxdefs.md .\n'
        '//********************************************************\n\n')
    autogen_warning = autogen_warning.format(curr_utc_time)

    op_def = io.open(args.op_def_file, 'w', newline='')
    op_def.write(autogen_warning)

    op_importer = io.open(args.op_importer_file, 'w', newline='')
    op_importer.write(autogen_warning)

    for domain, supportmap in build_operator_schemas():
        for _, namemap in supportmap:
            for op_type, schema, versions in namemap:
                gen_op_importer(schema, op_importer)
                r = gen_op_def(schema)
                op_def.write(r)


if __name__ == '__main__':
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    class Args(object):
        op_def_file = os.path.join(curr_dir, 'onnxop.inc')
        op_importer_file = os.path.join(curr_dir, 'OpBuildTable.inc')

    main(Args)
