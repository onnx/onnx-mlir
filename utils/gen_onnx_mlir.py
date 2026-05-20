#!/usr/bin/env python3

# After modifying this file, the script will need to run to rebuild the
# onnx-mlir ONNX Dialect. This is performed by calling
# `make OMONNXOpsIncTranslation` in the build dir.
# Alternatively, gen_onnx_mlir_multiple_custom_ops.sh can be used to generate
# the ONNX Dialect with custom dialect extensions.
# Before running these scripts, ensure that onnx==1.19.1 is installed
# in the python environment.
# If the changes are not seen, then you need to rebuild the entire onnx-mlir.

# After changes that impact the documentation of the ops, run
# "make onnx-mlir-docs".

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
import yaml

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dry-run-onnx-ops",
    help="Output ONNXOps.td.inc content to stdout.",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--dry-run-op-build-table",
    help="Output OpBuildTable.inc content to stdout.",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--check-operation-version",
    help="check whether the imported onnx package has new operation or "
    " newer version of operation compared with version stored in version_dict",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--list-operation-version",
    help="list the version stored in version_dict without performing checks",
    action="store_true",
    default=False,
)
parser.add_argument(
    "--custom-ops-yaml",
    type=str,
    help="Path to YAML file containing custom operation schemas",
    default=None,
    action="append",
)

args = parser.parse_args()

check_operation_version = args.check_operation_version
list_operation_version = args.list_operation_version

# ==UPDATE_ONNX_VERSION_OPSET==
# Look for tag above and update all references when upgrading the ONNX support within ONNX-MLIR.
current_onnx_version = "1.19.1"

# Check the version of onnx package being used.
# Allow version mismatch if only processing custom ops from YAML
if (
    not check_operation_version and not list_operation_version
) and current_onnx_version != onnx.__version__:
    if args.custom_ops_yaml and all(
        os.path.exists(yaml_path) for yaml_path in args.custom_ops_yaml
    ):
        print(
            "WARNING: version of expected onnx is {}, ".format(current_onnx_version)
            + "while onnx package being used is {}".format(onnx.__version__)
        )
        print("Continuing with custom ops from YAML...")
    else:
        print(
            "version of expected onnx is {}, ".format(current_onnx_version)
            + "while onnx package being used is {}".format(onnx.__version__)
        )
        quit()

# Record the version of each operation that is treated as the current version.
# To check whether the onnx package being used has newer version operation,
# run this script with --check-operation-version flag.
# Update this dictionary when a newer version is implemented
# TODO: how to keep the old version

version_dict = {
    "": {
        "Abs": [13],
        "Acos": [22],
        "Acosh": [22],
        "Add": [14],
        "AffineGrid": [20],
        "And": [7],
        "ArgMax": [13],
        "ArgMin": [13],
        "Asin": [22],
        "Asinh": [22],
        "Atan": [22],
        "Atanh": [22],
        "Attention": [23],
        "AveragePool": [22],
        "BatchNormalization": [15, 9],
        "Bernoulli": [22],
        "BitShift": [11],
        "BitwiseAnd": [18],
        "BitwiseNot": [18],
        "BitwiseOr": [18],
        "BitwiseXor": [18],
        "BlackmanWindow": [17],
        "Cast": [21],
        "CastLike": [21],
        "Ceil": [13],
        "Celu": [12],
        "CenterCropPad": [18],
        "Clip": [13, 12, 11, 6],
        "Compress": [11],
        "Concat": [13],
        "ConcatFromSequence": [11],
        "Constant": [21],
        "ConstantOfShape": [21],
        "Conv": [22],
        "ConvInteger": [10],
        "ConvTranspose": [22],
        "Cos": [22],
        "Cosh": [22],
        "Col2Im": [18],
        "CumSum": [14],
        "DeformConv": [22],
        "DepthToSpace": [13],
        "DequantizeLinear": [21],
        "Det": [22],
        "DFT": [20, 17],
        "Div": [14],
        "Dropout": [22],
        "DynamicQuantizeLinear": [11],
        "Einsum": [12],
        "Elu": [22],
        "Equal": [19],
        "Erf": [13],
        "Exp": [13],
        "Expand": [13],
        "EyeLike": [22],
        "Flatten": [21],
        "Floor": [13],
        "GRU": [22],
        "Gather": [13],
        "GatherElements": [13],
        "GatherND": [13],
        "Gelu": [20],
        "Gemm": [13],
        "GlobalAveragePool": [22],
        "GlobalLpPool": [2],
        "GlobalMaxPool": [22],
        "Greater": [13],
        "GreaterOrEqual": [16],
        "GridSample": [22, 20, 16],
        "GroupNormalization": [21, 18],
        "HammingWindow": [17],
        "HannWindow": [17],
        "HardSigmoid": [22],
        "Hardmax": [13],
        "HardSwish": [22],
        "Identity": [21],
        "If": [21],
        "ImageDecoder": [20],
        "InstanceNormalization": [22],
        "IsInf": [20],
        "IsNaN": [20],
        "LayerNormalization": [17],
        "LRN": [13],
        "LSTM": [22],
        "LeakyRelu": [16],
        "Less": [13],
        "LessOrEqual": [16],
        "Log": [13],
        "LogSoftmax": [13],
        "Loop": [21],
        "LpNormalization": [22],
        "LpPool": [22],
        "MatMul": [13],
        "MatMulInteger": [10],
        "Max": [13],
        "MaxPool": [22],
        "MaxRoiPool": [22],
        "MaxUnpool": [22],
        "Mean": [13],
        "MeanVarianceNormalization": [13],
        "MelWeightMatrix": [17],
        "Min": [13],
        "Mish": [22],
        "Mod": [13],
        "Mul": [14],
        "Multinomial": [22],
        "Neg": [13],
        "NegativeLogLikelihoodLoss": [22],
        "NonMaxSuppression": [11],
        "NonZero": [13],
        "Not": [1],
        "OneHot": [11],
        "Optional": [15],
        "OptionalGetElement": [18],
        "OptionalHasElement": [18],
        "Or": [7],
        "PRelu": [16],
        "Pad": [21, 18, 13, 11, 2],
        "Pow": [15],
        "QLinearConv": [10],
        "QLinearMatMul": [21],
        "QuantizeLinear": [21],
        "RNN": [22],
        "RandomNormal": [22],
        "RandomNormalLike": [22],
        "RandomUniform": [22],
        "RandomUniformLike": [22],
        "Range": [11],
        "Reciprocal": [13],
        "ReduceL1": [18, 13],
        "ReduceL2": [18, 13],
        "ReduceLogSum": [18, 13],
        "ReduceLogSumExp": [18, 13],
        "ReduceMax": [20, 18, 13],
        "ReduceMean": [18, 13],
        "ReduceMin": [20, 18, 13],
        "ReduceProd": [18, 13],
        "ReduceSum": [13, 11],
        "ReduceSumSquare": [18, 13],
        "RegexFullMatch": [20],
        "Relu": [14],
        "Reshape": [21],
        "Resize": [19, 18, 13, 11, 10],
        "ReverseSequence": [10],
        "RoiAlign": [22],
        "RotaryEmbedding": [23],
        "Round": [22],
        "Scan": [21],
        "Scatter": [11],
        "ScatterElements": [18],
        "ScatterND": [18],
        "Selu": [22],
        "SequenceAt": [11],
        "SequenceConstruct": [11],
        "SequenceEmpty": [11],
        "SequenceErase": [11],
        "SequenceInsert": [11],
        "SequenceLength": [11],
        "SequenceMap": [17],
        "Shape": [21],
        "Shrink": [9],
        "Sigmoid": [13],
        "Sign": [13],
        "Sin": [22],
        "Sinh": [22],
        "Size": [21],
        "Slice": [13],
        "Softmax": [13, 11],
        "SoftmaxCrossEntropyLoss": [13],
        "Softplus": [22],
        "Softsign": [22],
        "SpaceToDepth": [13],
        "Split": [18, 13, 11],
        "SplitToSequence": [11],
        "Sqrt": [13],
        "Squeeze": [21, 11],
        "StringConcat": [20],
        "StringNormalizer": [10],
        "StringSplit": [20],
        "STFT": [17],
        "Sub": [14],
        "Sum": [13],
        "Tan": [22],
        "Tanh": [13],
        "TfIdfVectorizer": [9],
        "ThresholdedRelu": [22],
        "Tile": [13],
        "TopK": [24],
        "Transpose": [21],
        "Trilu": [14],
        "Unique": [11],
        "Unsqueeze": [21, 11],
        "Upsample": [10, 7],
        "Where": [16],
        "Xor": [7],
    },
    "ai.onnx.ml": {
        "ArrayFeatureExtractor": [1],
        "Binarizer": [1],
        "CastMap": [1],
        "CategoryMapper": [1],
        "DictVectorizer": [1],
        "FeatureVectorizer": [1],
        "Imputer": [1],
        "LabelEncoder": [4],
        "LinearClassifier": [1],
        "LinearRegressor": [1],
        "Normalizer": [1],
        "OneHotEncoder": [1],
        "SVMClassifier": [1],
        "SVMRegressor": [1],
        "Scaler": [1],
        "TreeEnsemble": [5],
        "TreeEnsembleClassifier": [5],
        "TreeEnsembleRegressor": [5],
        "ZipMap": [1],
    },
    "ai.onnx.preview.training": {
        "Adagrad": [1],
        "Adam": [1],
        "Gradient": [1],
        "Momentum": [1],
    },
}
additional_op_version_dict = {
    "com.amd.quark": {
        "BFPQuantizeDequantize": [1],
        "ExtendedQuantizeLinear": [1],
        "ExtendedDequantizeLinear": [1],
    }
}


domain_abrv_dict = {
    "": "ONNX",
    "ai.onnx.ml": "ONNX",
    "ai.onnx.preview.training": "ONNX",
    "com.amd.quark": "AMDQuark",
    "com.amd.xfe": "XFE",
    "com.amd.xcompiler": "XCOMPILER",
}


def map_op_name_to_onnx_mlir_name(op_name: str, domain: str) -> str:
    assert domain in domain_abrv_dict, f"Domain {domain} not found in domain_abrv_dict"
    return domain_abrv_dict[domain] + op_name + "Op"


# Custom Op Schema class to mimic OpSchema for YAML-defined ops
class CustomOpSchema:
    """Wrapper class to make YAML-defined ops look like ONNX OpSchema objects."""

    class FormalParameter:
        def __init__(self, param_dict):
            self.name = param_dict["name"]
            self.description = param_dict.get("description", "")
            self.type_str = param_dict.get("type_str", "")
            self.option = OpSchema.FormalParameterOption.Single
            self.is_homogeneous = True

            if param_dict.get("optional", False):
                self.option = OpSchema.FormalParameterOption.Optional
            elif param_dict.get("variadic", False):
                self.option = OpSchema.FormalParameterOption.Variadic
                self.is_homogeneous = param_dict.get("is_homogeneous", True)

    class Attribute:
        def __init__(self, attr_dict):
            self.name = attr_dict["name"]
            self.description = attr_dict.get("description", "")
            self.required = attr_dict.get("required", False)
            self.default_value = type(
                "obj", (object,), {"name": attr_dict.get("default_value")}
            )()

            # Map type string to OpSchema.AttrType
            type_map = {
                "int": OpSchema.AttrType.INT,
                "int32": OpSchema.AttrType.INT,
                "float": OpSchema.AttrType.FLOAT,
                "string": OpSchema.AttrType.STRING,
                "tensor": OpSchema.AttrType.TENSOR,
                "list(int)": OpSchema.AttrType.INTS,
                "list(float)": OpSchema.AttrType.FLOATS,
                "list(string)": OpSchema.AttrType.STRINGS,
                "bool": OpSchema.AttrType.INT,
            }
            self.type_str = attr_dict.get("type", "int")
            self.type = type_map.get(self.type_str, OpSchema.AttrType.INT)

    class TypeConstraint:
        def __init__(self, tc_dict):
            self.type_param_str = tc_dict["type_param"]
            self.description = tc_dict.get("description", "")
            self.allowed_type_strs = tc_dict.get("allowed_types", [])

    def __init__(self, yaml_dict):
        self.name = yaml_dict["name"]
        self.domain = yaml_dict.get("domain", "")
        self.since_version = yaml_dict.get("since_version", 1)
        self.support_level = 1  # COMMON
        self.doc = yaml_dict.get("description", "")

        # Parse inputs
        self.inputs = []
        for inp in yaml_dict.get("inputs", []):
            self.inputs.append(self.FormalParameter(inp))

        # Parse outputs
        self.outputs = []
        for out in yaml_dict.get("outputs", []):
            self.outputs.append(self.FormalParameter(out))

        # Parse attributes
        self.attributes = {}
        for attr in yaml_dict.get("attributes", []):
            attr_obj = self.Attribute(attr)
            self.attributes[attr_obj.name] = attr_obj

        # Parse type constraints
        self.type_constraints = []
        for tc in yaml_dict.get("type_constraints", []):
            self.type_constraints.append(self.TypeConstraint(tc))

        # Min/max counts
        self.min_input = yaml_dict.get("min_input", len(self.inputs))
        self.max_input = yaml_dict.get("max_input", len(self.inputs))
        self.min_output = yaml_dict.get("min_output", len(self.outputs))
        self.max_output = yaml_dict.get("max_output", len(self.outputs))

        # Parse meta_attributes for verify, fold, etc.
        self.meta_attributes = {}
        for meta_attr in yaml_dict.get("meta_attributes", []):
            if isinstance(meta_attr, dict):
                self.meta_attributes.update(meta_attr)


def load_custom_ops_from_yaml(yaml_paths: List[str]) -> List[CustomOpSchema]:
    """Load custom operation schemas from a YAML file."""
    if not yaml_paths or not all(os.path.exists(yaml_path) for yaml_path in yaml_paths):
        return []
    custom_schemas = []
    for yaml_path in yaml_paths:
        with open(yaml_path, "r") as f:
            yaml_content = yaml.safe_load(f)

        # YAML file can contain a list of ops or a single op
        if isinstance(yaml_content, list):
            ops_list = yaml_content
        else:
            ops_list = [yaml_content]

        for op_dict in ops_list:
            try:
                schema = CustomOpSchema(op_dict)
                custom_schemas.append(schema)
            except Exception as e:
                print(
                    f"Warning: Failed to parse op {op_dict.get('name', 'unknown')}: {e}"
                )

    return custom_schemas


# Extra tensor types patched into ops' ONNX type constraints after parsing.
# Keyed by op name. Value is a dict of type_param -> extra allowed type strings:
#   "*" applies to all type params, named keys (e.g. "T4") target one param.
special_type_constraints = {
    "QLinearConv": {
        "*": ["tensor(uint16)"],
        "T4": ["tensor(int16)", "tensor(int8)"],
    },
    "QLinearMatMul": {
        "*": ["tensor(int16)", "tensor(uint16)"],
    },
    "Concat": {
        "T": ["tensor(uint4)", "tensor(int4)"],
    },
    "Slice": {
        "T": ["tensor(uint4)", "tensor(int4)"],
    },
}

# Manual specification of attribute type.
special_attr_types = dict([("Cast.to", "type")])

# Manual specification of attribute order:
# The names in each tuple will be ordered in that sequence.
special_attr_order = {
    ("then_branch", "else_branch"),
}

# Special operation importing handlers.
special_op_handler = dict(
    [
        ("BatchNormalization", "ImportNodeBatchNormalization"),
        ("CategoryMapper", "ImportCategoryMapper"),
        ("Dropout", "ImportNodeDropout"),
        ("Cast", "ImportNodeCast"),
        ("MaxPool", "ImportNodeMaxPool"),
        ("Pad", "ImportNodePad"),
        ("Slice", "ImportNodeSlice"),
    ]
)

# Operations with custom assembly format (alphabetical order).
OpsWithCustomAssemblyFormat = [
    "Constant",
    "ConstantOfShape",
]

# Operations supporting canonicalization (alphabetical order).
OpsWithCanonicalizer = [
    "Abs",
    "Add",
    "And",
    "AveragePool",
    "BatchNormalization",
    "BatchNormalizationV9",
    "Cast",
    "Clip",
    "Concat",
    "Constant",
    "DepthToSpace",
    "DequantizeLinear",
    "Div",
    "Dropout",
    "Equal",
    "GlobalAveragePool",
    "GlobalMaxPool",
    "Greater",
    "GRU",
    "Identity",
    "LeakyRelu",
    "Less",
    "Loop",
    "LSTM",
    "Mul",
    "Or",
    "Pow",
    "ReduceMean",
    "Reshape",
    "Resize",
    "RNN",
    "Shape",
    "Split",
    "Size",
    "Softmax",
    "SoftmaxV11",
    "SpaceToDepth",
    "Squeeze",
    "SqueezeV11",
    "Sub",
    "Tile",
    "Transpose",
    "Unsqueeze",
    "UnsqueezeV11",
    "Where",
    "Xor",
]

# Operations with custom verifiers (alphabetical order).
OpsWithVerifier = [
    "Add",
    "And",
    "ArgMax",
    "ArgMin",
    "Attention",
    "AveragePool",
    "Bernoulli",
    "BitShift",
    "BitwiseAnd",
    "BitwiseOr",
    "BitwiseXor",
    "Cast",
    "CastLike",
    "CategoryMapper",
    "Compress",
    "Concat",
    "ConcatFromSequence",
    "ConstantOfShape",
    "Conv",
    "ConvTranspose",
    "DepthToSpace",
    "DequantizeLinear",
    "Div",
    "Einsum",
    "Equal",
    "Expand",
    "EyeLike",
    "Flatten",
    "Gather",
    "GatherElements",
    "GatherND",
    "Gelu",
    "Greater",
    "GreaterOrEqual",
    "GridSample",
    "GroupNormalizationV18",
    "Hardmax",
    "If",
    "IsInf",
    "InstanceNormalization",
    "LayerNormalization",
    "Less",
    "LessOrEqual",
    "LogSoftmax",
    "Max",
    "MatMulInteger",
    "Mean",
    "Min",
    "Mod",
    "Mul",
    "NonMaxSuppression",
    "OneHot",
    "OneHotEncoder",
    "Optional",
    "OptionalGetElement",
    "OptionalHasElement",
    "Or",
    "PRelu",
    "Pad",
    "Pow",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
    "Range",
    "Reshape",
    "Resize",
    "ReverseSequence",
    "RoiAlign",
    "RotaryEmbedding",
    "ScatterElements",
    "ScatterND",
    "SequenceEmpty",
    "SequenceInsert",
    "Shape",
    "SpaceToDepth",
    "Split",
    "SplitToSequence",
    "Sub",
    "Sum",
    "TopK",
    "Unique",
    "Upsample",
    "Where",
    "Xor",
]

# Op with fold function
OpsWithFolder = ["Constant", "Squeeze", "SqueezeV11", "ReduceMean", "Slice", "Clip"]

# Op with ConstantLike trait
OpsWithConstantLike = ["Constant"]

# Op with SameOperandsAndResultShape trait
OpsWithSameOperandsAndResultShape = [
    "Abs",
    "Acos",
    "Acosh",
    "Asin",
    "Asinh",
    "Atan",
    "Atanh",
    "Cast",
    "Ceil",
    "Celu",
    "Cos",
    "Cosh",
    "Elu",
    "Erf",
    "Exp",
    "EyeLike",
    "Floor",
    "Gelu",
    "HardSigmoid",
    "HardSwish",
    "Identity",
    "LRN",
    "LeakyRelu",
    "Log",
    "Neg",
    "Not",
    "Relu",
    "Round",
    "Selu",
    "Sigmoid",
    "Sign",
    "Sin",
    "Sinh",
    "Sqrt",
    "Swish",
    "Tan",
    "Tanh",
]

# Op with Helper functions
OpsWithHelpers = {
    "EyeLike": """
    mlir::Type getResultElementType();
  """,
    "Loop": """
    mlir::Operation::result_range v_final();
    mlir::Operation::result_range scan_outputs();
  """,
    "Scan": """
    mlir::Operation::operand_range getVInitial();
    mlir::Operation::result_range v_final();
    mlir::Operation::operand_range scan_inputs();
    mlir::Operation::result_range scan_outputs();
  """,
}

# Type inference are usually done with the type string for Op definition.
# This dictionary provides special code for type inference for some Ops.
# The type inference is used only in Builder before constant canonicalization.
OpsWithResultTypeInference = [
    "Bernoulli",
    "Constant",
    "Cast",
    "CastLike",
    "ConcatFromSequence",
    "ConstantOfShape",
    "EyeLike",
    "If",
    "Loop",
    "RandomNormal",
    "RandomNormalLike",
    "RandomUniform",
    "RandomUniformLike",
    "Scan",
    "SequenceAt",
    "SequenceConstruct",
    "SequenceEmpty",
    "SequenceMap",
    "SplitToSequence",
]

FloatTypes = {"TensorOf<[F32]>"}

# Add an Op in this list if the Op needs result type deduction which is required
# when writing declarative rewriting rules. Deduced type is always
# an UnrankedTensorType whose element type is the same as the first operand's
# element type.
#
# Currently, there are only two build methods generated:
#  - one with operands and attributes having a separate parameter, and
#  - one with operands and attributes having aggregated parameters.
custom_builder_unranked_ops_list = [
    "Abs",
    "Conv",
    "Exp",
    "Identity",
    "Neg",
    "Pad",
    "ReduceLogSum",
    "ReduceMaxV13",
    "ReduceMaxV18",
    "ReduceMax",
    "ReduceSum",
    "ReduceSumSquare",
    "ReduceSumV11",
    "Softmax",
    "Split",
    "SplitV13",
    "Sqrt",
    "Squeeze",
    "SqueezeV11",
    "Unsqueeze",
    "UnsqueezeV11",
]
# Custom builder op list for operations with broadcast; we can deduce the right
# output type, no need to leave it undef as in the above list.
# Ops must have two operands, not one, not three... And there shall be two.
# TODO: handle variadic ops omitted here: Max, Min, Min, Sum.
custom_builder_broadcast_to_same_type_ops_list = [
    "Add",
    "And",
    "Div",
    "Mul",
    "Or",
    "Pow",
    "Sub",
    "Xor",
]
custom_builder_broadcast_to_bool_ops_list = [
    "Equal",
    "Greater",
    "GreaterOrEqual",
    "Less",
    "LessOrEqual",
]
custom_builder_broadcast_ops_list = (
    custom_builder_broadcast_to_same_type_ops_list
    + custom_builder_broadcast_to_bool_ops_list
)
# Union of both.
custom_builder_ops_list = (
    custom_builder_unranked_ops_list + custom_builder_broadcast_ops_list
)

# A dictionary to add any special definition for an operation.
custom_definition_misc = dict(
    [
        (
            "Constant",
            """  let builders = [
  OpBuilder<(ins "Attribute":$sparse_value, "Attribute":$value), [{
   if (value) {
    auto tensorType = mlir::cast<TypedAttr>(value).getType();
    build($_builder, $_state, tensorType, sparse_value, value,
      FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(), ArrayAttr());
   } else {
    auto tensorType = mlir::cast<TypedAttr>(sparse_value).getType();
    build($_builder, $_state, tensorType, sparse_value, value,
      FloatAttr(), ArrayAttr(), IntegerAttr(), ArrayAttr(), StringAttr(), ArrayAttr());
   }
  }]>
  ];""",
        ),
        (
            "Cast",
            """   let builders = [
  OpBuilder<(ins "Value":$input, "IntegerAttr":$saturate, "TypeAttr":$to), [{
   auto resultType = mlir::UnrankedTensorType::get(to.getValue());
   build($_builder, $_state, resultType, input, saturate, to);
  }] >
  ];""",
        ),
    ]
)

# Get this order from TensorProto in https://github.com/onnx/onnx/blob/main/onnx/onnx.in.proto#L481.
# enum DataType {
#     UNDEFINED = 0;
#     // Basic types.
#     FLOAT = 1;   // float
#     UINT8 = 2;   // uint8_t
#     INT8 = 3;    // int8_t
#     UINT16 = 4;  // uint16_t
#     INT16 = 5;   // int16_t
#     INT32 = 6;   // int32_t
#     INT64 = 7;   // int64_t
#     STRING = 8;  // string
#     BOOL = 9;    // bool
#
#     // IEEE754 half-precision floating-point format (16 bits wide).
#     // This format has 1 sign bit, 5 exponent bits, and 10 mantissa bits.
#     FLOAT16 = 10;
#
#     DOUBLE = 11;
#     UINT32 = 12;
#     UINT64 = 13;
#     COMPLEX64 = 14;     // complex with float32 real and imaginary components
#     COMPLEX128 = 15;    // complex with float64 real and imaginary components
#
#     // Non-IEEE floating-point format based on IEEE754 single-precision
#     // floating-point number truncated to 16 bits.
#     // This format has 1 sign bit, 8 exponent bits, and 7 mantissa bits.
#     BFLOAT16 = 16;
#
#     // Non-IEEE floating-point format based on papers
#     // FP8 Formats for Deep Learning, https://arxiv.org/abs/2209.05433,
#     // 8-bit Numerical Formats For Deep Neural Networks, https://arxiv.org/pdf/2206.02915.pdf.
#     // Operators supported FP8 are Cast, CastLike, QuantizeLinear, DequantizeLinear.
#     // The computation usually happens inside a block quantize / dequantize
#     // fused by the runtime.
#     FLOAT8E4M3FN = 17;    // float 8, mostly used for coefficients, supports nan, not inf
#     FLOAT8E4M3FNUZ = 18;  // float 8, mostly used for coefficients, supports nan, not inf, no negative zero
#     FLOAT8E5M2 = 19;      // follows IEEE 754, supports nan, inf, mostly used for gradients
#     FLOAT8E5M2FNUZ = 20;  // follows IEEE 754, supports nan, inf, mostly used for gradients, no negative zero
#
#     // 4-bit integer data types
#     UINT4 = 21;  // Unsigned integer in range [0, 15]
#     INT4 = 22;   // Signed integer in range [-8, 7], using two's-complement representation
#
#     // Future extensions go here.
#   }
onnx_types = (
    "undefined",
    "float",
    "uint8",
    "int8",
    "uint16",
    "int16",
    "int32",
    "int64",
    "string",
    "bool",
    "float16",
    "double",
    "uint32",
    "uint64",
    "complex64",
    "complex128",
    "bfloat16",
    "float8e4m3fn",
    "float8e4m3fnuz",
    "float8e5m2",
    "float8e5m2fnuz",
    "uint4",
    "int4",
)
tblgen_types = (
    "BF16",
    "F32",
    "AnyUI8",
    "AnyI8",
    "AnyUI16",
    "AnyI16",
    "AnyI32",
    "AnyI64",
    "StringType",
    "AnyI1",
    "F16",
    "F64",
    "AnyUI32",
    "AnyUI64",
    "Complex<F32>",
    "Complex<F64>",
    "BF16",
    "F8E4M3FN",
    "F8E4M3FNUZ",
    "F8E5M2",
    "F8E5M2FNUZ",
    "AnyUI4",
    "AnyI4",
)

# Maximum count for actual type. Number more than MAX_NUM_TYPES will be used to encode
# the mapping method. MAX_NUM_TYPES should be greater than the length of onnx_types
# This value has to be kept the same with MAX_TYPE in FrontendDialectTransformer.cpp
MAX_NUM_TYPES = 30


# Attribute names are ordered alphabetically except for the
# manually specified special orderings in special_attr_order.
def order_attr_names(attrNames):
    attrNames = sorted(attrNames)
    for namesOrder in special_attr_order:
        # If attrNames includes all the namesOrder names, then reorder
        # those names in attrNames to their order in namesOrder,
        if set(namesOrder).issubset(attrNames):
            # The namesIndexes are where the namesOrder names appear in attrNames.
            namesIndexes = (attrNames.index(name) for name in namesOrder)
            # Write the namesOrder names into those indexes in the correct order.
            for name, index in zip(namesOrder, sorted(namesIndexes)):
                attrNames[index] = name
    return attrNames


def should_render_domain(domain):  # type: (Text) -> bool
    return True


def display_attr_type(v):  # type: (OpSchema.AttrType) -> Text
    assert isinstance(v, OpSchema.AttrType)
    s = Text(v)
    s = s[s.rfind(".") + 1 :].lower()
    if s[-1] == "s":
        s = "list of " + s
    return s


# ONNX allows input and output to have the same name. But MLIR does not
# When the conflict occurs, add 'out_' to the output name
def get_unique_output_name(schema, name):
    for input in schema.inputs:
        if input.name == name:
            return "out_" + name
    return name


def onnx_attr_type_to_mlir_attr_type(t):
    onnx_attr_type = Text(t)
    onnx_attr_type = onnx_attr_type[onnx_attr_type.rfind(".") + 1 :].lower()

    if onnx_attr_type == "int":
        mlir_attr_type = "SI64Attr"
    elif onnx_attr_type == "float":
        mlir_attr_type = "F32Attr"
    elif onnx_attr_type == "ints":
        mlir_attr_type = "I64ArrayAttr"
    elif onnx_attr_type == "floats":
        mlir_attr_type = "F32ArrayAttr"
    elif onnx_attr_type == "string":
        mlir_attr_type = "StrAttr"
    elif onnx_attr_type == "strings":
        mlir_attr_type = "StrArrayAttr"
    elif onnx_attr_type == "bool":
        mlir_attr_type = "BoolAttr"
    elif onnx_attr_type in {"type", "type_proto"}:
        # 'type' is the attribute type used in special_attr_types,
        # 'type_proto' is Optional op's type attribute's type
        mlir_attr_type = "TypeAttr"
    else:
        mlir_attr_type = "AnyAttr"
    # TODO: tensor and sparse tensor.
    return mlir_attr_type


# TODO: any better way to do this.
def tblgen_attr_type_to_cpp_type(t):
    if "I64Attr" in t:
        cpp_type = "IntegerAttr"
    elif "I32Attr" in t:
        cpp_type = "IntegerAttr"
    elif "F32Attr" in t:
        cpp_type = "FloatAttr"
    elif "I64ArrayAttr" in t or "F32ArrayAttr" in t:
        cpp_type = "ArrayAttr"
    elif "StrAttr" in t:
        cpp_type = "StringAttr"
    elif "BoolAttr" in t:
        cpp_type = "BoolAttr"
    elif "strings" in t:
        cpp_type = "ArrayAttr"
    else:
        cpp_type = "Attribute"
    return cpp_type


def tblgen_operand_type_to_cpp_type(op_type):
    if op_type.startswith("Variadic"):
        mytype = "ValueRange"
    else:
        mytype = "Value"
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


# The possible data structures are tensor, map and seq(tensor()).
def get_data_structure_element(allowed_type_str):
    structure_list = ["tensor", "seq", "map"]
    for structure in structure_list:
        if allowed_type_str.startswith(structure):
            element = allowed_type_str.replace(structure + "(", "", 1).replace(
                ")", "", 1
            )
            return (structure, element)
    return (None, None)


def get_allowed_elem_types(schema, input):
    # allowed_types_str = None
    # return allowed_types_str
    # TODO: enable type constraints.
    if input.type_str:
        tstr = input.type_str
        structure, element = get_data_structure_element(tstr)
        # In case the type is directly specified.
        if structure and element:
            t = np_type_to_tblgen_attr_type(element)
            if t == None:
                return allowed_structure, None
            else:
                return structure, [t]
    else:
        return None
    if schema.type_constraints:
        for type_constraint in schema.type_constraints:
            if type_constraint.type_param_str != tstr:
                continue
            allowed_type_list = []
            allowedTypes = type_constraint.allowed_type_strs
            allowed_structure = None
            for allowedType in allowedTypes:
                structure, element = get_data_structure_element(allowedType)
                if structure == None or element == None:
                    return None, None

                if allowed_structure != None and allowed_structure != structure:
                    return None, None
                allowed_structure = structure
                t = np_type_to_tblgen_attr_type(element)
                if t == None:
                    return allowed_structure, None
                if not t in allowed_type_list:
                    allowed_type_list.append(t)

            return allowed_structure, allowed_type_list

    return None, None


def inc_indent(indent=None):
    return "" if indent is None else indent + " " * 2


def dec_indent(indent):
    return indent[:-2]


def join_args(args):
    return ", ".join(args)


def get_operands_or_results(schema, type_str_dict, op_name, is_input):
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
        str_types = get_onnx_mlir_types(schema, type_str_dict, value)

        # In case the type string is used more than once.
        types = str_types.copy()

        # No need to add AnyMemRef type. Keep the code in case.
        # types.append("AnyMemRef")

        qType = False
        if op_name == "DequantizeLinear":
            qType = is_input and i == 0
        elif op_name == "QuantizeLinear":
            qType = not is_input and i == 0
        else:
            qType = FloatTypes.issubset(types)
        if qType:
            types.append("TensorOf<[quant_QuantizedType]>")

        if OpSchema.FormalParameterOption.Optional == value.option:
            types.append("NoneType")

        if OpSchema.FormalParameterOption.Variadic == value.option:
            if value.is_homogeneous:
                types = ["Variadic<{}>".format(any_type_of(types))]
            else:
                # TODO handle(variadic, heterogeneous) "
                types = ["Variadic<{}>".format(any_type_of(types))]
                sys.stderr.write(
                    "warning: (variadic, heterogeneous) for "
                    + schema.name
                    + " "
                    + value.name
                    + "\n"
                )

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
        return "OptionalAttr<{}>".format(onnx_attr_type_to_mlir_attr_type(attr_type))

    def get_attr_type_with_default(attr_type, attr_default):
        if attr_type == OpSchema.AttrType.STRING:
            return 'DefaultValuedStrAttr<{}, "{}">'.format(
                onnx_attr_type_to_mlir_attr_type(attr_type), attr_default
            )
        else:
            return 'DefaultValuedAttr<{}, "{}">'.format(
                onnx_attr_type_to_mlir_attr_type(attr_type), attr_default
            )

    if not schema.attributes:
        return OrderedDict()

    def get_default_value_if_present(attr):
        """Return (has_default, default_value).

        For real ONNX schemas, `attr.default_value` is an AttributeProto. Its
        `name` may be populated even when no default literal exists, so use the
        proto `type` to detect real defaults.
        For YAML custom ops, the default is stored directly in `.name`.
        """
        default_proto_or_obj = attr.default_value

        # Real ONNX AttributeProto path.
        if isinstance(default_proto_or_obj, onnx.AttributeProto):
            if default_proto_or_obj.type == onnx.AttributeProto.UNDEFINED:
                return False, None
            default_value = helper.get_attribute_value(default_proto_or_obj)
            # Be defensive: if decoder yields None, treat as no default.
            if default_value is None:
                return False, None
            return True, default_value

        # CustomOpSchema path.
        default_value = getattr(default_proto_or_obj, "name", None)
        if default_value is None:
            return False, None
        return True, default_value

    name_to_type = OrderedDict()
    for _, attr in sorted(schema.attributes.items()):
        if attr.type == OpSchema.AttrType.GRAPH:
            continue

        qualified_attr_name = "{}.{}".format(schema.name, attr.name)
        if qualified_attr_name in special_attr_types:
            name_to_type[attr.name] = onnx_attr_type_to_mlir_attr_type(
                special_attr_types[qualified_attr_name]
            )
        # Option holds either required or default value.
        elif attr.required:
            name_to_type[attr.name] = onnx_attr_type_to_mlir_attr_type(attr.type)
        else:
            has_default, default_value = get_default_value_if_present(attr)
            if not has_default:
                # Optional attribute; use type_str for custom ops
                # (e.g. bool -> BoolAttr, int32 -> I32Attr). Without these
                # overrides custom-op ints always lower to SI64Attr.
                type_str = getattr(attr, "type_str", None)
                if type_str == "bool":
                    name_to_type[attr.name] = "OptionalAttr<BoolAttr>"
                elif type_str == "int32":
                    name_to_type[attr.name] = "OptionalAttr<I32Attr>"
                else:
                    name_to_type[attr.name] = get_attr_type_optional(attr.type)
                continue

            def format_value(value):  # type: (Any) -> Text
                if isinstance(value, float):
                    formatted = str(np.round(value, 5))
                    # Use default formatting, unless too long.
                    if len(formatted) > 10:
                        formatted = str("({:e})".format(value))
                    return formatted
                elif isinstance(value, (bytes, bytearray)) and sys.version_info[0] == 3:
                    return str(value.decode("utf-8"))
                return str(value)

            if isinstance(default_value, list):
                default_value = [format_value(val) for val in default_value]
                default_value_str = "{}".format(default_value)
                default_value_str = default_value_str.replace("[", "{", 1)
                default_value_str = default_value_str.replace("]", "}", 1)
                if Text(attr.type) == "AttrType.STRINGS":
                    default_value_str = default_value_str.replace("'", '\\"')
                else:
                    default_value_str = default_value_str.replace("'", "")
            else:
                default_value = format_value(default_value)
                default_value_str = default_value

            # Custom ops may use type_str (e.g. bool -> BoolAttr)
            type_str = getattr(attr, "type_str", None)
            if type_str == "bool":
                default_value_str = str(default_value).lower()
                name_to_type[attr.name] = 'DefaultValuedAttr<BoolAttr, "{}">'.format(
                    default_value_str
                )
            else:
                name_to_type[attr.name] = get_attr_type_with_default(
                    attr.type, default_value_str
                )
    return name_to_type


def get_numberof_list(my_list):
    expected_num = len(my_list)
    for element in my_list:
        if OpSchema.FormalParameterOption.Variadic == element.option:
            expected_num = -1
    return expected_num


def get_output_type_mapping(schema):
    mapping = []
    for output in schema.outputs:
        # If only one type is allowed, just set that.
        structure, allowed_elem_types = get_allowed_elem_types(schema, output)
        if allowed_elem_types != None and len(allowed_elem_types) == 1:
            mapping.append(str(get_tblgen_type_index(allowed_elem_types[0])))
            continue

        # Map the type string.
        if output.type_str:
            tstr = output.type_str
            found = False
            for i, input in enumerate(schema.inputs):
                if input.type_str and input.type_str == tstr:
                    mapping.append(str(i + MAX_NUM_TYPES))
                    found = True
                    break
            if found:
                continue

        # Unknown output type.
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
    cpp_name_to_idx_literal = (
        "{"
        + ", ".join(
            [
                '{{"{}", {}}}'.format(*name_to_idx)
                for name_to_idx in const_operands_name_to_idx
            ]
        )
        + "}"
    )

    # s += indent + "let extraClassDeclaration = [{\n"
    indent = inc_indent(indent)
    s += indent + "std::map<std::string, size_t> promotableConstOperands() {\n"
    indent = inc_indent(indent)
    s += indent + "return {};\n".format(cpp_name_to_idx_literal)
    indent = dec_indent(indent)
    s += indent + "}\n"
    # indent = dec_indent(indent)
    # s += indent + "}];\n"

    return s


def parse_type_str(allowedType):
    # AnyI may be used for uint because the onnx_mlir is not generating uint output.
    # This will be fixed later and UI will be replace AnyI.
    onnx_to_mlir_type_dict = {
        "(": "<[",
        ")": "]>",
        "tensor": "TensorOf",
        "seq": "SeqOf",
        "map": "TupleOf",
        "bool": "I1",
        "uint4": "UI<4>",
        "uint8": "UI8",
        "uint16": "UI16",
        "uint32": "UI32",
        "uint64": "UI64",
        "int4": "I<4>",
        "int8": "I8",
        "int16": "I16",
        "int32": "I32",
        "int64": "I64",
        "double": "F64",
        "float": "F32",
        "float16": "F16",
        "bfloat16": "BF16",
        "float8e4m3fn": "F8E4M3FN",
        "float8e4m3fnuz": "F8E4M3FNUZ",
        "float8e5m2": "F8E5M2",
        "float8e5m2fnuz": "F8E5M2FNUZ",
        "complex64": "Complex<F32>",
        "complex128": "Complex<F64>",
        "string": "StringType",
    }

    # Optional(...) always appears outermost.
    if allowedType.find("optional") == 0:
        allowedType = allowedType.replace("optional(", "OptOf<", 1)
        allowedType = allowedType[:-1] + ">"

    # Apply substitutions in decreasing order of key-length, so that float16
    # is replaced before float, and uint16 is replaced before int16, etc.
    mapping = list(onnx_to_mlir_type_dict.items())
    mapping.sort(key=lambda pair: len(pair[0]), reverse=True)
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
    # However onnx keeps a consistently meaningful order
    # There is no redundancy as long as each onnx type is mapped uniquely
    # mlirTypes = sorted(list(set(mlirTypes)))

    return mlirTypes


def parse_type_constraints(schema):
    type_str_dict = dict()
    for type_constraint in schema.type_constraints:
        type_str_dict[type_constraint.type_param_str] = parse_a_type_constraint(
            type_constraint
        )
    if schema.name in special_type_constraints:
        patches = special_type_constraints[schema.name]
        for type_param, types in type_str_dict.items():
            extra = patches.get(type_param, []) + patches.get("*", [])
            for t in extra:
                mlir_t = parse_type_str(t)
                if mlir_t not in types:
                    types.append(mlir_t)
    return type_str_dict


def get_onnx_mlir_types(schema, type_str_dict, input):
    if input.type_str:
        if not input.type_str in type_str_dict:
            # Some arguments use type description directly
            # instead of constraint.
            type_str = parse_type_str(input.type_str)
            return [type_str]
        else:
            return type_str_dict[input.type_str]
    else:
        print("No type_str ", schema.name)
        return []


# Generate extra class declaration for shape helper.
def gen_shape_helper_code(s, indent, opName, domain=""):
    # Print getShapeHelper.
    indent = inc_indent(indent)
    s += (
        indent
        + "onnx_mlir::ONNXOpShapeHelper * $cppClass::getShapeHelper(mlir::Operation *op, llvm::ArrayRef<mlir::Value> oper, \n"
    )
    indent = inc_indent(indent)
    indent = inc_indent(indent)
    s += (
        indent
        + "onnx_mlir::IndexExprBuilder *ieb, onnx_mlir::IndexExprScope *scope) {\n"
    )
    indent = dec_indent(indent)

    # For custom domains, return nullptr as placeholder
    # Custom ops should implement their own shape helpers
    if domain and domain not in ["", "ai.onnx.ml", "ai.onnx.preview.training"]:
        s += indent + "// TODO: Implement specific shape helper for this custom op\n"
        s += (
            indent
            + "// For now, returning nullptr - shape inference will use default behavior\n"
        )
        s += indent + "return nullptr;\n"
    else:
        s += (
            indent
            + "onnx_mlir::ONNXOpShapeHelper *sh = new onnx_mlir::ONNX{0}OpShapeHelper(op, oper, ieb, scope);\n".format(
                opName
            )
        )
        s += indent + 'assert(sh && "failed to allocate shape helper");\n'
        s += indent + "return sh;\n"
    indent = dec_indent(indent)
    s += indent + "}\n"
    return s


def gen_op_name(schema, with_version):
    if with_version:
        return schema.name + "V" + str(schema.since_version)
    return schema.name


# Generate entry for a given operation given by opName (from schema).
def gen_op_def(schema, with_version=False):
    indent = inc_indent()
    opName = gen_op_name(schema, with_version)
    # Get domain abbreviation for the op prefix only
    # All ops use ONNX_Op as base class regardless of domain
    domain = schema.domain if schema.domain else ""
    domain_abbrev = domain_abrv_dict.get(domain, "ONNX")

    # For custom domains (non-empty, non-standard ONNX domains), prefix the MLIR op name
    # This makes them appear as onnx.XFEConvChannelLast instead of onnx.ConvChannelLast
    if domain and domain not in ["", "ai.onnx.ml", "ai.onnx.preview.training"]:
        mlir_op_name = domain_abbrev + opName
    else:
        mlir_op_name = opName

    s = 'def {1}{0}Op:ONNX_Op<"{2}",\n'.format(opName, domain_abbrev, mlir_op_name)

    regions = OrderedDict()
    for name in order_attr_names(schema.attributes.keys()):
        attr = schema.attributes[name]
        if attr.type == OpSchema.AttrType.GRAPH:
            if attr.required:
                regions[attr.name] = "SizedRegion<1>"
            else:
                regions[attr.name] = "AnyRegion"

    # Generate decl for op traits.
    traits = ["Pure", f"OpVersionTrait<{schema.since_version}>"]

    # Generate SameOperandsAndResultShape traits.
    if mlir_op_name in OpsWithSameOperandsAndResultShape:
        traits.append("SameOperandsAndResultShape")

    # Generate ConstantLike traits.
    if mlir_op_name in OpsWithConstantLike:
        traits.append("ConstantLike")

    # OpsWithShapeInference:
    # Now the ShapeInference traits are added to all operation.
    # Dummy implementations are added to ONNXOps.cpp.
    # Error will be report if these operations are encountered at runtime.
    traits.append("DeclareOpInterfaceMethods<ShapeInferenceOpInterface>")
    traits.append("DeclareOpInterfaceMethods<ShapeHelperOpInterface>")
    if mlir_op_name in OpsWithResultTypeInference:
        traits.append("DeclareOpInterfaceMethods<ResultTypeInferenceOpInterface>")
    if len(regions):
        traits.append('OpInterface<"HasOnnxSubgraphOpInterface">')
    s += inc_indent(indent) + "[{}]> {{\n".format(join_args(traits))

    indent = inc_indent(indent)

    # Generate decl for custom assembly format.
    if mlir_op_name in OpsWithCustomAssemblyFormat:
        s += indent + "let hasCustomAssemblyFormat = 1;\n"

    # Generate decl for canonicalizer.
    if mlir_op_name in OpsWithCanonicalizer:
        s += indent + "let hasCanonicalizer = 1;\n"

    # Generate decl for summary.
    s += indent + 'let summary = "{} operation";\n'.format(
        ("ONNX " if domain_abbrev == "ONNX" else "") + schema.name
    )

    # Generate description.
    s += indent + "let description = [{\n"
    if schema.doc:
        lines = schema.doc.lstrip().splitlines()
        for line in lines:
            escaped_line = line.replace('"', '\\"').replace("}]", "\\}\\]")
            # Description does not really need to have "" for each line.
            # s += indent + '"{}"\n'.format(escaped_line)
            s += indent + "{}\n".format(escaped_line)
    s += indent + "}];\n"

    # Handle the type constraint for input and output.
    # Parse type constraint into onnx-mlir type string list.
    type_str_dict = parse_type_constraints(schema)

    ###########################################
    # Generate ins (consisting of operands and attributes).
    ins = get_operands_or_results(schema, type_str_dict, opName, is_input=True)
    ins.update(get_attrs(schema))

    ins_strs = ["{1}:${0}".format(*i) for i in ins.items()]
    s += indent + "let arguments = (ins {});\n".format(
        (",\n" + inc_indent(indent)).join(ins_strs)
    )

    ###########################################
    # Generate outs (operation results).
    outs = get_operands_or_results(schema, type_str_dict, opName, is_input=False)
    outs_strs = ["{1}:${0}".format(*i) for i in outs.items()]
    s += indent + "let results = (outs {});\n".format(
        (",\n" + inc_indent(indent)).join(outs_strs)
    )

    regions_strs = ["{1}:${0}".format(*i) for i in regions.items()]

    if len(regions):
        s += indent + "let regions = (region {});\n".format(
            (",\n" + inc_indent(indent)).join(regions_strs)
        )

    # custom_builder_broadcast_ops_list

    ###########################################
    # Add custom builders.
    # Use element type of the first operand to construct an UnrankedTensorType
    # for the output.
    if mlir_op_name in custom_builder_ops_list:
        if len(ins) == 0:
            raise RuntimeWarning(
                "warning: not generate custom build methods for "
                + schema.name
                + " since it does not have operands."
            )

        r = ""  # r is the resultType, use it with r.format(*operands, indent=indent)
        if mlir_op_name in custom_builder_broadcast_ops_list:
            numOperands = 2
            r += "{indent}auto lhsTy = {0}.getType();\n"
            r += "{indent}auto rhsTy = {1}.getType();\n"
            if mlir_op_name in custom_builder_broadcast_to_bool_ops_list:
                r += "{indent}auto elTy = $_builder.getI1Type();\n"
                elTy = "elTy"
            else:
                elTy = ""
            r += (
                "{indent}auto resultType = getBroadcastedRankedType(lhsTy, rhsTy"
                + (", " + elTy if elTy else "")
                + ");\n"
            )
            r += "{indent}auto shapedType = mlir::dyn_cast_or_null<ShapedType>(resultType);\n"
            r += "{indent}if (!shapedType || !shapedType.hasStaticShape())\n"
            r += (
                "{indent}  resultType = UnrankedTensorType::get("
                + (elTy if elTy else "mlir::cast<ShapedType>(lhsTy).getElementType()")
                + ");\n"
            )
        else:
            numOperands = 1
            r += (
                "{indent}auto resultType = UnrankedTensorType::get("
                + "mlir::cast<ShapedType>({0}.getType()).getElementType());\n"
            )
        resultType = r

        s += indent + "let builders = [\n"
        # Custom builders with operands and attributes having a separate parameter.
        # E.g. OpBuilder<(ins "Value":$X, "Value":$Y, "Attribute":$A), [{}]>
        indent = inc_indent(indent)
        s += indent + "OpBuilder<(ins "
        operands_dict = get_operands_or_results(
            schema, type_str_dict, opName, is_input=True
        )
        attrs_dict = get_attrs(schema)
        s += ", ".join(
            '"{}":${}'.format(tblgen_operand_type_to_cpp_type(ty), name)
            for name, ty in operands_dict.items()
        )
        if operands_dict and attrs_dict:
            s += ", "
        s += ", ".join(
            '"{}":${}'.format(tblgen_attr_type_to_cpp_type(ty), name)
            for name, ty in attrs_dict.items()
        )
        s += "), [{\n"
        indent = inc_indent(indent)
        # Get output type from first operand's type.
        operands = operands_dict.keys()
        s += resultType.format(*operands, indent=indent)
        s += indent + "build($_builder, $_state, resultType"
        for name, _ in ins.items():
            s += ", " + name
        s += ");\n"
        indent = dec_indent(indent)
        s += indent + "}]>,\n"

        # Custom builders with all operands and attributes having aggregate parameters.
        # E.g. OpBuilder<(ins "ValueRange operands,
        #    ArrayRef<NamedAttribute> attributes", [{}]>'
        s += (
            indent
            + "OpBuilder<(ins "
            + '"ValueRange":$operands, "ArrayRef<NamedAttribute>":$attributes), [{\n'
        )
        indent = inc_indent(indent)
        operands = (f"operands[{i}]" for i in range(numOperands))
        s += resultType.format(*operands, indent=indent)
        s += indent + "build($_builder, $_state, {resultType}, operands, attributes);\n"
        indent = dec_indent(indent)
        s += indent + "}]>"

        s += "\n" + indent + "];\n"

    ###########################################
    # Generate extraClassDeclaration.
    s += indent + "let extraClassDeclaration = [{\n"
    # indent = inc_indent(indent)

    # Generate input/output number and output type mapping
    s = get_numberof_inout(s, indent, schema)

    if mlir_op_name in OpsWithHelpers:
        s += OpsWithHelpers[mlir_op_name]

    if len(regions):
        s += indent + "int64_t getSubgraphRegionIdx(const std::string& name) {\n"
        indent = inc_indent(indent)
        for idx, region_name in enumerate(regions.keys()):
            s += indent + 'if (name == "{}") return {};\n'.format(region_name, idx)
        s += (
            indent
            + 'llvm_unreachable("region with the specified name does not exist");\n'
        )
        indent = dec_indent(indent)
        s += indent + "}\n"
    s += indent + "}];\n"

    ###########################################
    # Generate extraClassDefinition.
    s += indent + "let extraClassDefinition = [{\n"

    # Generate shape helper code
    s = gen_shape_helper_code(s, indent, opName, schema.domain)

    s += indent + "}];\n"

    if mlir_op_name in custom_definition_misc:
        s += custom_definition_misc[mlir_op_name] + "\n"

    ###########################################
    # Generate decl for verifier/folder.
    custom_meta = getattr(schema, "meta_attributes", {})
    custom_has_verifier = bool(custom_meta.get("verify", False))
    custom_has_folder = bool(custom_meta.get("fold", False))

    if opName in OpsWithVerifier or custom_has_verifier:
        s += indent + "let hasVerifier = 1;\n"
    if opName in OpsWithFolder or custom_has_folder:
        s += indent + "let hasFolder = 1;\n"
    s += "}\n\n"
    return s


def gen_op_versions(file, op_version_dict, domain):
    indent = inc_indent()
    s = ""
    for key, item in op_version_dict.items():
        s += indent + 'dialect_op_version_map_["{}"]["{}"] = '.format(domain, key)
        s += "{" + "{}".format(", ".join(str(x) for x in item)) + "};\n"
    file.write(s)


def gen_opsets(file, defined_versions_collected, dialect):
    indent = inc_indent()
    s = ""
    for name, versions in defined_versions_collected.items():
        s += indent + 'dialect_op_opsets_map_["{}"]["{}"] = '.format(dialect, name)
        s += "{" + "{}".format(", ".join(str(x) for x in versions)) + "};\n"
    file.write(s)


"""
special cases:
* Split: attr split default value: sizeof(output1) namely 1
* Conv: attr dilations default value is {num_dim of first input - 2, 1}
* Conv: attr kernel_shape type is ints
* Transpose: attr perm default value is {} empty int list
"""


def gen_op_importer(domain, name, file, since_version=None):
    indent = inc_indent()
    if since_version is not None:
        opName = name + "V" + str(since_version)
    else:
        opName = name

    mappedName = map_op_name_to_onnx_mlir_name(opName, domain)
    s = indent + 'import_handler_map_["{}"]["{}"] = \n '.format(domain, opName)
    if domain and domain not in ["", "ai.onnx.ml", "ai.onnx.preview.training"]:
        mlir_op_name = domain_abrv_dict[domain] + opName
    else:
        mlir_op_name = opName

    # Only support special op handler for the op without version.
    if since_version is not None:
        handler_func = "buildOperation<mlir::{}>".format(mappedName)
    else:
        handler_func = special_op_handler.get(
            mlir_op_name, "buildOperation<mlir::{}>".format(mappedName)
        )

    s += inc_indent(indent) + "&onnx_mlir::detail::FrontendGenImpl::"
    s += handler_func + ";\n"

    file.write(s)


def build_operator_schemas(custom_ops=None):
    # domain -> support level -> name -> [schema]
    index = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )  # type: Dict[Text, Dict[int, Dict[Text, List[OpSchema]]]]
    for schema in defs.get_all_schemas_with_history():
        index[schema.domain][int(schema.support_level)][schema.name].append(schema)

    # Add custom ops from YAML
    if custom_ops:
        for schema in custom_ops:
            index[schema.domain][int(schema.support_level)][schema.name].append(schema)

    # Preprocess the Operator Schemas:
    # [(domain, [(support_level, [(schema name, current schema, all versions schemas)])])]
    operator_schemas = (
        list()
    )  # type: List[Tuple[Text, List[Tuple[int, List[Tuple[Text, OpSchema, List[OpSchema]]]]]]]
    existing_ops = set()  # type: Set[Tuple[Text, Text]]
    # Domain, name, versions
    opsets: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: defaultdict(list)
    )  # type: (Dict[Text, Dict[Text, List[int]]])
    for domain, _support_map in sorted(index.items()):
        if not should_render_domain(domain):
            continue
        processed_support_map = list()
        for _support, _name_map in sorted(_support_map.items()):
            processed_name_map = list()
            for n, unsorted_versions in sorted(_name_map.items()):
                versions = sorted(unsorted_versions, key=lambda s: s.since_version)
                opsets[domain][n].extend(reversed([s.since_version for s in versions]))
                schema = versions[-1]
                if (domain, schema.name) in existing_ops:
                    continue

                if check_operation_version:
                    # Generate operation of the latest version of your onnx.
                    existing_ops.add((domain, schema.name))
                    processed_name_map.append((n, schema, versions))
                    if domain not in version_dict:
                        continue

                    # Add checks against version_dict.
                    if schema.name not in version_dict[domain]:
                        print(
                            "Check-operation-version: Operation {} is new  with version {}".format(
                                schema.name, schema.since_version
                            )
                        )
                    elif schema.since_version > version_dict[domain][schema.name][0]:
                        print(
                            "Check-operation-version: Operation {}".format(schema.name)
                            + " has a newer version {} over old version {}".format(
                                schema.since_version,
                                version_dict[domain][schema.name][0],
                            )
                        )
                else:
                    # Generate operation according to the version in version_dict.
                    if (
                        domain not in version_dict
                        or schema.name not in version_dict[domain]
                    ):
                        continue
                    found = False
                    v_counter = 0
                    for schema in reversed(versions):
                        # Check the version number against the version_dict.
                        specified_version = version_dict[domain][schema.name][v_counter]
                        if schema.since_version == specified_version:
                            existing_ops.add((domain, schema.name))
                            processed_name_map.append((n, schema, versions))
                            found = True
                            v_counter += 1
                            if len(version_dict[domain][schema.name]) == v_counter:
                                break
                    if not found:
                        print(
                            "Your onnx installation may be too old. "
                            "The desired version for operation {} is not found.".format(
                                schema.name
                            )
                        )
                        sys.exit()
            processed_support_map.append((_support, processed_name_map))
        operator_schemas.append((domain, processed_support_map))
    return operator_schemas, opsets


def main(args):  # type: (Type[Args]) -> None
    if list_operation_version:
        pprint.pprint(version_dict)
        return

    # Load custom ops from YAML if provided
    custom_ops = []
    custom_version_dict = {}
    if hasattr(args, "custom_ops_yaml") and args.custom_ops_yaml:
        custom_ops = load_custom_ops_from_yaml(args.custom_ops_yaml)
        print(f"Loaded {len(custom_ops)} custom operations from {args.custom_ops_yaml}")

        # Build version dict for custom ops.
        for op in custom_ops:
            if op.domain not in custom_version_dict:
                custom_version_dict[op.domain] = {}
            custom_version_dict[op.domain][op.name] = [op.since_version]

            # Ensure domain is in domain_abrv_dict
            if op.domain not in domain_abrv_dict:
                # Extract abbreviation from domain (e.g., com.amd.xfe -> XFE)
                parts = op.domain.split(".")
                abbrev = parts[-1].upper() if parts else op.domain.upper()
                domain_abrv_dict[op.domain] = abbrev
                print(f"Registered new domain: {op.domain} -> {abbrev}")

            if op.meta_attributes.get("verify", False):
                print(f"  {op.name}: verify enabled")

            if op.meta_attributes.get("fold", False):
                print(f"  {op.name}: fold enabled")

        # Merge custom_version_dict into version_dict so custom ops are generated
        for domain, ops_dict in custom_version_dict.items():
            if domain not in version_dict:
                version_dict[domain] = {}
            version_dict[domain].update(ops_dict)

    curr_utc_time = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%m/%d/%Y, %H:%M:%S"
    )
    autogen_warning = (
        "//********************************************************\n"
        "//   Do not modify this file directly.\n"
        "//   This file is automatically generated via script.\n"
        "//   Details can be found in docs/ImportONNXDefs.md .\n"
        "//********************************************************\n\n"
    )
    autogen_warning = autogen_warning.format(curr_utc_time)

    op_def = args.op_def
    op_def.write(autogen_warning)

    op_importer = args.op_importer
    op_importer.write(autogen_warning)
    for domain, version_map in version_dict.items():
        gen_op_versions(op_importer, version_map, domain)
    for domain, version_map in additional_op_version_dict.items():
        gen_op_versions(op_importer, version_map, domain)
    for domain, version_map in custom_version_dict.items():
        gen_op_versions(op_importer, version_map, domain)

    new_version_dict = defaultdict(dict)
    operator_schemas, operation_opsets = build_operator_schemas(custom_ops)

    # Open separate files for custom domains
    custom_domain_files = {}
    if custom_ops:
        curr_dir = os.path.dirname(os.path.realpath(__file__))
        # Group custom ops by domain
        custom_domains = set(op.domain for op in custom_ops)
        for domain in custom_domains:
            if (
                domain
                and domain != ""
                and domain not in ["", "ai.onnx.ml", "ai.onnx.preview.training"]
            ):
                domain_abbrev = domain_abrv_dict.get(
                    domain, domain.split(".")[-1].upper()
                )
                if hasattr(args, "dry_run") and args.dry_run:
                    custom_domain_files[domain] = StringIO()
                else:
                    filename = f"{domain_abbrev}Ops.td"
                    filepath = os.path.join(curr_dir, filename)
                    custom_domain_files[domain] = io.open(filepath, "w", newline="")
                    print(f"Generating custom ops for domain '{domain}' in {filename}")

                # Write header for custom domain file
                custom_domain_files[domain].write(autogen_warning)
                custom_domain_files[domain].write(
                    f"//===-- {domain_abbrev}Ops.td -- {domain_abbrev} Ops -*- tablegen -===//\n"
                )
                custom_domain_files[domain].write(f"//\n")
                custom_domain_files[domain].write(
                    f"// Operations for domain: {domain}\n"
                )
                custom_domain_files[domain].write(f"//\n")
                custom_domain_files[domain].write(
                    f"//===----------------------------------------------------------------------===//\n\n"
                )
                custom_domain_files[domain].write(
                    f'include "mlir/Interfaces/CallInterfaces.td"\n'
                )
                custom_domain_files[domain].write(
                    f'include "mlir/IR/SymbolInterfaces.td"\n'
                )
                custom_domain_files[domain].write(f'include "src/IR/AttrBase.td"\n\n')

    for domain, support_map in operator_schemas:
        for _, name_map in support_map:
            # Generate Op with version number if not the latest version.
            previous_name = ""
            for op_type, schema, versions in name_map:
                new_version_dict[domain][schema.name] = [schema.since_version]
                if not check_operation_version:
                    with_version = previous_name == schema.name
                    since_version = schema.since_version if with_version else None
                    gen_op_importer(domain, schema.name, op_importer, since_version)
                    r = gen_op_def(schema, with_version)

                    # Write to appropriate file based on domain
                    if domain in custom_domain_files:
                        custom_domain_files[domain].write(r)
                    else:
                        op_def.write(r)
                    previous_name = schema.name

    # Close custom domain files and generate corresponding C++ stubs
    for domain, file_handle in custom_domain_files.items():
        if not (hasattr(args, "dry_run") and args.dry_run):
            file_handle.close()
            domain_abbrev = domain_abrv_dict.get(domain, domain.split(".")[-1].upper())
            print(f"Generated {domain_abbrev}Ops.td for domain '{domain}'")

            # Generate C++ files for shape inference in utils directory
            hpp_filename = f"{domain_abbrev}ShapeInference.hpp"
            cpp_stub_filename = f"{domain_abbrev}.cpp"
            cpp_impl_filename = f"{domain_abbrev}ShapeInference.cpp"

            # 1. Generate header file (.hpp) with function declarations
            hpp_filepath = os.path.join(curr_dir, hpp_filename)
            with io.open(hpp_filepath, "w", newline="") as hpp_file:
                hpp_file.write(autogen_warning)
                hpp_file.write(
                    f"// Shape inference function declarations for {domain_abbrev} operations\n"
                )
                hpp_file.write(f"// This file is auto-generated - DO NOT EDIT\n\n")
                hpp_file.write(f"#pragma once\n\n")
                hpp_file.write(f'#include "mlir/IR/Operation.h"\n')
                hpp_file.write(f'#include "llvm/ADT/FunctionExtras.h"\n\n')
                hpp_file.write(f"namespace mlir {{\n\n")

                for op in custom_ops:
                    if op.domain == domain:
                        domain_abbrev_for_op = domain_abrv_dict.get(
                            domain, domain.split(".")[-1].upper()
                        )
                        op_class_name = f"{domain_abbrev_for_op}{op.name}Op"
                        hpp_file.write(f"// Shape inference for {op.name}\n")
                        hpp_file.write(
                            f"LogicalResult {op_class_name}ShapeInference(\n"
                        )
                        hpp_file.write(
                            f"    Operation *op, std::function<void(Region &)> doShapeInference);\n\n"
                        )

                hpp_file.write(f"}} // namespace mlir\n")

            # 2. Generate forwarding stubs (.cpp) - DO NOT EDIT
            # This now includes shape inference, verify, and fold stubs all in one file
            cpp_stub_filepath = os.path.join(curr_dir, cpp_stub_filename)
            with io.open(cpp_stub_filepath, "w", newline="") as cpp_stub:
                cpp_stub.write(autogen_warning)
                cpp_stub.write(f"// DO NOT EDIT - Auto-generated forwarding stubs\n")
                cpp_stub.write(
                    f"// This file forwards to manual implementations in shape inference, verify, and fold files\n"
                )
                cpp_stub.write(
                    f"// Copy this file to src/Dialect/ONNX/ONNXOps/Additional/{cpp_stub_filename}\n\n"
                )
                cpp_stub.write(f'#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"\n')
                cpp_stub.write(f'#include "{hpp_filename}"\n')

                # Check if we need verify or fold headers
                ops_with_verify = [
                    op
                    for op in custom_ops
                    if op.domain == domain and op.meta_attributes.get("verify", False)
                ]
                ops_with_fold = [
                    op
                    for op in custom_ops
                    if op.domain == domain and op.meta_attributes.get("fold", False)
                ]

                if ops_with_verify:
                    verify_hpp_filename = f"{domain_abbrev}Verify.hpp"
                    cpp_stub.write(f'#include "{verify_hpp_filename}"\n')

                if ops_with_fold:
                    fold_hpp_filename = f"{domain_abbrev}Fold.hpp"
                    cpp_stub.write(f'#include "{fold_hpp_filename}"\n')

                cpp_stub.write(f"\n")
                cpp_stub.write(f"using namespace mlir;\n")
                cpp_stub.write(f"using namespace onnx_mlir;\n\n")

                # Generate shape inference stubs
                cpp_stub.write(
                    f"// ============================================================\n"
                )
                cpp_stub.write(f"// Shape Inference\n")
                cpp_stub.write(
                    f"// ============================================================\n\n"
                )
                for op in custom_ops:
                    if op.domain == domain:
                        domain_abbrev_for_op = domain_abrv_dict.get(
                            domain, domain.split(".")[-1].upper()
                        )
                        op_class_name = f"{domain_abbrev_for_op}{op.name}Op"
                        cpp_stub.write(f"LogicalResult {op_class_name}::inferShapes(\n")
                        cpp_stub.write(
                            f"    std::function<void(Region &)> doShapeInference) {{\n"
                        )
                        cpp_stub.write(
                            f"  return {op_class_name}ShapeInference(this->getOperation(), doShapeInference);\n"
                        )
                        cpp_stub.write(f"}}\n\n")

                # Generate verify stubs if needed
                if ops_with_verify:
                    cpp_stub.write(
                        f"// ============================================================\n"
                    )
                    cpp_stub.write(f"// Verify\n")
                    cpp_stub.write(
                        f"// ============================================================\n\n"
                    )
                    for op in ops_with_verify:
                        domain_abbrev_for_op = domain_abrv_dict.get(
                            domain, domain.split(".")[-1].upper()
                        )
                        op_class_name = f"{domain_abbrev_for_op}{op.name}Op"
                        cpp_stub.write(f"LogicalResult {op_class_name}::verify() {{\n")
                        cpp_stub.write(
                            f"  return {op_class_name}Verify(this->getOperation());\n"
                        )
                        cpp_stub.write(f"}}\n\n")

                # Generate fold stubs if needed
                if ops_with_fold:
                    cpp_stub.write(
                        f"// ============================================================\n"
                    )
                    cpp_stub.write(f"// Fold\n")
                    cpp_stub.write(
                        f"// ============================================================\n\n"
                    )
                    for op in ops_with_fold:
                        domain_abbrev_for_op = domain_abrv_dict.get(
                            domain, domain.split(".")[-1].upper()
                        )
                        op_class_name = f"{domain_abbrev_for_op}{op.name}Op"
                        cpp_stub.write(
                            f"OpFoldResult {op_class_name}::fold(FoldAdaptor adaptor) {{\n"
                        )
                        cpp_stub.write(
                            f"  return {op_class_name}Fold(this->getOperation(), adaptor.getOperands());\n"
                        )
                        cpp_stub.write(f"}}\n\n")

            # 3. Generate implementation templates - EDIT THIS FILE
            cpp_impl_filepath = os.path.join(curr_dir, cpp_impl_filename)
            with io.open(cpp_impl_filepath, "w", newline="") as cpp_impl:
                cpp_impl.write(autogen_warning)
                cpp_impl.write(f"// IMPLEMENT YOUR SHAPE INFERENCE HERE\n")
                cpp_impl.write(
                    f"// This file contains templates - safe to edit and customize\n"
                )
                cpp_impl.write(
                    f"// Move to: src/Dialect/ONNX/ONNXOps/Additional/{cpp_impl_filename}\n"
                )
                cpp_impl.write(
                    f"// and add it to the CMakeLists.txt in src/Dialect/ONNX/\n\n"
                )
                cpp_impl.write(f'#include "{hpp_filename}"\n')
                cpp_impl.write(f'#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"\n')
                cpp_impl.write(
                    f'#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"\n\n'
                )
                cpp_impl.write(f"using namespace mlir;\n")
                cpp_impl.write(f"using namespace onnx_mlir;\n\n")
                cpp_impl.write(f"namespace mlir {{\n\n")

                for op in custom_ops:
                    if op.domain == domain:
                        domain_abbrev_for_op = domain_abrv_dict.get(
                            domain, domain.split(".")[-1].upper()
                        )
                        op_class_name = f"{domain_abbrev_for_op}{op.name}Op"
                        cpp_impl.write(
                            f"LogicalResult {op_class_name}ShapeInference(\n"
                        )
                        cpp_impl.write(
                            f"    Operation *op, std::function<void(Region &)> doShapeInference) {{\n"
                        )
                        cpp_impl.write(
                            f"  // TODO: Implement shape inference for {op.name}\n"
                        )
                        cpp_impl.write(f"  // \n")
                        cpp_impl.write(
                            f"  // Cast to specific op type to access operation-specific methods:\n"
                        )
                        cpp_impl.write(
                            f"  // auto customOp = dyn_cast<{op_class_name}>(op);\n"
                        )
                        cpp_impl.write(f"  // if (!customOp) return failure();\n")
                        cpp_impl.write(f"  // \n")
                        cpp_impl.write(f"  // Get operand types and shapes:\n")
                        cpp_impl.write(
                            f"  // auto operandType = customOp.getOperand(0).getType().dyn_cast<ShapedType>();\n"
                        )
                        cpp_impl.write(f"  // \n")
                        cpp_impl.write(
                            f"  // Compute output shape based on operation semantics\n"
                        )
                        cpp_impl.write(f"  // Set result type:\n")
                        cpp_impl.write(f"  // customOp.getResult().setType(...);\n")
                        cpp_impl.write(f"  \n")
                        cpp_impl.write(f"  return success();\n")
                        cpp_impl.write(f"}}\n\n")

                cpp_impl.write(f"}} // namespace mlir\n")

            print(
                f"Generated {hpp_filename}, {cpp_stub_filename}, {cpp_impl_filename} for domain '{domain}'"
            )

            # Generate Verify and Fold C++ files if needed
            ops_with_verify = [
                op
                for op in custom_ops
                if op.domain == domain and op.meta_attributes.get("verify", False)
            ]
            ops_with_fold = [
                op
                for op in custom_ops
                if op.domain == domain and op.meta_attributes.get("fold", False)
            ]

            if ops_with_verify:
                # Generate verify header file
                verify_hpp_filename = f"{domain_abbrev}Verify.hpp"
                verify_hpp_filepath = os.path.join(curr_dir, verify_hpp_filename)
                with io.open(verify_hpp_filepath, "w", newline="") as verify_hpp:
                    verify_hpp.write(autogen_warning)
                    verify_hpp.write(
                        f"// Verify function declarations for {domain_abbrev} operations\n"
                    )
                    verify_hpp.write(
                        f"// This file is auto-generated - DO NOT EDIT\n\n"
                    )
                    verify_hpp.write(f"#pragma once\n\n")
                    verify_hpp.write(f'#include "mlir/IR/Operation.h"\n')
                    verify_hpp.write(f'#include "mlir/Support/LogicalResult.h"\n\n')
                    verify_hpp.write(f"namespace mlir {{\n\n")

                    for op in ops_with_verify:
                        domain_abbrev_for_op = domain_abrv_dict.get(
                            domain, domain.split(".")[-1].upper()
                        )
                        op_class_name = f"{domain_abbrev_for_op}{op.name}Op"
                        verify_hpp.write(f"// Verify for {op.name}\n")
                        verify_hpp.write(
                            f"LogicalResult {op_class_name}Verify(Operation *op);\n\n"
                        )

                    verify_hpp.write(f"}} // namespace mlir\n")

                # Generate verify implementation template
                verify_impl_filename = f"{domain_abbrev}Verify.cpp"
                verify_impl_filepath = os.path.join(curr_dir, verify_impl_filename)
                with io.open(verify_impl_filepath, "w", newline="") as verify_impl:
                    verify_impl.write(autogen_warning)
                    verify_impl.write(f"// IMPLEMENT YOUR VERIFY LOGIC HERE\n")
                    verify_impl.write(
                        f"// Move to: src/Dialect/ONNX/ONNXOps/Additional/{verify_impl_filename}\n\n"
                    )
                    verify_impl.write(f'#include "{verify_hpp_filename}"\n')
                    verify_impl.write(
                        f'#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"\n\n'
                    )
                    verify_impl.write(f"using namespace mlir;\n")
                    verify_impl.write(f"using namespace onnx_mlir;\n\n")
                    verify_impl.write(f"namespace mlir {{\n\n")

                    for op in ops_with_verify:
                        domain_abbrev_for_op = domain_abrv_dict.get(
                            domain, domain.split(".")[-1].upper()
                        )
                        op_class_name = f"{domain_abbrev_for_op}{op.name}Op"
                        verify_impl.write(
                            f"LogicalResult {op_class_name}Verify(Operation *op) {{\n"
                        )
                        verify_impl.write(
                            f"  // TODO: Implement verification for {op.name}\n"
                        )
                        verify_impl.write(f"  // \n")
                        verify_impl.write(f"  // Cast to specific op type:\n")
                        verify_impl.write(
                            f"  // auto customOp = dyn_cast<{op_class_name}>(op);\n"
                        )
                        verify_impl.write(f"  // if (!customOp) return failure();\n")
                        verify_impl.write(f"  // \n")
                        verify_impl.write(
                            f"  // Verify operand types, shapes, attributes, etc.\n"
                        )
                        verify_impl.write(
                            f"  // Example: Check that input tensors have expected rank\n"
                        )
                        verify_impl.write(f"  // if (operandType.getRank() < 2)\n")
                        verify_impl.write(
                            f'  //   return op->emitError("Expected input rank >= 2");\n'
                        )
                        verify_impl.write(f"  \n")
                        verify_impl.write(f"  return success();\n")
                        verify_impl.write(f"}}\n\n")

                    verify_impl.write(f"}} // namespace mlir\n")

                print(
                    f"Generated {verify_hpp_filename} and {verify_impl_filename} for verify (stubs in {cpp_stub_filename})"
                )

            if ops_with_fold:
                # Generate fold header file
                fold_hpp_filename = f"{domain_abbrev}Fold.hpp"
                fold_hpp_filepath = os.path.join(curr_dir, fold_hpp_filename)
                with io.open(fold_hpp_filepath, "w", newline="") as fold_hpp:
                    fold_hpp.write(autogen_warning)
                    fold_hpp.write(
                        f"// Fold function declarations for {domain_abbrev} operations\n"
                    )
                    fold_hpp.write(f"// This file is auto-generated - DO NOT EDIT\n\n")
                    fold_hpp.write(f"#pragma once\n\n")
                    fold_hpp.write(f'#include "mlir/IR/Operation.h"\n')
                    fold_hpp.write(f'#include "mlir/IR/OpDefinition.h"\n')
                    fold_hpp.write(f'#include "llvm/ADT/SmallVector.h"\n\n')
                    fold_hpp.write(f"namespace mlir {{\n\n")

                    for op in ops_with_fold:
                        domain_abbrev_for_op = domain_abrv_dict.get(
                            domain, domain.split(".")[-1].upper()
                        )
                        op_class_name = f"{domain_abbrev_for_op}{op.name}Op"
                        fold_hpp.write(f"// Fold for {op.name}\n")
                        fold_hpp.write(f"OpFoldResult {op_class_name}Fold(\n")
                        fold_hpp.write(
                            f"    Operation *op, ArrayRef<Attribute> operands);\n\n"
                        )

                    fold_hpp.write(f"}} // namespace mlir\n")

                # Generate fold implementation template
                fold_impl_filename = f"{domain_abbrev}Fold.cpp"
                fold_impl_filepath = os.path.join(curr_dir, fold_impl_filename)
                with io.open(fold_impl_filepath, "w", newline="") as fold_impl:
                    fold_impl.write(autogen_warning)
                    fold_impl.write(f"// IMPLEMENT YOUR FOLD LOGIC HERE\n")
                    fold_impl.write(
                        f"// Move to: src/Dialect/ONNX/ONNXOps/Additional/{fold_impl_filename}\n\n"
                    )
                    fold_impl.write(f'#include "{fold_hpp_filename}"\n')
                    fold_impl.write(
                        f'#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"\n\n'
                    )
                    fold_impl.write(f"using namespace mlir;\n")
                    fold_impl.write(f"using namespace onnx_mlir;\n\n")
                    fold_impl.write(f"namespace mlir {{\n\n")

                    for op in ops_with_fold:
                        domain_abbrev_for_op = domain_abrv_dict.get(
                            domain, domain.split(".")[-1].upper()
                        )
                        op_class_name = f"{domain_abbrev_for_op}{op.name}Op"
                        fold_impl.write(f"OpFoldResult {op_class_name}Fold(\n")
                        fold_impl.write(
                            f"    Operation *op, ArrayRef<Attribute> operands) {{\n"
                        )
                        fold_impl.write(
                            f"  // TODO: Implement constant folding for {op.name}\n"
                        )
                        fold_impl.write(f"  // \n")
                        fold_impl.write(f"  // Cast to specific op type:\n")
                        fold_impl.write(
                            f"  // auto customOp = dyn_cast<{op_class_name}>(op);\n"
                        )
                        fold_impl.write(f"  // if (!customOp) return {{}};\n")
                        fold_impl.write(f"  // \n")
                        fold_impl.write(
                            f"  // Check if all operands are constant (attributes)\n"
                        )
                        fold_impl.write(
                            f"  // If so, compute the result and return it as an attribute\n"
                        )
                        fold_impl.write(f"  // Example:\n")
                        fold_impl.write(
                            f"  // if (!operands[0] || !operands[1]) return {{}};\n"
                        )
                        fold_impl.write(
                            f"  // auto result = computeConstantResult(operands[0], operands[1]);\n"
                        )
                        fold_impl.write(f"  // return result;\n")
                        fold_impl.write(f"  \n")
                        fold_impl.write(f"  return {{}}; // No folding\n")
                        fold_impl.write(f"}}\n\n")

                    fold_impl.write(f"}} // namespace mlir\n")

                print(
                    f"Generated {fold_hpp_filename} and {fold_impl_filename} for fold (stubs in {cpp_stub_filename})"
                )

    for domain, version_map in additional_op_version_dict.items():
        for op_name in version_map:
            gen_op_importer(domain, op_name, op_importer)

    # Generate importers for custom ops
    for domain, version_map in custom_version_dict.items():
        for op_name in version_map:
            gen_op_importer(domain, op_name, op_importer)

    opsets_collected = defaultdict(
        dict
    )  # type: (defaultdict[str, Dict[str, List[int]]])
    for domain, ops in operation_opsets.items():
        for op, versions in ops.items():
            assert op not in opsets_collected[domain], "Operation exist multiple times?"
            opsets_collected[domain][op] = versions

    for domain, collected in opsets_collected.items():
        gen_opsets(op_importer, collected, domain)
    for domain, versions_map in additional_op_version_dict.items():
        gen_opsets(op_importer, versions_map, domain)
    for domain, versions_map in custom_version_dict.items():
        gen_opsets(op_importer, versions_map, domain)

    if check_operation_version:
        for domain, version_map in version_dict.items():
            for key in version_map:
                if (
                    not domain in new_version_dict
                    or not key in new_version_dict[domain]
                ):
                    print("op {}.{} is not in the version".format(domain, key))
                # Assume the top version will be upgraded to the latest version.
                # The existing extra version (from index 1) will be kept.
                for x in version_map[key][1:]:
                    new_version_dict[domain][key].append(x)
        pprint.pprint(new_version_dict)


if __name__ == "__main__":
    curr_dir = os.path.dirname(os.path.realpath(__file__))

    class Args(object):
        dry_run = args.dry_run_onnx_ops or args.dry_run_op_build_table
        custom_ops_yaml = args.custom_ops_yaml

        # If either dry_run_onnx_ops or dry_run_op_build_table is true, then treat
        # both of them as true. Otherwise, one of them runs as a dry-run and one
        # of them runs as a real run creating unnecessary artifacts in the wrong
        # locations in the build tree.
        if dry_run:
            op_def = StringIO()
            op_importer = StringIO()
        else:
            op_def_file_path = os.path.join(curr_dir, "ONNXOps.td.inc")
            op_def = io.open(op_def_file_path, "w", newline="")
            op_importer_file_path = os.path.join(curr_dir, "OpBuildTable.inc")
            op_importer = io.open(op_importer_file_path, "w", newline="")

    main(Args)

    # This is based on diff.py from llvm-project (llvm\utils\lit\lit\builtin_commands\diff.py).
    # On Windows, by default, stdout uses \r\n for newlines, however, all the
    # files we compare against use \n. This piece of code forces the windows stdout
    # to use \n for newlines.
    if sys.platform == "win32":
        if hasattr(sys.stdout, "buffer"):
            # python 3
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, newline="\n")
        else:
            # python 2.7
            import msvcrt

            msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)

    # Only output the generated values for the specifically requested dry run.
    if args.dry_run_onnx_ops:
        sys.stdout.write(Args.op_def.getvalue())
    if args.dry_run_op_build_table:
        sys.stdout.write(Args.op_importer.getvalue())
