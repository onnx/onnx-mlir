#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

##################### RunONNXModel.py #########################################
#
# Copyright 2019-2025 The IBM Research Authors.
#
################################################################################
#
# This script is to run and debug an onnx model.

################################################################################

import os
import sys
import argparse
import onnx
import time
import signal
import subprocess
import numpy as np
import tempfile
import json
import importlib.util
import shlex
import shutil

from onnx import numpy_helper
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from collections import OrderedDict

################################################################################
# Test environment and set global environment variables.

if not os.environ.get("ONNX_MLIR_HOME", None):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the"
        " path to the HOME directory for onnx-mlir. The HOME directory for"
        " onnx-mlir refers to the parent folder containing the bin, lib, etc"
        " sub-folders in which ONNX-MLIR executables and libraries can be found,"
        " typically `onnx-mlir/build/Debug`."
    )
ONNX_MLIR_EXENAME = "onnx-mlir.exe" if sys.platform == "win32" else "onnx-mlir"
ONNX_MLIR = os.path.join(os.environ["ONNX_MLIR_HOME"], "bin", ONNX_MLIR_EXENAME)
# Include runtime directory in python paths, so PyRuntime can be imported.
RUNTIME_DIR = os.path.join(os.environ["ONNX_MLIR_HOME"], "lib")
sys.path.append(RUNTIME_DIR)

VERBOSE = os.environ.get("VERBOSE", False)

################################################################################
# Check and import Onnx Mlir Execution session / python interface.

try:
    from PyRuntime import OMExecutionSession
except ImportError:
    raise ImportError(
        "Looks like you did not build the PyRuntimeC target, build it by running"
        " `make PyRuntimeC`. You may need to set ONNX_MLIR_HOME to"
        " `onnx-mlir/build/Debug` since `make PyRuntimeC` outputs to"
        " `build/Debug` by default."
    )

################################################################################
# Support functions for parsing environment.


def valid_onnx_input(fname):
    valid_exts = ["onnx", "mlir", "onnxtext"]
    ext = os.path.splitext(fname)[1][1:]

    if ext not in valid_exts:
        parser.error(
            "Only accept an input model with one of extensions {}".format(valid_exts)
        )
    return fname


def check_positive(argname, value):
    value = int(value)
    if value <= 0:
        parser.error("Value passed to {} must be positive".format(argname))
    return value


def check_non_negative(argname, value):
    value = int(value)
    if value < 0:
        parser.error("Value passed to {} must be non-negative".format(argname))
    return value


################################################################################
# Command arguments.

parser = argparse.ArgumentParser()
parser.add_argument(
    "--log-to-file",
    action="store",
    nargs="?",
    const="compilation.log",
    default=None,
    help="Output compilation messages to file, default compilation.log.",
)
parser.add_argument(
    "-m",
    "--model",
    type=lambda s: valid_onnx_input(s),
    help="Path to an ONNX model (.onnx or .mlir).",
)
parser.add_argument(
    "-c",
    "--compile-args",
    type=str,
    default="",
    help="Arguments passed directly to onnx-mlir command." " See bin/onnx-mlir --help.",
)
parser.add_argument(
    "-C", "--compile-only", action="store_true", help="Only compile the input model."
)
parser.add_argument("--print-input", action="store_true", help="Print out inputs.")
parser.add_argument(
    "--print-output",
    action="store_true",
    help="Print out inference outputs produced by onnx-mlir.",
)
parser.add_argument(
    "--print-signatures",
    action="store_true",
    help="Print out the input and output signatures of the model.",
)
parser.add_argument(
    "--save-onnx",
    metavar="PATH",
    type=str,
    help="File path to save the onnx model. Only effective if --verify=onnxruntime.",
)
parser.add_argument(
    "--verify",
    choices=["onnxruntime", "ref"],
    help="Verify the output by using onnxruntime or reference"
    " inputs/outputs. By default, no verification. When being"
    " enabled, --verify-with-softmax or --verify-every-value"
    " must be used to specify verification mode.",
)
parser.add_argument(
    "--verify-all-ops",
    action="store_true",
    help="Verify all operation outputs when using onnxruntime.",
)
parser.add_argument(
    "--verify-with-softmax",
    metavar="AXIS_INDEX",
    type=str,
    default=None,
    help="Verify the result obtained by applying softmax along with"
    " specific axis. The axis can be specified"
    " by --verify-with-softmax=<axis>.",
)
parser.add_argument(
    "--verify-every-value",
    action="store_true",
    help="Verify every value of the output using atol and rtol.",
)
parser.add_argument(
    "--rtol", type=str, default="0.05", help="Relative tolerance for verification."
)
parser.add_argument(
    "--atol", type=str, default="0.01", help="Absolute tolerance for verification."
)

lib_group = parser.add_mutually_exclusive_group()
lib_group.add_argument(
    "--save-model",
    metavar="PATH",
    type=str,
    help="Path to a folder to save the compiled model.",
)
lib_group.add_argument(
    "--load-model",
    metavar="PATH",
    type=str,
    help="Path to a folder to load a compiled model for "
    "inference, and the ONNX model will not be re-compiled.",
)
lib_group.add_argument(
    "--cache-model",
    metavar="PATH",
    type=str,
    help="When finding a compiled model in given path, reuse it. "
    "Otherwise, compile model and save it into the given path.",
)

parser.add_argument(
    "-o",
    "--default-model-name",
    metavar="MODEL_NAME",
    type=str,
    default="model",
    help="Change the default model name that is used for two generated files: "
    " .so and .constants.bin. Default is model.",
)

parser.add_argument(
    "--save-ref",
    metavar="PATH",
    type=str,
    help="Path to a folder to save the inputs and outputs in protobuf.",
)
data_group = parser.add_mutually_exclusive_group()
data_group.add_argument(
    "--load-ref",
    metavar="PATH",
    type=str,
    help="Path to a folder containing reference inputs and outputs stored in protobuf."
    " If --verify=ref, inputs and outputs are reference data for verification.",
)
data_group.add_argument(
    "--inputs-from-arrays", help="List of numpy arrays used as inputs for inference."
)
data_group.add_argument(
    "--load-ref-from-numpy",
    metavar="PATH",
    type=str,
    help="Path to a python script that defines variables inputs and outputs that are"
    " a list of numpy arrays. "
    " For example, inputs = [np.array([1], dtype=np.int64), np.array([2], dtype=np.float32]."
    " Variable outputs can be omitted if --verify is not used.",
)
data_group.add_argument(
    "--shape-info",
    type=str,
    help="Shape for each dynamic input of the model, e.g. 0:1x10x20,1:7x5x3. "
    "Used to generate random inputs for the model if --load-ref is not set.",
)

parser.add_argument(
    "--lower-bound",
    type=str,
    help="Lower bound values for each data type. Used inputs."
    " E.g. --lower-bound=int64:-10,float32:-0.2,uint8:1."
    " Supported types are bool, uint8, int8, uint16, int16, uint32, int32,"
    " uint64, int64,float16, float32, float64.",
)
parser.add_argument(
    "--upper-bound",
    type=str,
    help="Upper bound values for each data type. Used to generate random inputs."
    " E.g. --upper-bound=int64:10,float32:0.2,uint8:9."
    " Supported types are bool, uint8, int8, uint16, int16, uint32, int32,"
    " uint64, int64, float16, float32, float64.",
)
parser.add_argument(
    "-w",
    "--warmup",
    type=lambda s: check_non_negative("--warmup", s),
    default=0,
    help="The number of warmup inference runs.",
)
parser.add_argument(
    "-n",
    "--n-iteration",
    type=lambda s: check_positive("--n-iteration", s),
    default=1,
    help="The number of inference runs excluding warmup.",
)
parser.add_argument(
    "--seed",
    type=str,
    default="42",
    help="seed to initialize the random num generator for inputs.",
)


def verify_arg():
    if (
        args.verify
        and (args.verify_with_softmax is None)
        and (not args.verify_every_value)
    ):
        raise RuntimeError(
            "Choose verification mode: --verify-with-softmax or "
            "--verify-every-value or both"
        )
    if args.verify_with_softmax is not None and (not args.verify):
        raise RuntimeError("Must specify --verify to use --verify-with-softmax")
    if args.verify_every_value and (not args.verify):
        raise RuntimeError("Must specify --verify to use --verify-every-value")

    if args.verify and args.verify.lower() == "onnxruntime":
        if not args.model or (args.model and not args.model.endswith(".onnx")):
            raise RuntimeError(
                "Set input onnx model using argument --model when verifying"
                " using onnxruntime."
            )


################################################################################
# Support functions for RunONNXModel functionality.
# Functions are free of args (all needed parameters are passed to the function).


# A type mapping from MLIR to Numpy.
MLIR_TYPE_TO_NP_TYPE = {
    "f64": np.dtype("float64"),
    "f32": np.dtype("float32"),
    "f16": np.dtype("float16"),
    "i64": np.dtype("int64"),
    "i32": np.dtype("int32"),
    "i16": np.dtype("int16"),
    "i8": np.dtype("int8"),
    "ui64": np.dtype("uint64"),
    "ui32": np.dtype("uint32"),
    "ui16": np.dtype("uint16"),
    "ui8": np.dtype("uint8"),
    "i1": np.dtype("bool"),
    "string": np.dtype("str_"),
}

# Default lower bound for generating random inputs.
DEFAULT_LB = {
    "float64": -0.1,
    "float32": -0.1,
    "float16": -0.1,
    "int64": -10,
    "int32": -10,
    "int16": -10,
    "int8": -10,
    "uint64": 0,
    "uint32": 0,
    "uint16": 0,
    "uint8": 0,
    # For some reason, random.uniform with lb/ub to 0/1 resulted in 1 only.
    "bool": -10,  # treated as int32
}

# Default upper bound for generating random inputs.
DEFAULT_UB = {
    "float64": 0.1,
    "float32": 0.1,
    "float16": 0.1,
    "int64": 10,
    "int32": 10,
    "int16": 10,
    "int8": 10,
    "uint64": 10,
    "uint32": 10,
    "uint16": 10,
    "uint8": 10,
    # For some reason, random.uniform with lb/ub to 0/1 resulted in 1 only.
    "bool": 9,  # treated as int32
}


def ordinal(n):
    suffix = ["th", "st", "nd", "rd", "th"][min(n % 10, 4)]
    if 11 <= (n % 100) <= 13:
        suffix = "th"
    return str(n) + suffix


def softmax(x, axis_value):
    return np.exp(x) / np.sum(np.exp(x), axis=axis_value, keepdims=True)


def execute_commands(cmds):
    if VERBOSE:
        print(cmds)
    out = subprocess.Popen(cmds, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = out.communicate()
    msg = stderr.decode("utf-8") + stdout.decode("utf-8")
    if out.returncode == -signal.SIGSEGV:
        return (False, "Segfault")
    if out.returncode != 0:
        return (False, msg)
    return (True, msg)


def extend_model_output(model, intermediate_outputs):
    # Run shape inference to make sure we have valid tensor value infos for all
    # intermediate tensors available
    model = onnx.shape_inference.infer_shapes(model)
    value_infos = {vi.name: vi for vi in model.graph.value_info}
    graph_inputs = {vi.name: vi for vi in model.graph.input}
    graph_outputs = {vi.name: vi for vi in model.graph.output}

    # Retrieve tensor value info for each intermediate output
    new_outputs = []
    for name in intermediate_outputs:
        if name in value_infos:
            new_outputs.append(value_infos[name])
        elif name in graph_inputs:
            new_outputs.append(graph_inputs[name])
        elif name in graph_outputs:
            new_outputs.append(graph_outputs[name])
        else:
            raise RuntimeError(f"Unable to find value infos for {name}")

    # Clear old graph outputs and replace by new set of intermediate outputs
    while len(model.graph.output):
        model.graph.output.pop()

    model.graph.output.extend(new_outputs)
    return model


def get_names_in_signature(signature):
    names = []
    # Load the input signature.
    signature_dict = json.loads(signature)
    for sig in signature_dict:
        names.append(sig["name"])
    return names


def read_input_from_refs(num_inputs, load_ref_filename, is_load_ref):
    print("Reading inputs from {} ...".format(load_ref_filename))
    inputs = []

    if is_load_ref:
        for i in range(num_inputs):
            input_file = load_ref_filename + "/input_{}.pb".format(i)
            input_ts = onnx.TensorProto()
            with open(input_file, "rb") as f:
                input_ts.ParseFromString(f.read())
            input_np = numpy_helper.to_array(input_ts)
            inputs += [input_np]
    else:
        spec = importlib.util.spec_from_file_location("om_load_ref", load_ref_filename)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        inputs = module.inputs

    for i in range(len(inputs)):
        input_np = inputs[i]
        print(
            "  - {} input: [{}x{}]".format(
                ordinal(i + 1),
                "x".join([str(i) for i in input_np.shape]),
                input_np.dtype,
            )
        )

    print("  done.\n")
    return inputs


def read_output_from_refs(num_outputs, load_ref_filename, is_load_ref):
    print("Reading reference outputs from {} ...".format(load_ref_filename))
    reference_output = []

    if is_load_ref:
        for i in range(num_outputs):
            output_file = load_ref_filename + "/output_{}.pb".format(i)
            output_ts = onnx.TensorProto()
            with open(output_file, "rb") as f:
                output_ts.ParseFromString(f.read())
            output_np = numpy_helper.to_array(output_ts)
            reference_output += [output_np]
    else:
        spec = importlib.util.spec_from_file_location(
            "om_load_ref_output", load_ref_filename
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        reference_output = module.outputs

    for i in range(len(reference_output)):
        output_np = reference_output[i]
        print(
            "  - {} output: [{}x{}]".format(
                ordinal(i + 1),
                "x".join([str(i) for i in output_np.shape]),
                output_np.dtype,
            )
        )
    print("  done.\n")
    return reference_output


def generate_random_input(input_signature, shape_info, seed, lower_bound, upper_bound):
    # Load random values: first get shape info, where shape_info in the form of
    # 'input_index:d1xd2, input_index:d1xd2'
    input_shapes = {}
    if shape_info:
        for input_shape in shape_info.strip().split(","):
            input_index_shape = input_shape.split(":")
            input_index = input_index_shape[0]
            assert not (input_index in input_shapes), "Duplicate input indices"
            dims = [int(d) for d in input_index_shape[1].split("x")]
            input_shapes[int(input_index)] = dims

    # Then fill shapes with random numbers.
    # Numpy expect an int, tolerate int/float strings.
    curr_seed = int(float(seed))
    print("Generating random inputs using seed", curr_seed, "...")
    # Generate random data as input.
    inputs = []

    # Load the input signature.
    signature = json.loads(input_signature)

    np.random.seed(curr_seed)
    for i, sig in enumerate(signature):
        # Get shape.
        explicit_shape = []
        for d, dim in enumerate(sig["dims"]):
            if dim >= 0:
                explicit_shape.append(dim)
                continue
            if i in input_shapes:
                if d < len(input_shapes[i]):
                    explicit_shape.append(input_shapes[i][d])
                else:
                    print(
                        "The {} dim".format(ordinal(d + 1)),
                        "of the {} input is unknown.".format(ordinal(i + 1)),
                        "Use --shape-info to set.",
                    )
                    print(" - The input signature: ", sig)
                    exit(1)
            else:
                print(
                    "The shape of the {} input".format(ordinal(i + 1)),
                    "is unknown. Use --shape-info to set.",
                )
                print(" - The input signature: ", sig)
                exit(1)
        # Get element type.
        elem_type = sig["type"]
        np_elem_type = MLIR_TYPE_TO_NP_TYPE[elem_type]

        # Set a range for random values.
        custom_lb = {}
        custom_ub = {}
        # Get user's range if any.
        if lower_bound:
            for type_lbs in lower_bound.strip().split(","):
                type_lb = type_lbs.split(":")
                assert not (type_lb[0] in custom_lb), "Duplicate types"
                custom_lb[type_lb[0]] = type_lb[1]
        if upper_bound:
            for type_ubs in upper_bound.strip().split(","):
                type_ub = type_ubs.split(":")
                assert not (type_ub[0] in custom_ub), "Duplicate types"
                custom_ub[type_ub[0]] = type_ub[1]
        DEFAULT_LB.update(custom_lb)
        DEFAULT_UB.update(custom_ub)

        lb = ub = 0
        random_element_type = np_elem_type
        if np.issubdtype(np_elem_type, np.dtype(bool).type):
            # For some reason, random.uniform with lb/ub to 0/1 resulted in 1 only.
            lb = int(DEFAULT_LB["bool"])
            ub = int(DEFAULT_UB["bool"])
            random_element_type = np.dtype("int32")
        elif np.issubdtype(np_elem_type, np.uint8):
            lb = int(DEFAULT_LB["uint8"])
            ub = int(DEFAULT_UB["uint8"])
        elif np.issubdtype(np_elem_type, np.uint16):
            lb = int(DEFAULT_LB["uint16"])
            ub = int(DEFAULT_UB["uint16"])
        elif np.issubdtype(np_elem_type, np.uint32):
            lb = int(DEFAULT_LB["uint32"])
            ub = int(DEFAULT_UB["uint32"])
        elif np.issubdtype(np_elem_type, np.uint64):
            lb = int(DEFAULT_LB["uint64"])
            ub = int(DEFAULT_UB["uint64"])
        elif np.issubdtype(np_elem_type, np.int8):
            lb = int(DEFAULT_LB["int8"])
            ub = int(DEFAULT_UB["int8"])
        elif np.issubdtype(np_elem_type, np.int16):
            lb = int(DEFAULT_LB["int16"])
            ub = int(DEFAULT_UB["int16"])
        elif np.issubdtype(np_elem_type, np.int32):
            lb = int(DEFAULT_LB["int32"])
            ub = int(DEFAULT_UB["int32"])
        elif np.issubdtype(np_elem_type, np.int64):
            lb = int(DEFAULT_LB["int64"])
            ub = int(DEFAULT_UB["int64"])
        elif np.issubdtype(np_elem_type, np.float64):
            lb = float(DEFAULT_LB["float64"])
            ub = float(DEFAULT_UB["float64"])
        elif np.issubdtype(np_elem_type, np.float32):
            lb = float(DEFAULT_LB["float32"])
            ub = float(DEFAULT_UB["float32"])
        elif np.issubdtype(np_elem_type, np.float16):
            lb = float(DEFAULT_LB["float16"])
            ub = float(DEFAULT_UB["float16"])
        elif np.issubdtype(np_elem_type, np.str_):
            lb = 0
            ub = 64
            random_element_type = np.dtype("int32")
        else:
            raise AssertionError("Unsupported element type")
        rinput = np.random.uniform(lb, ub, explicit_shape).astype(random_element_type)
        # For boolean, transform range into True/False using greater_equal
        if np.issubdtype(np_elem_type, np.dtype(bool).type):
            rinput = np.greater_equal(rinput, [0])
        elif np.issubdtype(np_elem_type, np.str_):
            rinput = np.array(rinput, dtype=np.str_)
            # rinput = np.array(["ab", "defg"], dtype=np.str_)
            rinput = np.array(rinput, dtype=object)
        print(
            "  - {} input's shape {}, element type {}.".format(
                ordinal(i + 1), rinput.shape, np_elem_type
            ),
            "Value ranges [{}, {}]".format(lb, ub),
        )
        inputs.append(rinput)
    print("  done.\n")
    return inputs


def verify_outs(actual_outs, ref_outs, atol, rtol):
    total_elements = 0
    mismatched_elements = 0
    for index, actual_val in np.ndenumerate(actual_outs):
        total_elements += 1
        ref_val = ref_outs[index]
        if np.issubdtype(actual_outs.dtype, np.dtype(bool).type):
            if ref_val == actual_val:
                continue
        else:
            # Use equation atol + rtol * abs(desired), that is used in assert_allclose.
            diff = float(atol) + float(rtol) * abs(ref_val)
            if abs(actual_val - ref_val) <= diff:
                continue
        mismatched_elements += 1
        print(
            "  at {}".format(index),
            "mismatch {} (actual)".format(actual_val),
            "vs {} (reference)".format(ref_val),
        )
    if mismatched_elements == 0:
        print("  correct.\n")
    else:
        raise AssertionError(
            "  got mismatched elements {}/{}, abort.\n".format(
                mismatched_elements, total_elements
            )
        )


def data_without_top_bottom_quartile(data, percent):
    data = np.array(sorted(data))
    trim = int(percent * data.size / 100.0)
    if trim == 0 or data.size - 2 * trim < 1:
        # Want at least one element, return as is.
        return data
    return data[trim:-trim]


def cache_string(model_name, compile_option):
    return "model: " + model_name + "; compile option: " + compile_option


################################################################################
# Inference Session implementing RunONNXModel.
#
# Constructor: fetch the model and compile if needed, save model if requested.
# process_inputs: initialize the inputs, which can come from various sources.
# run_inference: run one inference using the inputs set in process_inputs.
# process_output: verify values generated in run, save outputs,...
# process_perf_results: compute and print performance data.
#
# run_performance_test: process inputs, perform several inferences (warmup and perf),
#   process performance results and validate outputs,


class InferenceSession:
    """
    Init the class by loading / compiling and build an execution session.
    model_file: the file name of the model, possibly needing compilation.
    options: parsed and added into args.
    """

    # Init load the model or compile it, and build an execution session.
    # For init, either options have been parsed because this file is executed
    # as a main, or a model_file is expected as parameter to init.
    # In either case, args will be parsed and thus be available.
    #
    # Object variables are:
    #  default_model_name
    #  model_dir
    #  session
    #  inputs (definition of inputs delayed to process_inputs).
    #  input_names, output_names
    #  temp_dir

    def __init__(self, model_file=None, **kwargs):
        global args

        # Get options passes, if any.
        options = kwargs["options"] if "options" in kwargs.keys() else ""
        # Add model file to options, if given.
        if model_file:
            if model_file.endswith(".onnx") or model_file.endswith(".mlir"):
                options += " --model=" + model_file
            else:
                options += " --load-model=" + model_file
        # Parse options
        if options:
            args = parser.parse_args(shlex.split(options))
        # Default model name that will be used for the compiled model.
        # e.g. model.so, model.constants.bin, ...
        self.default_model_name = args.default_model_name

        # Handle cache_model.
        if args.cache_model:
            shared_lib_path = args.cache_model + f"/{self.default_model_name}.so"
            if not os.path.exists(shared_lib_path):
                print(
                    'Cached compiled model not found in "'
                    + args.cache_model
                    + '": save model this run.'
                )
                args.save_model = args.cache_model
            else:
                print(
                    'Cached compiled model found in "'
                    + args.cache_model
                    + '": load model this run.'
                )
                args.load_model = args.cache_model
            args.cache_model = None

        # Load the onnx model.
        if args.model and args.model.endswith(".onnx"):
            model = onnx.load(args.model)
            # Get names of all intermediate tensors and modify model such that each of
            # them will be an output of the model. If using onnxruntime for
            # verification, we can then verify every operation output.
            output_names = [o.name for o in model.graph.output]
            output_names = list(OrderedDict.fromkeys(output_names))
            if args.verify and args.verify == "onnxruntime" and args.verify_all_ops:
                print("Extending the onnx model to check every node output ...\n")
                output_names = sum(
                    [[n for n in node.output if n != ""] for node in model.graph.node],
                    [],
                )
                output_names = list(OrderedDict.fromkeys(output_names))
                model = extend_model_output(model, output_names)

                # Save the modified onnx file of the model if required.
                if args.save_onnx:
                    print("Saving modified onnx model to ", args.save_onnx, "\n")
                    onnx.save(model, args.save_onnx)

        # If a shared library is given, use it without compiling the ONNX model.
        # Otherwise, compile the ONNX model.
        if args.load_model:
            self.model_dir = args.load_model
            compiler_option_file = os.path.join(self.model_dir, "compiler_option.txt")
            # Verify that we have the same options:
            #  if we have saved the options in the "compiler_option.txt" file, and
            #  if we have provided compiler options
            if os.path.exists(compiler_option_file):
                expected_string = cache_string(args.model, args.compile_args)
                with open(compiler_option_file, "r") as f:
                    options_from_file = f.read()
                    if args.compile_args:
                        if options_from_file != expected_string:
                            print(
                                "Try to load model from",
                                args.load_model,
                                " using different options than when saved, abort",
                            )
                            print('  Save options: "' + options_from_file + '"')
                            print('  Load options: "' + expected_string + '"')
                            exit(1)
                    else:
                        print('  Cached model options: "' + options_from_file + '"')
        else:
            # Compile the ONNX model.
            self.temp_dir = tempfile.TemporaryDirectory()
            print("Temporary directory has been created at {}\n".format(self.temp_dir))
            print("Compiling the model ...")
            self.model_dir = self.temp_dir.name
            # Prepare input and output paths.
            output_path = os.path.join(self.model_dir, self.default_model_name)
            if args.model.endswith(".onnx"):
                if args.verify and args.verify == "onnxruntime" and args.verify_all_ops:
                    input_model_path = os.path.join(
                        self.model_dir, f"{self.default_model_name}.onnx"
                    )
                    onnx.save(model, input_model_path)
                else:
                    input_model_path = args.model
            elif args.model.endswith(".mlir") or args.model.endswith(".onnxtext"):
                input_model_path = args.model
            else:
                print(
                    "Invalid input model path. Must end with .onnx or .mlir or .onnxtext"
                )
                exit(1)

            # Prepare compiler arguments.
            command_str = [ONNX_MLIR]
            if args.compile_args:
                command_str += args.compile_args.split()
            command_str += [input_model_path]
            command_str += ["-o", output_path]

            # Compile the model.
            start = time.perf_counter()
            ok, msg = execute_commands(command_str)
            # Dump the compilation log into a file.
            if args.log_to_file:
                log_file = (
                    args.log_to_file
                    if args.log_to_file.startswith("/")
                    else os.path.join(os.getcwd(), args.log_to_file)
                )
                print("  Compilation log is dumped into {}".format(log_file))
                with open(log_file, "w") as f:
                    f.write(msg)
            if not ok:
                print(msg)
                exit(1)
            end = time.perf_counter()
            print("  took ", end - start, " seconds.\n")

            # Save the following information:
            # - .so file,
            # - .constants.bin file, and
            # - compilation.log containing the compilation output.
            if args.save_model:
                if not os.path.exists(args.save_model):
                    os.makedirs(args.save_model)
                if not os.path.isdir(args.save_model):
                    print("Path to --save-model is not a folder")
                    exit(0)
                # .so file.
                shared_lib_path = self.model_dir + f"/{self.default_model_name}.so"
                if os.path.exists(shared_lib_path):
                    print("Saving the shared library to", args.save_model)
                    shutil.copy2(shared_lib_path, args.save_model)
                # .constants.bin file.
                constants_file_path = os.path.join(
                    self.model_dir, f"{self.default_model_name}.constants.bin"
                )
                if os.path.exists(constants_file_path):
                    print("Saving the constants file to", args.save_model, "\n")
                    shutil.copy2(constants_file_path, args.save_model)
                # Compilation log.
                log_file_path = os.path.join(args.save_model, "compile.log")
                with open(log_file_path, "w") as f:
                    print("Saving the compilation log to", args.save_model, "\n")
                    f.write(msg)
                compiler_option_file_path = os.path.join(
                    args.save_model, "compiler_option.txt"
                )
                expected_string = cache_string(args.model, args.compile_args)
                with open(compiler_option_file_path, "w") as ff:
                    print("Saving the compilation options to", args.save_model, "\n")
                    ff.write(expected_string)

            # Exit if only compiling the model.
            if args.compile_only:
                exit(0)

        # Use the generated shared library to create an execution session.
        start = time.perf_counter()
        shared_lib_path = self.model_dir + f"/{self.default_model_name}.so"
        if not os.path.exists(shared_lib_path):
            print(f"Input model {shared_lib_path} does not exist")
            exit(0)
        print("Loading the compiled model ...")
        if args.load_model:
            session = OMExecutionSession(shared_lib_path, tag="None")
        else:
            session = OMExecutionSession(shared_lib_path)
        end = time.perf_counter()
        print("  took ", end - start, " seconds.\n")
        self.session = session

        # Additional model info.
        self.inputs = []
        input_signature = self.session.input_signature()
        output_signature = self.session.output_signature()
        self.input_names = get_names_in_signature(input_signature)
        self.output_names = get_names_in_signature(output_signature)
        if args.print_signatures:
            print("Model's input signature: ", input_signature.strip())
            print("Model's output signature: ", output_signature.strip())

        # Let onnx-mlir know where to find the constants file.
        os.environ["OM_CONSTANT_PATH"] = self.model_dir

    """
    process_inputs: define the model inputs for the model and store them in self.inputs.
    Print input if requested.
    """

    def process_inputs(self, input_feed=None):
        # Define inputs.
        self.inputs = []
        if input_feed:
            # Get input from input_feed.
            if isinstance(input_feed, dict):
                for name in self.input_names:
                    if name in input_feed:
                        self.inputs.append(input_feed[name])
                    else:
                        print("input name given: ", input_feed.keys())
                        print("input name expected by model: ", self.input_names)
                        print("do not match")
                        exit(1)
                # Since Python guarantees the order of values in a dictionary,
                # the name check could be ignored as follows:
                # inputs = list(input_feed.values())
            else:
                self.inputs = input_feed
        elif args.load_ref:
            # Get input from reference file.
            self.inputs = read_input_from_refs(
                len(self.input_names), args.load_ref, is_load_ref=True
            )
        elif args.load_ref_from_numpy:
            # Get input from numpy.
            self.inputs = read_input_from_refs(
                len(self.input_names), args.load_ref_from_numpy, is_load_ref=False
            )
        elif args.inputs_from_arrays:
            # Get input from array.
            self.inputs = args.inputs_from_arrays
        else:
            self.inputs = generate_random_input(
                self.session.input_signature(),
                args.shape_info,
                args.seed,
                args.lower_bound,
                args.upper_bound,
            )

        # Print the input if required.
        if args.print_input:
            for i, inp in enumerate(self.inputs):
                print(
                    "The {} input {}:[{}x{}] is: \n {} \n".format(
                        ordinal(i + 1),
                        self.input_names[i],
                        "x".join([str(i) for i in inp.shape]),
                        inp.dtype,
                        inp,
                    )
                )

    """
    Perform one inference without any timing.
    """

    def run_inference(self):
        return self.session.run(self.inputs)

    """
    When requested outputs are printed, verified, and/or saved.
    """

    def process_outputs(self, outs):
        # Print the output if required.
        if args.print_output:
            for i, out in enumerate(outs):
                print(
                    "The {} output {}:[{}x{}] is: \n {} \n".format(
                        ordinal(i + 1),
                        self.output_names[i],
                        "x".join([str(i) for i in out.shape]),
                        out.dtype,
                        out,
                    )
                )

        # Store the input and output if required.
        if args.save_ref:
            load_ref = args.save_ref
            if not os.path.exists(load_ref):
                os.mkdir(load_ref)
            for i in range(len(self.inputs)):
                tensor = numpy_helper.from_array(self.inputs[i])
                tensor_path = os.path.join(load_ref, "input_{}.pb".format(i))
                with open(tensor_path, "wb") as f:
                    f.write(tensor.SerializeToString())
            for i in range(len(outs)):
                tensor = numpy_helper.from_array(outs[i])
                tensor_path = os.path.join(load_ref, "output_{}.pb".format(i))
                with open(tensor_path, "wb") as f:
                    f.write(tensor.SerializeToString())

        # Verify the output if required.
        if args.verify:
            ref_outs = []
            if args.verify.lower() == "onnxruntime":
                input_model_path = args.model
                # Reference backend by using onnxruntime.
                import onnxruntime

                input_feed = dict(zip(self.input_names, self.inputs))
                print("Running inference using onnxruntime ...")
                start = time.perf_counter()
                ref_session = onnxruntime.InferenceSession(input_model_path)
                ref_outs = ref_session.run(self.output_names, input_feed)
                end = time.perf_counter()
                print("  took ", end - start, " seconds.\n")
            elif args.verify.lower() == "ref":
                # Reference output available in protobuf.
                if args.load_ref:
                    ref_outs = read_output_from_refs(
                        len(self.output_names), args.load_ref, is_load_ref=True
                    )
                elif args.load_ref_from_numpy:
                    ref_outs = read_output_from_refs(
                        len(self.output_names),
                        args.load_ref_from_numpy,
                        is_load_ref=False,
                    )
            else:
                print("Invalid verify option")
                exit(1)

            # Verify using softmax first.
            if args.verify_with_softmax is not None:
                axis = int(args.verify_with_softmax)
                for i, name in enumerate(self.output_names):
                    print(
                        "Verifying using softmax along with "
                        "axis {}".format(args.verify_with_softmax),
                        "for output {}:{}".format(name, list(outs[i].shape)),
                        "using atol={}, rtol={} ...".format(args.atol, args.rtol),
                    )
                    softmax_outs = softmax(outs[i], axis)
                    softmax_ref_outs = softmax(ref_outs[i], axis)
                    verify_outs(softmax_outs, softmax_ref_outs, args.atol, args.rtol)

            # For each output tensor, compare every value.
            if args.verify_every_value:
                for i, name in enumerate(self.output_names):
                    print(
                        "Verifying value of {}:{}".format(name, list(outs[i].shape)),
                        "using atol={}, rtol={} ...".format(args.atol, args.rtol),
                    )
                    verify_outs(outs[i], ref_outs[i], args.atol, args.rtol)

    """
    Perform a short analysis of time spent in the model.
    """

    def process_perf_results(self, perf_results):
        # Print statistics info, e.g., min/max/stddev inference time.
        if args.n_iteration > 1:
            print(
                "  Statistics 1 (excluding warmup),"
                " min, {:.6e}, max, {:.6e}, mean, {:.6e}, stdev, {:.6e}".format(
                    np.min(perf_results),
                    np.max(perf_results),
                    np.mean(perf_results),
                    np.std(perf_results, dtype=np.float64),
                )
            )
            t_perf_results = data_without_top_bottom_quartile(perf_results, 25)
            print(
                "  Statistics 2 (no warmup/quart.),"
                " min, {:.6e}, max, {:.6e}, mean, {:.6e}, stdev, {:.6e}".format(
                    np.min(t_perf_results),
                    np.max(t_perf_results),
                    np.mean(t_perf_results),
                    np.std(t_perf_results, dtype=np.float64),
                )
            )

    """
    From onnxruntime API:

    run_performance_test(output_names, input_feed)
    Compute the predictions.

    PARAMETERS:
    output_names – name of the outputs (optional)
    input_feed – dictionary { input_name: input_value }
    RETURNS:
    list of results, every result is either a numpy array, a sparse tensor, or
    a list or a dictionary.
    
    For onnxmlir, the run_options is ignored. If 'input_feed' is None, the
    input could be randomly generated or read from file, as args specified.
    In future, add '--shape-info' here. Better than in InferenceSession to
    allow different shape from run to run. 
    """

    def run_performance_test(self, output_name=None, input_feed=None, **kwargs):
        # Process inputs, saved in self.inputs.
        self.process_inputs(input_feed)
        # Running inference.
        print("Running inference ...")
        for i in range(args.warmup):
            start = time.perf_counter()
            outs = self.run_inference()  # Using inputs from self.inputs.
            end = time.perf_counter()
            print("  {} warmup: {} seconds".format(ordinal(i + 1), end - start))

        perf_results = []
        for i in range(args.n_iteration):
            start = time.perf_counter()
            outs = self.run_inference()  # Using inputs from self.inputs.
            end = time.perf_counter()
            elapsed = end - start
            perf_results += [elapsed]
            print("  {} iteration, {}, seconds".format(ordinal(i + 1), elapsed))

        # Print performance results and verify output.
        self.process_perf_results(perf_results)
        self.process_outputs(outs)
        if output_name:
            res = {output_name[i]: outs[i] for i in range(len(outs))}
            return res
        else:
            return outs


################################################################################
# Standalone driver


def main():
    # In main mode, parse the args here.
    global args
    args = parser.parse_args()
    if not (args.model or args.load_model):
        print("error: no input model, use argument --model and/or --load-model.")
        print(parser.format_usage())
        exit(1)

    # Create inference session and perform a performance run test, which load,
    # compute, and possibly verify data.
    session = InferenceSession()
    return session.run_performance_test()


if __name__ == "__main__":
    main()
