#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

##################### RunONNXModel.py #########################################
#
# Copyright 2019-2022 The IBM Research Authors.
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

from onnx import numpy_helper
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from collections import OrderedDict

# Command arguments.
parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, help="Path to the ONNX model")
lib_group = parser.add_mutually_exclusive_group()
parser.add_argument('--print-input',
                    action='store_true',
                    help="Print out inputs")
parser.add_argument('--print-output',
                    action='store_true',
                    help="Print out inference outputs produced by onnx-mlir")
parser.add_argument('--save-onnx',
                    metavar='PATH',
                    type=str,
                    help="File path to save the onnx model. Only effective if "
                    "--verify=onnxruntime")
lib_group.add_argument('--save-so',
                       metavar='PATH',
                       type=str,
                       help="File path to save the generated shared library of"
                       " the model")
lib_group.add_argument('--load-so',
                       metavar='PATH',
                       type=str,
                       help="File path to load a generated shared library for "
                       "inference, and the ONNX model will not be re-compiled")
parser.add_argument('--save-data',
                    metavar='PATH',
                    type=str,
                    help="Path to a folder to save the inputs and outputs"
                    " in protobuf")
data_group = parser.add_mutually_exclusive_group()
data_group.add_argument(
    '--data-folder',
    type=str,
    help="Path to a folder containing inputs and outputs stored in protobuf."
    " If --verify=ref, inputs and outputs are reference data for verification")
data_group.add_argument(
    '--shape-info',
    type=str,
    help="Shape for each dynamic input of the model, e.g. 0:1x10x20,1:7x5x3. "
    "Used to generate random inputs for the model if --data-folder is not set")
parser.add_argument('--compile-args',
                    type=str,
                    default="",
                    help="Arguments passed directly to onnx-mlir command."
                    " See bin/onnx-mlir --help")
parser.add_argument('--verify',
                    choices=['onnxruntime', 'ref'],
                    help="Verify the output by using onnxruntime or reference"
                    " inputs/outputs. By default, no verification")
parser.add_argument(
    '--verify-all-ops',
    action='store_true',
    help="Verify all operation outputs when using onnxruntime.")
parser.add_argument(
    '--compile-using-input-shape',
    action='store_true',
    help="Compile the model by using the shape info getting from"
    " the inputs in data folder. Must set --data-folder")
parser.add_argument('--rtol',
                    type=str,
                    default="0.05",
                    help="Relative tolerance for verification")
parser.add_argument('--atol',
                    type=str,
                    default="0.01",
                    help="Absolute tolerance for verification")
parser.add_argument('--compile-only',
                    action='store_true',
                    help="Only compile the input model")
args = parser.parse_args()

if (not os.environ.get('ONNX_MLIR_HOME', None)):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
        "the parent folder containing the bin, lib, etc sub-folders in which ONNX-MLIR "
        "executables and libraries can be found, typically `onnx-mlir/build/Debug`"
    )

VERBOSE = os.environ.get('VERBOSE', False)

ONNX_MLIR_EXENAME = "onnx-mlir"
if sys.platform == "win32":
    ONNX_MLIR_EXENAME = "onnx-mlir.exe"

ONNX_MLIR = os.path.join(os.environ['ONNX_MLIR_HOME'], "bin",
                         ONNX_MLIR_EXENAME)

# Include runtime directory in python paths, so PyRuntime can be imported.
RUNTIME_DIR = os.path.join(os.environ['ONNX_MLIR_HOME'], "lib")
sys.path.append(RUNTIME_DIR)

try:
    from PyRuntime import ExecutionSession
except ImportError:
    raise ImportError(
        "Looks like you did not build the PyRuntime target, build it by running `make PyRuntime`."
        "You may need to set ONNX_MLIR_HOME to `onnx-mlir/build/Debug` since `make PyRuntime` outputs to `build/Debug` by default"
    )

# A type mapping from MLIR to Numpy.
MLIR_TYPE_TO_NP_TYPE = {
    'f64': np.dtype("float64"),
    'f32': np.dtype("float32"),
    'f16': np.dtype("float16"),
    'i64': np.dtype("int64"),
    'i32': np.dtype("int32"),
    'i16': np.dtype("int16"),
    'i8': np.dtype("int8"),
    'i1': np.dtype("bool"),
}


def ordinal(n):
    suffix = ['th', 'st', 'nd', 'rd', 'th'][min(n % 10, 4)]
    if 11 <= (n % 100) <= 13:
        suffix = 'th'
    return str(n) + suffix


def execute_commands(cmds):
    if (VERBOSE):
        print(cmds)
    out = subprocess.Popen(cmds,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    stdout, stderr = out.communicate()
    if out.returncode == -signal.SIGSEGV:
        return (False, "Segfault")
    if out.returncode != 0:
        return (False, stderr.decode("utf-8") + stdout.decode("utf-8"))
    return (True, stdout.decode("utf-8"))


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
            raise RuntimeError(f"Unable to find value infor for {name}")

    # Clear old graph outputs and replace by new set of intermediate outputs
    while (len(model.graph.output)):
        model.graph.output.pop()

    model.graph.output.extend(new_outputs)
    return model


def get_names_in_signature(signature):
    names = []
    # Load the input signature.
    signature_dict = json.loads(signature)
    for i, sig in enumerate(signature_dict):
        name = 'input_{}'.format(i)
        if sig['name']:
            name = sig['name']
        names.append(name)
    return names


def read_input_from_refs(num_inputs, data_folder):
    print("Reading inputs from {} ...".format(data_folder))
    i = 0
    inputs = []

    for i in range(num_inputs):
        input_file = data_folder + '/input_{}.pb'.format(i)
        input_ts = onnx.TensorProto()
        with open(input_file, 'rb') as f:
            input_ts.ParseFromString(f.read())
        input_np = numpy_helper.to_array(input_ts)
        print("  - {} input: [{}x{}]".format(
            ordinal(i + 1), 'x'.join([str(i) for i in input_np.shape]),
            input_np.dtype))
        inputs += [input_np]
        i += 1
    print("  done.\n")
    return inputs


def read_output_from_refs(num_outputs, data_folder):
    print("Reading reference outputs from {} ...".format(data_folder))
    reference_output = []

    for i in range(num_outputs):
        output_file = data_folder + '/output_{}.pb'.format(i)
        output_ts = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            output_ts.ParseFromString(f.read())
        output_np = numpy_helper.to_array(output_ts)
        print("  - {} output: [{}x{}]".format(
            ordinal(i + 1), 'x'.join([str(i) for i in output_np.shape]),
            output_np.dtype))
        reference_output += [output_np]
    print("  done.\n")
    return reference_output


def generate_random_input(input_signature, input_shapes):
    print("Generating random inputs ...")
    # Generate random data as input.
    inputs = []

    # Load the input signature.
    signature = json.loads(input_signature)

    np.random.seed(42)
    for i, sig in enumerate(signature):
        # Get shape.
        explicit_shape = []
        for d, dim in enumerate(sig['dims']):
            if dim != -1:
                explicit_shape.append(dim)
                continue
            if i in input_shapes:
                if d < len(input_shapes[i]):
                    explicit_shape.append(input_shapes[i][d])
                else:
                    print("The {} dim".format(ordinal(d + 1)),
                          "of the {} input is unknown.".format(ordinal(i + 1)),
                          "Use --shape-info to set.")
                    print(" - The input signature: ", sig)
                    exit(1)
            else:
                print("The shape of the {} input".format(ordinal(i + 1)),
                      "is unknown. Use --shape-info to set.")
                print(" - The input signature: ", sig)
                exit(1)
        # Get element type.
        elem_type = sig['type']
        np_elem_type = MLIR_TYPE_TO_NP_TYPE[elem_type]
        # Set a range for random values.
        lb = ub = 0
        if (np.issubdtype(np_elem_type, np.floating)):
            lb = -1.0
            ub = 1.0
        elif (np.issubdtype(np_elem_type, np.integer)):
            lb = -10
            ub = 10
        else:
            raise AssertionError("Unsuported element type")
        rinput = np.random.uniform(lb, ub, explicit_shape).astype(np_elem_type)
        print(
            "  - {} input's shape {}, element type {}.".format(
                ordinal(i + 1), rinput.shape, np_elem_type),
            "Value ranges [{}, {}]".format(lb, ub))
        inputs.append(rinput)
    print("  done.\n")
    return inputs


def warning(msg):
    print("Warning:", msg)


def main():
    # Get shape information if given.
    # args.shape_info in the form of 'input_index:d1xd2, input_index:d1xd2'
    input_shapes = {}
    if args.shape_info:
        for input_shape in args.shape_info.strip().split(","):
            input_index_shape = input_shape.split(":")
            input_index = input_index_shape[0]
            assert not (input_index in input_shapes), "Duplicate input indices"
            dims = [int(d) for d in input_index_shape[1].split("x")]
            input_shapes[int(input_index)] = dims

    # Load the onnx model.
    if args.model:
        model = onnx.load(args.model)
        # Get names of all intermediate tensors and modify model such that each of
        # them will be an output of the model. If using onnxruntime for
        # verification, we can then verify every operation output.
        output_names = [o.name for o in model.graph.output]
        output_names = list(OrderedDict.fromkeys(output_names))
        if (args.verify and args.verify == "onnxruntime"
                and args.verify_all_ops):
            print("Extending the onnx model to check every node output ...\n")
            output_names = sum([[n for n in node.output if n != '']
                                for node in model.graph.node], [])
            output_names = list(OrderedDict.fromkeys(output_names))
            model = extend_model_output(model, output_names)

            # Save the modified onnx file of the model if required.
            if (args.save_onnx):
                print("Saving modified onnx model to ", args.save_onnx, "\n")
                onnx.save(model, args.save_onnx)

    # Compile, run, and verify.
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Temporary directory has been created at {}".format(temp_dir))

        shared_lib_path = ""

        # If a shared library is given, use it without compiling the ONNX model.
        # Otherwise, compile the ONNX model.
        if (args.load_so):
            shared_lib_path = args.load_so
        else:
            print("Compiling the model ...")
            # Save modified model & invoke onnx-mlir to compile it.
            temp_model_path = os.path.join(temp_dir, "model.onnx")
            onnx.save(model, temp_model_path)
            shared_lib_path = os.path.join(temp_dir, "model.so")

            # Prepare compiler arguments.
            command_str = [ONNX_MLIR]
            if args.compile_args:
                command_str += args.compile_args.split()
            if args.compile_using_input_shape:
                # Use shapes of the reference inputs to compile the model.
                assert args.data_folder, "No data folder given"
                assert "shapeInformation" not in command_str, "shape info was set"
                shape_info = "--shapeInformation="
                for i in range(len(inputs)):
                    shape_info += str(i) + ":" + 'x'.join(
                        [str(d) for d in inputs[i].shape]) + ","
                shape_info = shape_info[:-1]
                command_str += [shape_info]
                warning("the shapes of the model's inputs will be " \
                    "changed to the shapes of the inputs in the data folder")
            command_str += [temp_model_path]

            # Compile the model.
            start = time.perf_counter()
            ok, msg = execute_commands(command_str)
            if not ok:
                print(msg)
                exit(1)
            end = time.perf_counter()
            print("  took ", end - start, " seconds.\n")

            # Save the generated .so file of the model if required.
            if (args.save_so):
                print("Saving the shared library to", args.save_so, "\n")
                execute_commands(
                    ['rsync', '-ar', shared_lib_path, args.save_so])

            # Exit if only compiling the model.
            if (args.compile_only):
                exit(0)

        # Use the generated shared library to create an execution session.
        print("Loading the compiled model ...")
        start = time.perf_counter()
        sess = ExecutionSession(shared_lib_path)
        end = time.perf_counter()
        print("  took ", end - start, " seconds.\n")

        # Get the input and output signature.
        input_signature = sess.input_signature()
        output_signature = sess.output_signature()
        input_names = get_names_in_signature(input_signature)
        output_names = get_names_in_signature(output_signature)

        # Prepare input data.
        inputs = []
        if args.data_folder:
            inputs = read_input_from_refs(len(input_names), args.data_folder)
        else:
            inputs = generate_random_input(input_signature, input_shapes)

        # Print the input if required.
        if (args.print_input):
            for i, inp in enumerate(inputs):
                print("The {} input {}:[{}x{}] is: \n {} \n".format(
                    ordinal(i + 1), input_names[i],
                    'x'.join([str(i) for i in inp.shape]), inp.dtype, inp))

        print("Running inference ...")
        start = time.perf_counter()
        outs = sess.run(inputs)
        end = time.perf_counter()
        print("  took ", end - start, " seconds.\n")

        # Store the input and output if required.
        if args.save_data:
            data_folder = args.save_data
            if not os.path.exists(data_folder):
                os.mkdir(data_folder)
            for i in range(len(inputs)):
                tensor = numpy_helper.from_array(inputs[i])
                tensor_path = os.path.join(data_folder,
                                           'input_{}.pb'.format(i))
                with open(tensor_path, 'wb') as f:
                    f.write(tensor.SerializeToString())
            for i in range(len(outs)):
                tensor = numpy_helper.from_array(outs[i])
                tensor_path = os.path.join(data_folder,
                                           'output_{}.pb'.format(i))
                with open(tensor_path, 'wb') as f:
                    f.write(tensor.SerializeToString())

        # Run the model with reference backend and get results.
        if (args.verify):
            ref_outs = []
            if (args.verify.lower() == "onnxruntime"):
                # Reference backend by using onnxruntime.
                import onnxruntime
                input_feed = dict(zip(input_names, inputs))
                print("Running inference using onnxruntime ...")
                start = time.perf_counter()
                ref_session = onnxruntime.InferenceSession(temp_model_path)
                ref_outs = ref_session.run(output_names, input_feed)
                end = time.perf_counter()
                print("  took ", end - start, " seconds.\n")
            elif (args.verify.lower() == "ref"):
                ref_outs = read_output_from_refs(len(output_names),
                                                 args.data_folder)
            else:
                print("Invalid verify option")
                exit(1)

            # For each output tensor, compare results.
            for i, name in enumerate(output_names):
                print(
                    "Verifying value of {}:{}".format(name,
                                                      list(outs[i].shape)),
                    "using atol={}, rtol={} ...".format(args.atol, args.rtol))
                total_elements = 0
                mismatched_elements = 0
                for index, actual_val in np.ndenumerate(outs[i]):
                    total_elements += 1
                    ref_val = ref_outs[i][index]
                    # Use equation atol + rtol * abs(desired), that is used in assert_allclose.
                    diff = float(args.atol) + float(args.rtol) * abs(ref_val)
                    if (abs(actual_val - ref_val) <= diff):
                        continue
                    mismatched_elements += 1
                    print("  at {}".format(index),
                          "mismatch {} (actual)".format(actual_val),
                          "vs {} (reference)".format(ref_val))
                if mismatched_elements == 0:
                    print("  correct.\n".format(args.atol, args.rtol))
                else:
                    raise AssertionError(
                        "  mismatched elements {}/{}.\n".format(
                            mismatched_elements, total_elements))

        # Print the output if required.
        if (args.print_output):
            for i, out in enumerate(outs):
                print("The {} output {}:[{}x{}] is: \n {} \n".format(
                    ordinal(i + 1), output_names[i],
                    'x'.join([str(i) for i in out.shape]), out.dtype, out))


if __name__ == '__main__':
    main()
