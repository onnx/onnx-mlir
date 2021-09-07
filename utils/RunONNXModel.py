import os
import sys
import argparse
import onnx
import time
import subprocess
import numpy as np
import tempfile

from onnx import numpy_helper
from collections import OrderedDict

if (not os.environ.get('ONNX_MLIR_HOME', None)):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
        "the parent folder containing the bin, lib, etc sub-folders in which ONNX-MLIR "
        "executables and libraries can be found.")

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
    )


def execute_commands(cmds):
    if (VERBOSE):
        print(" ".join(cmds))
    subprocess.run(cmds, stdout=subprocess.PIPE, check=True)


def extend_model_output(model, intermediate_outputs):
    # onnx-mlir doesn't care about manually specified output types & shapes.
    DUMMY_TENSOR_TYPE = onnx.TensorProto.FLOAT

    while (len(model.graph.output)):
        model.graph.output.pop()

    for output_name in intermediate_outputs:
        output_value_info = onnx.helper.make_tensor_value_info(
            output_name, DUMMY_TENSOR_TYPE, None)
        model.graph.output.extend([output_value_info])
    return model


def read_input_from_refs(model, ref_folder):
    print("Reading inputs from {} ...".format(ref_folder))
    i = 0
    inputs = []
    input_names = []
    initializers = list(map(lambda x: x.name, model.graph.initializer))
    for input_proto in model.graph.input:
        if input_proto.name not in initializers:
            input_names.append(input_proto.name)
            input_file = ref_folder + '/input_{}.pb'.format(i)
            input_ts = onnx.TensorProto()
            with open(input_file, 'rb') as f:
                input_ts.ParseFromString(f.read())
            inputs += [numpy_helper.to_array(input_ts)]
            i += 1
    print("  done.")
    return (inputs, input_names)


def read_output_from_refs(model, ref_folder):
    print("Reading reference outputs from {} ...".format(ref_folder))
    reference_output = []
    for i, _ in enumerate(model.graph.output):
        output_file = ref_folder + '/output_{}.pb'.format(i)
        output_ts = onnx.TensorProto()
        with open(output_file, 'rb') as f:
            output_ts.ParseFromString(f.read())
        reference_output += [numpy_helper.to_array(output_ts)]
    print("  done.")
    return reference_output


def generate_random_input(model, input_shapes):
    print("Generating random inputs ...")
    # Generate random data as input.
    inputs = []
    input_names = []
    initializers = list(map(lambda x: x.name, model.graph.initializer))
    np.random.seed(42)
    for i, input_proto in enumerate(model.graph.input):
        if input_proto.name not in initializers:
            input_names.append(input_proto.name)
            shape_proto = input_proto.type.tensor_type.shape
            explicit_shape = []
            for d, dim in enumerate(shape_proto.dim):
                if dim.dim_value:
                    explicit_shape.append(dim.dim_value)
                else:
                    if i in input_shapes:
                        explicit_shape.append(input_shapes[i][d])
                    else:
                        print(
                            "No shape information for the {}-th input. Use --shape-info."
                            .format(i))
                        print(shape_proto)
                        exit()
            inputs.append(
                np.random.uniform(-1.0, 1.0,
                                  explicit_shape).astype(np.float32))
    print("  done.")
    return (inputs, input_names)


def main(model_path,
         mcpu,
         mtriple,
         shape_info,
         verify,
         ref_folder,
         atol=0.01,
         rtol=0.05):
    # Get shape information if given.
    # shape_info in the form of 'input_index:d1xd2, input_index:d1xd2'
    input_shapes = {}
    if shape_info:
        for input_shape in shape_info.strip().split(","):
            input_index_shape = input_shape.split(":")
            input_index = input_index_shape[0]
            assert not (input_index in input_shapes), "Duplicate input indices"
            dims = [int(d) for d in input_index_shape[1].split("x")]
            input_shapes[int(input_index)] = dims

    # Load the onnx model.
    model = onnx.load(model_path)

    # Compile, run, and verify.
    with tempfile.TemporaryDirectory() as temp_dir:
        print("Temporary directory has been created at {}".format(temp_dir))

        print("Compiling the model ...")
        # Save modified model & invoke onnx-mlir to compile it.
        temp_model_path = os.path.join(temp_dir, "model.onnx")
        onnx.save(model, temp_model_path)
        command_list = [ONNX_MLIR]
        if mcpu:
            command_list.append("--mcpu=" + args.mcpu)
        if mtriple:
            command_list.append("--mtriple=" + args.mtriple)
        command_list.append(temp_model_path)
        start = time.perf_counter()
        execute_commands(command_list)
        end = time.perf_counter()
        print("  took ", end - start, " seconds.")

        # Prepare input data.
        inputs = []
        input_names = []
        if (verify and verify.lower() == "ref"):
            assert ref_folder, "No reference folder given"
            inputs, input_names = read_input_from_refs(model, ref_folder)
        else:
            inputs, input_names = generate_random_input(model, input_shapes)

        print("Running inference ...")
        temp_shared_lib_path = os.path.join(temp_dir, "model.so")
        start = time.perf_counter()
        # Use the generated shared library to create an execution session.
        sess = ExecutionSession(temp_shared_lib_path, "run_main_graph")
        outs = sess.run(inputs)
        end = time.perf_counter()
        print("  took ", end - start, " seconds.")

        # Run the model with reference backend and get results.
        if (verify):
            ref_outs = []
            if (verify.lower() == "onnxruntime"):
                # Reference backend by using onnxruntime.
                import onnxruntime
                output_names = list(map(lambda x: x.name, model.graph.output))
                input_feed = dict(zip(input_names, inputs))
                print("Running inference using onnxruntime ...")
                start = time.perf_counter()
                ref_session = onnxruntime.InferenceSession(temp_model_path)
                ref_outs = ref_session.run(output_names, input_feed)
                end = time.perf_counter()
                print("  took ", end - start, " seconds.")

            elif (verify.lower() == "ref"):
                ref_outs = read_output_from_refs(model, ref_folder)
            else:
                print("Invalid verify option")
                exit()

            # For each intermediate output tensor, compare results.
            for i, output_proto in enumerate(model.graph.output):
                print("Verifying value of {}".format(output_proto.name))
                np.testing.assert_allclose(ref_outs[i],
                                           outs[i],
                                           rtol=rtol,
                                           atol=atol,
                                           verbose=True)
                print("  correct with atol={}, rtol={}.".format(atol, rtol))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path',
                        type=str,
                        help="Path to the ONNX model.")
    parser.add_argument('--mtriple',
                        type=str,
                        default="",
                        help='Triple to pass to the compiler')
    parser.add_argument('--mcpu',
                        type=str,
                        default="",
                        help='Target a specific cpu, passed to the compiler')
    parser.add_argument(
        '--shape_info',
        type=str,
        help="Shape for each dynamic input, e.g. 0:1x10x20,1:7x5x3")
    parser.add_argument('--verify', choices=['onnxruntime', 'ref'])
    parser.add_argument(
        '--ref_folder',
        type=str,
        help="Path to the folder containing reference inputs and outputs stored"
        " in protobuf. Used when --verify=ref")
    parser.add_argument('--rtol',
                        type=str,
                        default="0.05",
                        help="Relative tolerance for verification")
    parser.add_argument('--atol',
                        type=str,
                        default="0.01",
                        help="Absolute tolerance for verification")
    args = parser.parse_args()
    main(args.model_path, args.mtriple, args.mcpu, args.shape_info,
         args.verify, args.ref_folder, float(args.rtol), float(args.atol))
