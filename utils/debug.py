import os
import sys
import argparse
import onnx
import time
import subprocess
import numpy as np
import tempfile

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


def main(model_path, verify, shape_info, mcpu, mtriple):
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

    model = onnx.load(model_path)
    intermediate_outputs = sum(
        [list(node.output) for node in model.graph.node], [])
    intermediate_outputs = list(OrderedDict.fromkeys(intermediate_outputs))
    model = extend_model_output(model, intermediate_outputs)

    with tempfile.TemporaryDirectory() as temp_dir:
        print("Temporary directory has been created at {}".format(temp_dir))

        print("Compiling the model ...")
        # Save modified model & invoke onnx-mlir to compile it.
        temp_model_path = os.path.join(temp_dir, "model.onnx")
        onnx.save(model, temp_model_path)
        command_list = [ONNX_MLIR]
        if mcpu:
            command_list.append("--mcpu="+args.mcpu)
        if mtriple:
            command_list.append("--mtriple="+args.mtriple)
        command_list.append(temp_model_path)
        start = time.perf_counter()
        execute_commands(command_list)
        end = time.perf_counter()
        print("  took ", end - start, " seconds.")

        # Use the generated shared library to create an execution session.
        temp_shared_lib_path = os.path.join(temp_dir, "model.so")
        sess = ExecutionSession(temp_shared_lib_path, "run_main_graph")

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

        print("Running inference ...")
        # Run the compiled inference function on the randomly generated data.
        start = time.perf_counter()
        outs = sess.run(inputs)
        end = time.perf_counter()
        print("  took ", end - start, " seconds.")

        # Run the model with reference backend and get results.
        if (verify and verify.lower() == "onnxruntime"):
            # Reference backend, use onnxruntime by default
            import onnxruntime
            prepare = onnxruntime.InferenceSession

            ref_session = prepare(temp_model_path)
            output_names = list(map(lambda x: x.name, model.graph.output))
            input_feed = dict(zip(input_names, inputs))
            ref_outs = ref_session.run(output_names, input_feed)

            # For each intermediate output tensor, compare results.
            for i, name in enumerate(intermediate_outputs):
                print("Verifying value of {}".format(name))
                np.testing.assert_array_almost_equal(ref_outs[i],
                                                     outs[i],
                                                     decimal=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path',
                        type=str,
                        help="Path to the model to debug.")
    parser.add_argument('--verify', choices=['onnxruntime'])
    parser.add_argument('--shape-info',
                        type=str,
                        help="Shape for each input, e.g. 0:1x10x20,1:7x5x3")
    parser.add_argument('--mtriple', type=str, default="", 
                        help='triple to pass to the compiler')
    parser.add_argument('--mcpu', type=str, default="",
                        help='target a specific cpu, passed to the compiler')
    args = parser.parse_args()
    main(**vars(args))
