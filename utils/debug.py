import os
import sys
import argparse
import onnx
import subprocess
import numpy as np
import tempfile

from collections import OrderedDict

# Reference backend, use onnxruntime by default
import onnxruntime
prepare = onnxruntime.InferenceSession

if (not os.environ.get('ONNX_MLIR_HOME', None)):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
        "the parent folder containing the bin, lib, etc sub-folders in which ONNX-MLIR "
        "executables and libraries can be found.")

VERBOSE = os.environ.get('VERBOSE', False)
ONNX_MLIR = os.path.join(os.environ['ONNX_MLIR_HOME'], "bin/onnx-mlir")

# Include runtime directory in python paths, so pyruntime can be imported.
RUNTIME_DIR = os.path.join(os.environ['ONNX_MLIR_HOME'], "lib")
sys.path.append(RUNTIME_DIR)

try:
    from pyruntime import ExecutionSession
except ImportError:
    raise ImportError(
        "Looks like you did not build the pyruntime target, build it by running `make pyruntime`."
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


def main(model_path):
    model = onnx.load(model_path)
    intermediate_outputs = sum(
        [list(node.output) for node in model.graph.node], [])
    intermediate_outputs = list(OrderedDict.fromkeys(intermediate_outputs))
    model = extend_model_output(model, intermediate_outputs)

    with tempfile.TemporaryDirectory() as temp_dir:
        print("Temporary directory has been created at {}".format(temp_dir))

        # Save modified model & invoke onnx-mlir to compile it.
        temp_model_path = os.path.join(temp_dir, "model.onnx")
        onnx.save(model, temp_model_path)
        execute_commands([ONNX_MLIR, temp_model_path])

        # Use the generated shared library to create an execution session.
        temp_shared_lib_path = os.path.join(temp_dir, "model.so")
        sess = ExecutionSession(temp_shared_lib_path,
                                "_dyn_entry_point_main_graph")

        # Generate random data as input.
        inputs = []
        np.random.seed(42)
        for input_proto in model.graph.input:
            shape_proto = input_proto.type.tensor_type.shape
            explicit_shape = []
            for dim in shape_proto.dim:
                assert dim.dim_value, "Can only debug models with inputs that have explicit shapes."
                explicit_shape.append(dim.dim_value)
            inputs.append(
                np.random.uniform(-1.0, 1.0, explicit_shape).astype(np.float32))

        # Run the compiled inference function on the randomly generated data.
        outs = sess.run(inputs)

        # Run the model with reference backend and get results.
        ref_session = prepare(temp_model_path)
        output_names = list(map(lambda x: x.name, model.graph.output))
        input_names = list(map(lambda x: x.name, model.graph.input))
        input_feed = dict(zip(input_names, inputs))
        ref_outs = ref_session.run(output_names, input_feed)

        # For each intermediate output tensor, compare results.
        for i, name in enumerate(intermediate_outputs):
            print("Verifying value of {}".format(name))
            np.testing.assert_array_almost_equal(ref_outs[i], outs[i])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str, help="Path to the model to debug.")
    args = parser.parse_args()
    main(**vars(args))
