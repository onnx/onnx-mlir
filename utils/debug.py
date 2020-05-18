import os
import sys
import argparse
import onnx
import subprocess
import numpy as np
import tempfile

from functools import reduce
from collections import OrderedDict

# Reference backend.
import onnxruntime as ref_backend

if (not os.environ.get('ONNX_MLIR_HOME', None)):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The Home directory for onnx-mlir refers to "
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
    subprocess.run(cmds, stdout=subprocess.PIPE)


def extend_model_output(model, intermediate_outputs):
    # onnx-mlir doesn't care about manually specified output types & shapes.
    DUMMY_TENSOR_TYPE = onnx.TensorProto.FLOAT
    DUMMY_TENSOR_SHAPE = []

    while(len(model.graph.output)):
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
        temp_model_path = os.path.join(temp_dir, "model.onnx")
        onnx.save(model, temp_model_path)
        execute_commands([ONNX_MLIR, temp_model_path])

        temp_shared_lib_path = os.path.join(temp_dir, "model.so")
        sess = ExecutionSession(temp_shared_lib_path, "_dyn_entry_point_main_graph")
        img = np.ones([1, 1, 28, 28], dtype=np.float32)
        outs = sess.run([img])

        ref_session = ref_backend.InferenceSession(temp_model_path)
        output_names = list(map(lambda x: x.name, model.graph.output))
        input_names = list(map(lambda x: x.name, model.graph.input))
        input_feed = dict(zip(input_names, [img]))
        ref_outs = ref_session.run(output_names, input_feed)

        for i, name in enumerate(intermediate_outputs):
            print("Verifying value of {}".format(name))
            np.testing.assert_array_almost_equal(ref_outs[i], outs[i])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', type=str)
    args = parser.parse_args()
    main(**vars(args))
