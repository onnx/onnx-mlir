# SPDX-License-Identifier: Apache-2.0

##################### add_sub_onnx.py ##########################################
#
# Copyright 2023 The IBM Research Authors.
#
################################################################################
#
# This script is to test multiple models in the same process.

################################################################################

import os
import sys
import argparse

parser = argparse.ArgumentParser()
group = parser.add_mutually_exclusive_group()
group.add_argument("--create-models", action="store_true", help="mode to create models")
group.add_argument("--run-models", action="store_true", help="mode to run models")
args = parser.parse_args()

if not os.environ.get("MODELS_PATH", None):
    raise RuntimeError(
        "Environment variable MODELS_PATH is not set, please set it to the path to "
        "the directory where the .onnx models are located."
    )

MODELS = os.environ["MODELS_PATH"]
add_model = MODELS + "/add.onnx"
sub_model = MODELS + "/sub.onnx"

if args.create_models:
    import onnx
    from onnx import helper
    from onnx import AttributeProto, TensorProto, GraphProto

    # Create the model (ModelProto) of a single Add operator.
    add_node_def = helper.make_node(
        "Add",  # node name
        ["X1", "X2"],  # inputs
        ["Y"],  # outputs
    )
    add_graph_def = helper.make_graph(
        [add_node_def],
        "add-model",
        [
            helper.make_tensor_value_info("X1", TensorProto.INT64, [3, 2]),
            helper.make_tensor_value_info("X2", TensorProto.INT64, [3, 2]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, [3, 2])],
    )
    add_model_def = helper.make_model(add_graph_def, producer_name="multi-models-test")
    onnx.checker.check_model(add_model_def)
    onnx.save(add_model_def, add_model)

    # Create the model (ModelProto) of a single Sub operator.
    sub_node_def = helper.make_node(
        "Sub",  # node name
        ["X1", "X2"],  # inputs
        ["Y"],  # outputs
    )
    sub_graph_def = helper.make_graph(
        [sub_node_def],
        "sub-model",
        [
            helper.make_tensor_value_info("X1", TensorProto.INT64, [3, 2]),
            helper.make_tensor_value_info("X2", TensorProto.INT64, [3, 2]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.INT64, [3, 2])],
    )
    sub_model_def = helper.make_model(
        sub_graph_def, producer_name="multi-models-test-1"
    )
    onnx.checker.check_model(sub_model_def)
    onnx.save(sub_model_def, sub_model)

if args.run_models:
    import numpy as np

    if not os.environ.get("PY_LIB", None):
        raise RuntimeError(
            "Environment variable PY_LIB is not set, "
            "please set it to the folder containing "
            "PyCompileAndRuntimeXXX.so"
        )
    RUNTIME_DIR = os.environ["PY_LIB"]
    sys.path.append(RUNTIME_DIR)
    try:
        from PyCompileAndRuntime import OMCompileExecutionSession
    except ImportError as ie:
        raise ie
    add_sess = OMCompileExecutionSession(add_model, "", reuse_compiled_model=0)
    sub_sess = OMCompileExecutionSession(sub_model, "", reuse_compiled_model=0)
    a = np.array([10, 20, 30, 40, 50, 60], dtype=np.int64).reshape((3, 2))
    b = np.array([1, 2, 3, 4, 5, 6], dtype=np.int64).reshape((3, 2))

    # Do (a+b)-b using the two models: Add and Sub.
    ab = add_sess.run([a, b])[0]
    abb = sub_sess.run([ab, b])[0]
    try:
        # Verify that a+b-b = a.
        np.testing.assert_array_equal(a, abb)
        print("Checking multiple models: passed")
    except AssertionError as ae:
        raise ae
