#!/usr/bin/env python3

##################### input_verification_backend.py ############################
#
# Copyright 2022 The IBM Research Authors.
#
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import sys
import numpy as np
from collections import defaultdict
from collections import namedtuple
from common import compile_model
from contextlib import contextmanager
from variables import *
import ctypes
import tempfile

from onnx import helper, TensorProto
from onnx.backend.base import Device, DeviceType, Backend
from onnx.backend.test import BackendTest

TestCase = namedtuple(
    "TestCase", ["name", "model_name", "model", "input", "output", "kind"]
)


def load_model_tests(kind):
    test_arr = [
        (
            "wrong_number_of_inputs_less",
            [np.ones((3, 4, 5)).astype("float32")],
            "Wrong number of input tensors: expect 2, but got 1",
        ),
        (
            "wrong_number_of_inputs_more",
            [
                np.ones((3, 4, 5)).astype("float32"),
                np.ones((3, 4, 5)).astype("float32"),
                np.ones((3, 4, 5)).astype("float32"),
            ],
            "Wrong number of input tensors: expect 2, but got 3",
        ),
        (
            "wrong_rank_less",
            [np.ones((3, 4, 5)).astype("float32"), np.ones((3, 4)).astype("float32")],
            "Wrong rank for the input 1: expect 3, but got 2",
        ),
        (
            "wrong_rank_more",
            [
                np.ones((3, 4, 5)).astype("float32"),
                np.ones((3, 4, 5, 1)).astype("float32"),
            ],
            "Wrong rank for the input 1: expect 3, but got 4",
        ),
        (
            "wrong_data_type",
            [np.ones((3, 4, 5)).astype("int32"), np.ones((3, 4, 5)).astype("float32")],
            "Wrong data type for the input 0: expect f32",
        ),
        (
            "wrong_dim_size",
            [
                np.ones((3, 4, 1)).astype("float32"),
                np.ones((3, 4, 5)).astype("float32"),
            ],
            "Wrong size for the dimension 2 of the input 0: expect 5, but got 1",
        ),
    ]

    testcases = []

    for name, in_data, out_data in test_arr:
        model = AddModel.model(name)
        testcases.append(
            TestCase(
                name="test_{}".format(model.graph.name),
                model_name=model.graph.name,
                model=model,
                input=in_data,
                output=out_data,
                kind=kind,
            )
        )

    return testcases


class AddModel:
    @staticmethod
    def model(name):
        node = helper.make_node("Add", inputs=["x1", "x2"], outputs=["y"])

        x1 = helper.make_tensor_value_info("x1", TensorProto.FLOAT, [3, 4, 5])
        x2 = helper.make_tensor_value_info("x2", TensorProto.FLOAT, ["unknown", 4, 5])
        y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 4, 5])

        # Create the graph (GraphProto)
        graph = helper.make_graph([node], "{}_{}".format("add", name), [x1, x2], [y])

        # Create the model (ModelProto)
        model = helper.make_model(graph)

        return model


class InputVerificationBackendTest(BackendTest):
    def __init__(self, backend, parent_module=None):
        self.backend = backend
        self._parent_module = parent_module
        self._include_patterns = set()  # type: Set[Pattern[Text]]
        self._exclude_patterns = set()  # type: Set[Pattern[Text]]
        self._xfail_patterns = set()  # type: Set[Pattern[Text]]

        self._test_items = defaultdict(dict)  # type: Dict[Text, Dict[Text, TestItem]]

        for rt in load_model_tests(kind="node"):
            self._add_model_test(rt, "Node")

    @classmethod
    def assert_similar_outputs(cls, ref_output, output):  # type: (str, str) -> None
        assert (
            ref_output in output
        ), "Verification message {} does not match expected value {}.".format(
            output, ref_output
        )

    def _add_model_test(self, model_test, kind):  # type: (TestCase, Text) -> None
        model_marker = [None]  # type: List[Optional[Union[ModelProto, NodeProto]]]

        def run(test_self, device):  # type: (Any, Text) -> None
            model_marker[0] = model_test.model
            prepared_model = self.backend.prepare(model_test.model, device)
            assert prepared_model is not None

            output = prepared_model.run(model_test.input)
            ref_output = model_test.output
            self.assert_similar_outputs(ref_output, output)

        self._add_test(kind + "Model", model_test.name, run, model_marker)


@contextmanager
def redirect_c_stdout(stream):
    """Borrowed from:
    - https://eli.thegreenplace.net/2015/redirecting-all-kinds-of-stdout-in-python/
    """
    # The original fd stdout points to.
    original_stdout_fd = sys.stdout.fileno()

    def _redirect_stdout(to_fd):
        """Redirect stdout to the given file descriptor."""
        # Flush the C-level buffer stdout
        # Note: this does not work on Windows.
        libc = ctypes.CDLL(None)
        c_stdout = ctypes.c_void_p.in_dll(
            libc, "__stdoutp" if sys.platform == "darwin" else "stdout"
        )
        libc.fflush(c_stdout)
        # Flush and close sys.stdout - also closes the file descriptor (fd)
        sys.stdout.close()
        # Make original_stdout_fd point to the same file as to_fd
        os.dup2(to_fd, original_stdout_fd)
        # Create a new sys.stdout that points to the redirected fd
        sys.stdout = io.TextIOWrapper(os.fdopen(original_stdout_fd, "wb"))

    # Save a copy of the original stdout fd in saved_stdout_fd
    saved_stdout_fd = os.dup(original_stdout_fd)
    try:
        # Create a temporary file and redirect stdout to it
        tfile = tempfile.TemporaryFile(mode="w+b")
        _redirect_stdout(tfile.fileno())
        # Yield to caller, then redirect stdout back to the saved fd
        yield
        _redirect_stdout(saved_stdout_fd)
        # Copy contents of temporary file to the given stream
        tfile.flush()
        tfile.seek(0, io.SEEK_SET)
        stream.write(tfile.read())
    finally:
        tfile.close()
        os.close(saved_stdout_fd)


class InputVerificationExecutionSession(object):
    def __init__(self, model):
        self.model = model
        self.exec_name = compile_model(self.model, args.emit)

    def run(self, inputs, **kwargs):
        sys.path.append(RUNTIME_DIR)
        from PyRuntime import OMExecutionSession

        session = OMExecutionSession(self.exec_name)
        f = io.BytesIO()
        with redirect_c_stdout(f):
            try:
                session.run(inputs)
            except RuntimeError as re:
                pass
        output = f.getvalue().decode("utf-8")
        return output


class InputVerificationBackend(Backend):
    @classmethod
    def prepare(cls, model, device="CPU", **kwargs):
        super(InputVerificationBackend, cls).prepare(model, device, **kwargs)
        return InputVerificationExecutionSession(model)

    @classmethod
    def supports_device(cls, device):
        d = Device(device)
        if d.type == DeviceType.CPU:
            return True
        return False
