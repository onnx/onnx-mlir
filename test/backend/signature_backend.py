#!/usr/bin/env python3

##################### signature_backend.py #####################################
#
# Copyright 2021 The IBM Research Authors.
#
################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict
import sys
from onnx import helper, TensorProto
from onnx.backend.base import Device, DeviceType, Backend
from onnx.backend.test import BackendTest
from collections import namedtuple
from common import compile_model
from variables import *

TestCase = namedtuple("TestCase", ["name", "model_name", "model", "output", "kind"])


def load_model_tests(kind):
    test_arr = [
        (
            "uint8",
            TensorProto.UINT8,
            '[{"type":"ui8","dims":[2,2,1],"name":"x"},{"type":"i64","dims":[1],"name":"starts"},{"type":"i64","dims":[1],"name":"ends"}]',
        ),
        (
            "uint16",
            TensorProto.UINT16,
            '[{"type":"ui16","dims":[2,2,1],"name":"x"},{"type":"i64","dims":[1],"name":"starts"},{"type":"i64","dims":[1],"name":"ends"}]',
        ),
        (
            "uint32",
            TensorProto.UINT32,
            '[{"type":"ui32","dims":[2,2,1],"name":"x"},{"type":"i64","dims":[1],"name":"starts"},{"type":"i64","dims":[1],"name":"ends"}]',
        ),
        (
            "uint64",
            TensorProto.UINT64,
            '[{"type":"ui64","dims":[2,2,1],"name":"x"},{"type":"i64","dims":[1],"name":"starts"},{"type":"i64","dims":[1],"name":"ends"}]',
        ),
        (
            "int8",
            TensorProto.INT8,
            '[{"type":"i8","dims":[2,2,1],"name":"x"},{"type":"i64","dims":[1],"name":"starts"},{"type":"i64","dims":[1],"name":"ends"}]',
        ),
        (
            "int16",
            TensorProto.INT16,
            '[{"type":"i16","dims":[2,2,1],"name":"x"},{"type":"i64","dims":[1],"name":"starts"},{"type":"i64","dims":[1],"name":"ends"}]',
        ),
        (
            "int32",
            TensorProto.INT32,
            '[{"type":"i32","dims":[2,2,1],"name":"x"},{"type":"i64","dims":[1],"name":"starts"},{"type":"i64","dims":[1],"name":"ends"}]',
        ),
        (
            "int64",
            TensorProto.INT64,
            '[{"type":"i64","dims":[2,2,1],"name":"x"},{"type":"i64","dims":[1],"name":"starts"},{"type":"i64","dims":[1],"name":"ends"}]',
        ),
        (
            "float16",
            TensorProto.FLOAT16,
            '[{"type":"f16","dims":[2,2,1],"name":"x"},{"type":"i64","dims":[1],"name":"starts"},{"type":"i64","dims":[1],"name":"ends"}]',
        ),
        (
            "float",
            TensorProto.FLOAT,
            '[{"type":"f32","dims":[2,2,1],"name":"x"},{"type":"i64","dims":[1],"name":"starts"},{"type":"i64","dims":[1],"name":"ends"}]',
        ),
        (
            "double",
            TensorProto.DOUBLE,
            '[{"type":"f64","dims":[2,2,1],"name":"x"},{"type":"i64","dims":[1],"name":"starts"},{"type":"i64","dims":[1],"name":"ends"}]',
        ),
    ]

    testcases = []

    for dtype_name, dtype, output in test_arr:
        model = SliceModel.model(dtype_name, dtype)
        testcases.append(
            TestCase(
                name="test_{}".format(model.graph.name),
                model_name=model.graph.name,
                model=model,
                output=output,
                kind=kind,
            )
        )

    return testcases


class SliceModel:
    @staticmethod
    def model(dtype_name, dtype):
        node = helper.make_node("Slice", inputs=["x", "starts", "ends"], outputs=["y"])

        x = helper.make_tensor_value_info("x", dtype, [2, 2, 1])
        starts = helper.make_tensor_value_info(
            "starts",
            TensorProto.INT64,
            [
                1,
            ],
        )
        ends = helper.make_tensor_value_info(
            "ends",
            TensorProto.INT64,
            [
                1,
            ],
        )
        y = helper.make_tensor_value_info("y", dtype, [None, 2, 1])

        # Create the graph (GraphProto)
        graph = helper.make_graph(
            [node], "{}_{}".format("slice", dtype_name), [x, starts, ends], [y]
        )

        # Create the model (ModelProto)
        model = helper.make_model(graph)

        return model


class SignatureBackendTest(BackendTest):
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
        output_format = output.replace(" ", "").replace("\n", "").replace("\r", "")
        assert (
            ref_output == output_format
        ), "Input signature {} does not match expected value {}.".format(
            output_format, ref_output
        )

    def _add_model_test(self, model_test, kind):  # type: (TestCase, Text) -> None
        model_marker = [None]  # type: List[Optional[Union[ModelProto, NodeProto]]]

        def run(test_self, device):  # type: (Any, Text) -> None
            model_marker[0] = model_test.model
            prepared_model = self.backend.prepare(model_test.model, device)
            assert prepared_model is not None

            output = prepared_model.run()
            ref_output = model_test.output
            self.assert_similar_outputs(ref_output, output)

        self._add_test(kind + "Model", model_test.name, run, model_marker)


class SignatureExecutionSession(object):
    def __init__(self, model):
        self.model = model
        self.exec_name = compile_model(self.model, args.emit)

    def run(self, **kwargs):
        sys.path.append(RUNTIME_DIR)
        from PyRuntime import OMExecutionSession

        session = OMExecutionSession(self.exec_name)
        output = session.input_signature()
        return output


class SignatureBackend(Backend):
    @classmethod
    def prepare(cls, model, device="CPU", **kwargs):
        super(SignatureBackend, cls).prepare(model, device, **kwargs)
        return SignatureExecutionSession(model)

    @classmethod
    def supports_device(cls, device):
        d = Device(device)
        if d.type == DeviceType.CPU:
            return True
        return False
