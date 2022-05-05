#!/usr/bin/env python3

####################### test.py ################################################
#
# Copyright 2021-2022 The IBM Research Authors.
#
################################################################################
# Reorganize backend testing into following modules:
# - variables.py: all global variables
#   * Immutable variables are initialized once
#   * Mutable variables are set in one module file and used in another
# - common.py: common functions called by xxxExecuteSession
# - inference_backend.py:
#   * model list for inference testing
#   * class InferenceBackend
#   * class EndiannessAwareExecutionSession & JniExecuteSession
# - signature_backend.py:
#   * class SliceModel which generate slice models for signature testing
#   * class SignatureBackendTest which inherits from BackendTest
#   * class SignatureBackend
#   * class SignatureExecutionSession
# - test.py: main process
################################################################################
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
import onnx
import unittest
from onnx.backend.test import BackendTest

import inspect
from inference_backend import (
    InferenceBackend,
    get_test_models,
)
from signature_backend import SignatureBackendTest, SignatureBackend
import variables
from variables import args

###########   main process   #############
sys.argv[1:] = args.unittest_args

if args.signature:
    backend_test = SignatureBackendTest(SignatureBackend, __name__)
else:
    # Models to test
    test_to_enable = get_test_models()

    # Backend Test
    backend_test = BackendTest(InferenceBackend, __name__)

    # Extract name of all test cases.
    all_tests = []
    variables.real_model_tests = inspect.getmembers(
        backend_test.test_cases["OnnxBackendRealModelTest"]
    )
    all_tests += variables.real_model_tests
    variables.node_model_tests = inspect.getmembers(
        backend_test.test_cases["OnnxBackendNodeModelTest"]
    )
    all_tests += variables.node_model_tests
    all_test_names = list(map(lambda x: x[0], all_tests))

    # Ensure that test names specified in test_to_enable actually exist.
    for test_name in test_to_enable:
        assert (
            test_name in all_test_names
        ), """test name {} not found, it is likely
        that you may have misspelled the test name or the specified test does not
        exist in the version of onnx package you installed.""".format(
            test_name
        )
        backend_test.include(r"^{}$".format(test_name))

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test.test_cases)

if __name__ == "__main__":

    unittest.main()
