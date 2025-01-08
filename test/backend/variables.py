#!/usr/bin/env python3

############################ variables.py #####################################
#
# Copyright 2021-2022 The IBM Research Authors.
#
################################################################################
# Immutable global variables:
#   - args, tempdir, result_dir, RUNTIME_DIR, TEST_DRIVER
# Mutable global variables:
#   - test_for_dynamic, test_for_constant, test_for_constants_to_file, test_need_converter
#   - real_model_tests, node_model_tests
#   - test_to_enable_dict
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import sys
import argparse
import tempfile


# Reimplement strtobool per PEP 632 and python 3.12 deprecation
def strtobool(s: str) -> bool:
    if s.lower() in ["y", "yes", "t", "true", "on", "1"]:
        return True
    elif s.lower() in ["n", "no", "f", "false", "off", "0"]:
        return False
    else:
        raise ValueError(f"{s} cannot be converted to bool")


def get_args_from_env():
    # Casting with "bool" does not work well. When you specify TEST_VERBOSE=xxx,
    # regardless of the value of xxx (e.g., true, false, y, n, etc.) the
    # casted bool value will be true. Only if xxx is empty, the casted bool
    # value will be false. This is a bit counter intuitive. So we use strtobool
    # to do the conversion. But note that strtobool can't take an empty string.

    TEST_VERBOSE = os.getenv("TEST_VERBOSE")
    TEST_CASE_CHECK = os.getenv("TEST_CASE_CHECK")
    TEST_INVOKECONVERTER = os.getenv("TEST_INVOKECONVERTER")
    # Force input tensors to constants. Set this to a list of input indices.
    # E.g.
    #   - "0, 2" for the first and third input tensors.
    #   - "-1" for all the input tensors.
    TEST_DYNAMIC = os.getenv("TEST_DYNAMIC")
    TEST_CONSTANT = os.getenv("TEST_CONSTANT")
    TEST_SIGNATURE = os.getenv("TEST_SIGNATURE")
    TEST_INPUT_VERIFICATION = os.getenv("TEST_INPUT_VERIFICATION")
    TEST_COMPILERLIB = os.getenv("TEST_COMPILERLIB")
    TEST_INSTRUCTION_CHECK = os.getenv("TEST_INSTRUCTION_CHECK")
    TEST_CONSTANTS_TO_FILE = os.getenv("TEST_CONSTANTS_TO_FILE")
    TEST_NOFLOAT16 = os.getenv("TEST_NOFLOAT16")

    # Set ONNX_HOME to /tmp if not set to prevent onnx from downloading
    # real model files into home directory.
    if os.getenv("ONNX_HOME") is None or not os.getenv("ONNX_HOME").strip():
        os.environ["ONNX_HOME"] = "/tmp"

    parser = argparse.ArgumentParser(description="with dynamic shape or not.")
    parser.add_argument(
        "--signature",
        action="store_true",
        default=(strtobool(TEST_SIGNATURE) if TEST_SIGNATURE else False),
        help="enable signature tests (default: false if TEST_SIGNATURE env var not set)",
    )
    parser.add_argument(
        "--dynamic",
        action="store_true",
        default=(strtobool(TEST_DYNAMIC) if TEST_DYNAMIC else False),
        help="enable dynamic shape tests (default: false if TEST_DYNAMIC env var not set)",
    )
    parser.add_argument(
        "--constant",
        action="store_true",
        default=(strtobool(TEST_CONSTANT) if TEST_CONSTANT else False),
        help="enable constant input tests (default: false if TEST_CONSTANT env var not set)",
    )
    parser.add_argument(
        "--constants_to_file",
        action="store_true",
        default=(
            strtobool(TEST_CONSTANTS_TO_FILE) if TEST_CONSTANTS_TO_FILE else False
        ),
        help="whether store constants to file or not, passed to the compiler",
    )
    parser.add_argument(
        "--nofloat16",
        action="store_true",
        default=(strtobool(TEST_NOFLOAT16) if TEST_NOFLOAT16 else False),
        help="whether to disable float16 backend tests",
    )
    parser.add_argument(
        "--constants_to_file_total_threshold",
        type=float,
        default=(0.000001 if TEST_CONSTANTS_TO_FILE else 2),
        help="total threshold to trigger constants to file. Set it to a small "
        "value, 1024 bytes, since models in the model zoo are small",
    )
    parser.add_argument(
        "--compilerlib",
        action="store_true",
        default=(strtobool(TEST_COMPILERLIB) if TEST_COMPILERLIB else False),
        help="enable compiler lib tests (default: false if TEST_COMPILERLIB env var not set)",
    )
    parser.add_argument(
        "--input_verification",
        action="store_true",
        default=(
            strtobool(TEST_INPUT_VERIFICATION) if TEST_INPUT_VERIFICATION else False
        ),
        help="enable input verification tests (default: false if TEST_INPUT_VERIFICATION env var not set)",
    )
    parser.add_argument(
        "--instruction_check",
        action="store_true",
        default=(
            strtobool(TEST_INSTRUCTION_CHECK) if TEST_INSTRUCTION_CHECK else False
        ),
        help="check if specific instruction is included in generated library (default: false if TEST_INSTRUCTION_CHECK env var not set)",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=int,
        default=os.getenv("TEST_INPUT", -1),
        help="inputs whose dimensions to be changed to unknown (default: all inputs if TEST_INPUT env var not set)",
    )
    parser.add_argument(
        "-d",
        "--dim",
        type=int,
        default=os.getenv("TEST_DIM", -1),
        help="dimensions to be changed to unknown (default: all dimensions if TEST_DIM env var not set)",
    )
    parser.add_argument(
        "--converter",
        action="store_true",
        default=(strtobool(TEST_INVOKECONVERTER) if TEST_INVOKECONVERTER else False),
        help="invoke version converter (default: false if TEST_INVOKECONVERTER env var not set)",
    )
    parser.add_argument(
        "-e",
        "--emit",
        type=str,
        choices=["lib", "jni"],
        default=os.getenv("TEST_EMIT", "lib"),
        help="emit lib or jni for testing (default: lib)",
    )
    parser.add_argument(
        "-l",
        "--list",
        type=str,
        choices=["node", "model"],
        default=os.getenv("TEST_LIST", ""),
        help="list node or model tests backend runs",
    )
    parser.add_argument(
        "--mtriple",
        type=str,
        default=os.getenv("TEST_MTRIPLE", ""),
        help="triple to pass to the compiler",
    )
    parser.add_argument(
        "--mcpu",
        type=str,
        default=os.getenv("TEST_MCPU", ""),
        help="target a specific cpu, passed to the compiler (deprecated, use --march)",
    )
    parser.add_argument(
        "--march",
        type=str,
        default=os.getenv("TEST_MARCH", ""),
        help="target a specific architecture, passed to the compiler",
    )
    parser.add_argument(
        "--maccel",
        type=str,
        default=os.getenv("TEST_MACCEL", ""),
        help="target a specific accelerator, passed to the compiler",
    )
    parser.add_argument(
        "-O",
        "--Optlevel",
        type=str,
        choices=["0", "1", "2", "3"],
        default=os.getenv("TEST_OPTLEVEL", "3"),
        help="set compiler optimization level (default: 3 if TEST_OPTLEVEL env var not set)",
    )
    parser.add_argument(
        "-t",
        "--type",
        type=str,
        choices=["node", "model"],
        default=os.getenv("TEST_TYPE", ""),
        help="select subset of backend tests (node or model)",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        default=(strtobool(TEST_VERBOSE) if TEST_VERBOSE else False),
        help="verbose output (default: false if TEST_VERBOSE env var not set)",
    )
    parser.add_argument(
        "--case-check",
        action="store_true",
        default=(strtobool(TEST_CASE_CHECK) if TEST_CASE_CHECK else False),
        help="report the change of test cases (default: false if TEST_CASE_CHECK env var not set)",
    )
    parser.add_argument("unittest_args", nargs="*")
    args = parser.parse_args()
    return args


def get_runtime_vars():
    TEST_CASE_BY_USER = os.getenv("TEST_CASE_BY_USER")
    # Use ONNX_HOME if it is not the default /tmp
    if os.environ["ONNX_HOME"] != "/tmp":
        result_dir = os.environ["ONNX_HOME"]
    elif TEST_CASE_BY_USER is not None and TEST_CASE_BY_USER != "":
        result_dir = "./"
    else:
        # tempdir = tempfile.TemporaryDirectory()
        result_dir = tempdir.name + "/"

    if args.verbose:
        print("Test info:", file=sys.stderr)
        print("  temporary results are in dir:" + result_dir, file=sys.stderr)

    if args.mcpu:
        print("  targeting cpu:", args.mcpu, file=sys.stderr)
    if args.march:
        print("  targeting arch:", args.march, file=sys.stderr)
    if args.mtriple:
        print("  targeting triple:", args.mtriple, file=sys.stderr)
    if args.maccel:
        print("  targeting maccel:", args.maccel, file=sys.stderr)

    RUNTIME_DIR = ""
    TEST_DRIVER = ""
    if args.compilerlib:
        import test_config_compilerlib

        CXX = test_config_compilerlib.CXX_PATH
        LLC = test_config_compilerlib.LLC_PATH
        RUNTIME_DIR = test_config_compilerlib.TEST_DRIVER_RUNTIME_PATH
        TEST_DRIVER = test_config_compilerlib.TEST_DRIVER_PATH
    # No need to depend on test_config if we are simply listing nodes/models
    elif not args.list:
        import test_config

        CXX = test_config.CXX_PATH
        LLC = test_config.LLC_PATH
        RUNTIME_DIR = test_config.TEST_DRIVER_RUNTIME_PATH
        TEST_DRIVER = test_config.TEST_DRIVER_PATH

    # Make lib folder under build directory visible in PYTHONPATH
    doc_check_base_dir = os.path.dirname(os.path.realpath(__file__))
    sys.path.append(RUNTIME_DIR)

    # No need to depend on PyRuntime if we are simply listing nodes/models
    if not args.list:
        from PyRuntime import OMExecutionSession

    return result_dir, RUNTIME_DIR, TEST_DRIVER


### CONSTANT ###
STATIC_SHAPE = "static"
STATIC_SHAPE_STRING = "static_string"
DYNAMIC_SHAPE = "dynamic"
DYNAMIC_SHAPE_STRING = "dynamic_string"
CONSTANT_INPUT = "constant"
CONSTANT_INPUT_STRING = "constant_string"
CONSTANTS_TO_FILE = "constants_to_file"
FLOAT16 = "float16"

### immutable variables ###

# parse arguments
try:
    _ = args
except NameError:
    args = get_args_from_env()

# temp dir
try:
    _ = tempdir
except NameError:
    tempdir = tempfile.TemporaryDirectory()

# runtime vars
try:
    _, _, _ = result_dir, RUNTIME_DIR, TEST_DRIVER
except NameError:
    result_dir, RUNTIME_DIR, TEST_DRIVER = get_runtime_vars()

### mutable variables ###

# test_xxx
try:
    _, _, _, _ = (
        test_for_dynamic,
        test_for_constant,
        test_for_constants_to_file,
        test_need_converter,
    )
except NameError:
    test_for_dynamic = []
    test_for_constant = []
    test_for_constants_to_file = []
    test_need_converter = []

# real_model_tests, node_model_tests
try:
    _ = real_model_tests, node_model_tests
except NameError:
    real_model_tests = []
    node_model_tests = []

# test_to_enable_dict
try:
    _ = test_to_enable_dict, test_to_enable_symbol_dict, test_to_enable_dimparams_dict
except NameError:
    test_to_enable_dict = {}
    test_to_enable_symbol_dict = {}
    test_to_enable_dimparams_dict = {}
