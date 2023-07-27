#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

##################### CheckONNXModel.py ########################################onn
#
# Copyright 2023 The IBM Research Authors.
#
################################################################################
#
# This script is to run and debug an onnx model. Model is run twice, once with
# safe compile options, and the second time with the options to tests. The
# script verifies that both output are the same.

# This script can be used as follows:
#
# CheckONNXModel.py --model=reducegpt2.mlir --test-compile-args="-O3 -march=x86-64" --shape-info=0:10x20
#
# It will compile and run the model reducegpt2.mlir twice.
# * Once with the default (-O0) option, which can be overridden with
#   --ref-compile-args. This will build reference results stored by default
#   in a subdir named "check-ref", which can be overridden with --save-ref="name"
#   option. Building the reference values can be skipped with "--skip-ref"
#   option if it was previously built with the same compile options.
# * Once with the default (-O3) option, which can be overridden with
#   --test-compile-args. The values of this run are compared with the reference
#   values.
#
# Script will fail if the values are not identical. Currently only the
# "--verify-every-value" option is supported.
#
# This script relies on RunONNXModel.py to compile, run, and test. The actual
# commands used by this script are printed on stdout, so that users may call
# them manually if they wish to employ more RunONNXModel.py options.
# 
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
import logging
import re

from onnx import numpy_helper
from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
from collections import OrderedDict

LOG_LEVEL       = { 'debug':    logging.DEBUG,
                    'info':     logging.INFO,
                    'warning':  logging.WARNING,
                    'error':    logging.ERROR,
                    'critical': logging.CRITICAL }
# For Parallel verbose
VERBOSITY_LEVEL = { 'debug':    10,
                    'info':     5,
                    'warning':  1,
                    'error':    0,
                    'critical': 0 }

def valid_onnx_input(fname):
    valid_exts = ['onnx', 'mlir']
    ext = os.path.splitext(fname)[1][1:]

    if ext not in valid_exts:
        parser.error(
            "Only accept an input model with one of extensions {}".format(
                valid_exts))
    return fname

# Command arguments.
parser = argparse.ArgumentParser(prog="CheckONNXModel.py",
    description="Compile and run an ONNX/MLIR model twice. " 
                "Once with reference compiler options (-r) to set the reference values. "
                "And once with test compiler options (-t) to verify the validity of these options.")
parser.add_argument('-m', '--model',
                    type=lambda s: valid_onnx_input(s),
                    help="Path to an ONNX model (.onnx or .mlir)")
parser.add_argument('-r', '--ref-compile-args',
                    type=str,
                    default="-O0",
                    help="Reference arguments passed directly to onnx-mlir command."
                    " See bin/onnx-mlir --help")
parser.add_argument('-t', '--test-compile-args',
                    type=str,
                    default="-O3",
                    help="Reference arguments passed directly to onnx-mlir command."
                    " See bin/onnx-mlir --help")
parser.add_argument('-s', '--save-ref',
                    metavar='PATH',
                    type=str,
                    help="Path to a folder to save the inputs and outputs"
                    " in protobuf")
parser.add_argument('--shape-info',
                    type=str,
                    help="Shape for each dynamic input of the model, e.g. 0:1x10x20,1:7x5x3. "
                    "Used to generate random inputs for the model if --load-ref is not set")
parser.add_argument('--skip-ref',
                    action='store_true',
                    help="Skip building the ref compilation, assuming it was built before.")
parser.add_argument('-l',
                     '--log-level',
                     choices=[ 'debug', 'info', 'warning', 'error', 'critical' ],
                     default='info',
                     help="log level, default info")

args = parser.parse_args()

VERBOSE = os.environ.get('VERBOSE', False)

if (not os.environ.get('ONNX_MLIR_HOME', None)):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
        "the parent folder containing the bin, lib, etc sub-folders in which ONNX-MLIR "
        "executables and libraries can be found, typically `onnx-mlir/build/Debug`"
    )

# log to stderr so that stdout can be used for check results
def get_logger():
    logging.basicConfig(stream=sys.stderr,
                        level=LOG_LEVEL[args.log_level],
                        format='[%(asctime)s] %(levelname)s: %(message)s')
    return logging.getLogger('RunONNXModelZoo.py')

logger = get_logger()

def print_cmd(cmd):
    str = ""
    for s in cmd:
        m = re.match(r'--compile-args=(.*)', s)
        if m is not None:
            str += " --compile-args=\"" + m.group(1) + "\""
        else:
            str += " " + s
    return str

def execute_commands(cmds, cwd=None, tmout=None):
    logger.debug('cmd={} cwd={}'.format(' '.join(cmds), cwd))
    out = subprocess.Popen(cmds, cwd=cwd,
                           stdout=subprocess.PIPE,
                           stderr=subprocess.PIPE)
    try:
        stdout, stderr = out.communicate(timeout=tmout)
    except subprocess.TimeoutExpired:
        # Kill the child process and finish communication
        out.kill()
        stdout, stderr = out.communicate()
        return (False, (stderr.decode("utf-8") + stdout.decode("utf-8") +
                        "Timeout after {} seconds".format(tmout)))
    msg = stderr.decode("utf-8") + stdout.decode("utf-8")
    if out.returncode == -signal.SIGSEGV:
        return (False, msg + "Segfault")
    if out.returncode != 0:
        return (False, msg + "Return code {}".format(out.returncode))
    return (True, stdout.decode("utf-8"))

def main():
    if not (args.model):
        print("error: no input model, use argument --model.")
        print(parser.format_usage())
        exit(1)

    #Process common options.
    path = os.path.join(os.environ['ONNX_MLIR_HOME'], "..", "..", "utils")
    cmd = path + "/RunONNXModel.py"
    model_str = "--model=" + args.model
    test_dir = "check-ref"
    if  args.save_ref:
        test_dir = args.save_ref
    # Reference command

    ref_cmd = [cmd]
    # Compile options for reference
    ref_cmd += ["--compile-args=" + args.ref_compile_args]
    # Where to save the reference
    ref_cmd += ["--save-ref=" + test_dir]
    # Possible shape info
    if args.shape_info:
        ref_cmd += ["--shape-info=" + args.shape_info]
    # Model name.
    ref_cmd += [model_str]

    # Test command
    test_cmd = [cmd]
    # Compile options for test
    test_cmd += ["--compile-args=" + args.test_compile_args]
    # Where to load the ref from
    test_cmd += ["--load-ref=" + test_dir]
    # How to verify
    test_cmd += ["--verify=ref"]
    test_cmd += ["--verify-every-value"]
    # Model name.
    test_cmd += [model_str]

    # Execute ref
    print()
    if args.skip_ref:
        if not os.path.exists(test_dir):
            print("could not find \""+test_dir+"\" ref dir, abort.")
            exit(1)
        print("> Reference already built, skip.")
    else:
        print("> Reference command:", print_cmd(ref_cmd))
        ok, msg = execute_commands(ref_cmd)
        if not ok:
            print("Filed while executing reference compile and run")
            print(msg)
            exit(1)
        print(">   Successfully ran the reference example, saved refs in \"" + 
              test_dir + "\"." )

    # Execute ref
    print()
    print("> Test command:", print_cmd(test_cmd))
    ok, msg = execute_commands(test_cmd)
    if not ok:
        print(">  Failed while executing test compile and run")
        print(msg)
        print(">   Failed test command:", print_cmd(test_cmd))
        print()
        exit(1)
    print(">   Successfully ran the test example and verified against \"" +
            test_dir + "\"." )
    print()

if __name__ == '__main__':
    main()
