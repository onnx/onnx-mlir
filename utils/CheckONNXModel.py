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
# CheckONNXModel.py --model=reducegpt2.mlir --test-compile-args="-O3 --march=x86-64" --shape-info=0:10x20
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
# --compile-args/-c gives onnx-mlir options applied to BOTH runs, so a large
# option string shared between reference and test does not need to be typed
# twice in --ref-compile-args and --test-compile-args:
#   ref  = c + r
#   test = c + t        (if -t given -- independent of r, same base)
#        = ref + a      (if -a given -- a delta on top of ref itself)
#        = c            (if neither given -- t defaults to empty, same as r
#                         does when -r is absent)
# Reference and test must resolve to different options; the script errors out
# otherwise, since there would be nothing to compare.
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

# import onnx
import time
import signal
import subprocess
import numpy as np

# import tempfile
# import json
import logging
import re

# from collections import OrderedDict

LOG_LEVEL = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
# For Parallel verbose
VERBOSITY_LEVEL = {"debug": 10, "info": 5, "warning": 1, "error": 0, "critical": 0}


def valid_onnx_input(fname):
    valid_exts = ["onnx", "mlir", "onnxtext"]
    ext = os.path.splitext(fname)[1][1:]

    if ext not in valid_exts:
        parser.error(
            "Only accept an input model with one of extensions {}".format(valid_exts)
        )
    return fname


# Command arguments.
parser = argparse.ArgumentParser(
    prog="CheckONNXModel.py",
    # Wrapped by hand: RawDescriptionHelpFormatter (needed for the epilog's
    # option arithmetic to keep its shape) leaves this text exactly as given.
    description="Compile and run an ONNX/MLIR model twice -- once with "
    "reference onnx-mlir\noptions, once with the options under test -- and "
    "verify that both runs\nproduce the same values.",
    epilog="How the two option sets are built, with -c the shared prefix:\n"
    "  ref  = c + r\n"
    "  test = c + t        (if -t given -- independent of r, same base)\n"
    "       = ref + a      (if -a given -- a delta on top of ref itself)\n"
    "       = c            (if neither given -- t defaults to empty)\n"
    "They must resolve to different options; there would otherwise be\n"
    "nothing to compare. Use either -t or -a, not both.\n"
    "See bin/onnx-mlir --help for the options themselves.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    # -h is added by hand below, so that it lands in the first group with
    # everything else this script owns rather than in a section of its own.
    add_help=False,
)
# Two groups, so that reading the help does not mean sorting out, flag by flag,
# which ones are this script's own doing and which are just handed through to
# RunONNXModel.py and mean there exactly what they mean there. The forwarded
# ones are described in one line each, since RunONNXModel.py's own help is the
# authority on them and copying it here only invites the copy going stale.
own = parser.add_argument_group(f"{parser.prog}'s own options")
forwarded = parser.add_argument_group(
    "options forwarded to RunONNXModel.py",
    "Same spelling and meaning as there; this script only passes them on.\n"
    "See RunONNXModel.py --help for each one in full.",
)

own.add_argument("-h", "--help", action="help", help="Show this help message and exit.")
own.add_argument(
    "-m",
    "--model",
    type=lambda s: valid_onnx_input(s),
    help="Path to the model to check (.onnx, .mlir, or .onnxtext).",
)
own.add_argument(
    "-c",
    "--compile-args",
    type=str,
    default="",
    help="onnx-mlir options for BOTH runs: the shared prefix -r/-t/-a build "
    "on (see epilog). Default: empty.",
)
own.add_argument(
    "-r",
    "--ref-compile-args",
    type=str,
    default="",
    help="Reference onnx-mlir options, appended after -c's. Default: empty.",
)
test_group = own.add_mutually_exclusive_group()
test_group.add_argument(
    "-t",
    "--test-compile-args",
    type=str,
    default="",
    help="Test onnx-mlir options, appended after -c's -- independent of -r's, "
    "not built on top of them. Default: empty.",
)
test_group.add_argument(
    "-a",
    "--additional-test-compile-args",
    type=str,
    default="",
    help="Test onnx-mlir options, added on top of -c and -r together.",
)

data_group = forwarded.add_mutually_exclusive_group()
data_group.add_argument(
    "--load-ref",
    metavar="PATH",
    type=str,
    help="Folder of reference inputs and outputs, in protobuf, to run on.",
)
data_group.add_argument(
    "--inputs-from-arrays", help="Numpy arrays to use as the inputs."
)
data_group.add_argument(
    "--load-ref-from-numpy",
    metavar="PATH",
    type=str,
    help="Python script defining inputs and outputs as numpy arrays.",
)
data_group.add_argument(
    "--shape-info",
    type=str,
    help="Shape of each dynamic input, e.g. 0:1x10x20,1:7x5x3. Used to "
    "generate random inputs when no reference data is loaded.",
)

own.add_argument(
    "-s",
    "--save-ref",
    metavar="PATH",
    type=str,
    help="Folder the reference run saves to and the test run verifies "
    'against. Default: "check-ref".',
)

own.add_argument(
    "--skip-ref",
    action="store_true",
    help="Skip the reference run, assuming an earlier one already filled that "
    "folder with the same compile options.",
)
own.add_argument(
    "-l",
    "--log-level",
    choices=["debug", "info", "warning", "error", "critical"],
    default="info",
    help="Log level. Default: info.",
)
forwarded.add_argument(
    "--seed",
    type=str,
    default="42",
    help="Seed for the random input generator. Default: 42.",
)

forwarded.add_argument(
    "--lower-bound",
    type=str,
    help="Lower bound per data type for random inputs, e.g. int64:-10.",
)

forwarded.add_argument(
    "--upper-bound",
    type=str,
    help="Upper bound per data type for random inputs, e.g. int64:10.",
)
forwarded.add_argument(
    "--input-value",
    type=str,
    help="Per-input fill spec, overriding --lower-bound/--upper-bound, e.g. "
    "0:min-1.0max1.0,1:val0.",
)

forwarded.add_argument(
    "--rtol", type=str, default="", help="Relative tolerance for verification."
)

forwarded.add_argument(
    "--atol", type=str, default="", help="Absolute tolerance for verification."
)

own.add_argument(
    "--cache-ref-model",
    metavar="PATH",
    type=str,
    help="Folder to load the reference compiled model from, compiling it into "
    "that folder if it is not there yet.",
)
own.add_argument(
    "--cache-test-model",
    metavar="PATH",
    type=str,
    help="Folder to load the test compiled model from, compiling it into that "
    "folder if it is not there yet.",
)


args = parser.parse_args()

VERBOSE = os.environ.get("VERBOSE", False)

if not os.environ.get("ONNX_MLIR_HOME", None):
    raise RuntimeError(
        "Environment variable ONNX_MLIR_HOME is not set, please set it to the path to "
        "the HOME directory for onnx-mlir. The HOME directory for onnx-mlir refers to "
        "the parent folder containing the bin, lib, etc sub-folders in which ONNX-MLIR "
        "executables and libraries can be found, typically `onnx-mlir/build/Debug`."
    )


# Log to stderr so that stdout can be used for check results.
def get_logger():
    logging.basicConfig(
        stream=sys.stderr,
        level=LOG_LEVEL[args.log_level],
        format="[%(asctime)s] %(levelname)s: %(message)s",
    )
    return logging.getLogger("RunONNXModelZoo.py")


logger = get_logger()


def print_cmd(cmd):
    str = ""
    for s in cmd:
        m = re.match(r"--compile-args=(.*)", s)
        if m is not None:
            str += ' --compile-args="' + m.group(1) + '"'
        else:
            str += " " + s
    return str


def execute_commands(cmds, cwd=None, tmout=None):
    logger.debug("cmd={} cwd={}".format(" ".join(cmds), cwd))
    # Merge stderr into stdout at the OS level (rather than capturing them
    # separately and concatenating after the fact) so the combined output
    # preserves the actual chronological order in which the child process
    # wrote to each stream.
    out = subprocess.Popen(
        cmds, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    try:
        stdout, _ = out.communicate(timeout=tmout)
    except subprocess.TimeoutExpired:
        # Kill the child process and finish communication.
        out.kill()
        stdout, _ = out.communicate()
        return (
            False,
            stdout.decode("utf-8") + "Timeout after {} seconds".format(tmout),
        )
    msg = stdout.decode("utf-8")
    if out.returncode == -signal.SIGSEGV:
        return (False, msg + "Segfault")
    if out.returncode != 0:
        return (False, msg + "Return code {}".format(out.returncode))
    return (True, msg)


def main():
    if not (args.model):
        print("error: no input model, use argument --model.")
        print(parser.format_usage())
        exit(1)

    # Process common options.
    path = os.path.join(os.environ["ONNX_MLIR_HOME"], "..", "..", "utils")
    cmd = path + "/RunONNXModel.py"
    model_str = "--model=" + args.model
    test_dir = "check-ref"
    if args.save_ref:
        test_dir = args.save_ref

    # Resolve -c into both ref and test: ref = c+r, test = c+t (if -t given),
    # ref+a (if -a given), or c (if neither given -- t defaults to empty, same
    # as r does when -r is absent).
    ref_compile_args = " ".join(
        p for p in (args.compile_args, args.ref_compile_args) if p
    ).strip()
    if args.additional_test_compile_args:
        test_compile_args = " ".join(
            p for p in (ref_compile_args, args.additional_test_compile_args) if p
        ).strip()
    elif args.test_compile_args:
        test_compile_args = " ".join(
            p for p in (args.compile_args, args.test_compile_args) if p
        ).strip()
    else:
        test_compile_args = args.compile_args
    if set(ref_compile_args.split()) == set(test_compile_args.split()):
        print(
            "error: reference and test resolve to the same onnx-mlir options"
            " ({!r}) -- there is nothing to compare. Set -t/-a to genuinely"
            " different options.".format(ref_compile_args)
        )
        exit(1)

    # Reference command.
    ref_cmd = [cmd]
    # Compile options for reference. Omit entirely when empty (default) so
    # that, combined with --cache-ref-model, a cache hit is loaded as-is
    # instead of tripping RunONNXModel.py's saved-options mismatch check.
    if ref_compile_args:
        ref_cmd += ["--compile-args=" + ref_compile_args]
    # Where to load the ref.
    if args.load_ref:
        ref_cmd += ["--load-ref=" + args.load_ref]
    elif args.inputs_from_arrays:
        ref_cmd += ["--inputs-from-arrays=" + args.inputs_from_arrays]
    elif args.load_ref_from_numpy:
        ref_cmd += ["--load-ref-from-numpy=" + args.load_ref_from_numpy]
    elif args.shape_info:
        ref_cmd += ["--shape-info=" + args.shape_info]
    # Where to save the reference so as to reuse them for the test command.
    ref_cmd += ["--save-ref=" + test_dir]
    # Seeds.
    ref_cmd += ["--seed=" + args.seed]
    # Handle lb/ub.
    if args.lower_bound:
        ref_cmd += ["--lower-bound=" + args.lower_bound]
    if args.upper_bound:
        ref_cmd += ["--upper-bound=" + args.upper_bound]
    if args.input_value:
        ref_cmd += ["--input-value=" + args.input_value]
    if args.cache_ref_model:
        ref_cmd += ["--cache-model=" + args.cache_ref_model]
    # Model name.
    ref_cmd += [model_str]

    # Test command.
    test_cmd = [cmd]
    # Compile options for test. Omit entirely when empty (default) so
    # that, combined with --cache-test-model, a cache hit is loaded as-is
    # instead of tripping RunONNXModel.py's saved-options mismatch check.
    if test_compile_args:
        test_cmd += ["--compile-args=" + test_compile_args]
    # Where to load the ref from.
    test_cmd += ["--load-ref=" + test_dir]
    # How to verify.
    test_cmd += ["--verify=ref"]
    test_cmd += ["--verify-every-value"]
    if args.atol:
        test_cmd += ["--atol=" + args.atol]
    if args.rtol:
        test_cmd += ["--rtol=" + args.rtol]
    if args.cache_test_model:
        test_cmd += ["--cache-model=" + args.cache_test_model]
    # Model name.
    test_cmd += [model_str]

    # Execute ref.
    print()
    if args.skip_ref:
        if not os.path.exists(test_dir):
            print('could not find "' + test_dir + '" ref dir, abort.')
            exit(1)
        print("> Reference already built, skip.")
    else:
        print("> Reference command:", print_cmd(ref_cmd))
        ok, msg = execute_commands(ref_cmd)
        if not ok:
            print("Filed while executing reference compile and run")
            print(msg)
            exit(1)
        print(
            '>   Successfully ran the reference example, saved refs in "'
            + test_dir
            + '".'
        )

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
    print(
        '>   Successfully ran the test example and verified against "' + test_dir + '".'
    )
    print()


if __name__ == "__main__":
    main()
