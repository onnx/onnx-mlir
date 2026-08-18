#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

# Compare two lowering variants of the same computation and verify that they
# produce identical (within tolerance) numerical results.
#
# The "baseline" can be specified in one of two ways:
#   - A separate .mlir file containing a function with the same name as the
#     one being tested (file mode: -b, or the default
#     "<model without extension>-baseline<ext>" next to --model).
#   - The same --model file, compiled with different onnx-mlir options
#     (flag mode: -r/-t/-a), mirroring CheckONNXModel.py's options. Note
#     these are onnx-mlir options, NOT onnx-mlir-opt options -- a lit test's
#     own "// RUN: onnx-mlir-opt ..." line is a different flag namespace and
#     cannot be reused directly here.
#
# -c/--compile-args applies in both modes, to every compile in the run:
#   - File mode: the same options are used for both the baseline and test
#     file (there is no ref/test split to layer onto).
#   - Flag mode: -c is the shared prefix both branches build on, so you don't
#     have to repeat a large common option string in both -r and -t:
#       ref  = c + r
#       test = c + t            (if -t given -- independent of r, same base)
#            = ref + a          (if -a given -- a delta on top of ref itself)
#            = c                (if neither given -- t defaults to empty, same
#                                 as r does when -r is absent)
#
# File mode is for features (like krnl.collapse) where the difference between
# variants cannot be expressed as a single onnx-mlir flag, so the
# CheckONNXModel.py "same file, two option sets" model does not apply -- the
# two variants are genuinely different KRNL/MLIR source. For features that
# *can* be toggled with a flag, flag mode here is equivalent to
# CheckONNXModel.py, just scoped to a single isolated function.
#
# Relies on:
#   - fixLitTest.py -m <func-name> <file>: isolate a single function into a
#     standalone, runnable module (wrapped in "module {...}" with an
#     "onnx.EntryPoint"). RunONNXModel.py specifically requires this
#     "onnx.EntryPoint" marker in the source file before it will compile it.
#   - RunONNXModel.py: compile, run, save/load reference inputs+outputs, and
#     verify against them.

import argparse
import os
import shlex
import shutil
import subprocess
import sys
import tempfile

UTILS_DIR = os.path.dirname(os.path.abspath(__file__))
FIX_LIT_TEST = os.path.join(UTILS_DIR, "fixLitTest.py")
RUN_ONNX_MODEL = os.path.join(UTILS_DIR, "RunONNXModel.py")
# Bare names for the "reproduce this manually" recipe -- the user is expected
# to know these tools live in utils/, so the full path is just noise there.
FIX_LIT_TEST_NAME = os.path.basename(FIX_LIT_TEST)
RUN_ONNX_MODEL_NAME = os.path.basename(RUN_ONNX_MODEL)

# Canonical, fixed names (like fixLitTest.py's own "flt_*.mlir" files), written
# to the current directory and overwritten on every run -- so there is always
# exactly one copy to inspect after the fact, not one per invocation.
CLV_TEST_FILE = "clv_test.mlir"
CLV_BASELINE_FILE = "clv_baseline.mlir"
CLV_REF_DIR = "clv_ref"


class Logger:
    """
    All of this run's subprocess commands and their output go to a private,
    uniquely-named temp file (not a fixed shared name -- multiple users on
    the same machine could otherwise collide). Shown live only with -v;
    otherwise dumped in full only if the run fails, then always removed.
    """

    def __init__(self, verbose):
        self.verbose = verbose
        fd, self.path = tempfile.mkstemp(prefix="clv_run_", suffix=".log")
        self.fh = os.fdopen(fd, "w")

    def log(self, text="", end="\n"):
        self.fh.write(text + end)
        self.fh.flush()
        if self.verbose:
            print(text, end=end)

    def dump_on_failure(self):
        print("--- full log ---")
        self.fh.flush()
        with open(self.path) as f:
            sys.stdout.write(f.read())
        print("--- end log ---")

    def close(self):
        self.fh.close()
        try:
            os.remove(self.path)
        except OSError:
            pass


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Compare two lowering variants of the same MLIR function: "
            "compile and run both, feed them identical inputs, and verify "
            "their outputs match within tolerance."
        ),
        epilog=(
            "Two mutually exclusive ways to specify the baseline:\n"
            "  file mode (-b):     a different .mlir file, same function name.\n"
            "  flag mode (-r/-t/-a): the SAME --model, compiled twice with\n"
            "                       different onnx-mlir options.\n"
            "-c/--compile-args applies in both modes, to every compile in the\n"
            "run. In flag mode it is the shared prefix -r/-t/-a build on:\n"
            "  ref  = c + r\n"
            "  test = c + t        (if -t given)\n"
            "       = ref + a      (if -a given)\n"
            "       = c            (if neither given -- t defaults to empty)\n"
            "Compatibility: -b excludes -r/-t/-a. -c combines with all of\n"
            "-b/-r/-t/-a. -t and -a are mutually exclusive with each other."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-m", "--model", required=True, help="Path to the test .mlir file."
    )
    parser.add_argument(
        "-f",
        "--func",
        required=True,
        help="Name of the function to isolate and compare.",
    )

    parser.add_argument(
        "-b",
        "--baseline-model",
        default=None,
        help=(
            "File mode: path to a baseline .mlir file containing a function "
            "with the same name as --func. Default: "
            '"<model without extension>-baseline<ext>", next to --model. '
            "Mutually exclusive with -r/-t/-a. See epilog for the full "
            "compatibility rules."
        ),
    )
    parser.add_argument(
        "-c",
        "--compile-args",
        default=None,
        help=(
            "onnx-mlir options applied to EVERY compile in this run, in "
            "either mode. In file mode: the same options for both the "
            "baseline and test file. In flag mode: the shared prefix -r/-t/-a "
            "build on (see epilog). Default: empty."
        ),
    )
    parser.add_argument(
        "-r",
        "--ref-compile-args",
        default=None,
        help=(
            "Flag mode: compile the SAME --model twice instead of using a "
            "second file. These are the reference/baseline onnx-mlir "
            "options, appended after -c's (NOT onnx-mlir-opt options). "
            "Default: empty."
        ),
    )
    parser.add_argument(
        "-t",
        "--test-compile-args",
        default=None,
        help=(
            "Flag mode: test onnx-mlir options, appended after -c's -- "
            "independent of -r's options, not built on top of them. Use "
            "either -t or -a, not both."
        ),
    )
    parser.add_argument(
        "-a",
        "--additional-test-compile-args",
        default=None,
        help=(
            "Flag mode: test onnx-mlir options, added on top of the full "
            "reference options (-c and -r together). Use either -t or -a, "
            "not both."
        ),
    )

    parser.add_argument(
        "-d",
        "--diff",
        action="store_true",
        help=(
            "Show a side-by-side diff of the two isolated function bodies "
            "(file mode), or the two compile-arg strings being compared "
            "(flag mode)."
        ),
    )
    parser.add_argument(
        "--rtol", default="0.05", help="Relative tolerance (forwarded to RunONNXModel.py)."
    )
    parser.add_argument(
        "--atol", default="0.01", help="Absolute tolerance (forwarded to RunONNXModel.py)."
    )
    parser.add_argument(
        "--seed",
        default=None,
        help="Seed for random input generation, for the baseline run (forwarded).",
    )
    parser.add_argument(
        "--shape-info", default=None, help="Dynamic input shapes (forwarded)."
    )
    parser.add_argument(
        "--lower-bound", default=None, help="Lower bound for random inputs (forwarded)."
    )
    parser.add_argument(
        "--upper-bound", default=None, help="Upper bound for random inputs (forwarded)."
    )
    parser.add_argument(
        "--input-value", default=None, help="Per-input data fill spec (forwarded)."
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help=(
            "Show every command and its full output live. Without this, "
            "that output is captured to a private temp file and only printed "
            "(then removed) if the comparison fails."
        ),
    )
    args = parser.parse_args()

    flag_mode = (
        args.ref_compile_args is not None
        or args.test_compile_args is not None
        or args.additional_test_compile_args is not None
    )
    if flag_mode and args.baseline_model is not None:
        parser.error("-b/--baseline-model cannot be combined with -r/-t/-a.")
    if args.test_compile_args is not None and args.additional_test_compile_args is not None:
        parser.error("use either -t or -a, not both.")
    return args, flag_mode


def default_baseline_path(model_path):
    base, ext = os.path.splitext(model_path)
    return base + "-baseline" + ext


def format_cmd(cmd, redirect_to=None):
    line = "+ " + " ".join(shlex.quote(c) for c in cmd)
    if redirect_to:
        line += " > " + shlex.quote(redirect_to)
    return line


def run_cmd(cmd, **kwargs):
    return subprocess.run(cmd, capture_output=True, text=True, **kwargs)


def isolate_function(logger, src_file, func_name, dest_file):
    if not os.path.exists(src_file):
        sys.exit(f'ERROR: file "{src_file}" does not exist.')
    cmd = [sys.executable, FIX_LIT_TEST, "-m", func_name, src_file]
    logger.log(format_cmd(cmd, redirect_to=dest_file))
    result = run_cmd(cmd)
    if result.returncode != 0 or "module {" not in result.stdout:
        logger.log("--- fixLitTest.py stdout ---")
        logger.log(result.stdout)
        logger.log("--- fixLitTest.py stderr ---")
        logger.log(result.stderr)
        logger.dump_on_failure()
        sys.exit(
            f'ERROR: could not isolate function "{func_name}" from "{src_file}" '
            f"via fixLitTest.py -m. See log above."
        )
    logger.log(result.stderr, end="")
    with open(dest_file, "w") as f:
        f.write(result.stdout)


def show_diff(file_a, label_a, file_b, label_b):
    print(f"--- diff: {label_a}  |  {label_b} ---")
    icdiff = shutil.which("icdiff")
    if icdiff:
        cmd = [icdiff, "--label", label_a, "--label", label_b, file_a, file_b]
    else:
        cmd = ["diff", "-y", "--label", label_a, "--label", label_b, file_a, file_b]
    # A nonzero return code here just means "the files differ", which is
    # expected and not itself a failure of this tool.
    subprocess.run(cmd, check=False)
    print()


def run_onnx_model(logger, extra_args):
    cmd = [sys.executable, RUN_ONNX_MODEL] + extra_args
    logger.log(format_cmd(cmd))
    result = run_cmd(cmd)
    logger.log(result.stdout, end="")
    if result.stderr:
        logger.log(result.stderr, end="")
    return result.returncode == 0


def forwarded_input_args(args):
    extra = []
    if args.seed is not None:
        extra += ["--seed", args.seed]
    if args.shape_info is not None:
        extra += ["--shape-info", args.shape_info]
    if args.lower_bound is not None:
        extra += ["--lower-bound", args.lower_bound]
    if args.upper_bound is not None:
        extra += ["--upper-bound", args.upper_bound]
    if args.input_value is not None:
        extra += ["--input-value", args.input_value]
    return extra


def clean_kept_files():
    # Remove any files kept from a previous run before doing anything else.
    # Otherwise, a run that fails partway through could leave a misleading
    # mix of stale files from an earlier, unrelated run alongside whatever
    # this run managed to produce -- e.g. a fresh clv_test.mlir sitting next
    # to a stale clv_ref/ from a different comparison entirely.
    for path in (CLV_TEST_FILE, CLV_BASELINE_FILE):
        if os.path.exists(path):
            os.remove(path)
    shutil.rmtree(CLV_REF_DIR, ignore_errors=True)


def join_args(*parts):
    return " ".join(p for p in parts if p).strip()


def main():
    args, flag_mode = parse_args()
    clean_kept_files()
    common_compile_args = args.compile_args or ""

    if flag_mode:
        baseline_model = args.model
        ref_compile_args = join_args(common_compile_args, args.ref_compile_args)
        if args.test_compile_args is not None:
            test_compile_args = join_args(common_compile_args, args.test_compile_args)
        elif args.additional_test_compile_args is not None:
            test_compile_args = join_args(
                ref_compile_args, args.additional_test_compile_args
            )
        else:
            # -t defaults to empty, same as -r does when -r is absent, so
            # test = c (not ref = c + r) when neither -t nor -a is given.
            test_compile_args = common_compile_args
        if set(ref_compile_args.split()) == set(test_compile_args.split()):
            sys.exit(
                "ERROR: in flag mode, reference and test resolve to the same "
                f"onnx-mlir options ({ref_compile_args!r}) -- there is nothing "
                "to compare. Set -t/-a to genuinely different options, or use "
                "file mode (-b) instead if the difference is in the MLIR "
                "source rather than the compiler flags."
            )
        print(
            f"Comparing function {args.func!r} from file {args.model!r}, "
            f"compiled with onnx-mlir options {ref_compile_args!r} (baseline) "
            f"vs {test_compile_args!r} (test)."
        )
    else:
        baseline_model = args.baseline_model or default_baseline_path(args.model)
        if not os.path.exists(baseline_model):
            sys.exit(
                f'ERROR: no baseline available.\n'
                f'  Tried default baseline file "{baseline_model}" (not found).\n'
                f"  Either create that file (same function name as --func), "
                f"or pass -b/--baseline-model, or pass -r/-t/-a "
                f"to compare --model against itself with different options."
            )
        ref_compile_args = common_compile_args
        test_compile_args = common_compile_args
        summary = (
            f"Comparing function {args.func!r} from file {baseline_model!r} "
            f"(baseline) with the same function from file {args.model!r} (test)"
        )
        if common_compile_args:
            summary += f", both compiled with onnx-mlir options {common_compile_args!r}"
        print(summary + ".")

    logger = Logger(args.verbose)
    try:
        isolate_function(logger, args.model, args.func, CLV_TEST_FILE)

        if flag_mode:
            baseline_isolated = CLV_TEST_FILE
        else:
            baseline_isolated = CLV_BASELINE_FILE
            isolate_function(logger, baseline_model, args.func, CLV_BASELINE_FILE)

        if args.diff:
            if flag_mode:
                print("--- diff: reference compile args  |  test compile args ---")
                print(f"  reference: {ref_compile_args!r}")
                print(f"  test:      {test_compile_args!r}")
                print()
            else:
                show_diff(
                    baseline_isolated,
                    f"baseline ({baseline_model})",
                    CLV_TEST_FILE,
                    f"test ({args.model})",
                )

        logger.log("Compiling and running the baseline/reference variant ...")
        baseline_cmd = [
            "-m",
            baseline_isolated,
            "--save-ref",
            CLV_REF_DIR,
        ]
        if ref_compile_args:
            # Use "--flag=value" (not two separate argv items) so a value that
            # itself starts with "-" (e.g. "-O0") isn't misparsed by
            # RunONNXModel.py's own argparse as a new option.
            baseline_cmd += [f"--compile-args={ref_compile_args}"]
        baseline_cmd += forwarded_input_args(args)
        if not run_onnx_model(logger, baseline_cmd):
            logger.dump_on_failure()
            sys.exit(
                "ERROR: the baseline/reference variant failed to compile or "
                "run -- this is a setup problem, not a mismatch. See log above."
            )

        logger.log(
            "Compiling and running the test variant, verifying against the "
            "saved reference ..."
        )
        test_cmd = [
            "-m",
            CLV_TEST_FILE,
            "--load-ref",
            CLV_REF_DIR,
            "--verify",
            "ref",
            "--verify-every-value",
            "--rtol",
            args.rtol,
            "--atol",
            args.atol,
        ]
        if test_compile_args:
            test_cmd += [f"--compile-args={test_compile_args}"]
        passed = run_onnx_model(logger, test_cmd)

        print()
        print("To reproduce this test manually:")
        print(format_cmd(
            [FIX_LIT_TEST_NAME, "-m", args.func, args.model],
            redirect_to=CLV_TEST_FILE,
        ))
        if not flag_mode:
            print(format_cmd(
                [FIX_LIT_TEST_NAME, "-m", args.func, baseline_model],
                redirect_to=CLV_BASELINE_FILE,
            ))
        print(format_cmd([RUN_ONNX_MODEL_NAME] + baseline_cmd))
        print(format_cmd([RUN_ONNX_MODEL_NAME] + test_cmd))

        kept_baseline = CLV_BASELINE_FILE if not flag_mode else "<none, same file>"
        print(f"Kept files: {CLV_TEST_FILE}, {kept_baseline}, {CLV_REF_DIR}/")
        print()

        if not passed:
            logger.dump_on_failure()
            if not args.diff:
                print(
                    "Hint: rerun with -d/--diff to see exactly what differs "
                    "between the two variants -- that's the most likely "
                    "place to look for the cause."
                )

        if passed:
            print(f"PASS: {args.func} matches its baseline within "
                  f"rtol={args.rtol}, atol={args.atol}.")
        else:
            print(f"FAIL: {args.func} does NOT match its baseline within "
                  f"rtol={args.rtol}, atol={args.atol}.")
        sys.exit(0 if passed else 1)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
