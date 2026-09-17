import os
import sys
import re
import platform
import subprocess

import lit.util
import lit.formats
from lit.llvm import llvm_config
from lit.llvm.subst import FindTool
from lit.llvm.subst import ToolSubst

# name: The name of this test suite.
config.name = "Open Neural Network Frontend"

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# suffixes: A list of file extensions to treat as test files.
config.suffixes = [".mlir", ".json", ".onnxtext"]

# test_source_root: The root path where tests are located.
config.test_source_root = os.path.dirname(__file__)

# Tag marking a file as a GroundLitTest.py baseline rather than a test of its
# own: "mytest.mlir" is grounded numerically against "mytest.baseline.mlir".
# Keep in step with BASELINE_TAG in utils/GroundLitTest.py; the workflow is in
# docs/GroundedLitTests.md.
BASELINE_TAG = ".baseline"

# A baseline is the reference variant a grounded test is compared against, not a
# test: it carries no RUN line, so collecting it would only report UNRESOLVED and
# redden the suite. lit cannot exclude by pattern -- config.excludes is a set of
# exact names -- so the tag is expanded into that set here, by walking the suite
# once at config time. Matched on the stem rather than as a literal
# ".baseline.mlir", so it holds for every suffix above: exactly the set of names
# GroundLitTest.py's default_baseline_path() can derive, since that keeps
# whatever extension the file it grounds had. Excluding by bare name, as lit
# does, cannot hit a real test here, because every name collected below is one no
# test may have. A set comprehension, so its loop variables do not leak into the
# config namespace.
config.excludes = set(config.excludes) | {
    filename
    for _, _, filenames in os.walk(config.test_source_root)
    for filename in filenames
    if os.path.splitext(filename)[0].endswith(BASELINE_TAG)
}

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.onnx_mlir_obj_root, "test", "mlir")

llvm_config.use_default_substitutions()

# Tweak the PATH to include the tools dir.
llvm_config.with_environment("PATH", config.llvm_tools_dir, append_path=True)

tool_dirs = [config.onnx_mlir_tools_dir, config.mlir_tools_dir, config.llvm_tools_dir]

tools = [
    "onnx-mlir",
    "onnx-mlir-opt",
    "mlir-opt",
    "mlir-translate",
    "binary-decoder",
]

llvm_config.add_tool_substitutions(tools, tool_dirs)

# %onnx-mlir-home expands to the parent of the tools dir (i.e. the Debug/ or
# Release/ prefix under build/).  build-run-onnx-lib.sh expects ONNX_MLIR_HOME
# to point there so it can locate bin/, lib/, and the source tree.
config.substitutions.append(
    ("%onnx-mlir-home", os.path.dirname(os.path.normpath(config.onnx_mlir_tools_dir)))
)

# This is based on the same code in llvm and it is meant to determine what
# the supported targets for llvm & friends are - this allow us to filter test
# execution based on the available targets
for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower())
