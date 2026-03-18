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

# test_exec_root: The root path where tests should be run.
config.test_exec_root = os.path.join(config.onnx_mlir_obj_root, "test", "mlir")

llvm_config.use_default_substitutions()

config.excludes = ["onnx_to_mhlo"]

# Xilinx fork: Don't care about krnl dialect. Simplifies LLVM bumps
config.excludes += ["onnx_to_krnl", "krnl_to_affine", "krnl_to_llvm"]

if not config.enable_krnl:
    config.excludes += [
        # Whole directories
        "krnl",
        "parallel",
        "instrument",
        # Individual files in onnx/
        "onnx_location.mlir",
        "onnx_lowering_call_canonicalize_O3.mlir",
        "onnx_pre_krnl_verify_error.mlir",
        # Individual files in driver/
        "buffer_loop_hoisting.mlir",
        "compile_phases.mlir",
        "invalid_output_path.mlir",
        "llvm.ident.mlir",
        "product.version.mlir",
    ]

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

# This is based on the same code in llvm and it is meant to determine what
# the supported targets for llvm & friends are - this allow us to filter test
# execution based on the available targets
for arch in config.targets_to_build.split():
    config.available_features.add(arch.lower())

# Remember whether krnl dialect is available
if config.enable_krnl:
    config.available_features.add("krnl")

# Remember whether pybind11 is available
if config.enable_pybind:
    config.available_features.add("pybind")
