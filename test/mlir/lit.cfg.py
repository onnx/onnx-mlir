
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
config.name = 'Open Neural Network Frontend'

config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)

# test_source_root: The root path where tests are located.
config.test_source_root = config.onnf_mlir_test_src_dir

# test_exec_root: The root path where tests should be run.
config.test_exec_root = config.onnf_mlir_test_build_dir

llvm_config.use_default_substitutions()

# Tweak the PATH to include the tools dir.
llvm_config.with_environment('PATH', config.llvm_tools_dir, append_path=True)

tool_dirs = [
    config.onnf_mlir_tools_dir, config.mlir_tools_dir, config.llvm_tools_dir
]
tool_names = [
    'onnf-opt', 'mlir-opt', 'mlir-translate'
]
tools = [ToolSubst(s, unresolved='ignore') for s in tool_names]
llvm_config.add_tool_substitutions(tools, tool_dirs)