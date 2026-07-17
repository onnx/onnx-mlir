# SPDX-License-Identifier: Apache-2.0

##################### config.py *******#########################################
#
# Copyright 2025-2026 The IBM Research Authors.
#
################################################################################
#
# This module contains various configuration flags and settings that control the backend.
# These flags and settings can be set in users' script by using package.config., e.g.:
# ```python
# import torch_onnxmlir
# torch_onnxmlir.config.same_hash_size = 1
# ```
#
################################################################################

# If the compiler detects that after this number of hashings, the graph module stays
# the same, the compiler does not hash the module in the next run in order to reduce
# the inference overhead.
same_hash_size = 3

# Control how many values in a constant tensor (parameters) are used for hashing
# the graph module. This affects the hashing time since it takes more time to
# read more values and hash more data.
sample_parameter_values_limit = 3

# Custom cache_dir
cache_dir = None

# Control the maximum number of compiler sessions to be cached at runtime.
session_cache_limit = 3

# Whether to keep ONNX files or not.
keep_onnx_files = False

# Whether to generate test data sets (inputs/outputs) for the .so file in the cache.
# This is useful for debugging where we can use them to run the .so file in the cache by using [RunONNXModel.py](https://github.com/onnx/onnx-mlir/blob/main/utils/RunONNXModel.py).
# When enabled, a folder `test_data_set` is created in the cache folder containing multiple files for inputs (input_0.pb, input_1.pb, ...) and outputs (output_0.pb, output_1.pb, ...).
# The generation is done once after compiling a model, which makes the first run slower and does not affect next runs.
generate_test_data_set = False

# Whether to regenerate test data sets or not. When enabled, the folder test_data_set is rewritten with new data.
# This regeneration is called on every run, and the data of the last run is used. Thus, only use it for debugging.
regenerate_test_data_set = False
