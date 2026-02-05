# SPDX-License-Identifier: Apache-2.0

##################### config.py *******#########################################
#
# Copyright 2025 The IBM Research Authors.
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

# Control the maximum number of compiler sessions to be cached at runtime.
session_cache_limit = 3
