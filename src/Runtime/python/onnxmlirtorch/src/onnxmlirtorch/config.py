# SPDX-License-Identifier: Apache-2.0

##################### config.py *******#########################################
#
# Copyright 2025 The IBM Research Authors.
#
################################################################################
#
# This module contains various configuration flags and settings that control the backend.
#
################################################################################

# Control the number of same consecutive hashings that trigger stoping to hash the graph module.
# In other words, if the compiler detects that after this numer of hashing, the graph module stays
# the same, the compiler does not hash the module in the next run.
same_hash_size = 3

# Control how many values in a constant tensor (parameters) are used for hashing
# the graph module.
sample_parameter_values_limit = 3

# Control the maximum number of compiler sessions to be cached at runtime. 
session_cache_limit = 3
