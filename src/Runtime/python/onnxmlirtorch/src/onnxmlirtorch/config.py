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

# Control the maximum number of hashing the graph module in order to detect if
# the module is changed or not. This does not include the hashing at the first time
# touching the graph module.
gm_hash_limit = 2

# Control how many values in a constant tensor (parameters) are used for hashing
# the graph module.
sample_parameter_values_limit = 3

# Control the maximum number of compiler sessions to be cached at runtime. 
session_cache_limit = 3
