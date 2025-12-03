#!/bin/bash

# SPDX-License-Identifier: Apache-2.0

####################### onnx-mlir-print.sh #####################################
#
# Copyright 2025 The IBM Research Authors.
#
################################################################################
#
# This script is to run onnx-mlir and print the program after every pass,
# while ensuring that there are no lines that are too long.
#
################################################################################

# Usage:
#
# All but last arguments are mlir arguments; last one is log file name.

onnx-mlir-truncate.sh ${@:1:$#-1} -mlir-print-ir-after-all ${@: -1}
  
