#!/bin/bash

# SPDX-License-Identifier: Apache-2.0

####################### onnx-mlir-truncate.sh ##################################
#
# Copyright 2025 The IBM Research Authors.
#
################################################################################
#
# This script ensure that there are no super long lines.
#
################################################################################

# Usage
#
# All but last arguments are mlir arguments; last one is log file name

# Script forces the lines to be never longer than maxLineLength (llvm pass
# may print some very long lines) sed ensure that there are pairs of `"` because
# when interpreting mlir files (.mlir), VSCode look at strings syntax.
# Unterminated strings then cause issues with the VSCode mlir code prettifier.
maxLineLength=800
onnx-mlir ${@:1:$#-1} --mlir-elide-elementsattrs-if-larger=20 --mlir-elide-resource-strings-if-larger=20 2>&1 | cut -c -${maxLineLength} | sed -E "s/^(([^\"]*\"[^\"]*\")*[^\"]*\"[^\"]*)$/\1  \"/" | tee ${@: -1}
  
