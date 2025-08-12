#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The IBM Research Authors.

# clear env
unset e4

arch="z17"

# binary ops
./driver-nnpa.sh add Add ${arch}
./driver-nnpa.sh sub Sub ${arch}
./driver-nnpa.sh mul Mul ${arch}
./driver-nnpa.sh div Div ${arch}
./driver-nnpa.sh max Max ${arch}
./driver-nnpa.sh min Min ${arch}

# unary ops
./driver-nnpa.sh relu Relu ${arch}
./driver-nnpa.sh gelu Gelu ${arch}
./driver-nnpa.sh tanh Tanh ${arch}
./driver-nnpa.sh sigmoid Sigmoid ${arch}
./driver-nnpa.sh softmax Softmax ${arch}
./driver-nnpa.sh exp Exp ${arch}
./driver-nnpa.sh log Log ${arch}

# power
./driver-nnpa.sh pow-2 Mul ${arch}
./driver-nnpa.sh pow-3 Mul ${arch}
./driver-nnpa.sh pow-4 Mul ${arch}
./driver-nnpa.sh pow-8 Mul ${arch}

# reduce (need 4D)
export e4=1
./driver-nnpa.sh reducemean "(ReduceMeanV13|MeanReduce2d)" ${arch}
unset e4

# matrix multiply
./driver-nnpa.sh matmul_3d MatMul ${arch}
./driver-nnpa.sh matmul_bcast23 MatMul ${arch}

# stick / unstick: remove alias, alias add.mlir to stick/unstick.mlir, compute.
rm -f stick.mlir unstick.mlir || true
ln -s add.mlir stick.mlir
ln -s add.mlir unstick.mlir
. ./driver-zdnn-csu.sh stick "Stick" ${arch}
. ./driver-zdnn-csu.sh unstick "Unstick" ${arch}

# z17 only, comment if not on z17
. ./driver-quant.sh
echo "Done"