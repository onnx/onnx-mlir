#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The IBM Research Authors.

#   param1: file name without the extension (.mlir)
#   param2: op_name to be grepped (can be a regexp, e.g. "(cpuOp|nnpaOp)")
#   param3: architecture (xxx in --march=xxx)
#
# env var: 
#  e4:       add dim to the shape; e.g. e4="1", e4="1x1" (default none).
#  skipCPU:  if non-empty, skip CPU computations (default empty).
#  skipNNPA: if non-empty, skip NNPA computations (default empty).

# name of file (name) and operation (op_name).
name=$1
op_name=$2
arch="-march=$3"

# Extra dimensions: add "x" if additional dim e4 is nonempty.
if [ ! -z "${e4}" ]
then
    e4+="x"
fi

date=`date +%Y-%m-%d-%H-%M-%S`
mkdir log || true
mkdir res || true
log_file="log/test_${name}_${date}.log"
res_file="res/test_${name}_${date}.csv"
warmup=2
iter=100

echo "Experiment with ${name} file and ${op_name} op on arch $3"
echo ""

function run_experiment { # (option, e2, e1)
    compile_option=$1
    e2=$2
    e1=$3
    for e3 in 1 4 8 16 32 48 64 128 256 1024 4096
    do
        opt=$compile_option
        opt+=" --shapeInformation=0:${e4}${e3}x${e2}x${e1}"
        echo "Tile, $e3, $e2, $e1, option, $opt"
        echo "Tile, $e3, $e2, $e1, option, $opt" >> $log_file
        # Add lb & ub for log.
        rm run.log || true
        ONNX_MLIR_INSTRUMENT_FILE=run.log RunONNXModel.py -m $name.mlir -c "$opt" -w $warmup -n $iter --lower-bound=float32:0.1 --upper-bound=float32:0.8 >> $log_file
        cat run.log >> $log_file
        make-report.py -r run.log >> $log_file
    done
}

function run_matmul3d_experiment {
    compile_option=$1
    e2=$2
    e1=$3
    bcast23=$4
    for e3 in 1 2 3 4 6 8 16 32 48 64
    do
        opt=$compile_option
        opp=$compile_option # No commas here.
        # MatMul (B=e3 x N=e2 x M=e1) * (B=e3 x M=e1 x K=e1) = (B=e3 x N=e2 x K=e1): 
        if [ "$bcast23" -gt "0" ]
        then 
            # Has bcast 23: 3D x 2D
            opt+=" --shapeInformation=0:${e3}x${e2}x${e1},1:${e1}x${e1}" # has comma.
            opp+=" --shapeInformation=0:${e3}x${e2}x${e1} 1:${e1}x${e1}" # has no comma.
        else
            opt+=" --shapeInformation=0:${e3}x${e2}x${e1},1:${e3}x${e1}x${e1}" # has comma.
            opp+=" --shapeInformation=0:${e3}x${e2}x${e1} 1:${e3}x${e1}x${e1}" # has no comma.
        fi
        echo "Tile, $e3, $e2, $e1, option, $opt"
        echo "Tile, $e3, $e2, $e1, option, $opp" >> $log_file
        rm run.log || true
        ONNX_MLIR_INSTRUMENT_FILE=run.log RunONNXModel.py -m $name.mlir -c "$opt" -w $warmup -n $iter >> $log_file
        cat run.log >> $log_file
        make-report.py -r run.log >> $log_file
    done
}

# CPU
if  [ -z "${skipCPU}" ]
then
    if [ "${name}" == "matmul_bcast23" ]
    then
        # matmul with broadcast 23
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "16"  "32" "1"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "16"  "64" "1"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "16"  "128" "1"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "32"  "32" "1"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "32"  "64" "1"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "32"  "128" "1"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "64"  "32" "1"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "64"  "64" "1"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "64"  "128" "1"
    elif [ "${name}" == "matmul_3d" ]
    then
        # stacked matmul
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "16"  "32" "0"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "16"  "64" "0"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "16"  "128" "0"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "32"  "32" "0"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "32"  "64" "0"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "32"  "128" "0"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "64"  "32" "0"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "64"  "64" "0"
        run_matmul3d_experiment "-O3 ${arch} -profile-ir=Onnx" "64"  "128" "0"
    else
        # other operations
        run_experiment "-O3 ${arch} -profile-ir=Onnx" "16"  "32"
        run_experiment "-O3 ${arch} -profile-ir=Onnx" "16"  "64"
        run_experiment "-O3 ${arch} -profile-ir=Onnx" "16"  "128"
        run_experiment "-O3 ${arch} -profile-ir=Onnx" "16"  "256"
        run_experiment "-O3 ${arch} -profile-ir=Onnx" "32"  "32"
        run_experiment "-O3 ${arch} -profile-ir=Onnx" "32"  "64"
        run_experiment "-O3 ${arch} -profile-ir=Onnx" "32"  "128"
        run_experiment "-O3 ${arch} -profile-ir=Onnx" "32"  "256"
        run_experiment "-O3 ${arch} -profile-ir=Onnx" "64"  "32"
        run_experiment "-O3 ${arch} -profile-ir=Onnx" "64"  "64"
        run_experiment "-O3 ${arch} -profile-ir=Onnx" "64"  "128"
        run_experiment "-O3 ${arch} -profile-ir=Onnx" "64"  "256"
    fi
else
    echo "skip CPU"
fi

# NNPA
if  [ -z "${skipNNPA}" ]
then
    nnpa_options="-maccel=NNPA -profile-ir=ZHigh -nnpa-placement-heuristic=QualifyingOps"
    if [ "${name}" == "matmul_bcast23" ]
    then
        #matmul with bcast23
        # skip the e2=256
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "16"  "32" "1"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "16"  "64" "1"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "16"  "128" "1"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "32"  "32" "1"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "32"  "64" "1"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "32"  "128" "1"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "64"  "32" "1"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "64"  "64" "1"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "64"  "128" "1"
    elif [ "${name}" == "matmul_3d" ]
    then
        # stacked matmul
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "16"  "32" "0"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "16"  "64" "0"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "16"  "128" "0"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "32"  "32" "0"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "32"  "64" "0"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "32"  "128" "0"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "64"  "32" "0"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "64"  "64" "0"
        run_matmul3d_experiment "-O3 ${arch} $nnpa_options" "64"  "128" "0"
    else
        # normal operations
        run_experiment "-O3 ${arch} $nnpa_options" "16"  "32"
        run_experiment "-O3 ${arch} $nnpa_options" "16"  "64"
        run_experiment "-O3 ${arch} $nnpa_options" "16"  "128"
        run_experiment "-O3 ${arch} $nnpa_options" "16"  "256"
        run_experiment "-O3 ${arch} $nnpa_options" "32"  "32"
        run_experiment "-O3 ${arch} $nnpa_options" "32"  "64"
        run_experiment "-O3 ${arch} $nnpa_options" "32"  "128"
        run_experiment "-O3 ${arch} $nnpa_options" "32"  "256"
        run_experiment "-O3 ${arch} $nnpa_options" "64"  "32"
        run_experiment "-O3 ${arch} $nnpa_options" "64"  "64"
        run_experiment "-O3 ${arch} $nnpa_options" "64"  "128"
        run_experiment "-O3 ${arch} $nnpa_options" "64"  "256"
    fi
else
    echo "skip NNPA"
fi

# Final gathering of the data.
tile_name="/tmp/tile.$RANDOM.log"
stat_name="/tmp/stat.$RANDOM.log"
grep_pattern="^\s+(onnx|zhigh).$op_name,"
egrep "Tile" $log_file > $tile_name
egrep $grep_pattern $log_file > $stat_name
paste -d "," $tile_name $stat_name > $res_file
rm $tile_name $stat_name

# data has gathered the info from 
# Title: "Tile, $e3, $e2, $e1, option, $opt"
#
# and from the operation being monitored:
# e.g.
# Statistics start all ops ordered_by time, tot_time,  0.0000008
#  onnx.Add, 1, 0.0000008, 0.0000008, 100.0%A
# namely "op, num of occurrence, avg per call, tot time (all calls), % of total exec time"
# where time is in seconds.