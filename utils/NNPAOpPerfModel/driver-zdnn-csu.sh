#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The IBM Research Authors.

#   param1: file name without the extension (.mlir)
#   param2: op_name to be grepped (can be a regexp, e.g. "(cpuOp|nnpaOp)")
#   param3: architecture (xxx in --march=xxx)
#
# env var: 
#  e4:       add dim to the shape; e.g. e4="1", e4="1x1" (default none).

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

echo "Experiment with ${name} file and ${op_name} op"
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


# Compiler stick/unstick

nnpa_opt="-nnpa-placement-heuristic=QualifyingOps -maccel=NNPA -profile-ir=ZHigh"

run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=false" "16"  "32"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=false" "16"  "64"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=false" "16"  "128"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=false" "16"  "256"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=false" "32"  "32"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=false" "32"  "64"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=false" "32"  "128"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=false" "32"  "256"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=false" "64"  "32"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=false" "64"  "64"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=false" "64"  "128"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=false" "64"  "256"

# ZDNN compiler stick/unstick
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=true" "16"  "32"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=true" "16"  "64"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=true" "16"  "128"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=true" "16"  "256"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=true" "32"  "32"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=true" "32"  "64"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=true" "32"  "128"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=true" "32"  "256"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=true" "64"  "32"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=true" "64"  "64"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=true" "64"  "128"
run_experiment "-O3 ${arch} $nnpa_options -disable-compiler-stick-unstick=true" "64"  "256"

tile_name="/tmp/tile.$RANDOM.log"
stat_name="/tmp/stat.$RANDOM.log"
grep_pattern="^\s+(onnx|zhigh).$op_name,"
egrep "Tile" $log_file > $tile_name
egrep $grep_pattern $log_file > $stat_name
# sed the first pattern to have -maccel==nnpa instead of -maccel=NNPA so that it be recognized as CPU code
paste -d "," $tile_name $stat_name | sed 's/-maccel=NNPA -profile-ir=ZHigh -disable-compiler-stick-unstick=false/-maccel==nnpa -profile-ir=ZHigh -disable-compiler-stick-unstick=false/' > $res_file
rm $tile_name $stat_name
