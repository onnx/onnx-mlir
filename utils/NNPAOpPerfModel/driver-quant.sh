#!/bin/bash

# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 The IBM Research Authors.

#export ONNX_MLIR_HOME=/metis-workspace/main/onnx-mlir/build/Debug
#onnx_mlir_utils=/metis-workspace/main/onnx-mlir/utils

mkdir log || true
mkdir res || true
date=`date +%Y-%m-%d-%H-%M-%S`

# Matrix Multiply model file.
mm_mlir=mm.mlir

# Results in res subdir,
quant_log=res/quant_${date}.log
dlfloat_log=res/dlfloat_${date}.log
csv_file=res/test_matmul_dlfloat16_int8_${date}.csv

compile_args="-O3 -mcpu=z17 -maccel=NNPA -profile-ir=ZHigh -nnpa-placement-heuristic=QualifyingOps"
dims=(128 256 512 1024 2048 3072 4096)
#dims=(512 1024)

for m in ${dims[@]} 
do
  for k in ${dims[@]}
  do
    for n in ${dims[@]}
    do
      echo "doing ${m}-${k}-${n}"

      cat > ./${mm_mlir} << EOF
      module attributes {} {
        func.func @main_graph(%arg0 : tensor<${m}x${k}xf32>) -> tensor<${m}x${n}xf32> {
          %0 = onnx.Constant dense<3.14> : tensor<${k}x${n}xf32>
          %1 = "onnx.MatMul"(%arg0, %0) : (tensor<${m}x${k}xf32>, tensor<${k}x${n}xf32>) -> tensor<${m}x${n}xf32>
          onnx.Return %1 : tensor<${m}x${n}xf32>
        }
        "onnx.EntryPoint"() {func = @main_graph} : () -> ()
      }
EOF
      
      echo "  Quantization"
      # Instrument file in log dir.
      export ONNX_MLIR_INSTRUMENT_FILE=log/${m}-${k}-${n}-compile-quant.log
      RunONNXModel.py --model ./mm.mlir -c "${compile_args} --nnpa-quant-dynamic=symWeight,asymActivation" -w 2 -n 100 > ${m}-${k}-${n}-quant.log
      # String t includes a comma after number.
      t=$(tail -n 1 ${m}-${k}-${n}-quant.log | awk '{print $(NF-2)}')
      rm ${m}-${k}-${n}-quant.log
      echo "${m},${k},${n},${t}" >> ${quant_log}
      # Matmul of MxK * KxN, which in the Jupyter Notebook is recognized as e3xe2 * e2xe1.
      # No commas in options, has to match options used above.
      echo "Title,${m},${k},${n},option,-O3 -mcpu=z17 -maccel=NNPA -profile-ir=ZHigh --nnpa-quant-dynamic=symWeight:symActivation,zhigh.Matmul2D,1,${t}${t}100.0%" >> ${csv_file}

      echo "  Dlfloat"
      export ONNX_MLIR_INSTRUMENT_FILE=log/${m}-${k}-${n}-compile.log
      RunONNXModel.py --model ./mm.mlir -c "${compile_args}" -w 2 -n 100 > ${m}-${k}-${n}.log
      # String t includes a comma after number.
      t=$(tail -n 1 ${m}-${k}-${n}.log | awk '{print $(NF-2)}')
      rm ${m}-${k}-${n}.log
      echo "${m},${k},${n},${t}" >> ${dlfloat_log}
      # No commas in options, has to match options used above.
      # Do not use NNPA but nnpa instead (as this is a marker for distinguishing between
      # with/without experiment, usually CPU vs NNPA; but here orig vs quant).
      echo "Title,${m},${k},${n},option,-O3 -mcpu=z17 -maccel=nnpa -profile-ir=ZHigh,zhigh.Matmul2D,1,${t}${t}100.0%" >> ${csv_file}
    done
  done
done

rm ${mm_mlir}
