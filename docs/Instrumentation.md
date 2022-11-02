<!--- SPDX-License-Identifier: Apache-2.0 -->

# Instrumentation

Instrumentation is prototyped in onnx-mlir and can be used to debug runtime issue.

## Compile for instrumentation

By default, instrumentation is turned off. You need to use following command line options to turn it on. The pass for instrumentation is inserted in some stages. The `--instrument-stage` is an option to specify it. For example, when you specify `afterOnnxToOnnx`, the instrumentation is inserted after pass for onnx-to-onnx conversion. The `--instrument-ops` option is an option to specify operations to be instrumented using regular expression. the `--InstrumentBeforeOp` and `--InstrumentAfterOp` are the options to insert instrumentation before and/or after the ops. For example, when you specify `onnx.` in `--instrument-ops` and use `--InstrumentBeforeOp` and `--InstrumentAfterOp`, the instrumantation is inserted before and after onnx operations such as onnx.Conv, onnx.Add etc.

```
  --instrument-stage=<value>                        - Specify stage to be instrumented:
    =afterOnnxToOnnx                                -   Profile for onnx ops.
    =nnpaAfterOnnxToOnnx                            -   [NNPA] Profile for onnx ops.
    =nnpaAfterOnnxToZhigh                           -   [NNPA] Profile for onnx and zhigh ops.
    =nnpaAfterZhighToZlow                           -   [NNPA] Profile for zlow ops.

  --instrument-ops=<string>                         - Specify regex for ops to be instrumented:
                                                      "NONE" or "" for no instrument,
                                                      "regex1,regex2, ..." for the specified ops.
                                                      e.g. "onnx.,zhigh." for onnx and zhigh ops.
                                                      e.g. "onnx.Conv" for onnx Conv ops.

  Specify what instrumentation actions at runtime:
      --InstrumentBeforeOp                             - insert instrument before op,
      --InstrumentAfterOp                              - insert instrument after op,
      --InstrumentReportTime                           - instrument runtime reports time usage,
      --InstrumentReportMemory                         - instrument runtime reports memory usage.
```

Currently, the call of initialization, OMInstrumentInit, need to be added before you load the dynamic library. It is being considered to add it to the beginning of main_graph by compiler. 

## Run with instrumentation
Run the model in the same way as usual.
The instrumentation library will print out the time and memory usage along at each instrumentation point.
For example, a model, `mymodel.onnx`, is compiled with `onnx-mlir  --instrument-stage=afterOnnxToOnnx --instrument-ops="onnx." --InstrumentAfterOp --InstrumentReportMemory --InstrumentReportTime mymodel.onnx`.
Its runtime output is listed below:

```
#  0) after onnx.Transpose Time elapsed: 0.000766 accumulated: 0.000766 VMem: 156608 (model/transpose1)
#  1) after onnx.Constant  Time elapsed: 0.005398 accumulated: 0.006164 VMem: 156608
#  2) after onnx.Constant  Time elapsed: 0.004225 accumulated: 0.010389 VMem: 156608
#  3) after onnx.Conv      Time elapsed: 0.360213 accumulated: 0.370602 VMem: 156608 (model/conv1)
#  4) after onnx.Softplus  Time elapsed: 0.190591 accumulated: 0.561193 VMem: 156608 (model/softplus1)
#  5) after onnx.Tanh      Time elapsed: 0.115314 accumulated: 0.676507 VMem: 156608 (model/tanh1)
#  6) after onnx.Mul       Time elapsed: 0.022779 accumulated: 0.699286 VMem: 156608 (model/mul1)
```

The output is explained here:
* First column is the dynamic counter of instrument point at runtime
* Second column indicate the position of this instrument, before or after its op
* Third column is the name of op
* elpased: time, in second, elapsed from previous instrumentation point.
* accumulated: time, in second, from instrumentationInit.
* VMem: the virtual memory size (in kb) used by this process.
* Last column is the node name of op. This is displayed when the op has `onnx_node_name` attribute.

## Control instrument at runtime
By providing certain env variable at runtime, you can disable reports from  instrument library.
* If env variable NOOMINSTRUMENT is set, no report at all
* If env variable NOOMINSTRUMENTTIME is set, the report of time usage is disabled
* If env variable NOOMINSTRUMENTMEMORY is set, the report of memory usage is disabled
Please note that you cannot turn on extra report that is not chosen at compile time. If none of the detailed report (such as time and memory so far) is turned on, progress of instrument point will still be print out. This feature is thought to be useful as progress indicator. No output from instrument lib is NOOMINSTRUMENT is set.

## Used in gdb
The function for instrument point is called `OMInstrumentPoint`. Breakpoint can be set inside this function to kind of step through onnx ops.
