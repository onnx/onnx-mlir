<!--- SPDX-License-Identifier: Apache-2.0 -->

# Instrumentation

Instrumentation is prototyped in onnx-mlir and can be used to debug runtime issue.

## Compile for instrumentation

By default, instrumentation is turned off. You need to use following command line options to turn it on. The pass for instrumentation will be inserted in some stages by using `--instrument-stage` option. For example, when you specify `Onnx`, the instrumentation will be inserted after onnx-to-onnx conversion to get onnx-level profiling. The `--instrument-ops` option is an option to specify operations to be instrumented. You can use `onnx.Conv` for onnx Conv operations for example. Also, you can use asterisk such as `onnx.*` for all onnx operations, and specify two expressions with `,` such as `onnx.Conv,onnx.Add` for both Conv and Add operations. The `--InstrumentBeforeOp` and `--InstrumentAfterOp` are options to insert instrumentation before and/or after the specified operations. When you use `--instrument-ops=onnx.* --InstrumentBeforeOp --InstrumentAfterOp`, the instrumantation will be inserted before and after all onnx operations.
For NNPA, additional stages for `ZHigh` and `ZLow` are provided. You can get profile for onnx and zhigh ops using `--instrument-stage=ZHigh` and `--instrument-ops=onnx.*,zhigh.*`, and for zlow ops using `--instrument-stage=ZLow` and `--instrument-ops=zlow.*`.

```
  --instrument-stage=<value>                        - Specify stage to be instrumented:
    =Onnx                                             -   Profile for onnx ops. For NNPA, profile onnx ops before lowering to zhigh.
    =ZHigh                                            -   NNPA profiling for onnx and zhigh ops.
    =ZLow                                             -   NNPA profiling for zlow ops.

  --instrument-ops=<string>                         - Specify operations operations to be instrumented:
                                                      "NONE" or "" for no instrument,
                                                      "ops1,ops2, ..." for the multiple ops.
                                                      e.g. "onnx.Conv,onnx.Add" for Conv and Add ops.
                                                      Asterisk is also available.
                                                      e.g. "onnx.*" for all onnx operations.

  Specify what instrumentation actions at runtime:
      --InstrumentBeforeOp                          - insert instrument before op,
      --InstrumentAfterOp                           - insert instrument after op,
      --InstrumentReportTime                        - instrument runtime reports time usage,
      --InstrumentReportMemory                      - instrument runtime reports memory usage.
```

Currently, the call of initialization, OMInstrumentInit, need to be added before you load the dynamic library. It is being considered to add it to the beginning of main_graph by compiler. 

## Run with instrumentation
Run the model in the same way as usual.
The instrumentation library will print out the time and memory usage along at each instrumentation point.
For example, a model, `mymodel.onnx`, is compiled with `onnx-mlir  --instrument-stage=Onnx --instrument-ops=onnx.* --InstrumentAfterOp --InstrumentReportMemory --InstrumentReportTime mymodel.onnx`.
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

Other example for NNPA
- Performance profiling for onnx ops before lowering to zhigh ops:
  `onnx-mlir --maccel=NNPA --instrument-stage=Onnx --instrument-ops=onnx.* --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime mymodel.onnx`
- Performance profiling for onnx and zhigh ops:
  `onnx-mlir --maccel=NNPA --instrument-stage=ZHigh --instrument-ops=onnx.*,zhigh.* --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime mymodel.onnx`
- Performance profiling for zlow ops:
  `onnx-mlir --maccel=NNPA --instrument-stage=ZLow --instrument-ops=zlow.* --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime mymodel.onnx`

## Control instrument at runtime
By providing certain env variable at runtime, you can disable reports from  instrument library.
* If env variable NOOMINSTRUMENT is set, no report at all
* If env variable NOOMINSTRUMENTTIME is set, the report of time usage is disabled
* If env variable NOOMINSTRUMENTMEMORY is set, the report of memory usage is disabled
Please note that you cannot turn on extra report that is not chosen at compile time. If none of the detailed report (such as time and memory so far) is turned on, progress of instrument point will still be print out. This feature is thought to be useful as progress indicator. No output from instrument lib is NOOMINSTRUMENT is set.

## Used in gdb
The function for instrument point is called `OMInstrumentPoint`. Breakpoint can be set inside this function to kind of step through onnx ops.
