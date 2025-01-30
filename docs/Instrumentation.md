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
==PERF-REPORT==, onnx.Cast, bert/encoder/Reshape__27, before, 0.000001, 1692654182.738546
==PERF-REPORT==, onnx.Cast, bert/encoder/Reshape__27, after, 0.000001, 1692654182.738547
==PERF-REPORT==, onnx.Concat, bert/encoder/Reshape__27, before, 0.000000, 1692654182.738547
==PERF-REPORT==, onnx.Concat, bert/encoder/Reshape__27, after, 0.000001, 1692654182.738548
==PERF-REPORT==, onnx.Reshape, bert/encoder/Reshape, before, 0.000001, 1692654182.738549
==PERF-REPORT==, onnx.Reshape, bert/encoder/Reshape, after, 0.000001, 1692654182.738550
```

The output for the time measurement is explained here.
* The first column is a string to identify the performance being gathered, `PERF-REPORT` here.
* The second column is the name of op.
* The third column is the node name of op. This is displayed when the op has `onnx_node_name` attribute.
* The fourth column indicates if the time being reported is `before` or `after` the onnx operation being analyzed here.
* The fifth column indicates the elapsed time since the previous instrumentation point.
* The sixth column indicates the accumulated: time, in second, from instrumentationInit.

The output for the memory measurement is explained here.
* First column is a string to identify the performance being gathered, `MEM-REPORT` here.
* The second and third column are defined as above.
* The fourth column indicates VMem, the virtual memory size (in kb) used by this process.

Other example for NNPA
- Performance profiling for onnx ops before lowering to zhigh ops:
  `onnx-mlir --march=z16 --maccel=NNPA --instrument-stage=Onnx --instrument-ops=onnx.* --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime mymodel.onnx`
- Performance profiling for onnx and zhigh ops:
  `onnx-mlir --march=z16 --maccel=NNPA --instrument-stage=ZHigh --instrument-ops=onnx.*,zhigh.* --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime mymodel.onnx`
- Performance profiling for zlow ops:
  `onnx-mlir --march=z16 --maccel=NNPA --instrument-stage=ZLow --instrument-ops=zlow.* --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime mymodel.onnx`

## Control instrument at runtime
By providing certain env variable at runtime, you can disable reports from  instrument library.
* If the environment variable `ONNX_MLIR_NO_INSTRUMENT` is set, no report at all
* If the environment variable `ONNX_MLIR_NO_INSTRUMENT_TIME` is set, the report of time usage is disabled
* If the environment variable `ONNX_MLIR_NO_INSTRUMENT_MEMORY` is set, the report of memory usage is disabled
* If the environment variable `ONNX_MLIR_INSTRUMENT_FILE` is set, then this variable provide the file name in which to save the instrumentation.

Please note that the only way to enable instrumentation is to request it at compile time. If none of the detailed report (such as time and memory so far) is turned on at runtime, progress of instrument point will still be print out. This feature is thought to be useful as progress indicator. To fully disable any outputs requested at compile time, you must set `ONNX_MLIR_NO_INSTRUMENT`.

## Used in gdb
The function for instrument point is called `OMInstrumentPoint`. Breakpoint can be set inside this function to kind of step through onnx ops.
