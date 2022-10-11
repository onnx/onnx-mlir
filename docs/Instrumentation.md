<!--- SPDX-License-Identifier: Apache-2.0 -->

# Instrumentation

Instrumentation is prototyped in onnx-mlir and can be used to debug runtime issue.

## Compile for instrumentation

By default, instrumentation is turned off. You need to use command line options to turn it on. 
```
 --instrument-onnx-ops=<string>                    - specify onnx ops to be instrumented
                                                      "NONE" or "" for no instrument
                                                      "ALL" for all ops. 
                                                      "op1 op2 ..." for the specif
Specify what instrumentation actions at runtime:
      --InstrumentBeforeOp                             - insert instrument before op
      --InstrumentAfterOp                              - insert instrument after op
      --InstrumentReportTime                           - instrument runtime reports time usage
      --InstrumentReportMemory                         - instrument runtime reports memory usage
```
For example, `Debug/bin/onnx-mlir --instrument-onnx-ops="Conv" --InstrumentBeforeOp --InstrumentAfterOp --InstrumentReportTime mymodel.onnx`
 will instrument before and after each onnx.Conv op and report time usage  when the model is executed. 

Currently, the call of initialization, OMInstrumentInit, need to be added before you load the dynamic library. It is being considered to add it to the beginning of main_graph by compiler. 

## Run with instrumentation
Run the model in the same way as usual.
The instrumentation library will print out the time and memory usage along at each instrumentation point.
For example, a model, `mymodel.onnx`, is compiled with `Debug/bin/onnx-mlir  --instrument-onnx-ops="ALL" --InstrumentAfterOp --InstrumentReportMemory --InstrumentReportTime mymodel.onnx`.
Its runtime output is listed below:

```
#  0) after  op= Transpo Time elapsed: 0.000766 accumulated: 0.000766 VMem: 156608 (model/transpose1)
#  1) after  op= Constan Time elapsed: 0.005398 accumulated: 0.006164 VMem: 156608
#  2) after  op= Constan Time elapsed: 0.004225 accumulated: 0.010389 VMem: 156608
#  3) after  op=    Conv Time elapsed: 0.360213 accumulated: 0.370602 VMem: 156608 (model/conv1)
#  4) after  op= Softplu Time elapsed: 0.190591 accumulated: 0.561193 VMem: 156608 (model/softplus1)
#  5) after  op=    Tanh Time elapsed: 0.115314 accumulated: 0.676507 VMem: 156608 (model/tanh1)
#  6) after  op=     Mul Time elapsed: 0.022779 accumulated: 0.699286 VMem: 156608 (model/mul1)
```

The output is explained here:
* First column is the dynamic counter of instrument point at runtime
* Second column indicate the position of this instrument, before or after its op
* Third column is the name of op, limited to 7 characters
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
