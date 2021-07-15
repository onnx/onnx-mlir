<!--- SPDX-License-Identifier: Apache-2.0 -->

# Instrumentation

Instrumentation is prototyped in onnx-mlir and can be used to debug runtime issue.

## Compile for instrumentation

By default, instrumentation is turned off. To turn it on, modify the default value of `OMInstrumentEnabled' in 'src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.cpp' and build the compiler. Command line flag will be added.

Currently, only some onnx ops are instrumented. They are Conv, element-wise binary and element-wise varadic operations.

The instrumentation is added before and after the op.o

Currently, the call of initialization, OMInstrumentInit, need to be added before you load the dynamic library. It is being considered to add it to the beginning of main_graph by compiler. 

## Run with instrumentation
The instrumenation library will print out the time and virtual memory usage along at each instrumentation point. A sample output is listed below:
```
ID=Conv TAG=0 Time elapsed: 0.000966 accumulated: 0.000966
335128
ID=Conv TAG=1 Time elapsed: 0.395338 accumulated: 0.396304
335128
ID=Mul TAG=0 Time elapsed: 0.302189 accumulated: 0.698493
335128
ID=Mul TAG=1 Time elapsed: 0.021133 accumulated: 0.719626
335128
```
The output is explained here:
* ID: currently is the name (limited to up to 7 chars) of the op.
* TAG: 0 for before the op, while 1 for after the op.
* elpased: time, in second, elapsed from previous instrumentation point.
* accumulated: time, in second, from instrumentationInit.
* the following line, 33512 in this example, is the virtual memory size (in kb) used by this process.

## Control of output
* If env variable OMINSTRUMENTTIME is set, the report of time is disabled
* If env variable OMINSTRUMENTMEMORY is set, the report of virtual memory is disabled

## Used in gdb
The function for instrument point is called `OMInstrumentPoint`. Breakpoint can be set inside this function to kind of step through onnx ops.
