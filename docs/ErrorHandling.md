# Handling errors in MLIR

Three are two different kinds of errors: errors that comes from user inputs, and compiler errors. User input errors should provide feedback to users, and are typically handled using `emitError` functions. Compiler errors should be reported using `asserts` or `llvm_unreachable` calls.

## User errors

MLIR provides for 3 distinct calls depending on the severity: Error, Warning, and Remark. Errors should typically be reported to calling functions for proper handling. Typical use would be:

```cpp
  return op->emitError("message");
  return op->emitError() << "message";
```

Above calls will include the location of the operation. It returns a `LogicalResult` which can be set/tested as below.
```cpp
  LogicalResult isEven(int i) { return (i%2 == 0) success() : failure(); }

  if (succeeded(isEven(0)) && failure(isEven(1))) printf("its all good\n");
```

Errors can also be reported outside of the context of an operation. In this case, a location must be provided. To report a warning or a remark, just substitute "Warning" or "Remark" instead of "Error" in the above examples.

## Compiler errors

Once an ONNX graph has been validated, every subsequent erronerous situations should be reported with an assert to stop the compilation, as this is a compiler error that needs to be properly handled. There are two calls that can be used:

```cpp
  assert(condition-that-should-hold-true && "error message");
  llvm_unreachable("error message");
```

The unreachable calls is useful in functions that should return a value, as the compiler will not report warnings if there is no dummy-value return statement along that path. Otherwise, assert are perfectly fine.


## References
* [LLVM Docs on assert](https://llvm.org/docs/CodingStandards.html#assert-liberally)
* [MLIR Docs on diagnostic](https://mlir.llvm.org/docs/Diagnostics/)
  