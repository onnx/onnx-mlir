<!--- SPDX-License-Identifier: Apache-2.0 -->

# Handling errors in MLIR

Three are two different kinds of errors: errors that comes from user inputs, and compiler errors. We should provide meaningful user feedback for user input errors and we should use the `emitError` functions. Compiler errors should be reported using `asserts` or `llvm_unreachable` calls. In practice, if there are functions where errors are checked, and there is the ability to return "failure," the preferred way is to use `emitError` and return failure.  If, on the other hand, the function does not allow to return failure, then an assert or unreachable call should be used. Returning error is important for passes that check user inputs, e.g. such as during the ingestion of the ONNX model.

## User errors 

MLIR provides for 3 distinct calls depending on the severity: `emitError`, `emitWarning`, and 'emitRemark`. Errors should typically be reported to calling functions for proper handling. Typical use is as depicted below.

```cpp
  return op->emitError("message");
  return op->emitError() << "message";
```

Above calls will include the location of the operation. It returns a `LogicalResult` which can be set/tested as below. Note that the `emitError` calls return a `failure()` value;
```cpp
  LogicalResult isEven(int i) { return (i%2 == 0) success() : failure(); }

  if (succeeded(isEven(0)) && failed(isEven(1))) printf("It is all good.\n");
```

Errors can also be reported outside of the context of an operation. In this case, a location must be provided. To report a warning or a remark, just substitute "Warning" or "Remark" instead of "Error" in the above examples.

## Compiler errors

Once an ONNX graph has been validated, every subsequent erroneous  situations should be reported with an assert to stop the compilation, as this is a compiler error that needs to be properly handled. There are two calls that can be used:

```cpp
  assert(condition-that-should-hold-true && "error message");
  llvm_unreachable("error message");
```

The unreachable call is useful in functions that should return a value, as the compiler will not report warnings if there is no dummy-value return statement along that path. Otherwise, in `void` functions, using an assert is perfectly fine.


## References

Additional relevant information is found in the LLVM and MLIR documentation  referred below.
  
* [LLVM Docs on assert](https://llvm.org/docs/CodingStandards.html#assert-liberally)
* [MLIR Docs on diagnostic](https://mlir.llvm.org/docs/Diagnostics/)
  