<!--- SPDX-License-Identifier: Apache-2.0 -->

# Coding Practices

This document contains coding practices to use when adding or updating code to the onnx-mlir project.

## Practices

* Use C++ style casting instead of C style when casting in cpp.

For example, use C++ style casting:
```
      Value one = create.llvm.constant(llvmI64Ty, static_cast<int64_t>(1));
```

Not, C style casting:
```
      Value one = create.llvm.constant(llvmI64Ty, (int64_t)1);
```

* Perform bitwise operations on unsigned types and not signed.
* Check the result of malloc() invocations.
* Check the result of input/output operations, such as fopen() and fprintf().
* Use parentheses around parameter names in macro definitions. 

## Contributing

We are welcoming contributions from the community.
Please consult the [CONTRIBUTING](CONTRIBUTING.md) page for help on how to proceed.
