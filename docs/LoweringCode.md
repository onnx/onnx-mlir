<!--- SPDX-License-Identifier: Apache-2.0 -->

# Lowering Code

## Generating Standard or MemRef code

The traditional way to generate code in MLIR is to use the `create` methods, which internally correspond `builder` methods associated with each MLIR operations. For example, creating an addition of two values is done as shown below.
``` C++
  // Declaration for the input values, to be filled accordingly
  Value firstIntVal, secondIntVal;
  Value firstFloatVal, secondFloatVal;
  OpBuilder rewriter; // Typically inherited from a caller context.
  Location loc; // Typically derived from an operation.

  Value intRes = rewriter.create<AddIOp>(loc, firstIntVal, secondIntVal);
  Value floatRes = rewriter.create<AddFOp>(loc, firstFloatVal, secondFloatVal);
```

In the above code, we need to distinguish between int and float type operations, and the code need to repetitively pass the location.

A newer approach suggested by the MLIR community is to create a math builder, described below. The same code can be generated using the following.
``` C++
  // Using hte same declaration as above for values, rewriter, and location.
  MathBuilder createMath(rewriter, loc);
  Value intRes = createMath.add(firstIntVal, secondIntVal);
  Value floatRes = createMath.add(firstFloatVal, secondFloatVal);
```

MLIR recommended this approach as it reads better, namely "we are creating a math add of two values", and the rewriter and location is now "hidden" inside the lightweight `createMath` object.

In general, this and all other builders can be created as follows.
  ``` C++
    // Constructors in class declaration.
    MathBuilder(OpBuilder &b, Location loc);
    MathBuilder(DialectBuilder &db);

    // Usage
    MathBuilder createMath(rewriter, loc); // Use original info.
    MathBuilder createMath(createKrnl);    // Use info stored in another builder.
  ```

The Math builder we have been looking at currently contains the following operations. Up to date info should be looked from the `MLIRDialectBuilder.cpp` file. Most operations are self explanatory. They handle both integer and float operations, and will generate an assert when a specific operation is not supported for a specific type.

```C++
struct MathBuilder : DialectBuilder {
  MathBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
  MathBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  Value _and(Value lhs, Value rhs);
  Value add(Value lhs, Value rhs);
  Value sub(Value lhs, Value rhs);
  Value mul(Value lhs, Value rhs);
  Value div(Value lhs, Value rhs);
  Value exp(Value val);
  Value select(Value cmp, Value lhs, Value rhs);
  Value sgt(Value lhs, Value rhs);
  Value slt(Value lhs, Value rhs);
  Value eq(Value lhs, Value rhs);
};
```



