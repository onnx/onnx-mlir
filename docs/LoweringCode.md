<!--- SPDX-License-Identifier: Apache-2.0 -->

# Lowering Code

## Generating Standard or MemRef code

### Traditional approach

The traditional way to generate code in MLIR is to use the `create` methods, which internally employ the `builder` methods associated with each MLIR operation. For example, creating an addition of two values is done as shown below.
``` C++
  // Declaration for the input values, to be filled accordingly
  Value firstIntVal, secondIntVal;
  Value firstFloatVal, secondFloatVal;
  OpBuilder rewriter; // Typically inherited from a caller context.
  Location loc; // Typically derived from an operation.

  Value intRes = rewriter.create<AddIOp>(loc, firstIntVal, secondIntVal);
  Value floatRes = rewriter.create<AddFOp>(loc, firstFloatVal, secondFloatVal);
```

In the above code, we need to distinguish between int and float type operations. We also need to repetitively pass the location.

### Math builder

A newer approach suggested by the MLIR community is to create a math builder, described below. The same code can be generated using the following.
``` C++
  // Using hte same declaration as above for values, rewriter, and location.
  MathBuilder createMath(rewriter, loc);
  Value intRes = createMath.add(firstIntVal, secondIntVal);
  Value floatRes = createMath.add(firstFloatVal, secondFloatVal);
```

MLIR recommends this approach as it reads better, namely "we are creating a math add of two values", and the rewriter and location is now "hidden" inside the lightweight `createMath` object. In addition, the method deals with the different MLIR operations for adding integer and float internally.

In general, this and all other builders can be created as follows.
  ``` C++
    // Constructors in class declaration.
    MathBuilder(OpBuilder &b, Location loc);
    MathBuilder(DialectBuilder &db);

    // Usage.
    MathBuilder createMath(rewriter, loc); // Use original info.
    MathBuilder createMath(createKrnl);    // Use info stored in another builder.
  ```

The Math builder we have been looking at currently contains the following operations. Most operations are self explanatory. They handle both integer and float operations, and will generate an assert when a specific operation is not supported for a specific type.  Up to date info should be looked from the `MLIRDialectBuilder.cpp` file.

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

### MemRef builder

An equivalent builder exists for most MemRef operation. At a high level, the following operations are supported.

``` C++
struct MemRefBuilder : DialectBuilder {
  MemRefBuilder(OpBuilder &b, Location loc);
  MemRefBuilder(DialectBuilder &db);

  memref::AllocOp alloc(MemRefType type, ValueRange dynSymbols);
  memref::AllocaOp alloca(MemRefType type);
  memref::DeallocOp dealloc(Value val);
  Value dim(Value val, int64_t index);
};
```

It defines 4 distinct methods: how to allocate memory on the heap (`alloc`), on the stack (`alloca`), how to free memory from the heap (`dealloc`), and how to get the dimension of a multi-dimensional memory reference at a given index. The `alloca` methods above allows for the multi-dimensional memory to have dynamic dimensions, passed as the value range `dynSymbols`.  There are variant of these calls for static dimensions only, and for providing a mandatory alignment. See the `MLIRDialectBuilder.cpp` file for up to date information.

## Generating Krnl Operations

The krnl dialect is our main dialect to lower ONNX operations into loops. This dialect is one step above the MLIR affine dialect in that in enables us to express higher level loop constructs and loop optimizations.

### Older interface to generate loops

In the older approach, we generate loops manually by first defining the loop variables, then the loop bounds, then the loop itself. To generate  code inside of the loop, we first need to manually move a code insertion pointer to the block inside of the loop body and then generate the code. When generating code after the loop, we need to move back a code insertion pointer back to a point after the loop. For example, consider the code below to initialize a 2 dimensional array to zero.

``` C++
  // Defined values 0 and a 2 dimensional array with dim ub0 and ub1
  Value zero, array, ub0, ub1;

  // Define data structure containing the info for the 2 dimensional loop.
  BuildKrnlLoop loop(rewriter, loc, 2);
  loop.createDefineOp();
  loop.pushBounds(zero, ub0);
  loop.pushBounds(zero, ub1);
  // Create loop.
  loop.createIterateOp();
  // Set insertion point inside the loop.
  rewriter.setInsertionPointToStart(loop.getIterateBlock());
  // write the code inside the loop
  Value loopInd0 = loop.getInductionVar(0);
  Value loopInd1 = loop.getInductionVar(0);
  rewriter.create<KrnlStoreOp>(loc, zero, array, {loopInd0, loopInd1});
```

Notably, it is difficult to visually see in the code where the loop body starts and end, which operations are in it. In addition, the loop bounds are pushed by pairs (lower and upper bounds) in a strict order. That same order is used to retrieve the loop indices inside of the loop. The `rewriter`'s insertion point is manually changed from outside of the loop to inside the loop, and would have to be moved back if we wanted to insert operations after the loop.

Readability becomes a particular issue when the nesting structure becomes more complicated.

## Builder based interface to generate Krnl loops

The new approach uses a Krnl builder class to construct Krnl dialect operation. The basic methods to build loops are the one listed below. Up to date info is found in the `KrnlHelper.hpp` file.

``` C++
struct KrnlBuilder : public DialectBuilder {
  KrnlBuilder(OpBuilder &b, Location loc);
  KrnlBuilder(DialectBuilder &db);

  ValueRange defineLoops(int64_t originalLoopNum);

  void iterate(ValueRange originalLoops, ValueRange optimizedLoops,
      ValueRange lbs, ValueRange ubs,
      function_ref<void(KrnlBuilder &createKrnl, ValueRange indices)>
          bodyBuilderFn);
};
```

The first method, `defineLoops` creates a set of loop descriptors that will describe a loop iteration space. Initially, the loop descriptors describe the original loop iteration space, but they are also used to described optimized iteration spaces, for example after loop tiling and loop permutation.

The second method above, `iterate` is used to create a loop structure as well as the loop body. Unless we optimize  loops, both the `originalLoops` and the `optimizedLoops` are set to the output of the `defineLoops` call. It describes the iteration space and its dimensionality. The next two parameters are used to describe the lower and the upper bounds of the loop. The last parameter defines a lambda function that implements the body of the loop. When invoked, this labmda function is given 2 parameters: first, an object to create further Krnl operations within the loop body; and second, a list of the current loop index values.

The usage of this builder will become clearer with our same example, setting an array to value zero. This is the same example as in the prior section.
``` C++
  // Defined values 0 and a 2 dimensional array with dim ub0 and ub1
  Value zero, array, ub0, ub1;

  // Define the krnl builder.
  KrnlBuilder createKrnl(rewriter, loc);
  // Define a 2-dimensional iteration space.
  ValueRange loopDef = createKrnl.defineLoops(2);
  // Create the loop.
  createKrnl.iterate(loopDef, loopDef, {zero, zero}, {ub0, ub1},
    [&](KrnlBuilder &createKrnl, ValueRange loopInd){
      // Loop body.
      createKrnl.store(zero, array, loopInd);
    });
```

Using this new scheme, we first define the 2D loop iteration space and then create the loop iteration structure using the `iterate` method. Since the loop is unoptimized, the same `loopDef` value range is passed as the first 2 parameters. The bounds are passed as 2 set of ordered values.

Note that the lambda function create an `createKrnl` builder that is similar to that of the external environment (outside the loop), but customized for inside the loop. Se we can continue to use this overloaded builder to continue constructing krnl operations. In our case, we simply use the `loopInd` (2nd parameter of the lambda function), which are the current loop induction values, to define the element of the array that is set to zero.

Some of the other operations that are often used are listed below.
``` C++
struct KrnlBuilder : public DialectBuilder {
  // in addition to above...

  // Memory operations.
  Value load(Value memref, ValueRange indices = {});
  void store(Value val, Value memref, ValueRange indices = {});

  // Loop optimizations.
  ValueRange block(Value loop, int64_t blockSize);
  void permute(ValueRange loops, ArrayRef<int64_t> map);

  // Simple setter for entire arrays.
  void memcpy(Value dest, Value src, Value size);
  void memset(Value dest, Value val);
};
```

Above, both the load and store operations are used to create Krnl memory load and store operations. They should be used instead of the MLIR Affine or Standard dialect operations.

The `block` method takes one loop definition (one value extracted from the output of a `defineLoop` operation) and will split that loop definition into 2, where the first one iterates over blocks of the given side, and the second one iterates inside of a given block. The two loop definitions are returned by the `block` method as a value range containing the two split loops described above.

The `permute` method takes a list of loop definitions are ensure that the iterate will generate code reflecting the permuted order.

The `memcopy` method results in the array given by `dest` to be overwritten by `size` values from the array given by `src`. The `memset` method set the entire array given by `dest` to the value passed in `val`, typically zero.

## Builder based interface to generate optimized Krnl loops

Let us now look how we can optimize loops using the Krnl builder. Consider our same example, setting an array to zero, and say we whish to tile the loop along both dimensions.

``` C++
  // Defined values 0 and a 2 dimensional array with dim ub0 and ub1
  Value zero, array, ub0, ub1;

  // Define a 2-dimensional iteration space.
  ValueRange loopDef = createKrnl.defineLoops(2);
  Value outerLoopDef(loopDef[0]), innerLoopDef(loopDef[1]);
  // NEW: block each of the 2 dimensions: outer by 4, inner by 8.
  ValueRange outerLoopBlockDef = createKrnl.bock(outerLoopDef, 4);
  ValueRange innerLoopBlockDef = createKrnl.bock(innerLoopDef, 8);
  // NEW: Permute the loops (first loop over blocks, the loop inside blocks).
  createKrnl.permute({outerLoopBlockDef[0], outerLoopBlockDef[1],
    innerLoopBlockDef[0], innerLoopBlockDef[1]}, {0,2,1,4});
  // Create the loop iterating over the blocks.
  createKrnl.iterate(loopDef, {outerLoopBlockDef[0], innerLoopBlockDef[0]},
    {zero, zero}, {ub0, ub1},
    [&](KrnlBuilder &createKrnl, ValueRange blockLoopInd){
      // Create the loop iterating inside the blocks.
      createKrnl.iterate({}, {outerLoopBlockDef[1], innerLoopBlockDef[1]},
        {}, {}, [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Loop body.
          createKrnl.store(zero, array, loopInd);
        });
    });
```

In the code above, we first renamed the 2-dimensional loop iteration space defined by the `defineLoops` method as outer and inner loop defs, corresponding respectively to the first and second value in the value range named `loopDef`. Then we block each of the outer and inner loops, resulting in 4 loops, 2 going over the blocks of the outer/inner loop nd two going inside the blocks. The `permute` method defines the desired order, namely the blocked loop first, and the loops for the elements inside the block second.

All of the 4 loops could be now instantiated by a single `iterate` methods. We have chosen here to create the 2 sets of loop separately, as this pattern will be more likely so as to insert some additional code between the blocked loops and the loops iterating over the elements inside the blocks.

The first `iterate` calls provide the original unoptimized loop defs, as these loops are key to provide the lower and upper bounds of the original loop iteration space.  In other word, in this first `iterate` call, we inform the program that the original loops (defined by the value range returned by `defineLoops`) have lower bounds `{zero, zero}` and upper bounds `{ub0, ub1}`. This first `iterate` calls also indicates that we are interested issue loops for the 2 blocked loops, defined as `{outerLoopBlockDef[0], innerLoopBlockDef[0]}`.

The second `iterate` call does not need to redefine the original loop defs, as we have already provided the lower and upper bounds. So all these fields are left blank. In the second parameter to this call, we provide the next set of loops for which we want code to be generated, namely `{outerLoopBlockDef[1], innerLoopBlockDef[1]}`. Recall that the second parameter in the value range returned by a `permute` method corresponds to the iterating over the elements inside a given block.

Note also that we use the loop indices `loopInd` directly in the memory operation, as the loop indices are always the actual iteration number corresponding to the original loop. For example, consider an iteration space of 0..12. If we block it by a factor 4, the indices of the blocked loop will be 0, 4 and 8. And the indices of the loops iterating inside a given block will be 0,1,2, and 3 for the first block, 4,5,6,and 7 for the second block, and 8, 9, 10, and 11 for the third block. Say if the original loop trip count was only up to 11 instead of 12, the third block would iterate over the indices 8, 9, and 10 only.

## Generating affine loops

There is one more builder to assist the lowering of the Krnl dialect into the affine dialect. This builder is named `AffineBuilder` and is found in `KrnlToAffine.cpp` file. It provides helper methods to generate multiple nested `affine.for` loops as well as `affine.if then else` constructs.