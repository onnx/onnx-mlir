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
***Code: Traditional way to add numbers.***

In the above code, we need to distinguish between int and float type operations. We also need to repetitively pass the location.

### Math builder

A newer approach suggested by the MLIR community is to create a math builder, described below. The same code can be generated using the following.
``` C++
// Using hte same declaration as above for values, rewriter, and location.
MathBuilder createMath(rewriter, loc);
Value intRes = createMath.add(firstIntVal, secondIntVal);
Value floatRes = createMath.add(firstFloatVal, secondFloatVal);
```
***Code: New approach to add numbers.***

MLIR recommends this approach as it reads better, namely "we are creating a math add of two values", and the rewriter and location fields are now "hidden" inside the lightweight `createMath` object. In addition, the method deals with the different MLIR operations for adding integer and float internally.

In general, this and all other builders can be created as follows.
``` C++
// Constructors in class declaration.
struct MathBuilder : DialectBuilder {
  MathBuilder(OpBuilder &b, Location loc);
  MathBuilder(const DialectBuilder &db);
};

// Usage.
MathBuilder createMath(rewriter, loc); // Use original info.
MathBuilder createMath(createKrnl);    // Use info stored in another builder.
```

The Math builder contains the operations listed below. Most are self explanatory. They handle both integer and float operations, and will generate an assert when a specific operation is not supported for a specific type.  Up to date info should be looked from the [MLIRDialectBuilder.hpp](../src/Dialect/Mlir/DialectBuilder.hpp) file.

```C++
struct MathBuilder : DialectBuilder {
  MathBuilder(OpBuilder &b, Location loc);
  MathBuilder(const DialectBuilder &db);

  Value andi(Value lhs, Value rhs);
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
***Code: Math builder class.***

Note using the builders does not preclude making calls to the old interface. For any builders, we can extract, respectively, the rewriter and the location needed for the old interfaces using the `DialectBuilder` inherited methods `getRewriter()` and `getLoc()`.

### MemRef builder

An equivalent builder exists for some MemRef operation. At a high level, the following operations are supported.

``` C++
struct MemRefBuilder : DialectBuilder {
  MemRefBuilder(OpBuilder &b, Location loc);
  MemRefBuilder(const DialectBuilder &db);

  memref::AllocOp alloc(MemRefType type, ValueRange dynSymbols);
  memref::AllocaOp alloca(MemRefType type);
  memref::DeallocOp dealloc(Value val);
  Value dim(Value val, int64_t index);
};
```
***Code: MemRef builder class.***

It defines 4 distinct methods: how to allocate memory (`alloc`) and free (`dealloc`) memory from the heap, how to allocate memory on the stack (`alloca`), and how to extract the dimension of a multi-dimensional memory reference for a given dimension. The `alloca` method above allows for the multi-dimensional memory to have dynamic dimensions; these dynamic dimensions are specified by the parameter `dynSymbols`.  There are variant of these methods for static dimensions only and for providing alignment constraints. See the [MLIRDialectBuilder.hpp](../src/Dialect/Mlir/DialectBuilder.hpp) file for the full set of supported operations.

## Generating Krnl Operations

The krnl dialect is our main dialect to lower ONNX operations into loops. This dialect is one step above the MLIR affine dialect in that in enables us to express higher level loop constructs and loop optimizations.

## Builder based interface to generate Krnl loops

The new approach uses a Krnl builder class to construct Krnl dialect operation. The basic methods to build loops are the one listed below. Up to date info is found in the [KrnlHelper.hpp](../src/Dialect/Krnl/KrnlHelper.hpp) file.

``` C++
struct KrnlBuilder : public DialectBuilder {
  KrnlBuilder(OpBuilder &b, Location loc);
  KrnlBuilder(DialectBuilder &db);

  ValueRange defineLoops(int64_t originalLoopNum);

  void iterate(ValueRange originalLoops, ValueRange optimizedLoops,
      ValueRange lbs, ValueRange ubs,
      function_ref<void(const KrnlBuilder &createKrnl, ValueRange indices)>
          bodyBuilderFn);
};
```
***Code: Krnl builder class to minimally create a loop.***

The first method, `defineLoops` creates a set of loop descriptors that characterizes a loop iteration space. Initially, a set of loop descriptors characterizes the original loop iteration space, shortly, one such modified set can also be used to characterize an optimized iteration spaces, for example to represent a loop tiled iteration space after applying loop blocking and loop permutation.

The second method above, `iterate` is used to create a set of loops and its corresponding loop body. Until we optimize loops, both the `originalLoops` and the `optimizedLoops` are set to the output of a `defineLoops` method call. These sets describe the iteration space and its dimensionality. The next two parameters are used to describe the lower and the upper bounds of the loop. The last parameter defines a lambda function that implements the body of the loop. This lambda function is invoked with two parameters: an object to create further Krnl operations within the loop body and a list of the current loop index values.

The usage of this builder will become clearer with our example, setting an array to value zero. This is the same example as in the prior section.
``` C++
// Defined values 0 and a 2 dimensional array with dim ub0 and ub1
Value zero, array, ub0, ub1;

// Define the krnl builder.
KrnlBuilder createKrnl(rewriter, loc);

// Define a 2-dimensional iteration space.
ValueRange loopDef = createKrnl.defineLoops(2);

// Create the loop.
createKrnl.iterate(loopDef, loopDef, {zero, zero}, {ub0, ub1},
  [&](const KrnlBuilder  &createKrnl, ValueRange loopInd){
    // Loop body.
    createKrnl.store(zero, array, loopInd);
  });
```
***Code: Zeroing an array using the new builder interface***

Using this new scheme, we first define the 2D loop iteration space and then create the loop iteration structure using the `iterate` method. Since the loop is unoptimized, the same `loopDef` value range is passed as the first 2 parameters. The bounds are passed as 2 sets of ordered values.

Note that the lambda function creates an `createKrnl` builder that is similar to that of the external environment (outside the loop), but customized for inside the loop. So we can continue to use this overloaded builder to continue constructing krnl operations. In our case, we simply use the `loopInd` (2nd parameter of the lambda function), which are the current loop induction values, to define the element of the array that is set to zero.

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
***Code:Additional Krnl ops supported by the Krnl builder interface.***

Above, both the load and store operations are used to create Krnl memory load and store operations. They should be used instead of the MLIR Affine or Standard dialect operations.

The `block` method takes one loop definition (one value extracted from the output of a `defineLoop` operation) and will split that loop definition into 2, where the first one iterates over blocks of the given side, and the second one iterates inside of a given block. The two loop definitions are returned by the `block` method as a value range containing the two split loops described above.

The `permute` method takes a list of loop definitions and ensures that the loops will iterate according to the permuted order.

The `memcopy` method results in the array given by `dest` to be overwritten by `size` values from the array given by `src`. The `memset` method sets the entire array given by `dest` to the value passed in `val`, typically zero.

## Builder based interface to generate optimized Krnl loops

Let us now look how we can optimize loops using the Krnl builder. Consider our same example, setting an array to zero, and say we whish to tile the loop along both dimensions. Let us first tile a 1-dimensional loop iteration space.

``` C++
// Defined values 0 and a 1 dimensional array with dim ub0.
Value zero, array, ub0;
// Define a 2-dimensional iteration space.
ValueRange loopDef = createKrnl.defineLoops(1);
// Block the loop by a factor 4. First returned value in ValueRange 
// loops over blocks, the second return value loops inside a block.
ValueRange loopBlockDef = createKrnl.block(loopDef, 4);
// Permute the blocked loops
createKrnl.permute({loopBlockDef[0], loopBlockDef[1], {0,1});
// Create the loop iterating over the blocks.
createKrnl.iterate(loopDef, {loopBlockDef[0], loopBlockDef[0]}, {zero}, {ub0},
  [&](const KrnlBuilder  &createKrnl, ValueRange blockLoopInd){
    // Loop body.
    createKrnl.store(zero, array, loopInd);
  });
```
***Code: Blocked loop zeroing 1D array.***

In the code above, we block the original 1D loop iteration space defined by `defineLoop(1)` into two loops, one looping over the blocks of size 4, and the other looping inside a block. We then need to instruct the order of the optimized loop iteration space using a `permute` method. We can then perform an `iterate` method call, where the first parameter describes the original loop iteration space along with the lower and upper bound sets. In that same call, the second parameter indicates the actual loop iterations that we want to perform in the optimized iteration space, namely the loops over the blocks (`loopBlockDef[0]`) and loops inside a block (`loopBlockDef[1]`).

We now consider tiling our original 2-dimensional example below.
``` C++
  // Defined values 0 and a 2 dimensional array with dim ub0 and ub1
  Value zero, array, ub0, ub1;

  // Define a 2-dimensional iteration space.
  ValueRange loopDef = createKrnl.defineLoops(2);
  Value outerLoopDef(loopDef[0]), innerLoopDef(loopDef[1]);
  // Block each of the 2 dimensions: outer by 4, inner by 8.
  ValueRange outerLoopBlockDef = createKrnl.block(outerLoopDef, 4);
  ValueRange innerLoopBlockDef = createKrnl.block(innerLoopDef, 8);
  // Permute the loops (first loop over blocks, the loop inside blocks).
  createKrnl.permute({outerLoopBlockDef[0], outerLoopBlockDef[1],
    innerLoopBlockDef[0], innerLoopBlockDef[1]}, {0,2,1,4});
  // Create the loop iterating over the blocks.
  createKrnl.iterate(loopDef, {outerLoopBlockDef[0], innerLoopBlockDef[0]},
    {zero, zero}, {ub0, ub1},
    [&](const KrnlBuilder  &createKrnl, ValueRange blockLoopInd){
      // Create the loop iterating inside the blocks.
      createKrnl.iterate({}, {outerLoopBlockDef[1], innerLoopBlockDef[1]},
        {}, {}, [&](const KrnlBuilder  &createKrnl, ValueRange loopInd) {
          // Loop body.
          createKrnl.store(zero, array, loopInd);
        });
    });
```
***Code:Tiled loops zeroing 2D array.***

In the code above, we first renamed the 2-dimensional loop iteration space defined by the `defineLoops` method as outer and inner loop defs, corresponding respectively to the first and second value in the value range named `loopDef`. Then we block each of the outer and inner loops, resulting in 4 loops, 2 going over the blocks of the outer/inner loop and two going inside the blocks. The `permute` method defines the desired order, namely the blocked loop first, and the loops for the elements inside the block second.

All of the 4 loops could be now instantiated by a single `iterate` methods. We have chosen here to create the 2 sets of loop separately, as this pattern may be more prevalent in realistic code where we insert some additional code between the blocked loops and the loops iterating over the elements inside the blocks.

The first `iterate` calls provide the original unoptimized loop defs, as these loops are key to provide the lower and upper bounds of the original loop iteration space.  In other word, in this first `iterate` call, we inform the program that the original loops (defined by the value range returned by `defineLoops`) have lower bounds `{zero, zero}` and upper bounds `{ub0, ub1}`. This first `iterate` calls also indicates that we are interested issue loops for the 2 blocked loops, defined as `{outerLoopBlockDef[0], innerLoopBlockDef[0]}`.

The second `iterate` call does not need to redefine the original loop defs, as we have already provided the lower and upper bounds. So all these fields are left blank. In the second parameter to this call, we provide the next set of loops for which we want code to be generated, namely `{outerLoopBlockDef[1], innerLoopBlockDef[1]}`. Recall that the second parameter in the value range returned by a `permute` method corresponds to the iterating over the elements inside a given block.

Note also that we use the loop indices `loopInd` directly in the memory operation, as the loop indices are always the actual iteration number corresponding to the original loop. For example, consider an iteration space of 0..12. If we block it by a factor 4, the indices of the blocked loop will be 0, 4 and 8. And the indices of the loops iterating inside a given block will be 0,1,2, and 3 for the first block, 4,5,6,and 7 for the second block, and 8, 9, 10, and 11 for the third block. Say if the original loop trip count was only up to 11 instead of 12, the third block would iterate over the indices 8, 9, and 10 only.

## Generating affine loops

There is one more builder to assist the lowering of the Krnl dialect into the affine dialect. This builder is named `AffineBuilder` and is found in [KrnlToAffine.cpp](../src/Conversion/KrnlToAffine/KrnlToAffine.cpp)  file. It provides helper methods to generate multiple nested `affine.for` loops as well as `affine.if then else` constructs.

## Generating SCF operations

There is an additional builder for generating MLIR's SCF dialect.

## Combining multiple builders

Instead of creating multiple builders, e.g.

```C++
  KrnlBuilder createKrnl(rewriter, loc);
  MathBuilder createMath(createKrnl);
  MemRefBuilder createMemRef(createKrnl);
```
and then using them like this

```C++
  createKrnl.defineLoop(1);
  createMath.add(i1, i2);
  createMemRef.alloca(type);
```

we can create a single builder composed of multiple types and then as follows.

```C++
  MultiDialectBuilder<KrnlBuilder, MathBuilder, MemRefBuilder>
    create(rewriter, loc);

  create.krnl.defineLoop(1);
  create.math.add(i1, i2);
  create.mem.alloca(type);
```

Types that can be used here are listed here.
  *  `KrnlBuilder`, accessed with `krnl` field.
  *  `MathBuilder`, accessed with `math` field.
  *  `MemRefBuilder`, accessed with `mem` field.
  *  `ONNXBuilder`, accessed with `onnx` field.
  *  `SCFBuilder`, accessed with the `scf` field.
