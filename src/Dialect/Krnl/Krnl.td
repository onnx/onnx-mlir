// SPDX-License-Identifier: Apache-2.0

//===------------------ KrnlOps.td - MLIR Operations ---------*- tablegen -===//
//
// Copyright 2019-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains TableGen definition of krnl operations.
//
// REQUEST:
// When you perform a change to the KRNL dialect, please follow the steps below
// to update the Krnl dialect documentation.
//
// 1) cd onnx-mlir/build 
// 2) make onnx-mlir-docs
// 3) the newly build krnl.md file resides in the docs/Dialects directory.
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "src/Interface/SpecializedKernelOpInterface.td"

def Krnl_Dialect : Dialect {
  let name = "krnl";
  let cppNamespace = "::mlir";
}

def StringType : Type<CPred<"$_self.isa<krnl::StringType>()">, "string type">;

// Require regions to have krnl.terminate terminator operation.
def ImplicitKrnlTerminator : SingleBlockImplicitTerminator<"KrnlTerminatorOp">;

def KrnlCallOp : Op<Krnl_Dialect, "call"> {
  let summary = "call operation";
  let description = [{
    The call operation provides a generic way to call an external function
    at Krnl level.  The `funcName` determines which function to call.
    The `alloc` is the Value to store the function return. Since allocation
    of the return MemRef involves shape inference usually with IndexExpr.
    Thus most of time the allocation should stay in compiler, not in runtime library.
    The parameters can be of any type: MemRef, NoneType or any llvm type.
    Different types of parameters will be converted, if needed, when KrnlCallOp
    is lowered. Attributes will be converted to parameters too (To be Added).
    The function signature will be determined with the types of parameters.
    An LLVM::CallOp to either a runtime library or a llvm intrinsic function
    will be generated.
  }];

  let arguments = (ins StrAttr:$funcName, AnyType:$alloc, Variadic<AnyType>:$parameters);
  let results =(outs AnyType:$result);

  // builders to build KrnlCallOp from op and operands, helping conversion from onnx to krnl.
  // The name of function can be determined by the op name and elemnt type of the return,
  // or given to builder if the simple rule does not work
  // Attributes of the op will be propagated to KrnlCallOp if the copyAttrs is true
  let builders = [OpBuilder<(ins "mlir::Value":$alloc, "mlir::Operation *":$op, "mlir::ValueRange":$operands, "bool":$copyAttrs)>,
      OpBuilder<(ins "std::string":$funcNameStr, "mlir::Value":$alloc, "mlir::Operation *":$op, "mlir::ValueRange":$operands, "bool":$copyAttrs)>];
}
 
def KrnlDefineLoopsOp : Op<Krnl_Dialect, "define_loops"> {
  let summary = "define_loops operation";
  let description = [{
    The "krnl.define_loops" operation is used to define input loops,
    those are the for loops appearing in the input program that we
    intend to optimize.
  }];

  let results = (outs Variadic<AnyType>);

  let skipDefaultBuilders = 1;
  let builders = [ OpBuilder<(ins "int64_t":$num_loops)> ];

  let hasCustomAssemblyFormat = 1;

  let extraClassDeclaration = [{
    static StringRef getNumLoopsAttrName() { return "num_loops"; }

    // Helper function to extract the number of loops being defined.
    int64_t getNumLoops() {
      auto num_loops = (*this)->getAttrOfType<IntegerAttr>(getNumLoopsAttrName())
                           .getValue()
                           .getSExtValue();
      return num_loops;
    }
  }];
}

def KrnlIterateOp : Op<Krnl_Dialect, "iterate", [ImplicitKrnlTerminator,
    DeclareOpInterfaceMethods<LoopLikeOpInterface>]> {
  let summary = "iterate operation";
  let description = [{
    The "krnl.iterate" operation is conceptually equivalent to a nested for loops.

    For instance, say we have the following two
    %l0, %l1 = krnl.define_loops 2
    %o0, %o1 = krnl.optimize_loops  {
        // Identity schedule.
        krnl.return_loops %l0, %l1
    }

    Then, consider the following krnl.iterate operation:
    krnl.iterate (%o0, %o1) with (%l0 -> %i0 = 0 to 10, %l1 -> %i1 = 0 to 10) {
      // Some operations.
    }

    It is equivalent to:
    for (i0 = 0; i0 < 10; i0++)
      for (i1 = 0; i1 < 10; i1++)
        // Some operations.
  }];

  let arguments = (ins Variadic<AnyType>);

  let regions = (region SizedRegion<1>:$bodyRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    // Main builder.
    OpBuilder<(ins "KrnlIterateOperandPack":$operandPack,
      CArg<"ValueRange", "llvm::None">:$iterArgs,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>", "nullptr">:$bodyBuilderFn)>,
    // Builder for the optimized iterate op.
    OpBuilder<(ins
      CArg<"ValueRange">:$originalLoops, CArg<"ValueRange">:$optimizedLoops,
      CArg<"ValueRange">:$lbs, CArg<"ValueRange">:$ubs, CArg<"ValueRange">:$iterArgs,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>">:$bodyBuilderFn)>,
    // Builder for the optimized iterate op with IndexExpr
    OpBuilder<(ins
      CArg<"ValueRange">:$originalLoops, CArg<"ValueRange">:$optimizedLoops,
      CArg<"ArrayRef<IndexExpr>">:$lbs, CArg<"ArrayRef<IndexExpr>">:$ubs, 
      CArg<"ValueRange">:$iterArgs,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange)>">:$bodyBuilderFn)>
  ];

  let hasCustomAssemblyFormat = 1;
  let hasVerifier = 1;

  let extraClassDeclaration = [{
    // In krnl.iterate operation, operands are stored as such
    // - Optimized krnl.loops.
    // - Input krnl.loops and their operand bounds. (TODO(Tian) explain better how we store them).

    // We record the number of optimized and input loops to separate these three
    // group of operands out.
    static StringRef getNumOptimizedLoopsAttrName() { return "num_optimized_loops"; }

    int64_t getNumOptimizedLoops() {
      auto num_optimized_loops =
        (*this)->getAttrOfType<IntegerAttr>(getNumOptimizedLoopsAttrName())
          .getValue()
          .getSExtValue();
      return num_optimized_loops;
    }

    // Get name of the attribute for storing bound represented using affine maps.
      static StringRef getBoundsAttrName() { return "bounds"; }
  }];
}

def KrnlTerminatorOp : Op<Krnl_Dialect, "terminate", [Terminator]> {
  let summary = "Krnl terminator operation";
  let description = [{
    Krnl terminator is a special terminator operation for blocks inside krnl
    iterate operations. It unconditionally transmits the control flow to the
    successor of the operation enclosing the region.

    This operation does _not_ have a custom syntax. However, krnl control
    operations omit the terminator in their custom syntax for brevity.
  }];
}

def KrnlEntryPointOp : Op<Krnl_Dialect, "entry_point"> {
  let summary = "Indicate ONNX entry point";
  let description = [{The "krnl.entry_point" function indicates the main entry
                           point of ONNX model.}];
  let builders = [ OpBuilder<(ins "SymbolRefAttr":$funcAttr, "IntegerAttr":$numInputs,
                                  "IntegerAttr":$numOutputs, "StringAttr":$signature)> ];

  let extraClassDeclaration = [{
    static StringRef getEntryPointFuncAttrName() { return "func"; }
    static StringRef getNumInputsAttrName() { return "numInputs"; }
    static StringRef getNumOutputsAttrName() { return "numOutputs"; }
    static StringRef getSignatureAttrName() { return "signature"; }
  }];
}

def KrnlMemcpyOp : Op<Krnl_Dialect, "memcpy", [MemRefsNormalizable]> {
  let summary = "Krnl memcpy operation";
  let description = [{
    In the KRNL dialect the reshape op
    doesn't generate a new memory entry and treats a reshape like a cast.
  }];

  let arguments = (ins AnyMemRef:$dest, AnyMemRef:$src, AnyInteger:$size);
}

def KrnlGlobalOp : Op<Krnl_Dialect, "global", [NoSideEffect, MemRefsNormalizable]> {
  let summary = "Krnl global operation";
  let description = [{
    Operation for holding global data values. A global constant can have a
    meaningful name recorded as its `name` attribute. Its content is stored
    in the `value` dense/opaque element attribute.
  }];

  let arguments = (ins AnyAttr:$shape,
    StrAttr:$name, OptionalAttr<AnyAttr>:$value, OptionalAttr<I64Attr>:$offset,
    OptionalAttr<I64Attr>:$alignment);
  let results = (outs AnyTypeOf<[AnyMemRef]>:$output);
}

def KrnlGetRefOp : Op<Krnl_Dialect, "getref", [MemRefsNormalizable]> {
  let summary = "Krnl a MemRef from within another MemRef starting at a specific offset.";
  let description = [{
    Retreieves a MemRef from within another MemRef:

    "krnl.getref"(%memref, %offset)

    The offset is an integer which is used as an index into the input MemRef. It works
    just like an array index.
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$mempool, AnyInteger:$offset,
                   Variadic<Index>:$value);
  let results = (outs AnyTypeOf<[AnyMemRef]>:$output);

  let builders = [
    OpBuilder<(ins "Type":$resultType, "Value":$mempool, "Value":$offset), [{
      build($_builder, $_state, resultType, mempool, offset, {});
    }]>,
    ];

  let extraClassDeclaration = [{
    /// Returns the symbolic operands (the ones in square brackets), which bind
    /// to the symbols of the memref's layout map.
    operand_range getDynamicSizes() {
      return {operand_begin() + 2, operand_end()};
    }
  }];
}

def KrnlBlockOp : Op<Krnl_Dialect, "block"> {
  let summary = "Krnl block operation";
  let description = [{
    Block a single for loop by a constant tile size. For instance,
    $ib, $il = krnl.block %i, 4
    means to block the for loop referred to by %i using a tile size of 4.
  }];

  let arguments = (ins AnyType:$loop, I64Attr:$tile_size);
  let results = (outs AnyType:$loop_block, AnyType:$loop_local);

  let builders = [ OpBuilder<(ins "Value": $loop, "int64_t":$tile_size)> ];
  let assemblyFormat = [{
      $loop $tile_size attr-dict `:` functional-type($loop, results)
  }];
}

def KrnlPermuteOp : Op<Krnl_Dialect, "permute"> {
  let summary = "Krnl permute operation";
  let description = [{
    Permute a set of affine for loops using a specified permutation map.
    The permutation map `map` should be constructed in such way that the
    for loop referred to by the i-th operand to permute operation is sent
    to the `map[i]`-th position.

    For example, the following krnl dialect IR:
    ```
    %ii, %jj, %kk = krnl.define_loops 3
    krnl.permute(%ii, %jj, %kk) [1, 2, 0] : !krnl.loop, !krnl.loop, !krnl.loop
    krnl.iterate (%ii, %jj, %kk) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20, %kk -> %k = 0 to 30) {}
    ```
    will be lowered to:
    ```
    // Referenced by %kk
    affine.for %arg0 = 0 to 30 {
      // Referenced by %ii
      affine.for %arg1 = 0 to 10 {
        // Referenced by %jj
        affine.for %arg2 = 0 to 20 {
        }
      }
    }
    ```

    For a more complicated example, we demonstrate 3-D tiling using krnl.block in
    conjunction with krnl.permute:
    ```
    %ii, %jj, %kk = krnl.define_loops 3
    // Blocking each loop by a factor of 4.
    %ib, %il = krnl.block %ii 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    %jb, %jl = krnl.block %jj 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    %kb, %kl = krnl.block %kk 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    // Move iteration over tile coordinates to be the outer loops and iterateion over
    // the inter-tile elements to be the inner loops.
    krnl.permute(%ib, %il, %jb, %jl, %kb, %kl) [0, 3, 1, 4, 2, 5] : !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
    krnl.iterate(%ib, %il, %jb, %jl, %kb, %kl) with (%ii -> %i = 0 to 1024, %jj -> %j = 0 to 2048, %kk -> %k = 0 to 4096)  {
    }
    ```

    The above IR gets lowered to:
    ```
    affine.for %arg0 = 0 to 1024 step 4 {
      affine.for %arg1 = 0 to 2048 step 4 {
        affine.for %arg2 = 0 to 4096 step 4 {
          affine.for %arg3 = #map0(%arg0) to #map1(%arg0) {
            affine.for %arg4 = #map0(%arg1) to #map1(%arg1) {
              affine.for %arg5 = #map0(%arg2) to #map1(%arg2) {
              }
            }
          }
        }
      }
    }
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$loops, I64ArrayAttr:$map);

  let builders = [ OpBuilder<(ins "ValueRange": $loops, "ArrayRef<int64_t>":$map)> ];
  let assemblyFormat = [{
      `(` $loops `)` $map attr-dict `:` type($loops)
  }];
}

def KrnlUnrollOp : Op<Krnl_Dialect, "unroll"> {
  let summary = "Krnl unroll operation";
  let description = [{
    Fully unroll the specified loops.
    ```
    krnl.unroll %i
    ```
    unrolls the loop referred to by %i fully.
  }];

  let arguments = (ins AnyType:$loop);

  let assemblyFormat = [{
      $loop attr-dict `:` type($loop)
  }];
}

def KrnlDimOp : Op<Krnl_Dialect, "dim", [MemRefsNormalizable]> {
  let summary = "Krnl dimensions operation.";
  let description = [{
    Emits the dimension of a MemRef independent of the MemRef alloc:

    "krnl.dim"(%memref, %index)

    The index identifies the dimension within the shape which is going to be emitted.
    Initially the krnl.dim operation depends on the alloc of the MemRef.
    Unlike the std.dim operation which maintains a dependency on the alloc of the MemRef, the dimension emitted by krnl.dim will not depend on the alloc operation of the MemRef once the krnl.dim operation is lowered.

    Any changes to the original MemRef size after the krnl.dim has been lowered will not be picked up by the emitted dimension. This allows the original MemRef to be safely modified via code transformations or affine map normalization without the risk of changing the value already emitted via krnl.dim.
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$alloc, Index:$index);
  let results = (outs Index:$dimension);
}

def KrnlShapeOp : Op<Krnl_Dialect, "shape", [MemRefsNormalizable]> {
  let summary = "Krnl operation to retreieve the shape of a MemRef.";
  let description = [{
    Extracts the shape of a MemRef:
    ```
      "krnl.shape"(%memref)
    ```
    The return result is of `shape.type`.
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$alloc);
  let results = (outs AnyTypeOf<[AnyMemRef]>:$shape);
}

def KrnlErfOp : Op<Krnl_Dialect, "erf"> {
  let summary = "Krnl erf scalar operation";
  let description = [{
    Krnl erf scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);
}

def KrnlAcosOp : Op<Krnl_Dialect, "acos"> {
  let summary = "Krnl acos scalar operation";
  let description = [{
    Krnl acos scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);
}

def KrnlAcoshOp : Op<Krnl_Dialect, "acosh"> {
  let summary = "Krnl acosh scalar operation";
  let description = [{
    Krnl acosh scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);
}

def KrnlAsinOp : Op<Krnl_Dialect, "asin"> {
  let summary = "Krnl asin scalar operation";
  let description = [{
    Krnl asin scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);
}

def KrnlAsinhOp : Op<Krnl_Dialect, "asinh"> {
  let summary = "Krnl asinh scalar operation";
  let description = [{
    Krnl asinh scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);
}

def KrnlAtanOp : Op<Krnl_Dialect, "atan"> {
  let summary = "Krnl atan scalar operation";
  let description = [{
    Krnl atan scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);
}

def KrnlAtanhOp : Op<Krnl_Dialect, "atanh"> {
  let summary = "Krnl atanh scalar operation";
  let description = [{
    Krnl atanh scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);
}

def KrnlTanOp : Op<Krnl_Dialect, "tan"> {
  let summary = "Krnl tan scalar operation";
  let description = [{
    Krnl tan scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);
}


def KrnlLoadOp : Op<Krnl_Dialect, "load",
  [TypesMatchWith<"result type matches element type of 'memref'",
                  "memref", "result",
                  "$_self.cast<MemRefType>().getElementType()">,
                  MemRefsNormalizable]> {
  let summary = "A Krnl operation to load data from the memref.";

  let description = [{
    The `krnl.load` op reads an element from a memref specified by an index
    list. The output of load is a new value with the same type as the elements
    of the memref. The arity of indices is the rank of the memref (i.e., if the
    memref loaded from is of rank 3, then 3 indices are required for the load
    following the memref identifier).
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference to load from",
                           [MemRead]>:$memref,
                       Variadic<Index>:$indices);
  let results = (outs AnyType:$result);

  let builders = [
    OpBuilder<(ins "Value":$memref, CArg<"ValueRange", "{}">:$indices), [{
      auto memrefType = memref.getType().cast<MemRefType>();
      $_state.addOperands(memref);
      $_state.addOperands(indices);
      $_state.types.push_back(memrefType.getElementType());
    }]>];

  let assemblyFormat = [{$memref `[` $indices `]` attr-dict `:` type($memref)}];

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return getMemRef().getType().cast<MemRefType>();
    }

    operand_range getIndices() { return {operand_begin() + 1, operand_end()}; }
  }];
}

def KrnlStoreOp : Op<Krnl_Dialect, "store",
     [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                     "memref", "value",
                     "$_self.cast<MemRefType>().getElementType()">,
                     MemRefsNormalizable]> {
  let summary = "A Krnl operation to store data to the memref.";
  let description = [{
    The `krnl.store` stores a value to a memref location given by indices. The
    value stored should have the same type as the elemental type of the memref.
    The number of arguments provided within brackets need to match the rank of
    the memref.
  }];

  let arguments = (ins AnyType:$value,
                       Arg<AnyMemRef, "the reference to store to",
                           [MemWrite]>:$memref,
                       Variadic<Index>:$indices);

  let builders = [
    OpBuilder<(ins "Value":$valueToStore, "Value":$memref), [{
      $_state.addOperands(valueToStore);
      $_state.addOperands(memref);
    }]>];

  let assemblyFormat = [{
    $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
  }];

  let extraClassDeclaration = [{
      Value getValueToStore() { return getOperand(0); }

      Value getMemRef() { return getOperand(1); }
      void setMemRef(Value value) { setOperand(1, value); }
      MemRefType getMemRefType() {
        return getMemRef().getType().cast<MemRefType>();
      }

      operand_range getIndices() {
        return {operand_begin() + 2, operand_end()};
      }
  }];
}

def KrnlMovableOp : Op<Krnl_Dialect, "movable", [ImplicitKrnlTerminator]> {
  let summary = "Krnl movable operation";
  let description = [{
     Encapsulates a list of operations, which should be moved under a newly lowered
     affine for operation eventually, but cannot presently because the destination
     affine for operation is not materialized yet.

     This operation is automatically generated by the lowering of Krnl to affine dialect
     to assist with maintaining the relative positioning of loop and inner-loop statements.
     This construct is particularly helpful, for example, for lowering statements that
     are nested imperfectly between an "eager" and a "lazy" loop.
  }];

  let regions = (region AnyRegion:$region);

  let assemblyFormat = [{
      $region attr-dict
  }];

}

def KrnlGetInductionVariableValueOp : Op<Krnl_Dialect, "get_induction_var_value"> {
  let summary = "Krnl ";
  let description = [{
     Krnl operation to convert loop references to corresponding induction
     variable values. This is useful for accessing optimized loop induction
     variables, as they are not otherwise accessible during Krnl Dialect.

     For example, this operation can be applied to loop references corresponding to
     inter-tile iterations. The return values will be the starting index of the
     current tile being iterated over.
  }];

  let arguments = (ins Variadic<AnyType> : $loops);
  let results = (outs Variadic<AnyType> : $ind_var_vals);

  let builders = [ OpBuilder<(ins "ValueRange": $loops)>];

  let assemblyFormat = [{
      `(` $loops `)` attr-dict `:` functional-type($loops, results)
  }];

}

// =============================================================================

def KrnlVectorTypeCastOp : Op<Krnl_Dialect, "vector_type_cast", [NoSideEffect,
    MemRefsNormalizable, DeclareOpInterfaceMethods<CastOpInterface>, ViewLikeOpInterface]> {
  let summary = "vector type cast operation";
  let description = [{
    The "vector_type_cast" operation converts a memref from an non-vector
    element type to another memref of a vector elemental type while not changing
    the source memref's element type. The last dimension size of the source
    dimension is divided (floor division) by the vector size to obtain the
    corresponding dimension for target memref type.

    %MV = vector_type_cast %M : memref<64x16xf32> to memref<64x2xvector<8xf32>>
    %AV = vector_type_cast %A : memref<?x?xf32> to memref<?x?xvector<8xf32>>
  }];

  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$result);

  let hasFolder = 1;
  let builders = [ OpBuilder<(ins "Value": $source, "int64_t": $vectorLen)> ];

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `to` type($result)
  }];

  let extraClassDeclaration = [{
    /// The result of a vector_type_cast is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }

    /// Return the view source.
    Value getViewSource() { return source(); }
  }];
}

// =============================================================================

def KrnlSpecializedKernel : Op<Krnl_Dialect, "specialized_kernel",
                            [DeclareOpInterfaceMethods<SpecializedKernelOpInterface>]> {
  let summary = "Krnl specialized kernel op";
  let description = [{
    Krnl operation to convert.
  }];

  let arguments = (ins Variadic<AnyType> : $loops);

  let assemblyFormat = [{
      `(` $loops `)` attr-dict `:` type($loops)
  }];
}

// =============================================================================

def KrnlMatMulOp : Op<Krnl_Dialect, "matmul", [AttrSizedOperandSegments, 
       DeclareOpInterfaceMethods<SpecializedKernelOpInterface>, MemRefsNormalizable]> {
  let summary = "Matmul operation for a single pannel.";
  let description = [{
    Perform a matrix multiplication AA * BB + CC with sizes [IxK] * [KxJ] + [IxJ].
    The original matrices AA, BB, and CC can be buffered in buffered arrays
    which may be padded. The original matrices and the padded array might
    have a higher rank than 2, but the actual matrix multiplication operation
    only deal with the innermost 2 ranks of the matrices to perform its matrix
    multiplication operations.

    The computations may also compute only a sub-tile of the buffered arrays.
    This region is depicted using stars '*' below.

    All indices passed to this operation are the global indices in the original
    computation, so as to better know if we have boundary conditions.
    
    ORIGINAL ARRAY: if AA: *xIxK; if BB: *xKxJ; if CC: *xI*J).
    BUFFER ARRAYS: denotated as A, B, and C. Note that this operation does
      not require the use of buffers arrays. If none are used, then A=AA,
      B=BB, C=CC. If buffers are used, it is the responsability of the caller
      to properly fill the buffers with the appropriate data. Buffers are
      typically used for cache tiling.

     ORIGINAL ARRAY
     -------------------------------------------------
     |                                               ]
     |                                               ]
     |             buffer array       buffer pad     ]
     |            (3)---------------- ++++           ]
     |             |                 |   +           ]
     |             |     (1)****     |   +           ]
     |             |      *    *     |   +           ]
     |             |      *    *     |   +           ]
     |             |      ****(5)    |   +           ]
     |             |                 |   +           ]
     |             |                 |   +           ]
     |             ------------------|   +           ]
     |             +                     +           ]
     |             +++++++++++++++++++++(4)          ]
     |                                               ]
     -----------------------------------------------(2)

    (1) iGlobalIndexComputeStart/jGlobalIndexComputeStart/
        kGlobalIndexComputeStart, required, global 1D indices.
    (2) iGlobalUB/jGlobalUB/jGlobalUB, required, global 1D indices.
    (3) aGlobalIndexMemStart/bGlobalIndexMemStart/cGlobalIndexMemStart,
        required, global nD indices with the same rank as the buffers
        A, B, and C.
    (4) aTileSize/bTileSize/cTileSize, required when padding, 2D sizes.
    (5) computeTileSizes, required when tiled computation within buffer,
        3D sizes (I, J, K).

    The iGlobalIndexComputeStart/jGlobalIndexComputeStart/
    kGlobalIndexComputeStart (1) indicate the global indices of the
    first element of a tile to be computed in the original computations.

    The iGlobalUB/jGlobalUB/jGlobalUB (2) indicate the global upper bounds
    in the original computations.

    We provide 3 buffers for matrix multipy: A, B, and C. For each buffer,
    we indicate the global indices pointing the beginning of the buffer:
    aGlobalIndexMemStart, bGlobalIndexMemStart, and cGlobalIndexMemStart (3).
    If no buffers are used, i.e. the computation starts directly in the
    orginal memory, the global index is 0. If a buffer for AA is used to
    put data into it starting at indices [i1, k1], where i1 & k1 are the
    global indices in the original computations, then aGlobalIndexMemStart0
    and aGlobalIndexMemStart1 are i1 & k1, respectively.

    If the A, B, or C buffers are larger than the actual data tile they
    contain (see copy_to_tile_buffer), then the actual tile size must be
    given using an optional attribute: aTileSize, bTileSize, or cTileSize (4).
    These optional tile size have a rank of 2, and their values must be
    equal or smaller than their corresponding buffer memrefs.

    If the computation are further tiled with respect to the size of the
    buffers A, B, or C, then the actual computation tile is given by
    the optional tile attribute computeTileSize (5). Its rank is 3, for the
    I, J, and K dimension. The actual A, B, and C buffer tile size
    (possibly specified by the optional parameters) must be a multiple of
    the I, J, and K computeTileSizes, in their respective
    dimensions (A: [IxK], B: [KxJ], C: [IxJ]).

    Note that the buffers A, B, and C can be of higher dimensionality than
    the traditional 2D mentioned up to now, because of broadcasting rules.
    At this time, we only support broadcast of arrays having ranks of 2 or
    more. Because of the broadcast rules, the higher dimenstions have a
    constant index during one matrix multiply. These fixed indices are
    given as prefix dimensions in the starting indices for AA, BB, and CC
    as described above. E.g. if AA has a rank of 3, and BB has a rank of 2,
    the starting indices for AA are [d, i1, k1] where i1 and k1 are as
    above, and d is index pointing to the current instance of the IxK
    AA matrix to be computed. B start indices would be unchanged at [k1, j1].

    Simdize is used to state if simdization is requested.
    Unrolling is used to unroll and jam loops as warrented.

    Below is an example calculating a matrix multiply with pre-zeroed
    C matrix with the sizes below.

    %A: memref<40x60xf32>, %B: memref<60x80xf32>, %C: memref<40x80xf32>

    // 3 tiled loops.
    %ii, %jj, %kk = krnl.define_loops 3
    %ib, %il = krnl.block %ii 10 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    %jb, %jl = krnl.block %jj 8 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    %kb, %kl = krnl.block %kk 10 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    // 3 subtiles.
    %ilb, %ill = krnl.block %il 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    %jlb, %jll = krnl.block %jl 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    %klb, %kll = krnl.block %kl 5 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
    // Permute.
    krnl.permute(%ib, %ilb, %ill, %jb, %jlb, %jll, %kb, %klb, %kll)
        [0, 3, 6, 1, 4, 7, 2, 5, 8] :
        !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop,
        !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop
    // Outer 2 for i, j.
    krnl.iterate(%ib, %jb) with (%ii -> %i = 0 to 40,
                                 %jj -> %j = 0 to 80,
                                 %kk -> %k = 0 to 60) {
        %i1, %j1 = krnl.get_induction_var_value(%ib, %jb) :
          (!krnl.loop,!krnl.loop) -> (index, index)
        // Fill C buffer.
        %Cbuff = alloca(): memref<10x8xf32>  // n x m_simd
        krnl.copy_to_tile_buffer %Cbuff, %C[%i1, %j1], %f0 :
          memref<10x8xf32>, memref<40x80xf32>
        // Outer 1 for k.
        krnl.iterate(%kb) with () {
            %k1 = krnl.get_induction_var_value(%kb) : (!krnl.loop) -> (index)
            // Fill A and B buffer
            %Abuff = alloca(): memref<10x10xf32> // i x k
            %Bbuff = alloca(): memref<10x8xf32>  // k x j_simd
            krnl.copy_to_tile_buffer %Abuff, %A[%i1, %k1], %f0 :
              memref<10x10xf32>, memref<40x60xf32>
            krnl.copy_to_tile_buffer %Bbuff, %B[%k1, %j1], %f0 :
              memref<10x8xf32>, memref<60x80xf32>

            // Inner iterations for subtiles.
            krnl.iterate(%ilb, %jlb, %klb) with () {
                %i2, %j2, %k2 = krnl.get_induction_var_value(%ilb, %jlb, %klb) :
                (!krnl.loop,!krnl.loop,!krnl.loop) -> (index,index,index)

                krnl.matmul %Abuff[%i1, %k1], %Bbuff[%k1, %j1], %Cbuff[%i1, %j1],
                    (%ill, %jll, %kll), (%i2, %j2, %k2), (%c40, %c80, %c60)
                    { computeTileSize=[5,4,5], simdize=false, unroll=false } :
                    memref<10x10xf32>, memref<10x8xf32>, memref<10x8xf32>,
                    (!krnl.loop,!krnl.loop,!krnl.loop)
            }
        }
        // Copy back the data into C.
        krnl.copy_from_tile_buffer %Cbuff, %C[%i1, %j1] :
          memref<10x8xf32>, memref<40x80xf32>
    }

    Note that code is simdized along the J dim (last dim of B and C matrices).
    For simd to be enabled, the simdized flag must be set to true, and the
    following condition must be true:
    1) The vector length is the second entry of (i, j, k) compute tile size.
       The vector length must be a compile time constant.
  }];

  let arguments = (ins 
    // Buffer/memory used and indices that correspoond to the begining 
    // of each buffer/memory.
    Arg<AnyMemRef, "Mult A [NxK]", [MemRead]>:$A,
    Variadic<Index>: $aGlobalIndexMemStart,
    Arg<AnyMemRef, "Mult B [KxM]", [MemRead]>:$B,
    Variadic<Index>: $bGlobalIndexMemStart,
    Arg<AnyMemRef, "Add into C [NxM]", [MemRead, MemWrite]>:$C,
    Variadic<Index>: $cGlobalIndexMemStart,
    // Loops involved, and indices pointing to the start of the compute
    // subtile. 
    Variadic<AnyType> : $loops,
    Index: $iGlobalIndexComputeStart,
    Index: $jGlobalIndexComputeStart,
    Index: $kGlobalIndexComputeStart,
    // Global upper bound for the entire computations.
    Index: $iGlobalUB, Index: $jGlobalUB, Index: $kGlobalUB,
    // Optional argument to override defaults.
    OptionalAttr<I64ArrayAttr>:$computeTileSize,
    OptionalAttr<I64ArrayAttr>:$aTileSize,
    OptionalAttr<I64ArrayAttr>:$bTileSize,
    OptionalAttr<I64ArrayAttr>:$cTileSize,
    // Optimizations.
    DefaultValuedAttr<BoolAttr, "true">:$simdize,
    DefaultValuedAttr<BoolAttr, "true">:$unroll,
    DefaultValuedAttr<BoolAttr, "false">:$overcompute);

  let builders = [
    // include all parameters
    OpBuilder<(ins
      "Value": $A, "ValueRange": $aStart, // Both same rank
      "Value": $B, "ValueRange": $bStart, // Both same rank.
      "Value": $C, "ValueRange": $cStart, // Both same rank.
      "ValueRange": $loops, // Rank 3: (i,j,k).
      "Value": $iGlobalIndexComputeStart,
      "Value": $jGlobalIndexComputeStart,
      "Value": $kGlobalIndexComputeStart,
      "Value": $iGlobalUB, "Value": $jGlobalUB, "Value": $kGlobalUB,
      "ArrayRef<int64_t>": $computeTileSize, // Rank 3: (i,j,k).
      "ArrayRef<int64_t>": $aTileSize, // Rank 2: (rightmost 2 dims).
      "ArrayRef<int64_t>": $bTileSize, // Rank 2: (rightmost 2 dims).
      "ArrayRef<int64_t>": $cTileSize, // Rank 2: (rightmost 2 dims).
      "bool": $simdize,
      "bool": $unroll,
      "bool": $overcompute)>,
    // use default for all tile sizes
    OpBuilder<(ins
      "Value": $A, "ValueRange": $aStart, // Both same rank.
      "Value": $B, "ValueRange": $bStart, // Both same rank.
      "Value": $C, "ValueRange": $cStart, // Both same rank.
      "ValueRange": $loops, // Rank 3: (i,j,k).
      "Value": $iGlobalIndexComputeStart,
      "Value": $jGlobalIndexComputeStart,
      "Value": $kGlobalIndexComputeStart,
      "Value": $iGlobalUB, "Value": $jGlobalUB, "Value": $kGlobalUB,
      "bool": $simdize,
      "bool": $unroll,
      "bool": $overcompute)>
    ];

  let hasVerifier = 1;

  let assemblyFormat = [{
    $A `[` $aGlobalIndexMemStart `]` `,`
    $B `[` $bGlobalIndexMemStart `]` `,`
    $C `[` $cGlobalIndexMemStart `]` `,`
    `(` $loops `)` `,`
    `(` $iGlobalIndexComputeStart `,` $jGlobalIndexComputeStart `,`
        $kGlobalIndexComputeStart `)` `,`
    `(` $iGlobalUB `,` $jGlobalUB `,` $kGlobalUB `)` 
    attr-dict `:` type($A) `,` type($B)`,` type($C) `,` `(` type($loops) `)`
  }];
}

def KrnlCopyToBufferOp : Op<Krnl_Dialect, "copy_to_tile_buffer", [
    TypesMatchWith<"type of 'padValue' matches element type of 'source'",
                  "source", "padValue",
                  "$_self.cast<MemRefType>().getElementType()">,
    TypesMatchWith<"type of 'padValue' matches element type of 'buffer'",
                  "buffer", "padValue",
                   "$_self.cast<MemRefType>().getElementType()">,
    MemRefsNormalizable]> {
  let summary = "Copy to buffer.";
  let description = [{
    Operation that copy a source memory to a buffer memory.
    Starts indicate where the source data starts to come from within
    the source memory. Start values must be at multiples of buffer size
    in all dimensions. The buffer rank and dimensions are compile time
    constants.

    The buffer will be entirely filled with the source data. By default,
    the amount of data to copy is given by the size of the buffer.
    In some cases, we may want to oversize a buffer for better cache,
    simd, or loop unroll and jam reasons. If that is the case, the
    actual tile size of the data to be copied over is given by an
    optional tileSize attribute. This attributes has the same rank as
    the buffer size, and each dimension must be smaller or equal to
    the actual buffer size.
    
    If there is not enough data in the source memory to fill the buffer,
    because the operation reaches the upper bounds of the source memory,
    several actions may happen.

    * If padToNext attribute is given, the pad value will be copied from
      the last source data of to the next index for which index modulo padToNext
      is zero, i.e. to the end of a "cache line" of side padToLine. Pad
      of 1 means no padding, pad of buffer size means fully pad the buffer.
      Default is no padding (1). PadValue is used to initialized the padded
      areas.

    * If overreadToNext attribute is given, the copy may read source past
      its upperbound value. This enable optimized code, e.g. using SIMD
      read operations even if going past the last value of the source
      memory, or unrolling and jaming copy loops to reduce memory latency.
      overreadToNext is expressed like padToNext: value of 1 means no
      reading past boundary; value of buffer size enables reading
      as many additional sourve value as needed to fill the full
      buffer. Default is buffer-size.

    padToNext and overreadToNext are of the same rank as source and memory
    memrefs.
  }];

  let arguments = (ins 
    Arg<AnyMemRef, "buffer", [MemWrite]>:$buffer, // Buffer.
    Arg<AnyMemRef, "memory", [MemRead]>:$source, // Source, possibly tranposed.
    Variadic<Index>:$starts, // Rank of source, possibly transposed.
    AnyType: $padValue, // Rank of bufffer.
    OptionalAttr<I64ArrayAttr>:$tileSize, // Rank of bufffer.
    OptionalAttr<I64ArrayAttr>:$padToNext, // Rank of bufffer.
    DefaultValuedAttr<BoolAttr, "false">:$transpose); // Transposed or not.

  let builders = [ 
    OpBuilder<(ins "Value": $buffer, "Value": $source, 
      "ValueRange": $starts, "Value": $padValue, 
      "ArrayRef<int64_t>": $tileSize, "ArrayRef<int64_t>": $padToNext,
      "bool": $transpose)>,
    OpBuilder<(ins "Value": $buffer, "Value": $source, 
      "ValueRange": $starts, "Value": $padValue,
      "bool": $transpose)>
  ];

  let hasVerifier = 1;

  let assemblyFormat = [{
    $buffer `,` $source `[` $starts `]` `,`  $padValue  attr-dict
     `:` type($buffer) `,` type($source)
  }];
}

def KrnlCopyFromBufferOp : Op<Krnl_Dialect, "copy_from_tile_buffer",
    [MemRefsNormalizable]> {
  let summary = "Copy from buffer.";
  let description = [{
    Operation that copy a destination memory from a buffer memory.
    Starts indicate where the buffer data starts to go into the destination
    memory. Start values must be at multiples of buffer size in all dimensions.
    The buffer rank and dimensions are compile time constants.
    
    If the buffer was oversized with respect of the actual data contained
    in the tile, the actual tile size can be given using the tileSize
    optional attribute. This attributes has the same rank as the buffer size,
    and each dimension must be smaller or equal to the actual buffer size.
  }];

  let arguments = (ins Arg<AnyMemRef, "buffer", [MemRead]>:$buffer,
    Arg<AnyMemRef, "dest", [MemWrite]>:$dest,
    Variadic<Index>:$starts, // Rank of destination.
    OptionalAttr<I64ArrayAttr>:$tileSize); // Rank of buffer.

  let builders = [ 
    OpBuilder<(ins "Value": $buffer, "Value": $dest, 
      "ValueRange": $starts, "ArrayRef<int64_t>": $tileSize)>,
    OpBuilder<(ins "Value": $buffer, "Value": $dest, "ValueRange": $starts)>
  ];

  let hasVerifier = 1;

  let assemblyFormat = [{
    $buffer `,` $dest `[` $starts `]`  attr-dict `:` type($buffer) `,` type($dest)
  }];
}

def KrnlInstrumentOp : Op<Krnl_Dialect, "runtime_instrument",
    []> {
  let summary = "instrumentation point.";
  let description = [{
    Operation that invokes the runtime instrument utility.
    May be used for gdb.
  }];

  let arguments = (ins I64Attr:$opID, I64Attr:$tag);

  let builders = [ OpBuilder<(ins "Operation *": $op, "int ": $tag)> ];
}

def KrnlMemsetOp : Op<Krnl_Dialect, "memset", [MemRefsNormalizable,
    TypesMatchWith<"type of 'value' matches element type of 'dest'",
                   "dest", "value",
                   "$_self.cast<MemRefType>().getElementType()">]> {
  let summary = "Set buffer to a given value.";
  let description = [{
    Krnl operation that set buffer to a given value.
  }];

  let arguments = (ins AnyMemRef:$dest, AnyType: $value); 

  let assemblyFormat = [{ $dest `,` $value attr-dict `:` type($dest) }];
}

def KrnlStrlenOp : Op<Krnl_Dialect, "strlen", [NoSideEffect]> {
  let summary = "Compute the length of a string.";
  let description = [{
    Krnl operation that computes the length of a string.
  }];

  let arguments = (ins StringType:$str);
  let results = (outs I64:$res);
}

def KrnlStrncmpOp : Op<Krnl_Dialect, "strncmp", [NoSideEffect]> {
  let summary = "Perform string comparison up to N bytes.";
  let description = [{
    Krnl operation that performs a string comparison up to N bytes.
  }];

  let arguments = (ins StringType:$str1, StringType:$str2, I64:$len);
  let results = (outs I32:$res);
}

def KrnlRandomNormalOp : Op<Krnl_Dialect, "random_normal",
    [MemRefsNormalizable]> {
  let summary = "Generate a random normal tensor.";
  let description = [{
    Operation that generates a random normally distributed tensor.
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$output,
    Index:$numberOfValues,
    AnyFloat:$mean,
    AnyFloat:$scale,
    AnyFloat:$seed);
}

def KrnlFindIndexOp : Op<Krnl_Dialect, "find_index",
    [NoSideEffect, MemRefsNormalizable]> {
  let summary = "Retrieve an index into a perfect hash table described by G and V.";
  let description = [{
    This operation can be used to generate a call to a runtime function which, 
    given two arrays of int32_t values (G and V), whih are used to represent a perfect 
    hash table for a dictionary, returns the index corresponding to the input value.
    The index returned is valid only if 'input' is in the dictionary described by G and V.
  }];

  let arguments = (ins AnyTypeOf<[StringType, I64]>:$input, I32MemRef:$G, I32MemRef:$V, I32:$len);
  let results = (outs Index:$index);
}

def KrnlPrintTensorOp : Op<Krnl_Dialect, "print_tensor", [MemRefsNormalizable]> {
  let summary = "Print a tensor.";
  let description = [{
    This operation can be used to generate a call to a runtime function which prints a tensor.
  }];

  let arguments = (ins StrAttr:$msg, AnyMemRef:$input);
} 

def KrnlPrintOp : Op<Krnl_Dialect, "print", [MemRefsNormalizable]> {
  let summary = "Print a value.";
  let description = [{
    This operation can be used to print the input value. The user needs to provide a 
    format string (à la printf) to specify how to print the input value. 
    If the input value is not specified the operator will print the format string.
  }];

  let arguments = (ins StrAttr:$format, Optional<AnyType>:$input);
} 
