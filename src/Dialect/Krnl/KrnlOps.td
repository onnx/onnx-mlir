//===------------------ KrnlOps.td - MLIR Operations ---------*- tablegen -===//
//
// Copyright 2019-2020 The IBM Research Authors.
//
// =============================================================================
//
// This file contains TableGen definition of krnl operations.
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Shape/IR/ShapeBase.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "src/Interface/SpecializedKernelOpInterface.td"

def Krnl_Dialect : Dialect {
  let name = "krnl";
  let cppNamespace = "::mlir";
}

// Require regions to have krnl.terminate terminator operation.
def ImplicitKrnlTerminator : SingleBlockImplicitTerminator<"KrnlTerminatorOp">;

def KrnlDefineLoopsOp : Op<Krnl_Dialect, "define_loops"> {
  let summary = "define_loops operation";
  let description = [{
    The "krnl.define_loops" operation is used to define input loops,
    those are the for loops appearing in the input program that we
    intend to optimize.
  }];

  let arguments = (ins);
  let results = (outs Variadic<AnyType>);
  let skipDefaultBuilders = 1;
  let builders = [ OpBuilderDAG<(ins "int64_t":$num_loops)> ];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

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

def KrnlIterateOp : Op<Krnl_Dialect, "iterate", [ImplicitKrnlTerminator]> {
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
  let builders = [ OpBuilderDAG<(ins "KrnlIterateOperandPack":$operandPack)> ];

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

    let printer = [{ return ::print(p, *this); }];
    let parser = [{ return ::parse$cppClass(parser, result); }];
    let verifier = [{ return ::verify(*this); }];
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

  // No custom parsing/printing form.
  let parser = ?;
  let printer = ?;

  // Fully specified by traits.
  let verifier = ?;
}

def KrnlEntryPointOp : Op<Krnl_Dialect, "entry_point"> {
  let summary = "Indicate ONNX entry point";
  let description = [{The "krnl.entry_point" function indicates the main entry
                           point of ONNX model.}];
  let builders = [ OpBuilderDAG<(ins "SymbolRefAttr":$funcAttr, "IntegerAttr":$numInputs,
                                     "IntegerAttr":$numOutputs)> ];

  let extraClassDeclaration = [{
    static StringRef getEntryPointFuncAttrName() { return "func"; }
    static StringRef getNumInputsAttrName() { return "numInputs"; }
    static StringRef getNumOutputsAttrName() { return "numOutputs"; }
  }];

  // No custom parsing/printing form.
  let parser = ?;
  let printer = ?;
}

def KrnlMemcpyOp : Op<Krnl_Dialect, "memcpy", [MemRefsNormalizable]> {
  let summary = "Krnl memcpy operation";
  let description = [{
    In the KRNL dialect the reshape op
    doesn't generate a new memory entry and treats a reshape like a cast.
  }];

  let arguments = (ins AnyMemRef:$dest, AnyMemRef:$src, AnyInteger:$size);

  let parser = ?;
  let printer = ?;
}

def KrnlGlobalOp : Op<Krnl_Dialect, "global"> {
  let summary = "Krnl global operation";
  let description = [{
    Operation for holding global data values. A global constant can have a
    meaningful name recorded as its `name` attribute. Its content is stored
    in the `value` dense element attribute. Alternatively, if the constants
    are packed together, `offset` records the byte offset in the global
    constant pool from which the current constant is to be read.
  }];

  let arguments = (ins AnyAttr:$shape,
    StrAttr:$name, OptionalAttr<AnyAttr>:$value, OptionalAttr<I64Attr>:$offset);
  let results = (outs AnyTypeOf<[AnyMemRef]>:$output);

  let parser = ?;
  let printer = ?;
}

def KrnlPackedConstantOp : Op<Krnl_Dialect, "packed_const"> {
  let summary = "Krnl packed constant operation";
  let description = [{
    Operation for holding packed constants.
  }];

  let arguments = (ins I64Attr:$size_in_bytes,
                       BoolAttr:$is_le,
                       OptionalAttr<AnyIntElementsAttr<8>>:$value,
                       OptionalAttr<StrAttr>:$file_name);
  let results = (outs I64:$output);

  let extraClassDeclaration = [{
    // The *path* to the file storing the constant pack on disk is encoded
    // as a global variable in the LLVM module of the lowered model.
    // getConstPackFilePathSymbolName returns the name of this symbol;
    // getConstPackFilePathStrLenSymbolName returns the name of the symbol holding
    // a constant value equal to the length of the file path.
    static StringRef getConstPackFilePathSymbolName() { return "constPackFilePath"; }
    static StringRef getConstPackFilePathStrLenSymbolName() { return "constPackFilePathStrLen"; }

    // The *name* of the file storing the constant pack is also recorded for
    // convenience. Similarly, getConstPackFileNameSymbolName and
    // getConstPackFileNameStrLenSymbolName returns records the symbol holding
    // the string constant representing the filename and the length of this
    // string constant.
    static StringRef getConstPackFileNameSymbolName() { return "constPackFileName"; }
    static StringRef getConstPackFileNameStrLenSymbolName() { return "constPackFileNameStrLen"; }

    // We record whether the constant pack is stored in LE byte order as a
    // int8 symbol; it is to be interpreted as a boolean switch - with 0
    // meaning that the constant pack is not stored in LE byte order and
    // non-0 values meaning that it is stored in LE byte order.
    static StringRef getConstPackIsLESymbolName() { return "constPackIsLE"; }
    // The name of a function we call to read packed constants embedded within
    // the current binary executable/library, or in the case of unsupported platform,
    // from a binary constant pack file.
    static StringRef getEmbeddedDataLoaderMethodName() {
      return "getEmbeddedConstPool";
    }
  }];
  let parser = ?;
  let printer = ?;
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
    OpBuilderDAG<(ins "Type":$resultType, "Value":$mempool, "Value":$offset), [{
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

  let parser = ?;
  let printer = ?;
}

def KrnlBlockOp : Op<Krnl_Dialect, "block"> {
  let summary = "Krnl block operation";
  let description = [{
    Block a single for loop by a constant tile size. For instance,
    $ib, $il = krnl.block %i, 4
    means to block the for loop referred to by %i using a tile size of 4.
  }];

  let arguments = (ins
    AnyType:$loop, I64Attr:$tile_size);
  let results = (outs AnyType:$loop_block, AnyType:$loop_local);
  let builders = [ OpBuilderDAG<(ins "Value": $loop, "int64_t":$tile_size)> ];
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
  let results = (outs);
  let builders = [ OpBuilderDAG<(ins "ArrayRef<Value>": $loops, "ArrayRef<int64_t>":$map)> ];
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
  let results = (outs);
  let assemblyFormat = [{
      $loop attr-dict `:` type($loop)
  }];
}

def KrnlDimOp : Op<Krnl_Dialect, "dim"> {
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

  let parser = ?;
  let printer = ?;
}

def KrnlShapeOp : Op<Krnl_Dialect, "shape"> {
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

  let parser = ?;
  let printer = ?;
}

def KrnlDummyCastOp : Op<Krnl_Dialect, "dummy_cast"> {
  let summary = "A dummy Krnl operation to perform type casting.";
  let hasCanonicalizer = 1;

  let description = [{
    Krnl operation to perform dummy type casting to remove the type
    dependencies existing between lowering of multiple IR objects.
  }];

  let builders = [ OpBuilderDAG<(ins "Value":$in, "Type":$outType)> ];

  let arguments = (ins AnyTypeOf<[AnyMemRef, AnyTensor]>:$in);
  let results = (outs AnyTypeOf<[AnyMemRef, AnyTensor]>:$out);

  let assemblyFormat = [{
          $in attr-dict `:` functional-type($in, results)
  }];
}


def KrnlErfOp : Op<Krnl_Dialect, "erf"> {
  let summary = "Krnl erf scalar operation";
  let description = [{
    Krnl erf scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);

  let parser = ?;
  let printer = ?;
}

def KrnlAcosOp : Op<Krnl_Dialect, "acos"> {
  let summary = "Krnl acos scalar operation";
  let description = [{
    Krnl acos scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);

  let parser = ?;
  let printer = ?;
}

def KrnlAcoshOp : Op<Krnl_Dialect, "acosh"> {
  let summary = "Krnl acosh scalar operation";
  let description = [{
    Krnl acosh scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);

  let parser = ?;
  let printer = ?;
}

def KrnlAsinOp : Op<Krnl_Dialect, "asin"> {
  let summary = "Krnl asin scalar operation";
  let description = [{
    Krnl asin scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);

  let parser = ?;
  let printer = ?;
}

def KrnlAsinhOp : Op<Krnl_Dialect, "asinh"> {
  let summary = "Krnl asinh scalar operation";
  let description = [{
    Krnl asinh scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);

  let parser = ?;
  let printer = ?;
}

def KrnlAtanOp : Op<Krnl_Dialect, "atan"> {
  let summary = "Krnl atan scalar operation";
  let description = [{
    Krnl atan scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);

  let parser = ?;
  let printer = ?;
}

def KrnlAtanhOp : Op<Krnl_Dialect, "atanh"> {
  let summary = "Krnl atanh scalar operation";
  let description = [{
    Krnl atanh scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);

  let parser = ?;
  let printer = ?;
}

def KrnlTanOp : Op<Krnl_Dialect, "tan"> {
  let summary = "Krnl tan scalar operation";
  let description = [{
    Krnl tan scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);

  let parser = ?;
  let printer = ?;
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
    OpBuilderDAG<(ins "Value":$memref, CArg<"ValueRange", "{}">:$indices), [{
      auto memrefType = memref.getType().cast<MemRefType>();
      $_state.addOperands(memref);
      $_state.addOperands(indices);
      $_state.types.push_back(memrefType.getElementType());
    }]>];

  let extraClassDeclaration = [{
    Value getMemRef() { return getOperand(0); }
    void setMemRef(Value value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return getMemRef().getType().cast<MemRefType>();
    }

    operand_range getIndices() { return {operand_begin() + 1, operand_end()}; }
  }];

  let assemblyFormat = [{$memref `[` $indices `]` attr-dict `:` type($memref)}];
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
    OpBuilderDAG<(ins "Value":$valueToStore, "Value":$memref), [{
      $_state.addOperands(valueToStore);
      $_state.addOperands(memref);
    }]>];

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

  let assemblyFormat = [{
    $value `,` $memref `[` $indices `]` attr-dict `:` type($memref)
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

  let arguments = (ins );

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
  let builders = [ OpBuilderDAG<(ins "ArrayRef<Value>": $loops)> ];

  let assemblyFormat = [{
      `(` $loops `)` attr-dict `:` functional-type($loops, results)
  }];

}

// =============================================================================

def KrnlVectorTypeCastOp : Op<Krnl_Dialect, "vector_type_cast", [NoSideEffect]> {
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

  let parser = ?;
  let printer = ?;

  let verifier = [{ return impl::verifyCastOp(*this, areCastCompatible); }];

  let extraClassDeclaration = [{
    /// Return true if `a` and `b` are valid operand and result pairs for
    /// the operation.
    static bool areCastCompatible(Type a, Type b);

    /// The result of a vector_type_cast is always a memref.
    MemRefType getType() { return getResult().getType().cast<MemRefType>(); }
  }];

  let hasFolder = 1;

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `to` type($result)
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
  let results = (outs );

  let assemblyFormat = [{
      `(` $loops `)` attr-dict `:` type($loops)
  }];
}

// =============================================================================

def KrnlMatMulOp : Op<Krnl_Dialect, "matmul", 
      [DeclareOpInterfaceMethods<SpecializedKernelOpInterface>]> {
  let summary = "Matmul operation for a single pannel.";
  let description = [{
    Perform a matrix multiplication A * B + C
    for a small tile A * B + C of sizes
    [IxK] * [KxJ] + [IxJ].

    The i/j/k ComputeStarts indicate the global indices of the first element
    of a tile to be computed in the original computations.
    The i/j/k GlobalUBs indicate the upper bounds in the original computations.

    We provide 3 buffers for matrix multipy: A, B, and C. For each buffer,
    we indicate the global indices pointing the beginning of the buffer.
    If no buffers are used, i.e. the computation starts directly in the orginal
    memory, the global index is 0. If a buffer for A is used to put data into
    it starting at indices [i1, k1], where i1 & k1 are the global indices in
    the original computations, then aMemStart0 and aMemStart1 are i1 & k1,
    respectively.

    If the A, B, or C buffers are larger than the actual data tile they
    contain (see copy_to_tile_buffer), then the actual tile size must be
    given using an optional attribute: ATileSize, BTileSize, or CTileSize.
    These optional tile size have a rank of 2, and their values must be
    equal or smaller than their corresponding buffer memrefs.

    If the computation are further tiled with respect to the size of the
    buffers A, B, or C, then the actual computation tile is given by
    the optional tile attribute computeTileSize. Its rank is 3, for the
    I, J, and K dimension. The actual A, B, and C buffer tile size 
    (possibly specified by the optional parameters) must be a multiple of
    the I, J, and K computeTileSizes, in their respective
    dimensions (A: IxK], B: [KxJ], C: [IxJ]).

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
    krnl.iterate(%ib, %jb) with (%ii -> %i = 0 to 40, %jj -> %j = 0 to 80, %kk -> %k = 0 to 60) {
        %i1, %j1 = krnl.get_induction_var_value(%ib, %jb) : (!krnl.loop,!krnl.loop) -> (index, index)
        // Fill C buffer.
        %Cbuff = alloca(): memref<10x8xf32>  // n x m_simd
        krnl.copy_to_tile_buffer %Cbuff, %C[%i1, %j1], %f0 : memref<10x8xf32>,  memref<40x80xf32>
        // Outer 1 for k.
        krnl.iterate(%kb) with () {
            %k1 = krnl.get_induction_var_value(%kb) : (!krnl.loop) -> (index)
            // Fill A and B buffer
            %Abuff = alloca(): memref<10x10xf32> // n x k
            %Bbuff = alloca(): memref<10x8xf32>  // k x m_simd     
            krnl.copy_to_tile_buffer %Abuff, %A[%i1, %k1], %f0 : memref<10x10xf32>, memref<40x60xf32>
            krnl.copy_to_tile_buffer %Bbuff, %B[%k1, %j1], %f0 : memref<10x8xf32>,  memref<60x80xf32>

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
        krnl.copy_from_tile_buffer %Cbuff, %C[%i1, %j1] : memref<10x8xf32>, memref<40x80xf32>
    }

  }];

  let arguments = (ins 
    // Buffer/memory used and indices that correspoond to the begining 
    // of each buffer/memory.
    Arg<AnyMemRef, "Mult A [NxK]", [MemRead]>:$A,
    Index: $aMemStart0, Index: $aMemStart1,
    Arg<AnyMemRef, "Mult B [KxM]", [MemRead]>:$B,
    Index: $bMemStart0, Index: $bMemStart1,
    Arg<AnyMemRef, "Add into C [NxM]", [MemRead, MemWrite]>:$C,
    Index: $cMemStart0, Index: $cMemStart1,
    // Loops involved, and indices pointing to the start of the compute
    // subtile. 
    Variadic<AnyType> : $loops,
    Index: $iComputeStart, Index: $jComputeStart, Index: $kComputeStart,
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

  let builders = [ OpBuilderDAG<(ins 
      "Value": $A, "Value": $aStart0, "Value": $aStart1,
      "Value": $B, "Value": $bStart0, "Value": $bStart1,
      "Value": $C, "Value": $cStart0, "Value": $cStart1,
      "ArrayRef<Value>": $loops,
      "Value": $iComputeStart, "Value": $jComputeStart, "Value": $kComputeStart, 
      "Value": $iGlobalUB, "Value": $jGlobalUB, "Value": $kGlobalUB,
      "ArrayRef<int64_t>": $computeTileSize,
      "ArrayRef<int64_t>": $aTileSize,
      "ArrayRef<int64_t>": $bTileSize,
      "ArrayRef<int64_t>": $cTileSize, 
      "bool": $simdize,
      "bool": $unroll, 
      "bool": $overcompute 
    )> ];

  let parser = ?;
  let printer = ?;
  let assemblyFormat = [{
    $A `[` $aMemStart0 `,` $aMemStart1 `]` `,`
    $B `[` $bMemStart0 `,` $bMemStart1 `]` `,`
    $C `[` $cMemStart0 `,` $cMemStart1 `]` `,`
    `(` $loops `)` `,`
    `(` $iComputeStart `,` $jComputeStart `,` $kComputeStart `)` `,`
    `(` $iGlobalUB `,` $jGlobalUB `,` $kGlobalUB `)` 
    attr-dict `:` type($A) `,` type($B)`,` type($C) `,` `(` type($loops) `)`
  }];
}

def KrnlCopyToBufferOp : Op<Krnl_Dialect, "copy_to_tile_buffer", [
    TypesMatchWith<"type of 'padValue' matches element type of 'memref'",
                  "memref", "padValue",
                  "$_self.cast<MemRefType>().getElementType()">,
    TypesMatchWith<"type of 'padValue' matches element type of 'bufferMemref'",
                  "bufferMemref", "padValue",
                  "$_self.cast<MemRefType>().getElementType()">,
    SameVariadicOperandSize]> {
  let summary = "Copy to buffer.";
  let description = [{
    Operation that copy a source memory to a buffer memory. Source memory
    and buffer must have the same rank. Starts indicate where the source data
    starts to come from within the source memory. Start values must be at
    multiples of buffer size in all dimensions. The buffer rank and
    dimensions are compile time constants.

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

  let arguments = (ins Arg<AnyMemRef, "buffer", [MemWrite]>:$bufferMemref,
    Arg<AnyMemRef, "memory", [MemRead]>:$memref,
    Variadic<Index>:$starts,
    AnyType: $padValue,
    OptionalAttr<I64ArrayAttr>:$tileSize,
    OptionalAttr<I64ArrayAttr>:$padToNext);

  let builders = [ OpBuilderDAG<(ins "Value": $bufferMemref, "Value": $memref, 
      "ArrayRef<Value>": $starts, "Value": $padValue, 
      "ArrayRef<int64_t>": $tileSize, "ArrayRef<int64_t>": $padToNext
    )> ];

  let parser = ?;
  let printer = ?;
    let assemblyFormat = [{
    $bufferMemref `,` $memref `[` $starts `]` `,`  $padValue  attr-dict
     `:` type($bufferMemref) `,` type($memref)
  }];
}

def KrnlCopyFromBufferOp : Op<Krnl_Dialect, "copy_from_tile_buffer",
    [SameVariadicOperandSize]> {
  let summary = "Copy from buffer.";
  let description = [{
    Operation that copy a destination memory from a buffer memory.  Source
    and buffer memories must have the same rank. Starts indicate where the buffer
    data starts to go into the destination memory. Start values must be at
    multiples of buffer size in all dimensions. The buffer rank and dimensions
    are compile time constants. If the buffer was oversized with respect
    of the actual data contained in the tile, the actual tile size can
    be given using the tileSize optional attribute. This attributes has
    the same rank as the buffer size, and each dimension must be smaller or
    equal to the actual buffer size.
  }];

  let arguments = (ins Arg<AnyMemRef, "buffer", [MemRead]>:$bufferMemref,
    Arg<AnyMemRef, "memory", [MemWrite]>:$memref,
    Variadic<Index>:$starts,
    OptionalAttr<I64ArrayAttr>:$tileSize);

  let builders = [ OpBuilderDAG<(ins "Value": $bufferMemref, "Value": $memref, 
      "ArrayRef<Value>": $starts, "ArrayRef<int64_t>": $tileSize
    )> ];

  let parser = ?;
  let printer = ?;
    let assemblyFormat = [{
    $bufferMemref `,` $memref `[` $starts `]`  attr-dict
     `:` type($bufferMemref) `,` type($memref)
  }];
}

