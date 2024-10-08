// SPDX-License-Identifier: Apache-2.0

//===--------------------- Krnl.td - MLIR Operations ---------*- tablegen -===//
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
include "mlir/Dialect/Bufferization/IR/AllocationOpInterface.td"
include "mlir/Interfaces/CastInterfaces.td"
include "mlir/Interfaces/ControlFlowInterfaces.td"
include "mlir/Interfaces/LoopLikeInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"
include "mlir/Interfaces/ViewLikeInterface.td"
include "mlir/Dialect/Affine/IR/AffineMemoryOpInterfaces.td"
include "src/Interface/SpecializedKernelOpInterface.td"

def Krnl_Dialect : Dialect {
  let name = "krnl";
  let cppNamespace = "::mlir";
  let usePropertiesForAttributes = 0;
  let useDefaultTypePrinterParser = 1;
  let dependentDialects = [
    "affine::AffineDialect",
    "arith::ArithDialect",
    "func::FuncDialect",
    "linalg::LinalgDialect",
    "math::MathDialect",
    "memref::MemRefDialect",
    "scf::SCFDialect",
    "shape::ShapeDialect",
  ];
}

def StringType : Type<CPred<"mlir::isa<krnl::StringType>($_self)">, "string type">;

// Require regions to have krnl.terminate terminator operation.
def ImplicitKrnlTerminator : SingleBlockImplicitTerminator<"KrnlTerminatorOp">;

def KrnlCallOp : Op<Krnl_Dialect, "call",
    [DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]
  > {
  let summary = "call operation";
  let description = [{
    The call operation provides a generic way to replace an ONNX Op with a call
    to an external function at Krnl level.
    `funcName` attributes determines which function to call.
    `parameters` is the inputs to Krnl.Call. It includes the outputs and inputs
    of the ONNX Op. The outputs and inputs are already lowered to MemRefs.
    The external function is assumed NOT to allocate or free any memory.
    'numOfOutput` attribute to tell how manu outputs Memref in parameters.
    mlir::OpTrait::AttrSizedOperandSegments is not used to put outputs and
    inputs into separate variadic parameters because I am thinking of mixing
    the inputs and outpus as required by external library.

    The attributes of the ONNX Op will be copied to KrnlCallOp under the control
    of the user.
    In Krnl To llvm lowering, the parameters and attributes will be lowered to
    parameters of the llvm function call.

    Several builder is defined to help translating an ONNX Op to Krnl.Call.
    User can provides the allocated MemRefs for outputs and the inputs
    separately. The inputs are usually the operands of the ONNX Op.
    The attributes of ONNX Op can be copied or not copied based on a bool
    parameter in the builder. Builder also provide a mechanism for user
    to selectively copy some attributes.

    The krnl.call op will be lowered to llvm at krnl-to-llvm conversion in which
    OMTensor is used as a container for MemRef arguments. Other representation
    of parameters, such as data pointer only, will be supported in future.
  }];

  let arguments = (ins StrAttr:$funcName,
      // Should mlir::OpTrait::AttrSizedOperandSegments be used?
      // The case for krnl.Call is simple.
      DefaultValuedAttr<SI64Attr, "1">:$numOfOutput,
      Variadic<AnyType>:$parameters);

  // Return Value for the Call.
  // No return if the type is NoneType (void in llvm)
  // Only scalar type is supported now.
  // In future, return of memref can be supported with pointer of OMTensor.
  // The returned memref will be created inside the call. 
  let results = (outs Variadic<AnyTypeOf<[AnyFloat, AnyInteger]>>:$returnValue);

  // builders to build KrnlCallOp from op and operands, helping conversion from
  // onnx to krnl.
  // The name of function can be determined by the op name and elemnt type of
  // the return, or given to builder if the simple rule does not work.
  // Attributes of the op will be propagated to KrnlCallOp if the copyAttrs is
  // true. Or the attribute names can be specified.
  let builders = [
      OpBuilder<(ins "std::string":$funcNameStr, "int64_t":$numOfOutput, "mlir::ValueRange":$operands)>,
      OpBuilder<(ins "mlir::StringAttr":$funcNameStr, "IntegerAttr":$numOfOutput, "mlir::ValueRange":$operands)>,
      OpBuilder<(ins "std::string":$funcNameStr, "mlir::ValueRange":$results, "mlir::Operation *":$op, "mlir::ValueRange":$operands, "std::vector<std::string>":$attributeNames)>,
    OpBuilder<(ins "mlir::ValueRange":$results, "mlir::Operation *":$op, "mlir::ValueRange":$operands, "bool":$copyAttrs)>,
      OpBuilder<(ins "std::string":$funcNameStr, "mlir::ValueRange":$results, "mlir::Operation *":$op, "mlir::ValueRange":$operands, "std::vector<std::string>":$attributeNames)>,
      OpBuilder<(ins "std::string":$funcNameStr, "mlir::ValueRange":$results, "mlir::Operation *":$op, "mlir::ValueRange":$operands, "bool":$copyAttrs)>];
}

def KrnlDefineLoopsOp : Op<Krnl_Dialect, "define_loops", [NoMemoryEffect]> {
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

def KrnlYieldOp : Op<Krnl_Dialect, "yield", [Pure, Terminator, ReturnLike,
    MemRefsNormalizable]> {
  let summary = "Yield values to parent operation";
  let description = [{
    The `krnl.yield` yields zero or more SSA values from an krnl.iterate op region and
    terminates the region. The semantics of how the values yielded are used
    is defined by the parent operation.
    If `krnl.yield` has any operands, the operands must match the parent
    operation's results.
    If the parent operation defines no values, then the `krnl.yield` may be
    left out in the custom syntax and the builders will insert one implicitly.
    Otherwise, it has to be present in the syntax to indicate which values are
    yielded.
  }];

  let arguments = (ins Variadic<AnyType>:$operands);

  let builders = [
    OpBuilder<(ins), [{ build($_builder, $_state, std::nullopt); }]>
  ];

  let assemblyFormat = "attr-dict ($operands^ `:` type($operands))?";
  let hasVerifier = 1;
}

def ImplicitKrnlYield : SingleBlockImplicitTerminator<"KrnlYieldOp">;

def KrnlIterateOp : Op<Krnl_Dialect, "iterate", [RecursiveMemoryEffects, ImplicitKrnlYield,
    DeclareOpInterfaceMethods<LoopLikeOpInterface, ["getInitsMutable", "getYieldedValuesMutable"]>]> {
  let summary = "iterate operation";
  let description = [{
    The "krnl.iterate" operation is conceptually equivalent to a nested for loops.

    For instance, say we have the following two
    ```
    %l0, %l1 = krnl.define_loops 2
    %o0, %o1 = krnl.optimize_loops  {
        // Identity schedule.
        krnl.return_loops %l0, %l1
    }
    ```

    Then, consider the following krnl.iterate operation:
    ```
    krnl.iterate (%o0, %o1) with (%l0 -> %i0 = 0 to 10, %l1 -> %i1 = 0 to 10) {
      // Some operations.
    }
    ```

    It is equivalent to:
    ```
    for (i0 = 0; i0 < 10; i0++)
      for (i1 = 0; i1 < 10; i1++)
        // Some operations.
    ```
  }];

  let arguments = (ins Variadic<AnyType>);
  let results = (outs Variadic<AnyType>:$results);
  let regions = (region SizedRegion<1>:$bodyRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    // Main builder.
    OpBuilder<(ins "onnx_mlir::krnl::KrnlIterateOperandPack":$operandPack,
      CArg<"ValueRange", "std::nullopt">:$iterArgs,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>", "nullptr">:$bodyBuilderFn)>,
    // Builder for the optimized iterate op.
    OpBuilder<(ins
      CArg<"ValueRange">:$originalLoops, CArg<"ValueRange">:$optimizedLoops,
      CArg<"ValueRange">:$lbs, CArg<"ValueRange">:$ubs, CArg<"ValueRange">:$iterArgs,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>">:$bodyBuilderFn)>,
    // Builder for the optimized iterate op with IndexExpr
    OpBuilder<(ins
      CArg<"ValueRange">:$originalLoops, CArg<"ValueRange">:$optimizedLoops,
      CArg<"ArrayRef<onnx_mlir::IndexExpr>">:$lbs, CArg<"ArrayRef<onnx_mlir::IndexExpr>">:$ubs,
      CArg<"ValueRange">:$iterArgs,
      CArg<"function_ref<void(OpBuilder &, Location, ValueRange, ValueRange)>">:$bodyBuilderFn)>
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

    int64_t getNumIterArgs() { return getNumResults(); }

    Block::BlockArgListType getRegionIterArgs() {
      return Block::BlockArgListType(getBody()->getArguments().begin() +
                                         (getBody()->getArguments().size() - getNumIterArgs()),
          getBody()->getArguments().end());
    }
    ValueRange getIterArgInits() {
      return iterator_range(
          operand_begin() + (getNumOperands() - getNumIterArgs()),
          operand_end());
    }

    // Get name of the attribute for storing bound represented using affine maps.
      static StringRef getBoundsAttrName() { return "bounds"; }
  }];
}

// ONNX sequence type (!onnx.Seq) is implemented with memref of memref.
// When an element(tensor/memref) is stored into a sequence, it becomes
// invisible to the bufferization analysis. When onnx-mlir invokes
// bufferization::Deallocation pass to insert memref::dealloc, the element
// may be freed. Therefore, element need to be copied when inserted into
// a sequence. 'copy' here means to allocate a new memref and copy the content
// of the memref. The copy is a costly op to maintain the SSA requirement.
// To reduce the overhead of copy, elements are not copied when
// a new sequence is created from an exist sequence. However, when an element
// is extracted from a sequence (krnl.seqextract), the element will be
// copied and return. This copy can be further avoided if the extraction is
// the last reference (by extract or erase from any sequence) of this element.
// ToDo: handle an element in multiple sequences with copy
// ToDo: use the one-shot interface for space manangement.

def KrnlSeqAllocOp : Op<Krnl_Dialect, "seqalloc", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<AllocationOpInterface, ["buildDealloc", "buildClone"]>]> {
  let summary = "Krnl create a sequence";
  let description = [{
    This op allocates a memref for a new sequence according to the input Type and length.
    The output is tagged with Allocate side effect, and a deallocation is defined for
    sequence. This deallocation will free all the elements in the sequence as well as
    the sequence itself.
  }];
  let arguments = (ins Variadic<Index>:$length);
  let results = (outs AnyMemRef:$output);
}

def KrnlSeqDeallocOp : Op<Krnl_Dialect, "seqdealloc", [MemRefsNormalizable]> {
  let summary = "Krnl dealloc a sequence";
  let description = [{
    This op deallocate the elements in the sequence and the sequence itself
    with memref::dealloc. This Op is a deep dealloc for sequence type.
  }];
  let arguments = (ins AnyMemRef:$input_sequence);
}

def KrnlSeqExtractOp : Op<Krnl_Dialect, "seqextract", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    DeclareOpInterfaceMethods<AllocationOpInterface, ["buildDealloc", "buildClone"]>]> {
  let summary = "Krnl load from a sequence";
  let description = [{
    This op loads an element from the input sequence 'seq' at position 'index'.
    The loaded element is copied and then return.
    The position value is guaranteed to be positive. Negative position allowed
    by ONNX Op definition should be handled before lowered to KrnlSeqExtract.

    Attribute 'copy' provides an optimization for copying.
    When the attribute 'copy' is 1 (default value): the extracted element is copied and then return.
    When the attribute 'copy' is 0: the extracted element is directly returned
    without copy.

    The returned element is marked as allocated by this Op with the bufferation
    interface so that deallocation can be generated correctly through the
    Bufferization::Deallocation pass.
  }];
  let arguments = (ins AnyMemRef:$seq,
                       Index:$index,
                       DefaultValuedAttr<UI1Attr, "1">:$copy);

  let results = (outs AnyType:$output);
}
def KrnlSeqStoreOp : Op<Krnl_Dialect, "seqstore",
    [MemRefsNormalizable, DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]
  > {
  let summary = "Krnl store into a seq";
  let description = [{
    This op is similar to KrnSeqInsertOp but assumes that the input seq has
    the space for the new element and
    only need to copy the element and store it into the sequence.
    There is no return of a new seq, different from KrnlSeqInsertOp.
    This Op is introduced to accumulate a dynamic tensor in a LoopOp with
    statically known iteration count.
  }];
  let arguments = (ins AnyType:$input,
                       AnyMemRef:$seq,
                       Index:$index);
}

def KrnlTerminatorOp : Op<Krnl_Dialect, "terminate", [ReturnLike, Terminator, NoMemoryEffect]> {
  let summary = "Krnl terminator operation";
  let description = [{
    Krnl terminator is a special terminator operation for blocks inside krnl
    iterate operations. It unconditionally transmits the control flow to the
    successor of the operation enclosing the region.

    This operation does _not_ have a custom syntax. However, krnl control
    operations omit the terminator in their custom syntax for brevity.
  }];
}

def KrnlRegionOp : Op<Krnl_Dialect, "region", [NoTerminator, SingleBlock,
    AffineScope]> {
  let summary = "Affine boundary for krnl loops";
  let description = [{
    This Op has a region with AffineScope trait and is used to limit the
    scope of `affine.for`. The loop inside `krnl.region` can be affined if
    its boundary is defined at the level of `krnl.region`. The `krnl.region` does
    not guarantee or require the loops inside it to be affine.
    With `krnl.region`, a krnl loop may not be  affine if its boundary symbol
    is not defined inside a enclosing region without AffineScope trait.
    In MLIR, FuncOp has the AffineScope trait.
    The `krnl.region` will be removed after affine.for is lowered.
    ToFix: current `krnl.region` does not have input and output. You cannot
    create a new memref inside the region and use it outside of the region.
  }];

  let regions = (region SizedRegion<1>:$bodyRegion);

  let skipDefaultBuilders = 1;
  let builders = [
    // Main builder.
    OpBuilder<(ins
      CArg<"function_ref<void(OpBuilder &, Location)>", "nullptr">:$bodyBuilderFn)>,
  ];
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

def KrnlMemcpyOp : Op<Krnl_Dialect, "memcpy", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Krnl memcpy operation";
  let description = [{
    Copy `num_elems` elements from `src` to `dest` MemRef.

    Starting positions for `src` and `dest` are defined by `src_offset` and
    `dest_offset`, respectively.

    It is the users' responsibility to make sure there is no out-of-bound read/write.
  }];

  let arguments = (ins AnyMemRef:$dest, AnyMemRef:$src, I64:$num_elems,
                       Index:$dest_offset, Index:$src_offset);
}

def KrnlGlobalOp : Op<Krnl_Dialect, "global", [Pure, MemRefsNormalizable]> {
  let summary = "Krnl global operation";
  let description = [{
    Operation for holding global data values. A global constant can have a
    meaningful name recorded as its `name` attribute. Its content is stored
    in the `value` dense element attribute.
  }];

  let arguments = (ins AnyAttr:$shape,
    StrAttr:$name, OptionalAttr<AnyAttr>:$value, OptionalAttr<I64Attr>:$offset,
    OptionalAttr<I64Attr>:$alignment);
  let results = (outs AnyTypeOf<[AnyMemRef]>:$output);
}

def KrnlBlockOp : Op<Krnl_Dialect, "block"> {
  let summary = "Krnl block operation";
  let description = [{
    Block a single for loop by a constant tile size. For instance,
    ```
    $ib, $il = krnl.block %i, 4
    ```
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

def KrnlParallelOp : Op<Krnl_Dialect, "parallel", [AttrSizedOperandSegments]> {
  let summary = "Mark Krnl loops as parallel loops";
  let description = [{
    Parallelize the specified loops. When multiple loop specifiers are passed
    as parameters, there loops can be parallelized as a collapsed loop.
    krnl.parallel should be placed as the last operator before krnl.iterate,
    Since we do not want to parallelize the loop until we interpret krnl.block,
    krnl.permute and krnl.unroll.

    Optionally, a value may specifiy the number of threads requested for the
    parallel loop. A proc_bind string may also be specified; valid values are
    "primary", "close", or "spread". Default values are used when not specified.

    ```
    krnl.parallel (%i0, %i1) : !Krnl.loop, !Krnl.loop
    ```
  }];

  let arguments = (ins Variadic<AnyType>:$loops,
    Optional<I32>:$num_threads,
    OptionalAttr<StrAttr>:$proc_bind);

  let assemblyFormat = [{
      `(` $loops `)` (`,` `num_threads` `(` $num_threads^ `)`)? attr-dict `:` type($loops) 
  }];
}

def KrnlParallelClauseOp : Op<Krnl_Dialect, "parallel_clause"> {
  let summary = "Attach OpenMP clauses to an index varialbe";
  let description = [{
    Attach OpenMP clauses to an index variable. That index variable
    is used to uniquely associate a parallel loop with its clauses.
  }];

  let arguments = (ins Index: $parallel_loop_index, 
    Optional<I32>:$num_threads,
    OptionalAttr<StrAttr>:$proc_bind);

  let assemblyFormat = [{
      `(` $parallel_loop_index `)` (`,` `num_threads` `(` $num_threads^ `)`)? 
      attr-dict `:` type($parallel_loop_index) 
  }];
}

def KrnlRoundEvenOp : Op<Krnl_Dialect, "round_even"> {
  let summary = "Krnl round to nearest even operation";
  let description = [{
    Krnl round to nearest even operation.  Accept scalar or vector float values.
    Vector must be 1D of a size that is a multiple of the hardware vector size.
  }];

  let arguments = (ins FloatLike:$in);
  let results = (outs FloatLike:$out);
}

def KrnlErfOp : Op<Krnl_Dialect, "erf"> {
  let summary = "Krnl erf scalar operation";
  let description = [{
    Krnl erf scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs AnyFloat:$out);
}

def KrnlIsInfOp : Op<Krnl_Dialect, "isinf"> {
  let summary = "Krnl isinf scalar operation";
  let description = [{
    Krnl isinf scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs I1:$out);
}

def KrnlIsNaNOp : Op<Krnl_Dialect, "isnan"> {
  let summary = "Krnl isnan scalar operation";
  let description = [{
    Krnl isnan scalar operation.
  }];

  let arguments = (ins AnyFloat:$in);
  let results = (outs I1:$out);
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
                  "mlir::cast<MemRefType>($_self).getElementType()">,
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
      auto memrefType = mlir::cast<MemRefType>(memref.getType());
      $_state.addOperands(memref);
      $_state.addOperands(indices);
      $_state.types.push_back(memrefType.getElementType());
    }]>];

  let assemblyFormat = [{$memref `[` $indices `]` attr-dict `:` type($memref)}];

  let extraClassDeclaration = [{
    void setMemRef(Value value) { setOperand(0, value); }
    MemRefType getMemRefType() {
      return mlir::cast<MemRefType>(getMemref().getType());
    }
  }];
}

def KrnlStoreOp : Op<Krnl_Dialect, "store",
     [TypesMatchWith<"type of 'value' matches element type of 'memref'",
                     "memref", "value",
                     "mlir::cast<MemRefType>($_self).getElementType()">,
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
      void setMemRef(Value value) { setOperand(1, value); }
      MemRefType getMemRefType() {
        return mlir::cast<MemRefType>(getMemref().getType());
      }
  }];
}

def KrnlGetLinearOffsetIndexOp : Op<Krnl_Dialect, "get_linear_offset_index",
  [DeclareOpInterfaceMethods<AffineReadOpInterface>,
  DeclareOpInterfaceMethods<AffineMapAccessInterface>, MemRefsNormalizable]> {
  let summary = "A Krnl operation to compute a linear offset index from a N-D index.";

  let description = [{
    Given a MemRef and an N-D index (id_1, id_2, ..., id_n), where n is
    the rank of the MemRef, this operation computes a linear offset index.
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference memref", [MemRead]>:$memref,
                       Variadic<Index>:$indices,
                       AffineMapAttr:$map);
  let results = (outs Index:$result);

  // let assemblyFormat = [{$memref `[` $indices `]` attr-dict `:` type($memref)}];
  let builders = [
    /// Builds an op with the specified map and operands.
    OpBuilder<(ins "AffineMap":$map, "ValueRange":$operands)>,
    /// Builds an op with an identity map and operands.
    OpBuilder<(ins "Value":$memref, CArg<"ValueRange", "{}">:$indices)>,
    /// Builds an op with the specified map and its operands.
    OpBuilder<(ins "Value":$memref, "AffineMap":$map,
      "ValueRange":$mapOperands)>
  ];
  let extraClassDeclaration = [{
    /// Returns the operand index of the memref.
    unsigned getMemRefOperandIndex() { return 0; }

    void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }

    MemRefType getMemRefType() {
      return mlir::cast<MemRefType>(getMemref().getType());
    }

    /// Returns the affine map used to index the memref for this operation.
    AffineMapAttr getAffineMapAttr() {
      return getMapAttr();
    }

    static StringRef getMapAttrStrName() { return "map"; }
  }];

  let hasCustomAssemblyFormat = 1;

}

def KrnlPrefetchOp : Op<Krnl_Dialect, "prefetch",
  [ MemRefsNormalizable, DeclareOpInterfaceMethods<AffineMapAccessInterface>]> {
  let summary = "A Krnl operation to compute a linear offset index from a N-D index.";

  let description = [{
    Given a MemRef and an N-D index (id_1, id_2, ..., id_n), prefetch the memory
    location pointed by this memory reference.
  }];

  let arguments = (ins Arg<AnyMemRef, "the reference memref", [MemWrite]>:$memref,
      Variadic<Index>:$indices,
      BoolAttr:$isWrite,
      ConfinedAttr<I32Attr, [IntMinValue<0>, IntMaxValue<3>]>:$localityHint,
      BoolAttr:$isDataCache,
      AffineMapAttr:$map
   );

  // let assemblyFormat = [{$memref `[` $indices `]` attr-dict `:` type($memref)}];
  let builders = [
    /// Builds an op with an identity map and operands.
    OpBuilder<(ins "Value":$memref, "ValueRange":$indices, "bool":$isWrite,
      "unsigned":$localityHint, "bool":$isDataCache)>,
    OpBuilder<(ins "Value":$memref, "bool":$isWrite,
      "unsigned":$localityHint, "bool":$isDataCache)>,
    /// Builds an op with the specified map and its operands.
    OpBuilder<(ins "Value":$memref, "AffineMap":$map,
      "ValueRange":$mapOperands, "bool":$isWrite,
      "unsigned":$localityHint, "bool":$isDataCache)>
  ];

  let extraClassDeclaration = [{
    /// Returns the operand index of the memref.
    unsigned getMemRefOperandIndex() { return 0; }
    Value getMemRef() { return getOperand(getMemRefOperandIndex()); }
    void setMemRef(Value value) { setOperand(getMemRefOperandIndex(), value); }
    MemRefType getMemRefType() {
      return mlir::cast<MemRefType>(getMemref().getType());
    }

    /// Implements the AffineMapAccessInterface.
    /// Returns the AffineMapAttr associated with 'memref'.
    // Default version is fine.

    /// Returns the affine map used to index the memref for this operation.
    AffineMapAttr getAffineMapAttr() { return getMapAttr(); }
    AffineMap getAffineMap() { return getAffineMapAttr().getValue(); }

    /// Get affine map operands.
    operand_range getMapOperands() {  return getIndices(); }

    static StringRef getMapAttrStrName() { return "map"; }
    static StringRef getIsWriteAttrStrName() { return "isWrite"; }
    static StringRef getLocalityHintAttrStrName() { return "localityHint"; }
    static StringRef getIsDataCacheAttrStrName() { return "isDataCache"; }
  }];

  let hasCustomAssemblyFormat = 1;
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

def KrnlGetInductionVariableValueOp : Op<Krnl_Dialect, "get_induction_var_value", [NoMemoryEffect]> {
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

def KrnlVectorTypeCastOp : Op<Krnl_Dialect, "vector_type_cast", [Pure,
    MemRefsNormalizable, DeclareOpInterfaceMethods<CastOpInterface>, ViewLikeOpInterface]> {
  let summary = "vector type cast operation";
  let description = [{
    The "vector_type_cast" operation converts a memref from an non-vector
    element type to another memref of a vector elemental type while not changing
    the source memref's element type. The last dimension size of the source
    dimension is divided (floor division) by the vector size to obtain the
    corresponding dimension for target memref type.

    ```
    %MV = vector_type_cast %M : memref<64x16xf32> to memref<64x2xvector<8xf32>>
    %AV = vector_type_cast %A : memref<?x?xf32> to memref<?x?xvector<8xf32>>
    ```
  }];

  let arguments = (ins AnyMemRef:$source);
  let results = (outs AnyMemRef:$result);

  let hasFolder = 1;
  let builders = [ OpBuilder<(ins "Value": $source, "int64_t": $vectorLen)> ];

  let assemblyFormat = [{
    $source attr-dict `:` type($source) `to` type($result)
  }];

  let extraClassDeclaration = [{
    /// Return the view source.
    Value getViewSource() { return getSource(); }
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
    Perform a matrix multiplication AA * BB + CC with sizes `[IxK] * [KxJ] + [IxJ]`.
    The original matrices AA, BB, and CC can be buffered in buffered arrays
    which may be padded. The original matrices and the padded array might
    have a higher rank than 2, but the actual matrix multiplication operation
    only deal with the innermost 2 ranks of the matrices to perform its matrix
    multiplication operations.

    The computations may also compute only a sub-tile of the buffered arrays.
    This region is depicted using stars '*' below.

    All indices passed to this operation are the global indices in the original
    computation, so as to better know if we have boundary conditions.

    ORIGINAL ARRAY: denoted as AA, BB, CC with sizes AA: `*xIxK`; BB: `*xKxJ`; CC: `*xI*J`).

    BUFFER ARRAYS: denoted as A, B, and C. Note that this operation does
      not require the use of buffers arrays. If none are used, then A=AA,
      B=BB, C=CC. If buffers are used, it is the responsibility of the caller
      to properly fill the buffers with the appropriate data. Buffers are
      typically used for cache tiling.

     ORIGINAL ARRAY

```
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
```

* (1) `iGlobalIndexComputeStart`/`jGlobalIndexComputeStart`/`kGlobalIndexComputeStart`,
   required, each three are global 1D indices.
* (2) `iGlobalUB`/`jGlobalUB`/`jGlobalUB`, required, each three are global 1D indices.
* (3) `aGlobalIndexMemStart`/`bGlobalIndexMemStart`/`cGlobalIndexMemStart`,
   required, global nD indices with the same rank as the buffers A, B, and C.
* (4) `aTileSize`/`bTileSize`/`cTileSize`, required when padding, each 2D sizes.
* (5) `computeTileSizes`, required when tiled computation within buffer, 3D sizes (I, J, K).

    The `iGlobalIndexComputeStart`/`jGlobalIndexComputeStart`/
    `kGlobalIndexComputeStart` (1) indicate the global indices of the
    first element of a tile to be computed in the original computations.

    The `iGlobalUB`/`jGlobalUB`/`kGlobalUB` (2) indicate the global upper bounds
    in the original computations.

    We provide 3 buffers for matrix multiply: A, B, and C. For each buffer,
    we indicate the global indices pointing the beginning of the buffer:
    `aGlobalIndexMemStart`, `bGlobalIndexMemStart`, and `cGlobalIndexMemStart` (3).
    If no buffers are used, i.e. the computation starts directly in the
    original memory, the global index is 0. If a buffer for AA is used to
    put data into it starting at indices `[i1, k1]`, where `i1` & `k1` are the
    global indices in the original computations, then `aGlobalIndexMemStart0`
    and `aGlobalIndexMemStart1` are `i1` & `k1`, respectively.

    If the A, B, or C buffers are larger than the actual data tile they
    contain (see `copy_to_tile_buffer`), then the actual tile size must be
    given using an optional attribute: `aTileSize`, `bTileSize`, or `cTileSize` (4).
    These optional tile size have a rank of 2, and their values must be
    equal or smaller than their corresponding buffer memrefs.

    If the computation are further tiled with respect to the size of the
    buffers A, B, or C, then the actual computation tile is given by
    the optional tile attribute `computeTileSize` (5). Its rank is 3, for the
    I, J, and K dimension. The actual A, B, and C buffer tile size
    (possibly specified by the optional parameters) must be a multiple of
    the I, J, and K `computeTileSizes`, in their respective
    dimensions (A: `[IxK]`, B: `[KxJ]`, C: `[IxJ]`).

    Note that the buffers A, B, and C can be of higher dimensionality than
    the traditional 2D mentioned up to now, because of broadcasting rules.
    At this time, we only support broadcast of arrays having ranks of 2 or
    more. Because of the broadcast rules, the higher dimensions have a
    constant index during one matrix multiply. These fixed indices are
    given as prefix dimensions in the starting indices for AA, BB, and CC
    as described above. E.g. if AA has a rank of 3, and BB has a rank of 2,
    the starting indices for AA are `[d, i1, k1]` where `i1` and `k1` are as
    above, and d is index pointing to the current instance of the `IxK`
    AA matrix to be computed. B start indices would be unchanged at `[k1, j1]`.

    Simdize is used to state if simdization is requested.
    Unrolling is used to unroll and jam loops as warranted.

    Below is an example calculating a matrix multiply with pre-zeroed
    C matrix with the sizes below.

```
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
```

    Note that code is simdized along the J dim (last dim of B and C matrices).
    For simd to be enabled, the simdized flag must be set to true, and the
    following condition must be true:
    1) The vector length is the second entry of (i, j, k) compute tile size.
       The vector length must be a compile time constant.
  }];

  let arguments = (ins
    // Buffer/memory used and indices that correspoond to the beginning
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
                  "mlir::cast<MemRefType>($_self).getElementType()">,
    TypesMatchWith<"type of 'padValue' matches element type of 'buffer'",
                  "buffer", "padValue",
                   "mlir::cast<MemRefType>($_self).getElementType()">,
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

    * If`padToNext` attribute is given, the pad value will be copied from
      the last source data of to the next index for which index modulo `padToNext`
      is zero, i.e. to the end of a "cache line" of side `padToLine`. Pad
      of 1 means no padding, pad of buffer size means fully pad the buffer.
      Default is no padding (1). `PadValue` is used to initialized the padded
      areas.

    * If `overreadToNext` attribute is given, the copy may read source past
      its upper bound value. This enable optimized code, e.g. using SIMD
      read operations even if going past the last value of the source
      memory, or unrolling and jamming copy loops to reduce memory latency.
      `overreadToNext` is expressed like padToNext: value of 1 means no
      reading past boundary; value of buffer size enables reading
      as many additional source value as needed to fill the full
      buffer. Default is buffer-size.

    `padToNext` and `overreadToNex`t are of the same rank as source and memory
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

  let arguments = (ins StrAttr:$opName,
                       I64Attr:$tag,
                       OptionalAttr<StrAttr>:$nodeName);

  let builders = [ OpBuilder<(ins "Operation *": $op, "int ": $tag)> ];
}

def KrnlMemsetOp : Op<Krnl_Dialect, "memset", [MemRefsNormalizable,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>,
    TypesMatchWith<"type of 'value' matches element type of 'dest'",
                   "dest", "value",
                   "mlir::cast<MemRefType>($_self).getElementType()">]> {
  let summary = "Set buffer to a given value.";
  let description = [{
    Krnl operation that sets a buffer to a given value.
    In case that the buffer is a MemRef with affine_map, `delayed` indicates
    whether we set values along original or extended iteration space.

    For example, given
    - an affine_map `#tile = affine_map < (i)->(i floordiv 4, i mod 4) >`, and
    - a buffer of type `memref<5xf32, #tile>`

    Original iteration space is along the first axis that has 5 elements.

    If we do normalization, the memref becomes `memref<2x4xf32>`. Now we have
    an extended iteration space along two axes of sizes 2 and 4, respectively.
    This extended iteration space has 8 elements in total.

    If `delayed = false`, the original iteration space is used to set values.
    In the above example, only 5 out of 8 elementes will be set to the given value.

    If `delayed = true`, the extended iteration space is used to set values.
    In the above example, all 8 elements will be set to the given value.

  }];

  let arguments = (ins AnyMemRef:$dest, AnyType: $value,
    DefaultValuedAttr<BoolAttr, "false">:$delayed);

  let assemblyFormat = [{ $dest `,` $value attr-dict `:` type($dest) }];
}

def KrnlStrlenOp : Op<Krnl_Dialect, "strlen", [Pure]> {
  let summary = "Compute the length of a string.";
  let description = [{
    Krnl operation that computes the length of a string.
  }];

  let arguments = (ins StringType:$str);
  let results = (outs I64:$res);
}

def KrnlStrncmpOp : Op<Krnl_Dialect, "strncmp", [Pure]> {
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
    [Pure, MemRefsNormalizable]> {
  let summary = "Retrieve an index into a perfect hash table described by G and V.";
  let description = [{
    This operation can be used to generate a call to a runtime function which,
    given two arrays of int32_t values (G and V), which are used to represent a perfect
    hash table for a dictionary, returns the index corresponding to the input value.
    The index returned is valid only if 'input' is in the dictionary described by G and V.
  }];

  let arguments = (ins AnyTypeOf<[StringType, I64]>:$input, I32MemRef:$G, I32MemRef:$V, I32:$len);
  let results = (outs Index:$index);
}

//===----------------------------------------------------------------------===//
// NoneOp.
def KrnlNoneOp : Op<Krnl_Dialect, "noValue"> {
  let summary = "An operation representing the absence of a value.";
  let description = [{
    This operation can be used to represent the absence of a value. It is
    typically used as an argument to operators that have optional parameters,
    and converted into nullptr while krnl to llvm lowering.
    Typically it is used for optional arguments used in KrnlCallop.
  }];

  let arguments = (ins UnitAttr:$value);
  let results = (outs NoneType:$none_val);

  let builders = [
    OpBuilder<(ins),[{
      build($_builder, $_state, $_builder.getNoneType(), $_builder.getUnitAttr());
    }]>];
}

def KrnlPrintTensorOp : Op<Krnl_Dialect, "print_tensor", [MemRefsNormalizable]> {
  let summary = "Print a tensor.";
  let description = [{
    This operation can be used to generate a call to a runtime function which prints a tensor.
    At the beginning of the msg string, user can add formatting instructions. The flags are:

    *  `%s`: detailed signature (including shape, type, offsets),
    *  `%t`: compact type (ala MLIR: `32x16xfloat`),
    *  `%d`: data values.

    When no formatting is provided, `%s%d` is used (detailed signature and data) by default.
    Print operation ends with a newline, except when only requesting a compact types (`%t`).
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
