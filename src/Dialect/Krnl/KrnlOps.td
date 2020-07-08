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

def Krnl_Dialect : Dialect {
  let name = "krnl";
  let cppNamespace = "";
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
  let builders = [ OpBuilder<"OpBuilder &builder, OperationState &result,"
                             "int64_t num_loops"> ];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let extraClassDeclaration = [{
    static StringRef getNumLoopsAttrName() { return "num_loops"; }

  // Helper function to extract the number of loops being defined.
  int64_t getNumLoops() {
    auto num_loops = getAttrOfType<IntegerAttr>(getNumLoopsAttrName())
                         .getValue()
                         .getSExtValue();
    return num_loops;
  }
}];
}

def KrnlIterateOp : Op<Krnl_Dialect, "iterate", [ImplicitKrnlTerminator, AffineScope]> {
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
  let builders = [ OpBuilder<"OpBuilder &builder, OperationState &result, "
                             "KrnlIterateOperandPack operandPack"> ];

  let extraClassDeclaration = [{
    // In krnl.iterate operation, operands are stored as such
    // - Optimized krnl.loops.
    // - Input krnl.loops and their operand bounds. (TODO(Tian) explain better how we store them).

    // We record the number of optimized and input loops to separate these three
    // group of operands out.
    static StringRef getNumOptimizedLoopsAttrName() { return "num_optimized_loops"; }

    int64_t getNumOptimizedLoops() {
      auto num_optimized_loops =
        getAttrOfType<IntegerAttr>(getNumOptimizedLoopsAttrName())
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
  let builders = [ OpBuilder<"OpBuilder &builder, OperationState &result, "
                             "SymbolRefAttr funcAttr, IntegerAttr numInputs, "
                             "IntegerAttr numOutputs"> ];

  let extraClassDeclaration = [{
    static StringRef getEntryPointFuncAttrName() { return "func"; }
    static StringRef getNumInputsAttrName() { return "numInputs"; }
    static StringRef getNumOutputsAttrName() { return "numOutputs"; }
  }];

  // No custom parsing/printing form.
  let parser = ?;
  let printer = ?;
}

def KrnlMemcpyOp : Op<Krnl_Dialect, "memcpy"> {
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

def KrnlGetRefOp : Op<Krnl_Dialect, "getref"> {
  let summary = "Krnl a MemRef from within another MemRef starting at a specific offset.";
  let description = [{
    Retreieves a MemRef from within another MemRef:

    "krnl.getref"(%memref, %offset)

    The offset is an integer which is used as an index into the input MemRef. It works
    just like an array index.
  }];

  let arguments = (ins AnyTypeOf<[AnyMemRef]>:$mempool, AnyInteger:$offset);
  let results = (outs AnyTypeOf<[AnyMemRef]>:$output);

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

  let assemblyFormat = [{
      $loop $tile_size attr-dict `:` functional-type($loop, results)
  }];
}

def KrnlPermuteOp : Op<Krnl_Dialect, "permute"> {
  let summary = "Krnl permute operation";
  let description = [{
    Block a single for loop by a constant tile size. For instance,
    $ib, $il = krnl.block %i, 4
    means to block the for loop referred to by %i using a tile size of 4.
  }];

  let arguments = (ins Variadic<AnyType>:$loops, I64ArrayAttr:$mapping);
  let results = (outs Variadic<AnyType>);

  let assemblyFormat = [{
      $loops $mapping attr-dict `:` functional-type($loops, results)
  }];
}