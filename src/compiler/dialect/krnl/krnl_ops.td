//===--------------------- krnl_ops.td - MLIR Operations ------------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

include "mlir/IR/OpBase.td"

def Krnl_Dialect : Dialect {
  let name = "krnl";
  let cppNamespace = "";
}

// Require regions to have krnl.terminate terminator operation.
def ImplicitKrnlTerminator
    : SingleBlockImplicitTerminator<"KrnlTerminatorOp">;

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

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result,"
              "int64_t num_loops">
  ];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];

  let extraClassDeclaration = [{
    static StringRef getNumLoopsAttrName() { return "num_loops"; }

    // Helper function to extract the number of loops being defined.
    int64_t getNumLoops() {
        auto num_loops =
        getAttrOfType<IntegerAttr>(
            getNumLoopsAttrName())
                .getValue()
                .getSExtValue();
        return num_loops;
    }
  }];


}

def KrnlOptimizeLoopsOp : Op<Krnl_Dialect, "optimize_loops"> {
  let summary = "optimize_loops operation";
  let description = [{

    The "krnl.optimize_loops" operation is essentially a cosmetic operation
    which exists to encapsulate a region where loops are being scheduled/optimized.

    The optimized loops are returned at the end of the
    region associated with the krnl.optimize_loops operation.

    For example:
      TBD once we have actual schedule intrinsics.

  }];

  let arguments = (ins Variadic<AnyType>);
  let results = (outs Variadic<AnyType>);
  let regions = (region SizedRegion<1>:$region);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "int timestamp_space_rank">
  ];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
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
    for (i0=0; i0<10; i0++)
      for (i1=0; i1<10; i1++)
        // Some operations.
  }];

  let arguments = (ins Variadic<AnyType>);

  let regions = (region SizedRegion<1>:$bodyRegion);

  let skipDefaultBuilders = 1;

  let builders = [
    OpBuilder<"Builder *builder, OperationState &result, "
              "ArrayRef<Value*> input_loops, ArrayRef<Value*> optimized_loops, "
              "ArrayRef<Value*> operand_bounds, ArrayRef<int64_t> const_bounds, "
              "ArrayRef<int> bound_types">
  ];

  let extraClassDeclaration = [{

    // In krnl.iterate operation, three types of SSA values are stored:
    // - Optimized krnl.loops.
    // - Input krnl.loops.
    // - SSA value based induction variable bound (parametric bound).
    // We record the number of optimized and input loops to separate these three
    // group of operands out.
    static StringRef getNumOptimizedLoopsAttrName() { return "num_optimized_loops"; }

    int64_t getNumOptimizedLoops() {
      auto num_optimized_loops =
        getAttrOfType<IntegerAttr>(
                getNumOptimizedLoopsAttrName())
              .getValue()
              .getSExtValue();
      return num_optimized_loops;
    }

    static StringRef getNumInputLoopsAttrName() { return "num_input_loops"; }

    int64_t getNumInputLoops() {
      auto num_loops =
        getAttrOfType<IntegerAttr>(
                getNumInputLoopsAttrName())
              .getValue()
              .getSExtValue();
      return num_loops;
    }

    // Constant bounds are stored here as a list attribute.
    static StringRef getConstantBoundsAttrName() { return "constant_bounds"; }

    // Store type of each bound as three types:
    // - 0 = constant attribute.
    // - 1 = operand type.
    // - 2 = affine maps (TODO).
    static StringRef getBoundTypesAttrName() { return "bound_types"; }

    // Get dynamic attribute name for the i-th lower and upper bound.
    static std::string getBoundAttrName(int64_t i, bool is_ub) {
      std::string bound_type = is_ub ? "_ub" : "_lb";
      std::string bound_idx = std::to_string(i);
      return "__bound_" + bound_idx + bound_type;
    }
    }];

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
  let verifier = [{ return ::verify(*this); }];
}

def KrnlReturnLoopsOp : Op<Krnl_Dialect, "return_loops", [Terminator]> {
  let summary = "Krnl return handler operation";
  let description = [{
    Krnl return_loops operation is a terminator operation for returning
    scheduled dimension handlers in the krnl.optimize_loops region.
  }];

  let arguments = (ins Variadic<AnyType>);

  let printer = [{ return ::print(p, *this); }];
  let parser = [{ return ::parse$cppClass(parser, result); }];
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
