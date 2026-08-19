// RUN: onnx-mlir-opt --convert-krnl-to-affine %s -split-input-file -verify-diagnostics

// Cases rejected during lowering, by the consumed-ref rule: once a loop ref has
// been collapsed away it has no affine.for of its own, so nothing outside the
// collapse may still refer to it.

func.func @unroll_shares_operand_with_collapse() {
  %ii, %jj = krnl.define_loops 2
  // expected-error @+1 {{collapses a loop reference that is also used by 'krnl.unroll'}}
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.unroll %ii : !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
  }
  return
}

// -----

func.func @block_shares_operand_with_collapse() {
  %ii, %jj = krnl.define_loops 2
  // expected-error @+1 {{collapses a loop reference that is also used by 'krnl.block'}}
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  %ib, %il = krnl.block %ii 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
  }
  return
}

// -----

// get_induction_var_value on a collapsed-away ref: use krnl.collapse_indices on
// the fused index instead.
func.func @get_induction_var_of_collapsed_ref() {
  %ii, %jj = krnl.define_loops 2
  // expected-error @+1 {{collapses a loop reference that is also used by 'krnl.get_induction_var_value'}}
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %bad = krnl.get_induction_var_value(%ii) : (!krnl.loop) -> index
  }
  return
}

// -----

// Naming the collapsed dimensions in the 'with' clause and using those
// induction variables directly: the fused loop has no such induction variables.
// Left unchecked this crashed inside affine::coalesceLoops, whose induction
// variable rewrite is scoped to the innermost loop's region and so never reached
// the body once markLoopBodyAsMovable had parked it in a krnl.movable.
func.func @use_original_induction_variables(%arg0: memref<10x20xf32>) -> memref<10x20xf32> {
  %alloc = memref.alloc() : memref<10x20xf32>
  %ii, %jj = krnl.define_loops 2
  // expected-error @+1 {{the induction variable of a collapsed dimension cannot be used directly}}
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %v = krnl.load %arg0[%i, %j] : memref<10x20xf32>
    krnl.store %v, %alloc[%i, %j] : memref<10x20xf32>
  }
  return %alloc : memref<10x20xf32>
}

// -----

// A genuinely non-zero lower bound is out of scope. This is not merely passing
// through affine::coalesceLoops' own "normalized loops only" precondition: the
// trip count recorded for krnl.collapse_indices is taken to be the upper bound,
// which is only the trip count when the lower bound is 0 and the step is 1.
// Collapsing such a loop would silently mis-decompose every recovered index.
// Note this rejects the bound's *value*, not how it is spelled -- see
// collapse_lb_zero_via_constant in krnl_to_affine_collapse.mlir for %c0.
func.func @lower_bound_not_zero(%arg0: memref<10x20xf32>) -> memref<10x20xf32> {
  %alloc = memref.alloc() : memref<10x20xf32>
  %ii, %jj = krnl.define_loops 2
  // expected-error @+1 {{can only collapse loops whose lower bound is a compile-time constant 0}}
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 5 to 10, %jj -> %j = 0 to 20) {
    %idx = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> index
    %a, %b = krnl.collapse_indices(%idx) : (index) -> (index, index)
    %v = krnl.load %arg0[%a, %b] : memref<10x20xf32>
    krnl.store %v, %alloc[%a, %b] : memref<10x20xf32>
  }
  return %alloc : memref<10x20xf32>
}

// -----

// Same for a lower bound that is not a compile-time constant at all.
func.func @lower_bound_dynamic(%arg0: memref<10x20xf32>, %lb: index) -> memref<10x20xf32> {
  %alloc = memref.alloc() : memref<10x20xf32>
  %ii, %jj = krnl.define_loops 2
  // expected-error @+1 {{can only collapse loops whose lower bound is a compile-time constant 0}}
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = %lb to 10, %jj -> %j = 0 to 20) {
    %idx = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> index
    %a, %b = krnl.collapse_indices(%idx) : (index) -> (index, index)
    %v = krnl.load %arg0[%a, %b] : memref<10x20xf32>
    krnl.store %v, %alloc[%a, %b] : memref<10x20xf32>
  }
  return %alloc : memref<10x20xf32>
}

// -----

// permute is likewise rejected in v1: it is interpreted after the collapse has
// already fused (and erased) the loops it would reorder.
func.func @permute_shares_operand_with_collapse() {
  %ii, %jj, %kk = krnl.define_loops 3
  // expected-error @+1 {{collapses a loop reference that is also used by 'krnl.permute'}}
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.permute(%ii, %kk) [1, 0] : !krnl.loop, !krnl.loop
  krnl.iterate(%ff, %kk) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20, %kk -> %k = 0 to 30) {
  }
  return
}
