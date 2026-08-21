// RUN: onnx-mlir-opt %s -split-input-file -verify-diagnostics

// Cases rejected by op verifiers, before any lowering runs.

func.func @collapse_needs_two_loops() {
  %ii = krnl.define_loops 1
  // expected-error @+1 {{expects 2 or more loops to collapse}}
  %ff = krnl.collapse(%ii) : (!krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10) {
  }
  return
}

// -----

// v1 restriction: a block output is not a collapsable operand.
func.func @collapse_of_block() {
  %ii, %jj = krnl.define_loops 2
  %ib, %il = krnl.block %ii 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  // expected-error @+1 {{cannot collapse a loop produced by krnl.block}}
  %ff = krnl.collapse(%ib, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff, %il) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
  }
  return
}

// -----

// Nested collapse: the variadic form already covers the same need.
func.func @collapse_of_collapse() {
  %ii, %jj, %kk = krnl.define_loops 3
  %f1 = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  // expected-error @+1 {{cannot collapse a loop produced by krnl.collapse}}
  %f2 = krnl.collapse(%f1, %kk) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%f2) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5, %kk -> %k = 0 to 6) {
  }
  return
}

// -----

// The reverse direction: blocking a collapse result.
func.func @block_of_collapse() {
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  // expected-error @+1 {{cannot block a loop produced by krnl.collapse}}
  %fb, %fl = krnl.block %ff 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.iterate(%fb, %fl) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
  }
  return
}

// -----

// The reverse direction: unrolling a collapse result.
func.func @unroll_of_collapse() {
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  // expected-error @+1 {{cannot unroll a loop produced by krnl.collapse}}
  krnl.unroll %ff : !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
  }
  return
}

// -----

// Asking a collapsed loop for a single index without the fusedIndex attribute.
// This is the pre-existing spelling of a fused-index query, and now names the
// per-dimension mode with too few results.
func.func @get_iv_too_few_results() {
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    // expected-error @+1 {{expects 2 results but has 1}}
    %idx = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> index
  }
  return
}

// -----

// The fusedIndex mode yields exactly one index per loop reference, never the
// collapsed dimensions.
func.func @get_iv_fused_too_many_results() {
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    // expected-error @+1 {{expects 1 results but has 2}}
    %a, %b = krnl.get_induction_var_value(%ff) {fusedIndex} : (!krnl.loop) -> (index, index)
  }
  return
}

// -----

// A mixed per-operand intent -- %f1 expanded, %f2 fused -- is not expressible in
// one operation: the attribute applies to all operands. Such a body issues two
// operations instead. The result count then matches neither mode, 3 being between
// the 2 of fusedIndex and the 4 of per-dimension.
//
// Two sibling krnl.collapse ops in one krnl.iterate is the shape this case needs,
// and it is rejected here by the op verifier, before lowering ever runs. Such a
// nest does lower correctly when the result count is right -- see
// collapse_two_sibling_groups in
// test/mlir/conversion/krnl_to_affine/krnl_to_affine_collapse.mlir.
func.func @get_iv_mixed_modes() {
  %ii, %jj, %kk, %ll = krnl.define_loops 4
  %f1 = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  %f2 = krnl.collapse(%kk, %ll) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%f1, %f2) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5, %kk -> %k = 0 to 6, %ll -> %l = 0 to 7) {
    // expected-error @+1 {{expects 4 results but has 3}}
    %a, %b, %c = krnl.get_induction_var_value(%f1, %f2) : (!krnl.loop, !krnl.loop) -> (index, index, index)
  }
  return
}
