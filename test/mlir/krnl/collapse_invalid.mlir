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
