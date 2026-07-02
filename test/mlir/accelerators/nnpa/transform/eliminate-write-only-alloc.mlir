// RUN: onnx-mlir-opt --march=z16 --maccel=NNPA --eliminate-write-only-alloc %s -split-input-file | FileCheck %s

// -----

// Direct affine.store with no load: alloc and stores are dead.
func.func @eliminate_direct_stores() {
  %c3  = arith.constant 3  : i64
  %c32 = arith.constant 32 : i64
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<2xi64>
  affine.store %c3,  %alloc[0] : memref<2xi64>
  affine.store %c32, %alloc[1] : memref<2xi64>
  return

  // CHECK-LABEL: eliminate_direct_stores
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: affine.store
}

// -----

// krnl.store variant.
func.func @eliminate_krnl_stores() {
  %c0 = arith.constant 0 : index
  %v  = arith.constant 42 : i64
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<1xi64>
  krnl.store %v, %alloc[%c0] : memref<1xi64>
  return

  // CHECK-LABEL: eliminate_krnl_stores
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: krnl.store
}

// -----

// The pattern that arises from fused-op concat shape lowering:
// affine.for loops that load from a krnl.global and store into the dead alloc.
// After eliminating the store, the dead load and empty loop are also removed.
func.func @eliminate_loop_pattern() {
  %src = "krnl.global"() <{name = "shape_cst", shape = [1],
      value = dense<96> : tensor<1xi64>}> : () -> memref<1xi64>
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<1xi64>
  affine.for %i = 0 to 1 {
    %v = affine.load %src[%i] : memref<1xi64>
    affine.store %v, %alloc[%i] : memref<1xi64>
  }
  return

  // CHECK-LABEL: eliminate_loop_pattern
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: affine.for
  // CHECK-NOT: affine.store
}

// -----

// Intermediate pure computation between the load and the dead store.
// The dead-load cleanup erases the arith.muli (memory-effect-free, unused
// result), then isDeadAffineFor erases the loop because the remaining
// affine.load has an unused result and only Read effects.
func.func @eliminate_loop_with_intermediate_computation() {
  %src = "krnl.global"() <{name = "shape_cst2", shape = [1],
      value = dense<48> : tensor<1xi64>}> : () -> memref<1xi64>
  %c2  = arith.constant 2 : i64
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<1xi64>
  affine.for %i = 0 to 1 {
    %v  = affine.load %src[%i] : memref<1xi64>
    %v2 = arith.muli %v, %c2 : i64
    affine.store %v2, %alloc[%i] : memref<1xi64>
  }
  return

  // CHECK-LABEL: eliminate_loop_with_intermediate_computation
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: affine.for
  // CHECK-NOT: affine.store
  // CHECK-NOT: arith.muli
}

// -----

// memref.dealloc is also cleaned up when the alloc is eliminated.
func.func @eliminate_with_dealloc() {
  %v = arith.constant 7 : i64
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<1xi64>
  affine.store %v, %alloc[0] : memref<1xi64>
  memref.dealloc %alloc : memref<1xi64>
  return

  // CHECK-LABEL: eliminate_with_dealloc
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: memref.dealloc
  // CHECK-NOT: affine.store
}

// -----

// Generalization: memref.load (not affine.load) feeds a memref.store into a
// dead alloc.  The interface-based dead-load cleanup erases the memref.load
// too, and the now-empty loop is removed.  The old hardcoded affine.load-only
// path would have left the memref.load and the loop behind.
func.func @generalized_memref_load_cleanup(%src: memref<1xi64>) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<1xi64>
  %c0 = arith.constant 0 : index
  affine.for %i = 0 to 1 {
    %v = memref.load %src[%c0] : memref<1xi64>
    memref.store %v, %alloc[%c0] : memref<1xi64>
  }
  return

  // CHECK-LABEL: generalized_memref_load_cleanup
  // CHECK-NOT: memref.alloc
  // CHECK-NOT: affine.for
  // CHECK-NOT: memref.load
  // CHECK-NOT: memref.store
}

// -----

// Correctness: a loop contains a dead store (to the eliminated alloc) alongside
// a live store (to a returned alloc).  The live store has no SSA result, so
// use_empty() is trivially true — the old isDeadAffineFor check would have
// incorrectly erased the loop.  The interface-based check sees the Write effect
// on the live store and correctly preserves the loop.
func.func @keep_loop_with_live_write(%arg0: i64) -> memref<1xi64> {
  %dead = memref.alloc() {alignment = 16 : i64} : memref<1xi64>
  %live = memref.alloc() {alignment = 16 : i64} : memref<1xi64>
  affine.for %i = 0 to 1 {
    affine.store %arg0, %dead[0] : memref<1xi64>
    affine.store %arg0, %live[0] : memref<1xi64>
  }
  return %live : memref<1xi64>

  // CHECK-LABEL: keep_loop_with_live_write
  // CHECK: memref.alloc
  // CHECK: affine.for
  // CHECK: affine.store %arg0
}

// -----

// Alloc that is read from must NOT be eliminated.
func.func @keep_when_loaded() -> i64 {
  %v = arith.constant 5 : i64
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<1xi64>
  affine.store %v, %alloc[0] : memref<1xi64>
  %result = affine.load %alloc[0] : memref<1xi64>
  memref.dealloc %alloc : memref<1xi64>
  return %result : i64

  // CHECK-LABEL: keep_when_loaded
  // CHECK: memref.alloc
  // CHECK: affine.store
  // CHECK: affine.load
}

// -----

// Alloc that escapes via return must NOT be eliminated.
func.func @keep_when_returned() -> memref<1xi64> {
  %v = arith.constant 9 : i64
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<1xi64>
  affine.store %v, %alloc[0] : memref<1xi64>
  return %alloc : memref<1xi64>

  // CHECK-LABEL: keep_when_returned
  // CHECK: memref.alloc
  // CHECK: affine.store
}

// -----

// An op (zlow.lstm) writes to two allocs: hn_output (live — returned) and
// cf_output (write-only from the alloc's perspective).  The fixpoint rule must
// keep BOTH allocs and the LSTM: cf_output cannot be removed because its only
// writer (the LSTM) also writes to the live hn_output.
func.func @keep_lstm_with_live_cowrite(
    %input: memref<1xf16>, %h0: memref<1xf16>, %c0: memref<1xf16>,
    %wi: memref<1xf16>, %ib: memref<1xf16>,
    %wh: memref<1xf16>, %hb: memref<1xf16>,
    %work: memref<1xi8>, %shape: memref<5xi64>) -> memref<1xf16> {
  %hn = memref.alloc() : memref<1xf16>
  %cf = memref.alloc() : memref<1xf16>
  "zlow.lstm"(%input, %h0, %c0, %wi, %ib, %wh, %hb, %work, %shape, %hn, %cf)
      <{direction = "forward", return_all_steps = -1 : si64,
        prev_layer = "none"}>
      : (memref<1xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<1xi8>, memref<5xi64>, memref<1xf16>, memref<1xf16>) -> ()
  return %hn : memref<1xf16>

  // CHECK-LABEL: keep_lstm_with_live_cowrite
  // CHECK-DAG:   memref.alloc() : memref<1xf16>
  // CHECK-DAG:   memref.alloc() : memref<1xf16>
  // CHECK:       zlow.lstm
}

// -----

// Both LSTM outputs are write-only (neither is returned or read).  Since every
// Write-target of the LSTM is itself a dead candidate, the fixpoint keeps both
// in the candidate set and all three (hn alloc, cf alloc, LSTM) are erased.
func.func @eliminate_lstm_both_outputs_dead(
    %input: memref<1xf16>, %h0: memref<1xf16>, %c0: memref<1xf16>,
    %wi: memref<1xf16>, %ib: memref<1xf16>,
    %wh: memref<1xf16>, %hb: memref<1xf16>,
    %work: memref<1xi8>, %shape: memref<5xi64>) {
  %hn = memref.alloc() : memref<1xf16>
  %cf = memref.alloc() : memref<1xf16>
  "zlow.lstm"(%input, %h0, %c0, %wi, %ib, %wh, %hb, %work, %shape, %hn, %cf)
      <{direction = "forward", return_all_steps = -1 : si64,
        prev_layer = "none"}>
      : (memref<1xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<1xi8>, memref<5xi64>, memref<1xf16>, memref<1xf16>) -> ()
  return

  // CHECK-LABEL: eliminate_lstm_both_outputs_dead
  // CHECK-NOT:   memref.alloc
  // CHECK-NOT:   zlow.lstm
}

// -----

// Fixpoint chain (two iterations required):
//   lstm1 writes to %alloc_a (candidate) and %alloc_b (candidate).
//   lstm2 writes to %alloc_b (candidate) and %hn2 (live — returned).
// Iteration 1: %alloc_b is pruned because lstm2 also writes to live %hn2.
// Iteration 2: %alloc_a is pruned because lstm1 also writes to %alloc_b
//              which is no longer a candidate.
// Result: neither alloc nor either LSTM is eliminated.
func.func @keep_lstm_fixpoint_chain(
    %input: memref<1xf16>, %h0: memref<1xf16>, %c0: memref<1xf16>,
    %wi: memref<1xf16>, %ib: memref<1xf16>,
    %wh: memref<1xf16>, %hb: memref<1xf16>,
    %work: memref<1xi8>, %shape: memref<5xi64>) -> memref<1xf16> {
  %alloc_a = memref.alloc() : memref<1xf16>
  %alloc_b = memref.alloc() : memref<1xf16>
  %hn2     = memref.alloc() : memref<1xf16>
  "zlow.lstm"(%input, %h0, %c0, %wi, %ib, %wh, %hb, %work, %shape,
              %alloc_a, %alloc_b)
      <{direction = "forward", return_all_steps = -1 : si64,
        prev_layer = "none"}>
      : (memref<1xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<1xi8>, memref<5xi64>, memref<1xf16>, memref<1xf16>) -> ()
  "zlow.lstm"(%input, %h0, %c0, %wi, %ib, %wh, %hb, %work, %shape,
              %hn2, %alloc_b)
      <{direction = "forward", return_all_steps = -1 : si64,
        prev_layer = "none"}>
      : (memref<1xf16>, memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<1xf16>, memref<1xf16>, memref<1xf16>,
         memref<1xi8>, memref<5xi64>, memref<1xf16>, memref<1xf16>) -> ()
  return %hn2 : memref<1xf16>

  // CHECK-LABEL: keep_lstm_fixpoint_chain
  // CHECK-DAG:   memref.alloc() : memref<1xf16>
  // CHECK-DAG:   memref.alloc() : memref<1xf16>
  // CHECK-DAG:   memref.alloc() : memref<1xf16>
  // CHECK:       zlow.lstm
  // CHECK:       zlow.lstm
}
