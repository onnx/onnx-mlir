// RUN: onnx-mlir-opt %s -split-input-file | FileCheck %s

// Parse/print round-trip of krnl.collapse, and of the two modes of
// krnl.get_induction_var_value against a collapsed loop reference.

// By default a collapsed loop reference yields one index per collapsed
// dimension, so one operand gives two results.
func.func @collapse_two() {
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %a, %b = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index)
    %foo = arith.addi %a, %b : index
  }
  return

  // CHECK-LABEL: collapse_two
  // CHECK: [[LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[FF:%.+]] = krnl.collapse([[LOOPS]]#0, [[LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  // CHECK: krnl.iterate([[FF]]) with ([[LOOPS]]#0 -> %{{.+}} = 0 to 10, [[LOOPS]]#1 -> %{{.+}} = 0 to 20)
  // CHECK: %{{.+}}:2 = krnl.get_induction_var_value([[FF]]) : (!krnl.loop) -> (index, index)
}

// -----

// Three dimensions at once; variadic collapse covers what nesting would.
func.func @collapse_three() {
  %ii, %jj, %kk = krnl.define_loops 3
  %ff = krnl.collapse(%ii, %jj, %kk) : (!krnl.loop, !krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5, %kk -> %k = 0 to 6) {
    %a, %b, %c = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index, index)
    %foo = arith.addi %a, %b : index
    %bar = arith.addi %foo, %c : index
  }
  return

  // CHECK-LABEL: collapse_three
  // CHECK: [[LOOPS:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[FF:%.+]] = krnl.collapse([[LOOPS]]#0, [[LOOPS]]#1, [[LOOPS]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> !krnl.loop
  // CHECK: %{{.+}}:3 = krnl.get_induction_var_value([[FF]]) : (!krnl.loop) -> (index, index, index)
}

// -----

// The fusedIndex attribute asks for the single fused index instead, so the
// result count drops back to one per operand.
func.func @collapse_fused_index() {
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %idx = krnl.get_induction_var_value(%ff) {fusedIndex} : (!krnl.loop) -> index
    %foo = arith.addi %idx, %idx : index
  }
  return

  // CHECK-LABEL: collapse_fused_index
  // CHECK: [[FF:%.+]] = krnl.collapse(%{{.+}}#0, %{{.+}}#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  // CHECK: %{{.+}} = krnl.get_induction_var_value([[FF]]) {fusedIndex} : (!krnl.loop) -> index
}

// -----

// Both modes in one body, as two queries on the same loop reference. This is how
// a body that wants a linearized access *and* per-dimension indices is written,
// the attribute applying to the whole operation.
func.func @collapse_both_modes() {
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %idx = krnl.get_induction_var_value(%ff) {fusedIndex} : (!krnl.loop) -> index
    %a, %b = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index)
    %foo = arith.addi %a, %b : index
    %bar = arith.addi %foo, %idx : index
  }
  return

  // CHECK-LABEL: collapse_both_modes
  // CHECK: %{{.+}} = krnl.get_induction_var_value([[FF:%.+]]) {fusedIndex} : (!krnl.loop) -> index
  // CHECK: %{{.+}}:2 = krnl.get_induction_var_value([[FF]]) : (!krnl.loop) -> (index, index)
}

// -----

// A collapsed and a plain loop reference queried together: 2 operands, 3
// results, since only the collapsed one expands.
func.func @collapse_with_plain_loop() {
  %ii, %jj, %kk = krnl.define_loops 3
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff, %kk) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5, %kk -> %k = 0 to 6) {
    %a, %b, %c = krnl.get_induction_var_value(%ff, %kk) : (!krnl.loop, !krnl.loop) -> (index, index, index)
    %foo = arith.addi %a, %b : index
    %bar = arith.addi %foo, %c : index
  }
  return

  // CHECK-LABEL: collapse_with_plain_loop
  // CHECK: [[LOOPS:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[FF:%.+]] = krnl.collapse([[LOOPS]]#0, [[LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  // CHECK: %{{.+}}:3 = krnl.get_induction_var_value([[FF]], [[LOOPS]]#2) : (!krnl.loop, !krnl.loop) -> (index, index, index)
}

// -----

// The motivating combination: parallelize the single fused loop.
func.func @collapse_then_parallel() {
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.parallel(%ff) : !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %a, %b = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> (index, index)
  }
  return

  // CHECK-LABEL: collapse_then_parallel
  // CHECK: [[LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[FF:%.+]] = krnl.collapse([[LOOPS]]#0, [[LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  // CHECK: krnl.parallel([[FF]]) : !krnl.loop
}
