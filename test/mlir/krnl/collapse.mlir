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

// fusedIndex against a mix of collapsed and plain loop references, with two
// collapse groups of different rank and an ordinary loop between them. In this
// mode arity is one result per operand whatever each operand is, so the mixed
// query takes 3 operands and yields 3 results -- against the 6 results the same
// operands give in the default mode of @collapse_with_plain_loop above.
//
// The second query drops %f2: a collapse result need not be queried at all, and
// the group that is queried keeps its own fused index either way. So one group
// here has its fused index used and the other does not.
func.func @collapse_fused_index_two_groups() {
  %d0, %d1, %d2, %d3, %d4, %d5 = krnl.define_loops 6
  %f1 = krnl.collapse(%d0, %d1, %d2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> !krnl.loop
  %f2 = krnl.collapse(%d4, %d5) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%f1, %d3, %f2) with (%d0 -> %i0 = 0 to 2, %d1 -> %i1 = 0 to 3, %d2 -> %i2 = 0 to 4, %d3 -> %i3 = 0 to 5, %d4 -> %i4 = 0 to 6, %d5 -> %i5 = 0 to 7) {
    // %p runs 0..23 over the first group, %q is the plain loop, %r runs 0..41
    // over the second.
    %p, %q, %r = krnl.get_induction_var_value(%f1, %d3, %f2) {fusedIndex} : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
    %s, %t = krnl.get_induction_var_value(%f1, %d3) {fusedIndex} : (!krnl.loop, !krnl.loop) -> (index, index)
    %foo = arith.addi %p, %r : index
    %bar = arith.addi %q, %s : index
    %baz = arith.addi %bar, %t : index
  }
  return

  // CHECK-LABEL: collapse_fused_index_two_groups
  // CHECK: [[LOOPS:%.+]]:6 = krnl.define_loops 6
  // CHECK: [[F1:%.+]] = krnl.collapse([[LOOPS]]#0, [[LOOPS]]#1, [[LOOPS]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> !krnl.loop
  // CHECK: [[F2:%.+]] = krnl.collapse([[LOOPS]]#4, [[LOOPS]]#5) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  // CHECK: krnl.iterate([[F1]], [[LOOPS]]#3, [[F2]]) with ([[LOOPS]]#0 -> %{{.+}} = 0 to 2, [[LOOPS]]#1 -> %{{.+}} = 0 to 3, [[LOOPS]]#2 -> %{{.+}} = 0 to 4, [[LOOPS]]#3 -> %{{.+}} = 0 to 5, [[LOOPS]]#4 -> %{{.+}} = 0 to 6, [[LOOPS]]#5 -> %{{.+}} = 0 to 7)
  // CHECK: %{{.+}}:3 = krnl.get_induction_var_value([[F1]], [[LOOPS]]#3, [[F2]]) {fusedIndex} : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
  // CHECK: %{{.+}}:2 = krnl.get_induction_var_value([[F1]], [[LOOPS]]#3) {fusedIndex} : (!krnl.loop, !krnl.loop) -> (index, index)
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
