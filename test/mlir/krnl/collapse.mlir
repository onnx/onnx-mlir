// RUN: onnx-mlir-opt %s -split-input-file | FileCheck %s

// Parse/print round-trip of krnl.collapse and krnl.collapse_indices.

func.func @collapse_two() {
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %idx = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> index
    %a, %b = krnl.collapse_indices(%idx) : (index) -> (index, index)
    %foo = arith.addi %a, %b : index
  }
  return

  // CHECK-LABEL: collapse_two
  // CHECK: [[LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[FF:%.+]] = krnl.collapse([[LOOPS]]#0, [[LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  // CHECK: krnl.iterate([[FF]]) with ([[LOOPS]]#0 -> %{{.+}} = 0 to 10, [[LOOPS]]#1 -> %{{.+}} = 0 to 20)
  // CHECK: [[IDX:%.+]] = krnl.get_induction_var_value([[FF]]) : (!krnl.loop) -> index
  // CHECK: %{{.+}}:2 = krnl.collapse_indices([[IDX]]) : (index) -> (index, index)
}

// -----

// Three dimensions at once; variadic collapse covers what nesting would.
func.func @collapse_three() {
  %ii, %jj, %kk = krnl.define_loops 3
  %ff = krnl.collapse(%ii, %jj, %kk) : (!krnl.loop, !krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5, %kk -> %k = 0 to 6) {
    %idx = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> index
    %a, %b, %c = krnl.collapse_indices(%idx) : (index) -> (index, index, index)
    %foo = arith.addi %a, %b : index
    %bar = arith.addi %foo, %c : index
  }
  return

  // CHECK-LABEL: collapse_three
  // CHECK: [[LOOPS:%.+]]:3 = krnl.define_loops 3
  // CHECK: [[FF:%.+]] = krnl.collapse([[LOOPS]]#0, [[LOOPS]]#1, [[LOOPS]]#2) : (!krnl.loop, !krnl.loop, !krnl.loop) -> !krnl.loop
  // CHECK: %{{.+}}:3 = krnl.collapse_indices(%{{.+}}) : (index) -> (index, index, index)
}

// -----

// The motivating combination: parallelize the single fused loop.
func.func @collapse_then_parallel() {
  %ii, %jj = krnl.define_loops 2
  %ff = krnl.collapse(%ii, %jj) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  krnl.parallel(%ff) : !krnl.loop
  krnl.iterate(%ff) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %idx = krnl.get_induction_var_value(%ff) : (!krnl.loop) -> index
  }
  return

  // CHECK-LABEL: collapse_then_parallel
  // CHECK: [[LOOPS:%.+]]:2 = krnl.define_loops 2
  // CHECK: [[FF:%.+]] = krnl.collapse([[LOOPS]]#0, [[LOOPS]]#1) : (!krnl.loop, !krnl.loop) -> !krnl.loop
  // CHECK: krnl.parallel([[FF]]) : !krnl.loop
}
