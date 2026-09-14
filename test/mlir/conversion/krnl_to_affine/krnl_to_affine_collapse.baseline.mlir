
// Baseline companion for krnl_to_affine_collapse.mlir, following
// GroundLitTest.py's default "<model>-baseline<ext>" convention.
//
// Each function here has the same name and the same computation as its
// counterpart in krnl_to_affine_collapse.mlir, but written with an ordinary
// uncollapsed loop nest -- no krnl.collapse at all. This file exists purely to
// feed GroundLitTest.py; it deliberately carries no FileCheck assertions about
// the lowering, and is never touched by
// fixLitTest.py's CHECK generation. The RUN line above only parse-checks it and
// confirms each function is still present under its expected name -- a silently
// broken or renamed baseline would otherwise disable the numerical comparison
// without any test going red. Each function carries its own CHECK-LABEL, inside
// its own split-input-file segment: a segment with no assertions at all is what
// fixLitTest.py treats as a test being written from scratch, and would have full
// lowering assertions generated into it by a plain "-r" run.
//
// (Do not spell the split marker out in a comment here: split-input-file cuts on
// that substring anywhere in the file, not just at the start of a line, so even
// a quoted mention inside prose splits the file mid-sentence.)

func.func @collapse_base(%arg0: memref<10x20xf32> {onnx.name = "x"}) -> (memref<10x20xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<10x20xf32>
  %ii, %jj = krnl.define_loops 2
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
    %v = krnl.load %arg0[%a, %b] : memref<10x20xf32>
    %c20 = arith.constant 20 : index
    %row = arith.muli %a, %c20 : index
    %lin = arith.addi %row, %b : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b] : memref<10x20xf32>
  }
  return %alloc : memref<10x20xf32>
}

// -----

func.func @collapse_then_parallel(%arg0: memref<10x20xf32> {onnx.name = "x"}) -> (memref<10x20xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<10x20xf32>
  %ii, %jj = krnl.define_loops 2
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
    %v = krnl.load %arg0[%a, %b] : memref<10x20xf32>
    %c20 = arith.constant 20 : index
    %row = arith.muli %a, %c20 : index
    %lin = arith.addi %row, %b : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b] : memref<10x20xf32>
  }
  return %alloc : memref<10x20xf32>
}

// -----

func.func @collapse_then_permute(%arg0: memref<4x5x6xf32> {onnx.name = "x"}) -> (memref<4x5x6xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<4x5x6xf32>
  %ii, %jj, %kk = krnl.define_loops 3
  krnl.iterate(%ii, %jj, %kk) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5, %kk -> %k = 0 to 6) {
    %a, %b, %c = krnl.get_induction_var_value(%ii, %jj, %kk) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
    %v = krnl.load %arg0[%a, %b, %c] : memref<4x5x6xf32>
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %t0 = arith.muli %a, %c5 : index
    %t1 = arith.addi %t0, %b : index
    %t2 = arith.muli %t1, %c6 : index
    %lin = arith.addi %t2, %c : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b, %c] : memref<4x5x6xf32>
  }
  return %alloc : memref<4x5x6xf32>
}

// -----

func.func @collapse_three_dims(%arg0: memref<4x5x6xf32> {onnx.name = "x"}) -> (memref<4x5x6xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<4x5x6xf32>
  %ii, %jj, %kk = krnl.define_loops 3
  krnl.iterate(%ii, %jj, %kk) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5, %kk -> %k = 0 to 6) {
    %a, %b, %c = krnl.get_induction_var_value(%ii, %jj, %kk) : (!krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index)
    %v = krnl.load %arg0[%a, %b, %c] : memref<4x5x6xf32>
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %t0 = arith.muli %a, %c5 : index
    %t1 = arith.addi %t0, %b : index
    %t2 = arith.muli %t1, %c6 : index
    %lin = arith.addi %t2, %c : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b, %c] : memref<4x5x6xf32>
  }
  return %alloc : memref<4x5x6xf32>
}

// -----

func.func @collapse_lb_zero_via_constant(%arg0: memref<10x20xf32> {onnx.name = "x"}) -> (memref<10x20xf32> {onnx.name = "y"}) {
  %c0 = arith.constant 0 : index
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<10x20xf32>
  %ii, %jj = krnl.define_loops 2
  krnl.iterate(%ii, %jj) with (%ii -> %i = %c0 to 10, %jj -> %j = %c0 to 20) {
    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
    %v = krnl.load %arg0[%a, %b] : memref<10x20xf32>
    %c20 = arith.constant 20 : index
    %row = arith.muli %a, %c20 : index
    %lin = arith.addi %row, %b : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b] : memref<10x20xf32>
  }
  return %alloc : memref<10x20xf32>
}

// -----

func.func @collapse_raw_fused_index(%arg0: memref<200xf32> {onnx.name = "x"}) -> (memref<200xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<200xf32>
  %ii, %jj = krnl.define_loops 2
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to 10, %jj -> %j = 0 to 20) {
    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
    %c20 = arith.constant 20 : index
    %row = arith.muli %a, %c20 : index
    %idx = arith.addi %row, %b : index
    %v = krnl.load %arg0[%idx] : memref<200xf32>
    %idxI = arith.index_cast %idx : index to i64
    %idxF = arith.sitofp %idxI : i64 to f32
    %w = arith.addf %v, %idxF : f32
    krnl.store %w, %alloc[%idx] : memref<200xf32>
  }
  return %alloc : memref<200xf32>
}

// -----

func.func @collapse_dynamic_dims(%arg0: memref<?x?xf32> {onnx.name = "x"}) -> (memref<?x?xf32> {onnx.name = "y"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %d1 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %alloc = memref.alloc(%d0, %d1) {alignment = 16 : i64} : memref<?x?xf32>
  %ii, %jj = krnl.define_loops 2
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to %d0, %jj -> %j = 0 to %d1) {
    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
    %v = krnl.load %arg0[%a, %b] : memref<?x?xf32>
    %row = arith.muli %a, %d1 : index
    %lin = arith.addi %row, %b : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b] : memref<?x?xf32>
  }
  return %alloc : memref<?x?xf32>
}

// -----

func.func @collapse_dynamic_and_static_dims(%arg0: memref<?x20xf32> {onnx.name = "x"}) -> (memref<?x20xf32> {onnx.name = "y"}) {
  %c0 = arith.constant 0 : index
  %d0 = memref.dim %arg0, %c0 : memref<?x20xf32>
  %alloc = memref.alloc(%d0) {alignment = 16 : i64} : memref<?x20xf32>
  %ii, %jj = krnl.define_loops 2
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to %d0, %jj -> %j = 0 to 20) {
    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
    %v = krnl.load %arg0[%a, %b] : memref<?x20xf32>
    %c20 = arith.constant 20 : index
    %row = arith.muli %a, %c20 : index
    %lin = arith.addi %row, %b : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b] : memref<?x20xf32>
  }
  return %alloc : memref<?x20xf32>
}

// -----

func.func @collapse_dynamic_then_parallel(%arg0: memref<?x?xf32> {onnx.name = "x"}) -> (memref<?x?xf32> {onnx.name = "y"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %d0 = memref.dim %arg0, %c0 : memref<?x?xf32>
  %d1 = memref.dim %arg0, %c1 : memref<?x?xf32>
  %alloc = memref.alloc(%d0, %d1) {alignment = 16 : i64} : memref<?x?xf32>
  %ii, %jj = krnl.define_loops 2
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to %d0, %jj -> %j = 0 to %d1) {
    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
    %v = krnl.load %arg0[%a, %b] : memref<?x?xf32>
    %row = arith.muli %a, %d1 : index
    %lin = arith.addi %row, %b : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b] : memref<?x?xf32>
  }
  return %alloc : memref<?x?xf32>
}

// -----

func.func @collapse_two_sibling_groups(%arg0: memref<4x5x6x7xf32> {onnx.name = "x"}) -> (memref<4x5x6x7xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<4x5x6x7xf32>
  %ii, %jj, %kk, %ll = krnl.define_loops 4
  krnl.iterate(%ii, %jj, %kk, %ll) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5, %kk -> %k = 0 to 6, %ll -> %l = 0 to 7) {
    %a, %b, %c, %d = krnl.get_induction_var_value(%ii, %jj, %kk, %ll) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index)
    %v = krnl.load %arg0[%a, %b, %c, %d] : memref<4x5x6x7xf32>
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %t0 = arith.muli %a, %c5 : index
    %t1 = arith.addi %t0, %b : index
    %t2 = arith.muli %t1, %c6 : index
    %t3 = arith.addi %t2, %c : index
    %t4 = arith.muli %t3, %c7 : index
    %lin = arith.addi %t4, %d : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b, %c, %d] : memref<4x5x6x7xf32>
  }
  return %alloc : memref<4x5x6x7xf32>
}

// -----

func.func @collapse_two_groups_and_plain_loop(%arg0: memref<2x3x4x5x6x7xf32> {onnx.name = "x"}) -> (memref<2x3x4x5x6x7xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<2x3x4x5x6x7xf32>
  %d0, %d1, %d2, %d3, %d4, %d5 = krnl.define_loops 6
  krnl.iterate(%d0, %d1, %d2, %d3, %d4, %d5) with (%d0 -> %i0 = 0 to 2, %d1 -> %i1 = 0 to 3, %d2 -> %i2 = 0 to 4, %d3 -> %i3 = 0 to 5, %d4 -> %i4 = 0 to 6, %d5 -> %i5 = 0 to 7) {
    %a, %b, %c, %d, %e, %g = krnl.get_induction_var_value(%d0, %d1, %d2, %d3, %d4, %d5) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index, index)
    %v = krnl.load %arg0[%a, %b, %c, %d, %e, %g] : memref<2x3x4x5x6x7xf32>
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c5 = arith.constant 5 : index
    %c6 = arith.constant 6 : index
    %c7 = arith.constant 7 : index
    %s0 = arith.muli %a, %c3 : index
    %s1 = arith.addi %s0, %b : index
    %s2 = arith.muli %s1, %c4 : index
    %s3 = arith.addi %s2, %c : index
    %s4 = arith.muli %s3, %c5 : index
    %s5 = arith.addi %s4, %d : index
    %s6 = arith.muli %s5, %c6 : index
    %s7 = arith.addi %s6, %e : index
    %s8 = arith.muli %s7, %c7 : index
    %lin1 = arith.addi %s8, %g : index
    %lin1I = arith.index_cast %lin1 : index to i64
    %lin1F = arith.sitofp %lin1I : i64 to f32
    %w = arith.addf %v, %lin1F : f32
    krnl.store %w, %alloc[%a, %b, %c, %d, %e, %g] : memref<2x3x4x5x6x7xf32>
  }
  return %alloc : memref<2x3x4x5x6x7xf32>
}

// -----

func.func @collapse_fused_index_two_groups(%arg0: memref<5040xf32> {onnx.name = "x"}) -> (memref<5040xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<5040xf32>
  %d0, %d1, %d2, %d3, %d4, %d5 = krnl.define_loops 6
  krnl.iterate(%d0, %d1, %d2, %d3, %d4, %d5) with (%d0 -> %i0 = 0 to 2, %d1 -> %i1 = 0 to 3, %d2 -> %i2 = 0 to 4, %d3 -> %i3 = 0 to 5, %d4 -> %i4 = 0 to 6, %d5 -> %i5 = 0 to 7) {
    // The two fused indices the collapsed variant queries, derived by hand: %p is
    // the row-major linearization of the first group, %r that of the second.
    %a, %b, %c, %q, %e, %g = krnl.get_induction_var_value(%d0, %d1, %d2, %d3, %d4, %d5) : (!krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop, !krnl.loop) -> (index, index, index, index, index, index)
    %c3 = arith.constant 3 : index
    %c4 = arith.constant 4 : index
    %c7 = arith.constant 7 : index
    %p0 = arith.muli %a, %c3 : index
    %p1 = arith.addi %p0, %b : index
    %p2 = arith.muli %p1, %c4 : index
    %p = arith.addi %p2, %c : index
    %r0 = arith.muli %e, %c7 : index
    %r = arith.addi %r0, %g : index
    %c5 = arith.constant 5 : index
    %c42 = arith.constant 42 : index
    %t0 = arith.muli %p, %c5 : index
    %t1 = arith.addi %t0, %q : index
    %t2 = arith.muli %t1, %c42 : index
    %lin = arith.addi %t2, %r : index
    %v = krnl.load %arg0[%lin] : memref<5040xf32>
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%lin] : memref<5040xf32>
  }
  return %alloc : memref<5040xf32>
}

// -----

func.func @collapse_nested_iterate_outer(%arg0: memref<4x5x6x7xf32> {onnx.name = "x"}) -> (memref<4x5x6x7xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<4x5x6x7xf32>
  %ii, %jj, %kk, %ll = krnl.define_loops 4
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5) {
    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
    krnl.iterate(%kk, %ll) with (%kk -> %k = 0 to 6, %ll -> %l = 0 to 7) {
      %c, %d = krnl.get_induction_var_value(%kk, %ll) : (!krnl.loop, !krnl.loop) -> (index, index)
      %v = krnl.load %arg0[%a, %b, %c, %d] : memref<4x5x6x7xf32>
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %t0 = arith.muli %a, %c5 : index
      %t1 = arith.addi %t0, %b : index
      %t2 = arith.muli %t1, %c6 : index
      %t3 = arith.addi %t2, %c : index
      %t4 = arith.muli %t3, %c7 : index
      %lin = arith.addi %t4, %d : index
      %linI = arith.index_cast %lin : index to i64
      %linF = arith.sitofp %linI : i64 to f32
      %w = arith.addf %v, %linF : f32
      krnl.store %w, %alloc[%a, %b, %c, %d] : memref<4x5x6x7xf32>
    }
  }
  return %alloc : memref<4x5x6x7xf32>
}

// -----

func.func @collapse_nested_iterate_inner(%arg0: memref<4x5x6x7xf32> {onnx.name = "x"}) -> (memref<4x5x6x7xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<4x5x6x7xf32>
  %ii, %jj, %kk, %ll = krnl.define_loops 4
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5) {
    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
    krnl.iterate(%kk, %ll) with (%kk -> %k = 0 to 6, %ll -> %l = 0 to 7) {
      %c, %d = krnl.get_induction_var_value(%kk, %ll) : (!krnl.loop, !krnl.loop) -> (index, index)
      %v = krnl.load %arg0[%a, %b, %c, %d] : memref<4x5x6x7xf32>
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %t0 = arith.muli %a, %c5 : index
      %t1 = arith.addi %t0, %b : index
      %t2 = arith.muli %t1, %c6 : index
      %t3 = arith.addi %t2, %c : index
      %t4 = arith.muli %t3, %c7 : index
      %lin = arith.addi %t4, %d : index
      %linI = arith.index_cast %lin : index to i64
      %linF = arith.sitofp %linI : i64 to f32
      %w = arith.addf %v, %linF : f32
      krnl.store %w, %alloc[%a, %b, %c, %d] : memref<4x5x6x7xf32>
    }
  }
  return %alloc : memref<4x5x6x7xf32>
}

// -----

func.func @collapse_nested_iterate_both(%arg0: memref<4x5x6x7xf32> {onnx.name = "x"}) -> (memref<4x5x6x7xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<4x5x6x7xf32>
  %ii, %jj, %kk, %ll = krnl.define_loops 4
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5) {
    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
    krnl.iterate(%kk, %ll) with (%kk -> %k = 0 to 6, %ll -> %l = 0 to 7) {
      %c, %d = krnl.get_induction_var_value(%kk, %ll) : (!krnl.loop, !krnl.loop) -> (index, index)
      %v = krnl.load %arg0[%a, %b, %c, %d] : memref<4x5x6x7xf32>
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %t0 = arith.muli %a, %c5 : index
      %t1 = arith.addi %t0, %b : index
      %t2 = arith.muli %t1, %c6 : index
      %t3 = arith.addi %t2, %c : index
      %t4 = arith.muli %t3, %c7 : index
      %lin = arith.addi %t4, %d : index
      %linI = arith.index_cast %lin : index to i64
      %linF = arith.sitofp %linI : i64 to f32
      %w = arith.addf %v, %linF : f32
      krnl.store %w, %alloc[%a, %b, %c, %d] : memref<4x5x6x7xf32>
    }
  }
  return %alloc : memref<4x5x6x7xf32>
}

// -----

func.func @collapse_nested_iterate_both_dynamic(%arg0: memref<?x?x?x?xf32> {onnx.name = "x"}) -> (memref<?x?x?x?xf32> {onnx.name = "y"}) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2 = arith.constant 2 : index
  %c3 = arith.constant 3 : index
  %d0 = memref.dim %arg0, %c0 : memref<?x?x?x?xf32>
  %d1 = memref.dim %arg0, %c1 : memref<?x?x?x?xf32>
  %d2 = memref.dim %arg0, %c2 : memref<?x?x?x?xf32>
  %d3 = memref.dim %arg0, %c3 : memref<?x?x?x?xf32>
  %alloc = memref.alloc(%d0, %d1, %d2, %d3) {alignment = 16 : i64} : memref<?x?x?x?xf32>
  %ii, %jj, %kk, %ll = krnl.define_loops 4
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to %d0, %jj -> %j = 0 to %d1) {
    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
    krnl.iterate(%kk, %ll) with (%kk -> %k = 0 to %d2, %ll -> %l = 0 to %d3) {
      %c, %d = krnl.get_induction_var_value(%kk, %ll) : (!krnl.loop, !krnl.loop) -> (index, index)
      %v = krnl.load %arg0[%a, %b, %c, %d] : memref<?x?x?x?xf32>
      %t0 = arith.muli %a, %d1 : index
      %t1 = arith.addi %t0, %b : index
      %t2 = arith.muli %t1, %d2 : index
      %t3 = arith.addi %t2, %c : index
      %t4 = arith.muli %t3, %d3 : index
      %lin = arith.addi %t4, %d : index
      %linI = arith.index_cast %lin : index to i64
      %linF = arith.sitofp %linI : i64 to f32
      %w = arith.addf %v, %linF : f32
      krnl.store %w, %alloc[%a, %b, %c, %d] : memref<?x?x?x?xf32>
    }
  }
  return %alloc : memref<?x?x?x?xf32>
}

// -----

func.func @collapse_nested_iterate_both_then_parallel(%arg0: memref<4x5x6x7xf32> {onnx.name = "x"}) -> (memref<4x5x6x7xf32> {onnx.name = "y"}) {
  %alloc = memref.alloc() {alignment = 16 : i64} : memref<4x5x6x7xf32>
  %ii, %jj, %kk, %ll = krnl.define_loops 4
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to 4, %jj -> %j = 0 to 5) {
    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
    krnl.iterate(%kk, %ll) with (%kk -> %k = 0 to 6, %ll -> %l = 0 to 7) {
      %c, %d = krnl.get_induction_var_value(%kk, %ll) : (!krnl.loop, !krnl.loop) -> (index, index)
      %v = krnl.load %arg0[%a, %b, %c, %d] : memref<4x5x6x7xf32>
      %c5 = arith.constant 5 : index
      %c6 = arith.constant 6 : index
      %c7 = arith.constant 7 : index
      %t0 = arith.muli %a, %c5 : index
      %t1 = arith.addi %t0, %b : index
      %t2 = arith.muli %t1, %c6 : index
      %t3 = arith.addi %t2, %c : index
      %t4 = arith.muli %t3, %c7 : index
      %lin = arith.addi %t4, %d : index
      %linI = arith.index_cast %lin : index to i64
      %linF = arith.sitofp %linI : i64 to f32
      %w = arith.addf %v, %linF : f32
      krnl.store %w, %alloc[%a, %b, %c, %d] : memref<4x5x6x7xf32>
    }
  }
  return %alloc : memref<4x5x6x7xf32>
}
