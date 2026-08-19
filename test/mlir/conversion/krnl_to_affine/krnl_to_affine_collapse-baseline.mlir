// RUN: onnx-mlir-opt %s -split-input-file | FileCheck %s

// Baseline companion for krnl_to_affine_collapse.mlir, following
// GroundLitTest.py's default "<model>-baseline<ext>" convention.
//
// Each function here has the same name and the same computation as its
// counterpart in krnl_to_affine_collapse.mlir, but written with an ordinary
// uncollapsed loop nest -- no krnl.collapse, no krnl.collapse_indices. This
// file exists purely to feed GroundLitTest.py; it deliberately
// carries no FileCheck assertions about the lowering, and is never touched by
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

// CHECK-LABEL: func.func @collapse_base
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

// CHECK-LABEL: func.func @collapse_then_parallel
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

// CHECK-LABEL: func.func @collapse_then_permute
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

// CHECK-LABEL: func.func @collapse_three_dims
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

// CHECK-LABEL: func.func @collapse_lb_zero_via_constant
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

// CHECK-LABEL: func.func @collapse_raw_fused_index
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

// CHECK-LABEL: func.func @collapse_dynamic_dims
}

// -----

func.func @collapse_dynamic_and_static_dims(%arg0: memref<?x20xf32> {onnx.name = "x"}) -> (memref<?x20xf32> {onnx.name = "y"}) {
  %c0 = arith.constant 0 : index
  %c20 = arith.constant 20 : index
  %d0 = memref.dim %arg0, %c0 : memref<?x20xf32>
  %alloc = memref.alloc(%d0) {alignment = 16 : i64} : memref<?x20xf32>
  %ii, %jj = krnl.define_loops 2
  krnl.iterate(%ii, %jj) with (%ii -> %i = 0 to %d0, %jj -> %j = 0 to 20) {
    %a, %b = krnl.get_induction_var_value(%ii, %jj) : (!krnl.loop, !krnl.loop) -> (index, index)
    %v = krnl.load %arg0[%a, %b] : memref<?x20xf32>
    %row = arith.muli %a, %c20 : index
    %lin = arith.addi %row, %b : index
    %linI = arith.index_cast %lin : index to i64
    %linF = arith.sitofp %linI : i64 to f32
    %w = arith.addf %v, %linF : f32
    krnl.store %w, %alloc[%a, %b] : memref<?x20xf32>
  }
  return %alloc : memref<?x20xf32>

// CHECK-LABEL: func.func @collapse_dynamic_and_static_dims
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

// CHECK-LABEL: func.func @collapse_dynamic_then_parallel
}
