// RUN: onnx-mlir-opt -O3 --convert-krnl-to-affine %s -split-input-file | FileCheck %s

// yield value on a loop with unroll.
func.func @unroll() -> index {
  %cst = arith.constant 0 : index
  %ii = krnl.define_loops 1
  %alloca = memref.alloca() : memref<index>
  krnl.store %cst, %alloca[] : memref<index>
  %ii1, %ii2 = krnl.block %ii 4 : (!krnl.loop) -> (!krnl.loop, !krnl.loop)
  krnl.unroll %ii2 : !krnl.loop
  krnl.iterate(%ii1) with (%ii -> %i = 0 to 8) {
    %0 = krnl.iterate(%ii2) with () iter_args(%arg1 = %cst) -> (index) {
      %i2 = krnl.get_induction_var_value(%ii2) : (!krnl.loop) -> index
      %foo = arith.addi %arg1, %i2 : index
	  krnl.yield %foo : index
    }
	%ld = krnl.load %alloca[] : memref<index>
	%add = arith.addi %0, %ld : index
	krnl.store %add, %alloca[] : memref<index>
  }
  %ld1 = krnl.load %alloca[] : memref<index>
  return %ld1 : index
  // #map = affine_map<(d0) -> (d0 + 1)>
  // #map1 = affine_map<(d0) -> (d0 + 2)>
  // #map2 = affine_map<(d0) -> (d0 + 3)>
  // CHECK-LABEL: unroll
  // CHECK: [[Cst:%.+]] = arith.constant 0 : index
  // CHECK: [[Alloca:%.+]] = memref.alloca() : memref<index>
  // CHECK:   affine.store [[Cst]], %alloca[] : memref<index>
  // CHECK:  affine.for %arg0 = 0 to 8 step 4 {
  // CHECK: [[Add0:%.+]] = arith.addi %c0, %arg0 : index
  // CHECK: [[CST:%.+]] = affine.apply #map(%arg0)
  // CHECK: [[Add1:%.+]] = arith.addi %1, %2 : index
  // CHECK: [[CST:%.+]] = affine.apply #map1(%arg0)
  // CHECK: [[Add2:%.+]] = arith.addi %3, %4 : index
  // CHECK: [[CST:%.+]] = affine.apply #map2(%arg0)
  // CHECK: [[Add3:%.+]] = arith.addi %5, %6 : index
  // CHECK: [[AllocaLd:%.+]] = affine.load %alloca[] : memref<index>
  // CHECK: [[AllocaAdd:%.+]] = arith.addi %7, %8 : index
  // CHECK:   affine.store [[AllocaAdd]], %alloca[] : memref<index>
  // CHECK: [[AllocaLd2:%.+]] = affine.load %alloca[] : memref<index>
  // CHECK: return [[AllocaLd2]] : index

}

// -----

// yield value when iterate is not a loop.
func.func @no_loop(%arg0: memref<f32>) -> memref<f32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.alloc() : memref<f32>
  %1 = krnl.iterate() with () iter_args(%arg1 = %cst) -> (f32) {
    %2 = memref.load %arg0[] : memref<f32>
    memref.store %2, %0[] : memref<f32>
    %3 = arith.addf %arg1, %2 : f32
    krnl.yield %3 : f32
  }
  %4 = memref.load %arg0[] : memref<f32>
  %5 = arith.addf %4, %1 : f32
  memref.store %5, %0[] : memref<f32>
  return %0 : memref<f32>
  // CHECK-LABEL: no_loop
  // CHECK: [[CST:%.+]] = arith.constant 0.000000e+00 : f32
  // CHECK-NEXT: [[ALLOC:%.+]] = memref.alloc() : memref<f32>
  // CHECK-NEXT: [[ArgLD:%.+]] = memref.load %arg0[] : memref<f32>
  // CHECK-NEXT: memref.store [[ArgLD]], [[ALLOC]][] : memref<f32>
  // CHECK-NEXT: [[CstAdd:%.+]] = arith.addf [[CST]], [[ArgLD]] : f32
  // CHECK-NEXT: [[ArgLD2:%.+]] = memref.load %arg0[] : memref<f32>
  // CHECK-NEXT: [[Add:%.+]] = arith.addf [[ArgLD2]], [[CstAdd]] : f32
  // CHECK-NEXT: memref.store [[Add]], [[ALLOC]][] : memref<f32>
  // CHECK-NEXT: return [[ALLOC]] : memref<f32>

}

// -----

// yield more than one value.
func.func @yield2(%arg0: memref<10x10xf32>, %arg1: memref<10x?xf32>) -> memref<10x10xf32> {
  %cst = arith.constant 0.000000e+00 : f32
  %0 = memref.alloc() : memref<10x10xf32>
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %1 = memref.dim %arg1, %c1 : memref<10x?xf32>
  %2:2 = krnl.define_loops 2
  %y0, %y1 = krnl.iterate(%2#0, %2#1) with (%2#0 -> %arg2 = 0 to 10, %2#1 -> %arg3 = 0 to 10) iter_args(%arg4 = %cst, %arg5 = %cst) -> (f32, f32) {
    %3 = krnl.load %arg0[%arg2, %arg3] : memref<10x10xf32>
    %4 = arith.cmpi sgt, %1, %c1 : index
    %5 = arith.select %4, %arg3, %c0 : index
    %6 = krnl.load %arg1[%arg2, %5] : memref<10x?xf32>
    %7 = arith.addf %3, %6 : f32
    krnl.store %7, %0[%arg2, %arg3] : memref<10x10xf32>
    %8 = arith.addf %arg4, %7 : f32
    %9 = arith.mulf %arg5, %7 : f32
    krnl.yield %8, %9 : f32, f32
  }
  %10 = arith.addf %y0, %y1 : f32
  memref.store %10, %arg0[%c0, %c0] : memref<10x10xf32>
  return %0 : memref<10x10xf32>
	// CHECK-LABEL: yield2
	// CHECK: [[Dim:%.+]] = memref.dim %arg1, %c1 : memref<10x?xf32>
	// CHECK: [[Cmp:%.+]] = arith.cmpi sgt, [[Dim]], %c1 : index
	// CHECK: [[For0:%.+]]:2 = affine.for %arg2 = 0 to 10 iter_args(%arg3 = %cst, %arg4 = %cst) -> (f32, f32) {
	// CHECK: [[For1:%.+]]:2 = affine.for %arg5 = 0 to 10 iter_args(%arg6 = %arg3, %arg7 = %arg4) -> (f32, f32) {
	// CHECK: [[Arg0Ld:%.+]] = affine.load %arg0[%arg2, %arg5] : memref<10x10xf32>
	// CHECK: [[Sel:%.+]] = arith.select [[Cmp]], %arg5, %c0 : index
	// CHECK: [[Arg1Ld:%.+]] = memref.load %arg1[%arg2, [[Sel]]] : memref<10x?xf32>
	// CHECK: [[Add:%.+]] = arith.addf [[Arg0Ld]], [[Arg1Ld]] : f32
	// CHECK:       affine.store [[Add]], %alloc[%arg2, %arg5] : memref<10x10xf32>
	// CHECK: [[Add2:%.+]] = arith.addf %arg6, [[Add]] : f32
	// CHECK: [[Mul:%.+]] = arith.mulf %arg7, [[Add]] : f32
	// CHECK:        affine.yield [[Add2]], [[Mul]] : f32, f32
	// CHECK:      affine.yield [[For1]]#0, [[For1]]#1 : f32, f32
	// CHECK:   [[Add3:%.+]] = arith.addf [[For0]]#0, [[For0]]#1 : f32
	// CHECK:    memref.store [[Add3]], %arg0[%c0, %c0] : memref<10x10xf32>
}



// -----

// yield in outer iterate only.
func.func @outer(%arg0: memref<3x4xf32>, %arg1: memref<4x3xf32> ) -> (memref<3x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<3x3xf32>
  %0:3 = krnl.define_loops 3
  %i = krnl.iterate(%0#0, %0#1) with (%0#0 -> %arg2 = 0 to 3, %0#1 -> %arg3 = 0 to 3, %0#2 -> %arg4 = 0 to 4) iter_args(%arg5 = %cst) -> (f32){
    %1:2 = krnl.get_induction_var_value(%0#0, %0#1) : (!krnl.loop, !krnl.loop) -> (index, index)
      krnl.iterate(%0#2) with () {
      %3 = krnl.get_induction_var_value(%0#2) : (!krnl.loop) -> index
      %4 = krnl.load %arg0[%1#0, %3] : memref<3x4xf32>
      %5 = krnl.load %arg1[%3, %1#1] : memref<4x3xf32>
      %6 = arith.mulf %4, %5 : f32
      %7 = arith.addf %arg5, %6 : f32
      krnl.store %7, %alloc[%1#0, %1#1] : memref<3x3xf32>
      krnl.yield
    }
    %8 = krnl.load %alloc[%1#0, %1#1] : memref<3x3xf32>
    %9 = arith.addf %arg5, %8 : f32
    krnl.yield %9 : f32
  }
  return %alloc : memref<3x3xf32>
	// CHECK-LABEL: outer
	// CHECK: [[For0:%.+]] = affine.for %arg2 = 0 to 3 iter_args(%arg3 = %cst) -> (f32) {
	// CHECK: [[For1:%.+]] = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %arg3) -> (f32) {
	// CHECK:        affine.for %arg6 = 0 to 4 {
	// CHECK: [[Arg0Ld:%.+]] = affine.load %arg0[%arg2, %arg6] : memref<3x4xf32>
	// CHECK: [[Arg1Ld:%.+]] = affine.load %arg1[%arg6, %arg4] : memref<4x3xf32>
	// CHECK: [[Mul:%.+]] = arith.mulf %4, %5 : f32
	// CHECK: [[Add:%.+]] = arith.addf %arg5, %6 : f32
	// CHECK:          affine.store [[Add]], %alloc[%arg2, %arg4] : memref<3x3xf32>
	// CHECK: [[AllocLd:%.+]] = affine.load %alloc[%arg2, %arg4] : memref<3x3xf32>
	// CHECK: [[Add2:%.+]] = arith.addf %arg5, [[AllocLd]] : f32
	// CHECK:        affine.yield [[Add2]] : f32
	// CHECK:      affine.yield [[For1]] : f32
}


// -----

// yield in inner iterate only.
func.func @inner(%arg0: memref<3x4xf32>, %arg1: memref<4x3xf32> ) -> (memref<3x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<3x3xf32>
  %0:3 = krnl.define_loops 3
  krnl.iterate(%0#0, %0#1) with (%0#0 -> %arg2 = 0 to 3, %0#1 -> %arg3 = 0 to 3, %0#2 -> %arg4 = 0 to 4) {
    %1:2 = krnl.get_induction_var_value(%0#0, %0#1) : (!krnl.loop, !krnl.loop) -> (index, index)
    %2 = krnl.iterate(%0#2) with () iter_args(%arg5 = %cst) -> (f32){
      %3 = krnl.get_induction_var_value(%0#2) : (!krnl.loop) -> index
      %4 = krnl.load %arg0[%1#0, %3] : memref<3x4xf32>
      %5 = krnl.load %arg1[%3, %1#1] : memref<4x3xf32>
      %6 = arith.mulf %4, %5 : f32
      %7 = arith.addf %arg5, %6 : f32
      krnl.yield %7 : f32
    }
    krnl.store %2, %alloc[%1#0, %1#1] : memref<3x3xf32>
    krnl.yield
  }
  return %alloc : memref<3x3xf32>
	// CHECK-LABEL: inner
	// CHECK:    affine.for %arg2 = 0 to 3 {
	// CHECK:      affine.for %arg3 = 0 to 3 {
	// CHECK: [[For2:%.+]] = affine.for %arg4 = 0 to 4 iter_args(%arg5 = %cst) -> (f32) {
	// CHECK: [[Arg0Ld:%.+]] = affine.load %arg0[%arg2, %arg4] : memref<3x4xf32>
	// CHECK: [[Arg1Ld:%.+]] = affine.load %arg1[%arg4, %arg3] : memref<4x3xf32>
	// CHECK: [[Mul:%.+]] = arith.mulf [[Arg0Ld]], [[Arg1Ld]] : f32
	// CHECK: [[Add:%.+]] = arith.addf %arg5, [[Mul]] : f32
	// CHECK:          affine.yield [[Add]] : f32
	// CHECK:        affine.store [[For2]], %alloc[%arg2, %arg3] : memref<3x3xf32>
}


// -----

// yield in both inner and outter iterate.
func.func @both(%arg0: memref<3x4xf32>, %arg1: memref<4x3xf32> ) -> (memref<3x3xf32>) {
  %cst = arith.constant 0.000000e+00 : f32
  %alloc = memref.alloc() : memref<3x3xf32>
  %0:3 = krnl.define_loops 3
  %i = krnl.iterate(%0#0, %0#1) with (%0#0 -> %arg2 = 0 to 3, %0#1 -> %arg3 = 0 to 3, %0#2 -> %arg4 = 0 to 4) iter_args(%arg5 = %cst) -> (f32){
    %1:2 = krnl.get_induction_var_value(%0#0, %0#1) : (!krnl.loop, !krnl.loop) -> (index, index)
    %2 = krnl.iterate(%0#2) with () iter_args(%arg6 = %arg5) -> (f32){
      %3 = krnl.get_induction_var_value(%0#2) : (!krnl.loop) -> index
      %4 = krnl.load %arg0[%1#0, %3] : memref<3x4xf32>
      %5 = krnl.load %arg1[%3, %1#1] : memref<4x3xf32>
      %6 = arith.mulf %4, %5 : f32
      %7 = arith.addf %arg6, %6 : f32
      krnl.yield %7 : f32
    }
    krnl.store %2, %alloc[%1#0, %1#1] : memref<3x3xf32>
    krnl.yield %2 : f32
  }
  return %alloc : memref<3x3xf32>
	// CHECK-LABEL: both
	// CHECK: [[For0:%.+]] = affine.for %arg2 = 0 to 3 iter_args(%arg3 = %cst) -> (f32) {
	// CHECK: [[For1:%.+]] = affine.for %arg4 = 0 to 3 iter_args(%arg5 = %arg3) -> (f32) {
	// CHECK: [[For2:%.+]] = affine.for %arg6 = 0 to 4 iter_args(%arg7 = %arg5) -> (f32) {
	// CHECK: [[Arg0Ld:%.+]] = affine.load %arg0[%arg2, %arg6] : memref<3x4xf32>
	// CHECK: [[Arg1Ld:%.+]] = affine.load %arg1[%arg6, %arg4] : memref<4x3xf32>
	// CHECK: [[Mul:%.+]] = arith.mulf [[Arg0Ld]], [[Arg1Ld]] : f32
	// CHECK: [[Add:%.+]] = arith.addf %arg7, [[Mul]] : f32
	// CHECK:        affine.yield [[Add]] : f32
	// CHECK:      affine.store [[For2]], %alloc[%arg2, %arg4] : memref<3x3xf32>
	// CHECK:      affine.yield [[For2]] : f32
	// CHECK:   affine.yield [[For1]] : f32
}
