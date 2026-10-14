// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s
// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl=enable-parallel %s -split-input-file | FileCheck %s --check-prefix=PAR
// RUN: onnx-mlir-opt -O3 --march=z16 --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s --check-prefix=SIMD

// NonZero lowers to a flat-tiled stream compaction: X is viewed as one flat run of
// M elements cut into blocks; a per-block nonzero count is prefix-summed so every
// block knows which output columns it owns, then each block writes them.
//
// The tile size is derived from a target *block count* (64), and is chosen to
// divide the product of the static dims where possible, which proves at compile
// time that no block is short and drops the guarded tail path. "arith.cmpi sle" is
// the tell: it is the per-block "is this block full?" test, so its absence means
// the tail path was proven away.
//
// The CHECK lines are the scalar lowering: onnx-mlir-opt only turns SIMD on at
// -O3, which the SIMD-prefixed run below supplies.
//
// This file pins the shape of the generated IR only. The answers are checked
// separately against numpy.nonzero, which is the specification for this op.
// See src/Conversion/ONNXToKrnl/Tensor/NonZero.cpp.

func.func @test_nonzero_rank1(%arg0: tensor<8192xi64>) -> tensor<1x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<8192xi64>) -> tensor<1x?xi64>
  return %0 : tensor<1x?xi64>
}

// M = 8192 with a 64-block target gives 128 elements per block and 64 blocks, so
// nzPerBlock has 65 slots. 128 divides 8192, hence no tail path. Rank 1 is already
// flat, so no reshape is emitted either.
//
// nzPerBlock is the only array. The three passes each keep one running scalar, and
// sequentially their lifetimes do not overlap, so a single alloca outside all the
// loops serves all three -- there must be no alloca inside a loop here. The PAR
// checks at the bottom of this file cover the parallel case, where passes 1 and 3
// must allocate inside the block body instead.
// CHECK-LABEL:  func.func @test_nonzero_rank1
// CHECK-NOT:      memref.reshape
// CHECK-NOT:      arith.cmpi sle
// CHECK:          [[NZ:%.+]] = memref.alloc() {{.*}} : memref<65xindex>
// CHECK:          [[TMP:%.+]] = memref.alloca() : memref<index>
// Pass 1: count, into the shared scalar. Constant trip count, no bound check. The
// block's total lands in nzPerBlock slot b+1.
// CHECK:          krnl.iterate
// CHECK:            krnl.store {{.*}}, [[TMP]]
// CHECK:            krnl.iterate
// CHECK:              arith.cmpi eq, {{.*}} : i64
// CHECK:              arith.select
// CHECK:              krnl.load [[TMP]]
// CHECK:              krnl.store {{.*}}, [[TMP]]
// CHECK:            krnl.load [[TMP]]
// CHECK:            krnl.store {{.*}}, [[NZ]]
// Slot 0 of nzPerBlock is seeded here, between the two passes -- pass 1 never
// writes it, and it exists only to seed this scan. Its position relative to the two
// krnl.iterate ops is what these lines pin down.
// CHECK:          krnl.store {{.*}}, [[NZ]]{{.}}%c0
// Pass 2: serial prefix sum.
// CHECK:          krnl.iterate
// CHECK:            krnl.store {{.*}}, [[NZ]]
// CHECK:          [[OUT:%.+]] = memref.alloc({{.*}}) {{.*}} : memref<1x?xi64>
// Pass 3: skip empty blocks, then the guarded store. Rank 1 needs no unravelling,
// so the flat index is stored directly with no division at all.
//
// Pass 3 reuses the same scalar as the running output column.
// CHECK:          krnl.iterate
// CHECK:            arith.cmpi slt
// CHECK:            scf.if
// CHECK-NOT:          arith.floordivsi
// CHECK:              krnl.store {{.*}}, [[TMP]]
// CHECK:              scf.if
// CHECK:                krnl.load [[TMP]]
// CHECK:                arith.index_cast
// CHECK:                krnl.store {{.*}}, [[OUT]]
// CHECK:                krnl.store {{.*}}, [[TMP]]

// -----

func.func @test_nonzero_rank2(%arg0: tensor<4096x2048xf32>) -> tensor<2x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<4096x2048xf32>) -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
}

// Rank >= 2 is flattened to 1-D first, via memref.reshape with a small buffer
// holding the target shape (memref.collapse_shape would be free metadata but
// cannot be lowered here -- expanding it emits an affine.delinearize_index that
// ConvertKrnlToLLVMPass rejects).
//
// M = 8388608, 64 blocks -> 131072 per block, which divides M: no tail path.
//
// The float compare must be OEQ inverted by a select, never a not-equal: ONE is
// false for NaN, which would drop NaN elements, whereas numpy.nonzero and the ONNX
// spec treat NaN as nonzero.
//
// Coordinates come from the flat index by successive division, innermost axis
// first. The remainder is formed as t - (t/D)*D rather than with a separate
// modulo, so each level costs one division; the outermost coordinate is the final
// quotient and needs none.
// CHECK-LABEL:  func.func @test_nonzero_rank2
// CHECK-NOT:      arith.cmpi sle
// CHECK:          memref.reshape{{.*}}memref<4096x2048xf32>{{.*}}memref<8388608xf32>
// CHECK:          memref.alloc() {{.*}} : memref<65xindex>
// CHECK:          arith.cmpf oeq, {{.*}} : f32
// CHECK:          [[OUT:%.+]] = memref.alloc({{.*}}) {{.*}} : memref<2x?xi64>
// CHECK:          scf.if
// CHECK:            scf.if
// CHECK:              [[Q:%.+]] = arith.floordivsi {{.*}}, [[C:%.+]] : index
// CHECK:              [[MUL:%.+]] = arith.muli [[Q]], [[C]] : index
// CHECK:              [[REM:%.+]] = arith.subi {{.*}}, [[MUL]] : index
// CHECK:              {{.*}} = arith.index_cast [[REM]] : index to i64
// CHECK:              krnl.store {{.*}}, [[OUT]]{{.}}%c1
// CHECK:              {{.*}} = arith.index_cast [[Q]] : index to i64
// CHECK:              krnl.store {{.*}}, [[OUT]]{{.}}%c0

// -----

func.func @test_nonzero_rank3(%arg0: tensor<2048x1024x4xf32>) -> tensor<3x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<2048x1024x4xf32>) -> tensor<3x?xi64>
  return %0 : tensor<3x?xi64>
}

// Rank 3 needs two division levels, one per axis above the outermost. A short
// innermost dimension (4 here) costs nothing: blocks are a fixed number of
// elements regardless of shape, which is the whole point of flat tiling -- the
// row-aligned scheme this replaced would have produced one block per row.
// CHECK-LABEL:  func.func @test_nonzero_rank3
// CHECK-NOT:      arith.cmpi sle
// CHECK:          memref.reshape{{.*}}memref<2048x1024x4xf32>{{.*}}memref<8388608xf32>
// CHECK:          memref.alloc() {{.*}} : memref<65xindex>
// CHECK:          [[OUT:%.+]] = memref.alloc({{.*}}) {{.*}} : memref<3x?xi64>
// CHECK:          scf.if
// CHECK:            scf.if
// Innermost axis, then the middle one, then the leftover quotient.
// CHECK:              arith.floordivsi
// CHECK:              krnl.store {{.*}}, [[OUT]]{{.}}%c2
// CHECK:              arith.floordivsi
// CHECK:              krnl.store {{.*}}, [[OUT]]{{.}}%c1
// CHECK:              krnl.store {{.*}}, [[OUT]]{{.}}%c0

// -----

// A dynamic dimension does not by itself force the tail path. A non-static shape
// uses the fixed 1024 tile, and the tile only has to divide the *static* part to
// divide the whole: 1024 divides M' = 1024, so every block is provably full
// whatever d turns out to be.
func.func @test_nonzero_dyn_no_tail(%arg0: tensor<1024x?xf32>) -> tensor<2x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<1024x?xf32>) -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
}

// CHECK-LABEL:  func.func @test_nonzero_dyn_no_tail
// CHECK-NOT:      arith.cmpi sle
// CHECK:          memref.reshape{{.*}}memref<1024x?xf32>{{.*}}memref<?xf32>
// nzPerBlock is dynamically sized because the block count is.
// CHECK:          memref.alloc({{.*}}) {{.*}} : memref<?xindex>

// -----

// Here the static part is only 4, so no tile size of a useful width can be proven
// to divide M: the tail path is emitted. It is selected by one test per block
// (lo + T <= M), not by a bound check per element, and the guarded path it guards
// runs at most once per tensor.
func.func @test_nonzero_dyn_tail(%arg0: tensor<4x?xf32>) -> tensor<2x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<4x?xf32>) -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
}

// CHECK-LABEL:  func.func @test_nonzero_dyn_tail
// CHECK:          memref.alloc({{.*}}) {{.*}} : memref<?xindex>
// The per-block "is this block full?" test, with the full path and the
// bound-checked path below it.
// CHECK:          arith.cmpi sle
// CHECK:          scf.if
// CHECK:            krnl.iterate
// CHECK:          } else {
// CHECK:            krnl.iterate
// CHECK:              arith.cmpi slt
// CHECK:              scf.if

// -----

// With the block loops parallelized, passes 1 and 3 can no longer share one scalar
// outside the loops: that one would be shared by every thread. Each allocates
// inside its own block body instead, where ProcessScfParallelPrivate wraps the body
// in memref.alloca_scope, making it thread private and reclaimed per iteration.
// Pass 2's scan stays sequential, so it keeps using the one outside.
func.func @test_nonzero_parallel(%arg0: tensor<8192xi64>) -> tensor<1x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<8192xi64>) -> tensor<1x?xi64>
  return %0 : tensor<1x?xi64>
}

// CHECK-LABEL:  func.func @test_nonzero_parallel

// PAR-LABEL:  func.func @test_nonzero_parallel
// The scalar pass 2 uses, outside every loop.
// PAR:          memref.alloca() : memref<index>
// Pass 1: parallel, with its own scalar inside the block body.
// PAR:          krnl.parallel
// PAR:          krnl.iterate
// PAR:            memref.alloca() : memref<index>
// Pass 2: sequential, no krnl.parallel on it.
// PAR:          krnl.iterate
// Pass 3: parallel, again with its own scalar inside the block body.
// PAR:          krnl.parallel
// PAR:          krnl.iterate
// PAR:            memref.alloca() : memref<index>

// -----

// Rank 0 is degenerate: the result has zero rows, so there is nothing to write and
// only the dynamic dimension is computed. No loop, no block machinery. (numpy
// rejects nonzero() on a 0-d array outright, so there is no reference semantics to
// match here.)
func.func @test_nonzero_rank0(%arg0: tensor<f32>) -> tensor<0x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<f32>) -> tensor<0x?xi64>
  return %0 : tensor<0x?xi64>
}

// CHECK-LABEL:  func.func @test_nonzero_rank0
// CHECK-NOT:      krnl.iterate
// CHECK-NOT:      memref.alloca
// CHECK:          [[X:%.+]] = krnl.load
// CHECK:          arith.cmpf oeq, [[X]]
// CHECK:          [[N:%.+]] = arith.select
// CHECK:          memref.alloc([[N]]) {{.*}} : memref<0x?xi64>

// -----

// A bool input, small enough (M' = 4) that the whole tensor is one block. This case
// used to live in Math/Elementwise_with_canonicalize.mlir, which is not where a
// NonZero test belongs.
//
// i1 has no SIMD support on z, so this is also the shape that stays scalar when the
// counting pass is eventually vectorized.
func.func @test_nonzero_rank2_i1(%arg0: tensor<2x2xi1>) -> tensor<2x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<2x2xi1>) -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
}

// CHECK-LABEL:  func.func @test_nonzero_rank2_i1
// CHECK-NOT:      arith.cmpi sle
// CHECK:          memref.reshape{{.*}}memref<2x2xi1>{{.*}}memref<4xi1>
// One block, so nzPerBlock has 2 slots.
// CHECK:          [[NZ:%.+]] = memref.alloc() {{.*}} : memref<2xindex>
// CHECK:          [[TMP:%.+]] = memref.alloca() : memref<index>
// Pass 1: the zero test on i1 is an integer compare against false.
// CHECK:          krnl.iterate
// CHECK:            krnl.iterate
// CHECK:              arith.cmpi eq, {{.*}} : i1
// CHECK:              arith.select
// CHECK:            krnl.store {{.*}}, [[NZ]]
// CHECK:          [[OUT:%.+]] = memref.alloc({{.*}}) {{.*}} : memref<2x?xi64>
// Pass 3: one division level, innermost coordinate to row 1 then the quotient to
// row 0.
// CHECK:          scf.if
// CHECK:            scf.if
// CHECK:              arith.floordivsi
// CHECK:              krnl.store {{.*}}, [[OUT]]{{.}}%c1
// CHECK:              krnl.store {{.*}}, [[OUT]]{{.}}%c0

// -----

// Pass 1 is a masked reduce-sum, so it vectorizes with krnl.simdReduceIE. VL comes
// from the *input* element type, since the pass is memory bound and bytes per load
// is what matters: 4 for f32 on z16. The accumulator is i64 regardless -- a count
// does not fit in the narrower lanes -- so the partial sums are vector<4xi64>,
// reduced to a scalar once per block.
//
// The tile size is also chosen to be a multiple of VL where that is compatible with
// dividing M', so a full block is entirely SIMD and no scalar remainder loop is
// emitted inside it -- hence exactly one inner loop below.
//
// Pass 3 stays scalar.
func.func @test_nonzero_simd_f32(%arg0: tensor<4096x2048xf32>) -> tensor<2x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<4096x2048xf32>) -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
}

// CHECK-LABEL:  func.func @test_nonzero_simd_f32
// PAR-LABEL:    func.func @test_nonzero_simd_f32

// SIMD-LABEL:  func.func @test_nonzero_simd_f32
// The VL-wide partial sums, and their zero init.
// SIMD:          [[TMP:%.+]] = memref.alloca() {{.*}} : memref<4xi64>
// SIMD:          vector.broadcast {{.*}} : i64 to vector<4xi64>
// SIMD:          vector.store {{.*}}, [[TMP]]{{.*}} : memref<4xi64>, vector<4xi64>
// The counting loop: load 4 f32, compare to zero, select 0/1, accumulate.
// SIMD:          vector.load {{.*}} : memref<8388608xf32>, vector<4xf32>
// SIMD:          arith.cmpf oeq, {{.*}} : vector<4xf32>
// SIMD:          arith.select {{.*}} : vector<4xi1>, vector<4xi64>
// SIMD:          arith.addi {{.*}} : vector<4xi64>
// SIMD:          vector.store {{.*}}, [[TMP]]
// One horizontal reduction per block, then the count is cast to index.
// SIMD:          vector.reduction <add>, {{.*}} : vector<4xi64> into i64
// SIMD:          arith.index_cast

// -----

// i1 has no vector support on z (computeArchVectorLength returns UNSUPPORTED for a
// 1-bit type), so a bool input counts scalar even at -O3. This is the element type
// the real model uses.
func.func @test_nonzero_simd_i1(%arg0: tensor<8192xi1>) -> tensor<1x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<8192xi1>) -> tensor<1x?xi64>
  return %0 : tensor<1x?xi64>
}

// CHECK-LABEL:  func.func @test_nonzero_simd_i1
// PAR-LABEL:    func.func @test_nonzero_simd_i1

// SIMD-LABEL:  func.func @test_nonzero_simd_i1
// SIMD-NOT:      vector.load
// SIMD-NOT:      vector.reduction
// SIMD:          arith.cmpi eq, {{.*}} : i1
