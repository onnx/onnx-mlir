// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl %s -split-input-file | FileCheck %s --check-prefix=OLD
// RUN: onnx-mlir-opt --shape-inference --convert-onnx-to-krnl --test-compiler-opt=true %s -split-input-file | FileCheck %s --check-prefix=NEW

// NonZero has two lowerings, selected by --test-compiler-opt:
//  - default: the historical marginal-sum lowering, which is incorrect for rank
//    >= 2 but is kept as a reference oracle for rank 1.
//  - --test-compiler-opt=true: flat-tiled stream compaction. X is treated as one
//    flat run of M elements cut into blocks of a compile-time tile size; a
//    per-block count is prefix-summed so each block knows which output columns it
//    owns, then each block writes them.
// See src/Conversion/ONNXToKrnl/Tensor/NonZero.cpp.

func.func @test_nonzero_rank1(%arg0: tensor<8xi64>) -> tensor<1x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<8xi64>) -> tensor<1x?xi64>
  return %0 : tensor<1x?xi64>
}

// The old lowering builds one reduction-sum array per axis and then, for every
// output column, rescans that array to find the matching bucket.
// OLD-LABEL:  func.func @test_nonzero_rank1
// OLD:          memref.alloca() : memref<index>
// OLD:          krnl.iterate
// OLD:          arith.cmpi eq
// OLD:          arith.select

// M = 8 is fully static and below the default tile size, so the tile size is
// M itself: exactly one block, and no partial-block path anywhere. Rank 1 is
// already flat, so no reshape is emitted either.
// NEW-LABEL:  func.func @test_nonzero_rank1
// NEW-NOT:      memref.reshape
// NEW:          [[NZ:%.+]] = memref.alloc() {{.*}} : memref<2xindex>
// NEW:          krnl.store {{.*}}, [[NZ]]
// Pass 1: count. One block, constant trip count, no bound check.
// NEW:          krnl.iterate
// NEW:            arith.cmpi eq, {{.*}} : i64
// NEW:            arith.select
// NEW:            krnl.store {{.*}}, [[NZ]]
// Pass 2: serial prefix sum over the 2 slots.
// NEW:          krnl.iterate
// NEW:            krnl.store {{.*}}, [[NZ]]
// Output allocation uses the scanned total.
// NEW:          [[OUT:%.+]] = memref.alloc({{.*}}) {{.*}} : memref<1x?xi64>
// Pass 3: skip empty blocks, then the guarded store. Rank 1 needs no
// unravelling, so the flat index is stored directly with no division.
// NEW:          krnl.iterate
// NEW:            arith.cmpi slt
// NEW:            scf.if
// NEW-NOT:          arith.floordivsi
// NEW:              scf.if
// NEW:                arith.index_cast
// NEW:                krnl.store {{.*}}, [[OUT]]

// -----

func.func @test_nonzero_rank2(%arg0: tensor<3x4xf32>) -> tensor<2x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<3x4xf32>) -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
}

// OLD-LABEL:  func.func @test_nonzero_rank2

// Rank >= 2 is flattened to 1-D first. M = 12 is static and under the tile size,
// so again one block with no partial path.
//
// Float compare must be OEQ (ordered equal) inverted by a select, never a
// not-equal: ONE is false for NaN, which would drop NaN elements, whereas
// numpy.nonzero and the ONNX spec treat NaN as nonzero.
//
// Coordinates come from the flat index by successive division. The remainder is
// formed as t - (t/D)*D rather than with a separate modulo, so each level costs
// one division; the outermost coordinate is the final quotient, with no division
// of its own.
// NEW-LABEL:  func.func @test_nonzero_rank2
// NEW:          memref.reshape{{.*}}memref<3x4xf32>{{.*}}memref<12xf32>
// NEW:          memref.alloc() {{.*}} : memref<2xindex>
// NEW:          arith.cmpf oeq, {{.*}} : f32
// NEW:          [[OUT:%.+]] = memref.alloc({{.*}}) {{.*}} : memref<2x?xi64>
// NEW:          scf.if
// NEW:            scf.if
// NEW:              [[Q:%.+]] = arith.floordivsi {{.*}}, [[C4:%.+]] : index
// NEW:              [[M:%.+]] = arith.muli [[Q]], [[C4]] : index
// NEW:              [[R:%.+]] = arith.subi {{.*}}, [[M]] : index
// NEW:              {{.*}} = arith.index_cast [[R]] : index to i64
// NEW:              krnl.store {{.*}}, [[OUT]][%c1{{[^,]*}}, {{.*}}] : memref<2x?xi64>
// NEW:              {{.*}} = arith.index_cast [[Q]] : index to i64
// NEW:              krnl.store {{.*}}, [[OUT]][%c0{{[^,]*}}, {{.*}}] : memref<2x?xi64>

// -----

// A dynamic dimension means M is only known at run time, so no tile size can be
// proven to divide it: the partial-block path must be emitted. It is selected by
// a single test per block (lo + T <= M), not by a bound check per element.
func.func @test_nonzero_dyn(%arg0: tensor<4x?xf32>) -> tensor<2x?xi64> {
  %0 = "onnx.NonZero"(%arg0) : (tensor<4x?xf32>) -> tensor<2x?xi64>
  return %0 : tensor<2x?xi64>
}

// OLD-LABEL:  func.func @test_nonzero_dyn

// NEW-LABEL:  func.func @test_nonzero_dyn
// NEW:          memref.dim
// NEW:          memref.reshape{{.*}}memref<4x?xf32>{{.*}}memref<?xf32>
// nzPerBlock is dynamically sized because the block count is.
// NEW:          memref.alloc({{.*}}) {{.*}} : memref<?xindex>
// The per-block "is this block full?" test, and both paths below it.
// NEW:          arith.cmpi sle
// NEW:          scf.if
// NEW:            krnl.iterate
// NEW:          } else {
// NEW:            krnl.iterate
// NEW:              arith.cmpi slt
// NEW:              scf.if
