/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====--------- DialectBuilder.hpp - Krnl Dialect Builder -----------------===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file declares the Krnl Dialect Builder.
//
//===----------------------------------------------------------------------===//

#pragma once

#include "src/Dialect/Krnl/KrnlOps.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"

namespace mlir {

//====-------------------- Support for Krnl Builder ----------------------===//

struct KrnlBuilder : public DialectBuilder {
  KrnlBuilder(OpBuilder &b, Location loc) : DialectBuilder(b, loc) {}
  KrnlBuilder(DialectBuilder &db) : DialectBuilder(db) {}

  Value load(Value memref, ValueRange indices = {}) const;
  Value loadIE(Value memref, ArrayRef<IndexExpr> indices) const;
  void store(Value val, Value memref, ValueRange indices = {}) const;
  void storeIE(Value val, Value memref, ArrayRef<IndexExpr> indices) const;

  Value vectorTypeCast(Value sourceMemref, int64_t vectorLen) const;

  ValueRange defineLoops(int64_t originalLoopNum) const;
  ValueRange block(Value loop, int64_t blockSize) const;
  void permute(ValueRange loops, ArrayRef<int64_t> map) const;
  ValueRange getInductionVarValue(ValueRange loops) const;

  // Lambda passes loop indices as 2nd parameter.
  void iterate(ValueRange originalLoops, ValueRange optimizedLoops,
      ValueRange lbs, ValueRange ubs,
      function_ref<void(KrnlBuilder &createKrnl, ValueRange indices)>
          bodyBuilderFn) const;
  mlir::KrnlIterateOp iterate(const KrnlIterateOperandPack &operands) const;

  // Lambda passes loop indices as 2nd parameter.
  void iterateIE(ValueRange originalLoops, ValueRange optimizedLoops,
      ArrayRef<IndexExpr> lbs, ArrayRef<IndexExpr> ubs,
      function_ref<void(KrnlBuilder &createKrnl, ValueRange indices)>
          bodyBuilderFn) const;

  void copyToBuffer(
      // Buffer and source memory. Source memref may have a higher rank than
      // buffer.
      Value bufferMemref, Value sourceMemref,
      // Indices that points to the first data to be copied from source.
      // Starts has the same rank as sourceMemref.
      ValueRange starts,
      // If padding is needed, value to pad.
      Value padValue,
      // Now the bufferMemref may be larger than the actual data to be stored
      // in the buffer, if the user want to pad the data to a higher size.
      // TileSize enables the user to
      ArrayRef<int64_t> tileSize, ArrayRef<int64_t> padToNext,
      bool transpose = false) const;
  void copyToBuffer(Value bufferMemref, Value sourceMemref, ValueRange starts,
      Value padValue, bool transpose = false) const;

  void copyFromBuffer(Value bufferMemref, Value memref, ValueRange starts,
      ArrayRef<int64_t> tileSize) const;
  void copyFromBuffer(
      Value bufferMemref, Value memref, ValueRange starts) const;

  void matmul(
      // The a/b/cStart are the indices at the begining of the buffer/mem
      // A/B/C.
      Value A, ValueRange aStart, Value B, ValueRange bStart, Value C,
      ValueRange cStart,
      // Loops are the krnl loop indices that this matmul replaces
      ValueRange loops,
      // the computeStarts indicate the i/j/k indices pointing to the begining
      // of the matmul computation.
      ValueRange computeStarts,
      // The globalUBs are the global bounds on the original I, J, K
      // dimensions.
      ValueRange globalUBs,
      // If not the full A, B, C buffers are used by this matmul, meaning the
      // matmul uses a subtile of the buffers, this compute tile size
      // specifies the actual size of the i/j/k computations. Empty means
      // compute tiles encompass the entire buffer A, B, and C as defined by
      // their tile sizes.
      ArrayRef<int64_t> computeTileSize,
      // If buffers A, B, or C were padded, then the tile sizes give the size
      // of the non-padded data, basically the size of the data when the tile
      // is full. Partial tiles (due to computation on the edges of the
      // matrices) are handled differently (using the UBs), so no need to
      // worry about this. Empty means no padding was used.
      ArrayRef<int64_t> aTileSize, ArrayRef<int64_t> bTileSize,
      ArrayRef<int64_t> cTileSize,
      // Optimizations for code gen.
      bool simdize, bool unroll, bool overcompute) const;
  void matmul(Value A, ValueRange aStart, Value B, ValueRange bStart, Value C,
      ValueRange cStart, ValueRange loops, ValueRange computeStarts,
      ValueRange globalUBs, bool simdize, bool unroll, bool overcompute) const;

  Value dim(Type type, Value alloc, Value index) const;

  mlir::KrnlMovableOp movable() const;

  mlir::KrnlGetRefOp getRef(
      Type type, Value memref, Value offset, ValueRange indices = {}) const;

  Value constant(MemRefType type, StringRef name, Optional<Attribute> value,
      Optional<IntegerAttr> offset = None,
      Optional<IntegerAttr> alignment = None) const;

  // C library functions.
  void memcpy(Value dest, Value src, Value size) const;
  void memset(Value dest, Value val) const;
  Value strncmp(Value str1, Value str2, Value len) const;
  Value strlen(Value str) const;
  void printf(StringRef msg) const;
  void printf(StringRef msg, Value input, Type inputType) const;

  // Onnx-mlir runtime functions.
  void randomNormal(Value alloc, Value numberOfRandomValues, Value mean,
      Value scale, Value seed) const;
  Value findIndex(Value input, Value G, Value V, Value len) const;
  void printTensor(StringRef msg, Value input) const;
};

// Recursive class specialized for KrnlBuilder referred to as krnl.
template <class... Ts>
struct MultiDialectBuilder<KrnlBuilder, Ts...> : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(OpBuilder &b, Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), krnl(b, loc) {}
  MultiDialectBuilder(DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), krnl(db) {}
  KrnlBuilder krnl;
};

//====--- Support for Affine Builder with Krnl Mem Ops ------------------===//

// We use here a Affine builder that generates Krnl Load and Store ops instead
// of the affine memory ops directly. This is because we can still generrate
// Krnl Ops while lowring the dialect, and the big advantage of the Krnl memory
// operations is that they distinguish themselves if they are affine or not.
using AffineBuilderKrnlMem = GenericAffineBuilder<KrnlLoadOp, KrnlStoreOp>;

// Recursive class specialized for AffineBuilderKrnlMem refereed to as
// affineKMem.
template <class... Ts>
struct MultiDialectBuilder<AffineBuilderKrnlMem, Ts...>
    : MultiDialectBuilder<Ts...> {
  MultiDialectBuilder(OpBuilder &b, Location loc)
      : MultiDialectBuilder<Ts...>(b, loc), affineKMem(b, loc) {}
  MultiDialectBuilder(DialectBuilder &db)
      : MultiDialectBuilder<Ts...>(db), affineKMem(db) {}
  AffineBuilderKrnlMem affineKMem;
};

} // namespace mlir
