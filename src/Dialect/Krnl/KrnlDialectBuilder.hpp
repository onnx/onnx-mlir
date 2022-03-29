/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====------ KrnlDialectBuilder.hpp - Krnl Dialect Builder --------------===//
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
#include "src/Dialect/ONNX/MLIRDialectBuilder.hpp"

namespace mlir {

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
