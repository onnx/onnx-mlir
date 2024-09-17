/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- ProcessKrnlParallelClause.cpp - handle Krnl Parallel Clauses ------===//
//
// Copyright 2024 The IBM Research Authors.
//
// =============================================================================
// This pass seeks KrnlParallelClauseOp and integrate its parameter in the
// enclosing OpenMP Parallel construct.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PROCESS_KRNL_PARALLEL_CLAUSE_H
#define ONNX_MLIR_PROCESS_KRNL_PARALLEL_CLAUSE_H

#include "mlir/IR/PatternMatch.h"

namespace onnx_mlir {

// Exports the patterns. They are all plain rewrite patterns that can be used
// with any PatternRewriter, not conversion patterns.
void getKrnlParallelClauseIntoOpenMPPatterns(mlir::RewritePatternSet &patterns);

} // namespace onnx_mlir
#endif
