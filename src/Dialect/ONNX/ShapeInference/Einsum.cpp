/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------- Einsum.cpp - Shape Inference for Einsum Op -------------===//
//
// This file implements shape inference for the ONNX Einsum Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"
#include <algorithm>
#include <string>
#include <tuple>
#include <utility>

using namespace mlir;

namespace onnx_mlir {

namespace {

// There are 52 indices: 26 upper and 26 lower case ASCII letters,
// ordered alphabetically with all upper before all lower case.
constexpr int kNumIndices = 26 + 26;

char indexToChar(int i) {
  assert(0 <= i);
  assert(i < kNumIndices);
  return i < 26 ? ('A' + i) : ('a' + i - 26);
}

int charToIndex(char c) {
  if ('A' <= c && c <= 'Z') {
    return c - 'A';
  }
  if ('a' <= c && c <= 'z') {
    return c - 'a' + 26;
  }
  return -1;
}

std::string inferResult(const SmallVectorImpl<StringRef>& args) {
  // The inferred result indices are the ellipsis followed by
  // all indices that occur exactly once across all args,
  // ordered alphabetically with all upper before all lower case.
  std::string inferred = "...";
  SmallVector<int, kNumIndices> occurrences(kNumIndices, 0);
  for (const StringRef arg : args) {
    for (const char c : arg) {
      int i = charToIndex(c);
      if (i != -1)
        occurrences[i] += 1;
    }
  }
  for (int i = 0; i < kNumIndices; ++i) {
    if (occurrences[i] == 1) {
      inferred.push_back(indexToChar(i));
    }
  }
  return inferred;
}

// If either dim is non-1, it wins because it cannot broadcast.
// If either dim is 1, it loses because it can broadcast.
// If both are questionmarks, return a questionmark.
IndexExpr broadcastDim(IndexExpr currentDim, IndexExpr nextDim) {
  assert(!currentDim.isUndefined());
  assert(!nextDim.isUndefined());
  if (!currentDim.isLiteral() && !nextDim.isLiteral()) {
    // Two questionmarks broadcast to one questionmark.
    // TODO: to support code gen (not just shape inference) we could
    // return IndexExpr::select(currentDim == 1, nextDim, currentDim)
    return currentDim;
  } else if (nextDim.isLiteralAndIdenticalTo(1)) {
    // currentDim is unchanged by broadcast with dim 1.
    return currentDim;
  } else if (currentDim.isLiteralAndDifferentThan(1)) {
    // currentDim is unchanged by broadcast if it's non-1.
    if (nextDim.isLiteral())
      // TODO: determine if assert should be runtime check
      assert(nextDim.getLiteral() == currentDim.getLiteral());
    return currentDim;
  } else {
    assert(nextDim.isLiteralAndDifferentThan(1) || currentDim.isLiteralAndIdenticalTo(1));
    // Go with nextDim because it wins or currentDim loses.
    return nextDim;
  }
}

} // namespace

LogicalResult ONNXEinsumOpShapeHelper::computeShape(
    ONNXEinsumOpAdaptor operandAdaptor) {
  // First remove all spaces from equation.
  // Note that we permit spaces in the middle of ellipsis (...) and
  // arrow (->), which is like the onnxruntime einsum implementation
  // but more lax than torch.einsum and numpy.einsum.
  StringRef equation = op->equation();
  int spaces = std::count(equation.begin(), equation.end(), ' ');
  std::string trimmed;
  if (spaces > 0) {
    trimmed.reserve(equation.size() - spaces);
    for (char c : equation) {
      if (c != ' ')
        trimmed.push_back(c);
    }
    equation = trimmed;
  }

  StringRef csvArgs;
  StringRef result;
  std::tie(csvArgs, result) = equation.split("->");
  SmallVector<StringRef> args;
  csvArgs.split(args, ',');
  std::string inferred;
  if (csvArgs.size() == equation.size()) {
    // There's no -> in equation, so result string must be inferred.
    inferred = inferResult(args);
    result = inferred;
  }

  auto inputs = operandAdaptor.Inputs();
  assert(inputs.size() == args.size() && "tested in verify");

  // Because of broadcast the default value for each dim is 1 and,
  // if a dim has two different values, the non-1 value rules.
  // In indexDims the dim for each index will be set to non-1 if any arg
  // has the index with a non-1 dim. If an index occurs both with
  // a questionmark dim and a non-questionmark dim which is not 1,
  // then the non-questionmark dim is chosen because that is the only
  // valid instantiation of the questionmark.
  SmallVector<IndexExpr, kNumIndices> indexDims(kNumIndices, UndefinedIndexExpr());
  DimsExpr ellipsisDims;
  bool ellipsisFound = false;
  for (size_t n = 0; n < inputs.size(); ++n) {
    StringRef arg = args[n];
    Value input = inputs[n];
    assert(hasShapeAndRank(input) && "tested by caller");
    MemRefBoundsIndexCapture bounds(input);
    DimsExpr dims;
    bounds.getDimList(dims);
    size_t rank = dims.size();
    assert(rank == bounds.getRank());

    StringRef prefix;
    StringRef suffix;
    std::tie(prefix, suffix) = arg.split("...");
    for (size_t x = 0; x < prefix.size(); ++x) {
      auto dim = dims[x];
      int i = charToIndex(prefix[x]);
      indexDims[i] =
          indexDims[i].isUndefined() ? dim : broadcastDim(indexDims[i], dim);
    }
    // TODO: determine if asserts below should be runtime checks
    if (prefix.size() < arg.size()) {
      // There is an ellipsis between prefix and suffix.
      assert(rank >= prefix.size() + suffix.size());
      size_t ellipsisRank = rank - prefix.size() - suffix.size();
      if (!ellipsisFound) {
        ellipsisFound = true;
        for (size_t y = 0; y < ellipsisRank; ++y) {
          ellipsisDims.push_back(dims[prefix.size() + y]);
        }
      } else {
        assert(ellipsisRank == ellipsisDims.size());
        for (size_t y = 0; y < ellipsisRank; ++y) {
          auto dim = dims[prefix.size() + y];
          ellipsisDims[y] = broadcastDim(ellipsisDims[y], dim);
        }
      }
      for (size_t z = 0; z < suffix.size(); ++z) {
        auto dim = dims[rank - suffix.size() + z];
        int i = charToIndex(suffix[z]);
        indexDims[i] =
            indexDims[i].isUndefined() ? dim : broadcastDim(indexDims[i], dim);
      }
    } else {
      // No ellipsis.
      assert(rank == arg.size());
    }
  }

  {
    StringRef prefix;
    StringRef suffix;
    std::tie(prefix, suffix) = result.split("...");
    DimsExpr outputDims;
    // TODO: determine if asserts below should be runtime checks
    for (size_t x = 0; x < prefix.size(); ++x) {
      int i = charToIndex(prefix[x]);
      auto dim = indexDims[i];
      assert(!dim.isUndefined());
      outputDims.push_back(dim);
    }
    if (prefix.size() < result.size()) {
      // There is an ellipsis between prefix and suffix.
      for (auto dim : ellipsisDims) {
        outputDims.push_back(dim);
      }
      for (size_t z = 0; z < suffix.size(); ++z) {
        int i = charToIndex(suffix[z]);
        auto dim = indexDims[i];
        assert(!dim.isUndefined());
        outputDims.push_back(dim);
      }
    }
    dimsForOutput() = outputDims;
  }

  return success();
}

} // namespace onnx_mlir
