/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------- ONNXEinsumOpHelper.cpp - Helper functions for Einsum --------===//
//
// This file contains helper functions for processing the ONNX Einsum Operator.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXOps/Math/EinsumHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <map>
#include <regex>
#include <stddef.h>
#include <stdint.h>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

using namespace mlir;

namespace onnx_mlir {

namespace einsum {

namespace {

// Note that we permit spaces in the middle of ellipsis (...) and
// arrow (->), which is like the onnxruntime einsum implementation
// but more lax than torch.einsum and numpy.einsum.
#define ELLIPSIS "\\. *\\. *\\."
#define ARROW "- *>"
// Spaces are allowed before, between, and after subscripts.
#define SUBSCRIPTS "[ A-Za-z]*"
// A "parameter" describes each input and the output. Can contain an ellipsis.
#define PARAMETER SUBSCRIPTS "(?:" ELLIPSIS SUBSCRIPTS ")?"
// Input parameters are separated by commas.
#define INPUTS "(" PARAMETER "(?:," PARAMETER ")*)"
#define OUTPUT "(" PARAMETER ")"
// The arrow and output are optional.
#define EQUATION INPUTS "(?:" ARROW OUTPUT ")?"

const char *const equation_regex = EQUATION;

} // namespace

LogicalResult verifyEquation(
    StringRef equation, size_t numInputs, ErrorFn emitErrorFn) {
  std::regex pattern(equation_regex);
  std::cmatch match;
  if (!std::regex_match(equation.begin(), equation.end(), match, pattern)) {
    return emitErrorFn() << "invalid equation syntax";
  }
  assert(match.size() == 3 && "1: whole equation, 2: inputs, 3: output");
  size_t numEquationInputs = equation.count(',') + 1;
  if (numEquationInputs != numInputs) {
    return emitErrorFn() << "number of equation inputs " << numEquationInputs
                         << " != number of actual inputs " << numInputs;
  }
  // equation matched the regex - next check that any output satisfies the
  // constaints that its subscripts cannot be repeated and each must occur in
  // some inputs
  std::csub_match outputGroup = match[2];
  if (!outputGroup.matched) {
    // equation has no output
    return success();
  }
  std::csub_match inputsGroup = match[1];
  StringRef inputs(inputsGroup.first, inputsGroup.length());
  StringRef output(outputGroup.first, outputGroup.length());
  for (size_t p = 0; p < output.size(); ++p) {
    char x = output[p]; // x must be in [ .A-Za-z] given regex OUTPUT
    if (x >= 'A') {     // tests whether x is a letter given x is in [ .A-Za-z]
      if (StringRef::npos != output.find(x, p + 1)) {
        return emitErrorFn()
               << "subscript " << x << " appears multiple times in the output";
      }
      if (StringRef::npos == inputs.find(x)) {
        return emitErrorFn()
               << "output subscript " << x << " doesn't appear in inputs";
      }
    }
  }
  // equation has valid output
  return success();
}

FailureOr<Shape> inferOutputShape(
    ONNXEinsumOpAdaptor operandAdaptor, ErrorFn emitErrorFn) {
  FailureOr<einsum::Signature> signature =
      inferSignature(operandAdaptor, emitErrorFn);
  if (failed(signature)) {
    return failure();
  }
  return signature->output.shape;
}

LogicalResult verifyShapes(
    ONNXEinsumOpAdaptor operandAdaptor, ErrorFn emitErrorFn) {
  return success(succeeded(inferSignature(operandAdaptor, emitErrorFn)));
}

namespace {

// When the Einsum equation omits arrow and output parameter,
// it is implied that the output consists of the ellipsis
// followed by the subscripts that occur once in the inputs,
// ordered alphabetically with all upper case before lower case.
// (Ellipsis is empty and redundant if it doesn't occur in any inputs.)
std::string inferEquationOutput(StringRef commaSeparatedInputs) {
  std::map<char, int> counts;
  for (char x : commaSeparatedInputs) {
    if (x >= 'A') {   // tests whether x is a letter, given x is in [ ,.A-Za-z]
      counts[x] += 1; // counts[x] initializes to 0 if not yet mapped
    }
  }
  std::string equationOutput = "...";
  // iterate through sorted map in order, i.e. alphabetically and
  // all upper case letters before lower case
  for (const auto &entry : counts) { // entry == pair (x, count)
    if (entry.second == 1)           // one occurrence
      equationOutput.push_back(entry.first);
  }
  return equationOutput;
}

// Argument must be an input or output parameter consisting of letters,
// spaces, and possibly ellipsis.
size_t countLetters(StringRef parameter) {
  return parameter.size() - parameter.count(' ') - parameter.count('.');
}

bool hasEllipsis(StringRef parameter) {
  return parameter.find('.') != StringRef::npos;
}

void appendLetterSubscripts(StringRef letters, Subscripts &subscripts) {
  size_t p = letters.find_first_not_of(' ');
  while (p != StringRef::npos) {
    subscripts.push_back(letters[p]);
    p = letters.find_first_not_of(' ', p + 1);
  }
}

// It's convenient to limit the max ellipsis rank to 10 so we can represent
// an ellipsis internally as a sequence of digit subscripts which is easy
// to read when debugging.
// (Easy to change this if there's ever a use case for larger ellipsis rank.)
constexpr int64_t kMaxEllipsisRank = 10;

// We represent an ellipsis as a sequence of digit subscripts, one per axis.
char ellipsisSubscript(int64_t axis) {
  assert(0 <= axis && axis < 10 && "axis corresponds to a decimal digit");
  return '0' + axis;
}

// Precondition: letters and any ellipsis in parameterEquation are
// compatible with rank (#letters <= or == rank, with or without ellipsis)
Subscripts extractSubscripts(StringRef parameterEquation, int64_t rank) {
  Subscripts subscripts;
  subscripts.reserve(rank);
  StringRef prefix, suffix;
  std::tie(prefix, suffix) = parameterEquation.split('.');
  int64_t prefixRank = prefix.size() - prefix.count(' ');
  appendLetterSubscripts(prefix, subscripts);
  if (prefix.size() == parameterEquation.size()) {
    assert(prefixRank == rank && "#letters == rank without ellipsis");
  } else {
    // Skip past remainder of ellipsis "..." and any spaces
    suffix = suffix.ltrim(" .");
    int64_t suffixRank = suffix.size() - suffix.count(' ');
    assert(rank >= prefixRank + suffixRank && "#letters <= rank with ellipsis");
    int64_t ellipsisRank = rank - prefixRank - suffixRank;
    for (int64_t axis = 0; axis < ellipsisRank; ++axis) {
      subscripts.push_back(ellipsisSubscript(axis));
    }
    appendLetterSubscripts(suffix, subscripts);
  }
  assert(static_cast<int64_t>(subscripts.size()) == rank &&
         "#subscripts == rank after replacing any ellipsis with digits");
  return subscripts;
}

} // namespace

// Preconditions:
// . equation must be well-formed as validated by verifyEquation()
// . number of equation input parameters must match number of actual inputs
// . all input types must be shaped types with rank
FailureOr<Signature> inferSignature(
    ONNXEinsumOpAdaptor operandAdaptor, ErrorFn emitErrorFn) {
  StringRef equation = operandAdaptor.getEquation();
  ValueRange inputs = operandAdaptor.getInputs();
  assert(llvm::all_of(inputs.getTypes(), isRankedShapedType) &&
         "precondition checked in verify()");
  StringRef equationOutput, commaSeparatedInputs;
  std::tie(commaSeparatedInputs, equationOutput) = equation.split('-');
  std::string inferredOutput;
  if (commaSeparatedInputs.size() < equation.size()) {
    // Skip past second char of arrow "->" and any spaces
    equationOutput = equationOutput.ltrim(" >");
  } else {
    // No arrow and output. Infer output.
    inferredOutput = inferEquationOutput(commaSeparatedInputs);
    equationOutput = StringRef(inferredOutput);
  }
  SmallVector<StringRef> equationInputs;
  commaSeparatedInputs.split(equationInputs, ',');
  assert(equationInputs.size() == inputs.size() &&
         "precondition checked in verify()");
  Signature signature;
  int64_t ellipsisRank = -1; // -1 means unknown
  // Map subscripts across all inputs to non-1 dim sizes.
  // Static dim sizes trump dynamic so if a subscript occurs
  // with both dynamic and non-1 static dim size, record the latter.
  // (In the hope that the dynamic dim size will become the static
  // dim size or 1 so that things will type check in the end.)
  std::unordered_map<char, int64_t> broadcast;
  std::unordered_set<char> dynamicSubscripts;
  for (size_t i = 0; i < inputs.size(); ++i) {
    Value input = inputs[i];
    StringRef equationInput = equationInputs[i];
    ShapedType type = mlir::cast<ShapedType>(input.getType());
    auto shape = type.getShape();
    size_t rank = shape.size();
    size_t letters = countLetters(equationInput);
    if (!hasEllipsis(equationInput)) {
      if (rank != letters) {
        return emitErrorFn() << "number of equation input parameter subscripts "
                             << letters << " != input type rank " << rank;
      }
    } else {
      if (rank < letters) {
        return emitErrorFn() << "number of equation input parameter subscripts "
                             << letters << " exceeds input type rank " << rank;
      }
      int64_t thisEllipsisRank = rank - letters;
      if (thisEllipsisRank > kMaxEllipsisRank) {
        return emitErrorFn()
               << "ellipsis rank exceeds maximum of " << kMaxEllipsisRank;
      }
      if (ellipsisRank == -1) {
        ellipsisRank = thisEllipsisRank;
      } else {
        if (ellipsisRank != thisEllipsisRank) {
          return emitErrorFn() << "inputs disagree on ellipsis rank, "
                               << ellipsisRank << " vs " << thisEllipsisRank;
        }
      }
    }
    Parameter &signatureInput = signature.inputs.emplace_back();
    signatureInput.shape.assign(shape.begin(), shape.end());
    signatureInput.subscripts = extractSubscripts(equationInput, rank);

    // Map subscripts to dim sizes while checking that repeated
    // subscripts have the same (or dynamic) dim sizes.
    std::unordered_map<char, int64_t> subscriptsToDims;
    for (size_t j = 0; j < rank; ++j) {
      int64_t d = signatureInput.shape[j];
      char s = signatureInput.subscripts[j];
      if (ShapedType::isDynamic(d)) {
        dynamicSubscripts.insert(s);
      } else {
        auto insertion = subscriptsToDims.emplace(s, d);
        int64_t d0 = insertion.first->second; // == subscriptsToDims[s]
        if (d0 != d) {
          return emitErrorFn()
                 << "subscript '" << s << "' has axes with different dim sizes "
                 << d0 << ", " << d << " in the same input";
        }
      }
    }
    // Merge the subscripts with static dim sizes into the
    // broadcast map shared by all inputs.
    for (const auto &entry : subscriptsToDims) {
      char s = entry.first;
      int64_t d = entry.second;
      if (d != 1) {
        auto insertion = broadcast.emplace(s, d);
        int64_t d0 = insertion.first->second; // == broadcast[s]
        if (d0 != d) {
          return emitErrorFn()
                 << "subscript '" << s << "' has conflicting dim sizes " << d0
                 << ", " << d;
        }
      }
    }
  }
  if (ellipsisRank == -1) {
    // no input had an ellipsis, so the ellipsis is deemed empty
    ellipsisRank = 0;
  }
  bool outputHasEllipsis = hasEllipsis(equationOutput);
  if (ellipsisRank != 0 && !outputHasEllipsis) {
    return emitErrorFn() << "output needs ellipsis because inputs have "
                         << "non-empty ellipsis with rank " << ellipsisRank;
  }
  int64_t outputRank =
      countLetters(equationOutput) + (outputHasEllipsis ? ellipsisRank : 0);
  signature.output.subscripts = extractSubscripts(equationOutput, outputRank);
  for (char s : signature.output.subscripts) {
    int64_t d;
    auto iter = broadcast.find(s);
    if (iter != broadcast.end()) {
      d = iter->second;
    } else if (dynamicSubscripts.find(s) != dynamicSubscripts.end()) {
      d = ShapedType::kDynamic;
    } else {
      d = 1;
    }
    signature.output.shape.push_back(d);
  }
  return signature;
}

} // namespace einsum

} // namespace onnx_mlir
