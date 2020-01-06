#include "mlir/Dialect/AffineOps/AffineOps.h"
#include "mlir/IR/AffineExpr.h"

#include "src/dialect/krnl/krnl_ops.hpp"

#include "krnl_helper.hpp"

namespace onnf {

using namespace mlir;

ParseResult
KrnlDialectOperandParser::ParseOptionalOperand(const Type &operandType,
                                               Value &operand) {
  // If operand queue is empty, parse more operands and cache them.
  if (_operandRefQueue.empty()) {
    // Parse operand types:
    llvm::SmallVector<OpAsmParser::OperandType, 2> operand_refs;
    _parser.parseOperandList(operand_refs);

    // Record operands:
    for (auto& operand_ref : operand_refs)
      _operandRefQueue.emplace(operand_ref);
  }

  // If we parsed some operand reference(s), resolve the ref to an operand:
  if (!_operandRefQueue.empty()) {
    auto operand_ref = _operandRefQueue.front();
    _operandRefQueue.pop();

    llvm::SmallVector<Value, 1> operands;
    _parser.resolveOperand(operand_ref, operandType, operands);
    operand = operands.front();
    return success();
  } else {
    operand = nullptr;
    return failure();
  }
}

ParseResult KrnlDialectOperandParser::ParseOptionalOperand(
    const Type &operandType, llvm::SmallVectorImpl<Value> &operandList) {
  Value operand = nullptr;
  if (ParseOptionalOperand(operandType, operand))
    return failure();

  operandList.emplace_back(operand);
  return success();
}

ParseResult KrnlDialectOperandParser::ParseOperand(const Type &operandType,
                                                   Value &operand) {
  if (ParseOptionalOperand(operandType, operand))
    return _parser.emitError(
        _parser.getCurrentLocation(), "Expecting an operand.");
  return success();
}

ParseResult KrnlDialectOperandParser::ParseOperand(
    const Type &operandType, llvm::SmallVectorImpl<Value> &operandList) {
  if (ParseOptionalOperand(operandType, operandList))
    return _parser.emitError(
        _parser.getCurrentLocation(), "Expecting an operand.");

  return success();
}

void printDimAndSymbolList(Operation::operand_iterator& begin, unsigned numDims,
    unsigned numSymbols, OpAsmPrinter& p) {
  p << '(';
  p.printOperands(begin, begin + numDims);
  p << ')';

  if (numSymbols) {
    p << '[';
    p.printOperands(begin + numDims, begin + numDims + numSymbols);
    p << ']';
  }

  begin = std::next(begin, numDims + numSymbols);
}

void printBound(AffineMapAttr boundMap,
    Operation::operand_iterator& boundOperandsBeg, const char* prefix,
    OpAsmPrinter& p) {
  AffineMap map = boundMap.getValue();

  // Check if this bound should be printed using custom assembly form.
  // The decision to restrict printing custom assembly form to trivial cases
  // comes from the will to roundtrip MLIR binary -> text -> binary in a
  // lossless way.
  // Therefore, custom assembly form parsing and printing is only supported for
  // zero-operand constant maps and single symbol operand identity maps.
  if (map.getNumResults() == 1) {
    AffineExpr expr = map.getResult(0);

    // Print constant bound.
    if (map.getNumDims() == 0 && map.getNumSymbols() == 0) {
      if (auto constExpr = expr.dyn_cast<AffineConstantExpr>()) {
        p << constExpr.getValue();
        return;
      }
    }

    // Print bound that consists of a single SSA symbol if the map is over a
    // single symbol.
    if (map.getNumDims() == 0 && map.getNumSymbols() == 1) {
      if (auto symExpr = expr.dyn_cast<AffineSymbolExpr>()) {
        p.printOperand(*(boundOperandsBeg++));
        return;
      }
    }
  } else {
    // Map has multiple results. Print 'min' or 'max' prefix.
    p << prefix << ' ';
  }

  // Print the map and its operands.
  p << boundMap;
  printDimAndSymbolList(
      boundOperandsBeg, map.getNumDims(), map.getNumSymbols(), p);
}
}  // namespace onnf

namespace mlir {
void KrnlIterateOperandPack::pushConstantBound(int64_t bound) {
  if (boundMaps.size() % 2 == 0)
    _operands.emplace_back(inputLoops[boundMaps.size() / 2]);
  AffineMap map = builder.getConstantAffineMap(bound);
  boundMaps.emplace_back(AffineMapAttr::get(map));
}

void KrnlIterateOperandPack::pushOperandBound(mlir::Value operand) {
  if (boundMaps.size() % 2 == 0)
    _operands.emplace_back(inputLoops[boundMaps.size() / 2]);
  AffineMap map = builder.getSymbolIdentityMap();
  boundMaps.emplace_back(AffineMapAttr::get(map));
  _operands.emplace_back(operand);
}
}  // namespace mlir
