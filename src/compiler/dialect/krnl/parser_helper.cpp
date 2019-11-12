//===------------------ parser_helper.cpp - MLIR Operations ---------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include "parser_helper.hpp"

#include "src/compiler/dialect/krnl/krnl_ops.hpp"

namespace onnf {

mlir::ParseResult KrnlDialectOperandParser::ParseOptionalOperand(
    mlir::Type operand_type, mlir::Value*& operand) {
  // If operand queue is empty, parse more operands and cache them.
  if (_operand_ref_queue.empty()) {
    // Parse operand types:
    llvm::SmallVector<mlir::OpAsmParser::OperandType, 2> operand_refs;
    _parser.parseOperandList(operand_refs);

    // Record operands:
    for (auto& operand_ref : operand_refs)
      _operand_ref_queue.emplace(operand_ref);
  }

  // If we parsed some operand reference(s), resolve the ref to an operand:
  if (!_operand_ref_queue.empty()) {
    auto operand_ref = _operand_ref_queue.front();
    _operand_ref_queue.pop();

    llvm::SmallVector<mlir::Value*, 1> operands;
    _parser.resolveOperand(operand_ref, operand_type, operands);
    operand = operands.front();
    return mlir::success();
  } else {
    operand = nullptr;
    return mlir::failure();
  }
}

mlir::ParseResult KrnlDialectOperandParser::ParseOperand(
    mlir::Type operand_type, mlir::Value*& operand) {
  ParseOptionalOperand(operand_type, operand);
  if (operand == nullptr)
    return _parser.emitError(
        _parser.getCurrentLocation(), "Expecting an operand.");
  return mlir::success();
}

}  // namespace onnf
