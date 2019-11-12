//===------------------ parser_helper.hpp - MLIR Operations ---------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#pragma once

#include <queue>

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

namespace onnf {

class KrnlDialectOperandParser {
 public:
  KrnlDialectOperandParser(mlir::OpAsmParser& parser)
      : _parser(parser), _builder(parser.getBuilder()){};

  // Parse an optional operand.
  mlir::ParseResult ParseOptionalOperand(
      mlir::Type operand_type, mlir::Value*& operand);

  // Parse a required operand.
  mlir::ParseResult ParseOperand(
      mlir::Type operand_type, mlir::Value*& operand);

  // Do we have more operands to parse?
  bool has_operand_left() { return !_operand_ref_queue.empty(); }

 private:
  mlir::OpAsmParser& _parser;

  mlir::Builder& _builder;

  // A queue storing the parsed SSA id references.
  std::queue<mlir::OpAsmParser::OperandType> _operand_ref_queue;
};

}  // namespace onnf
