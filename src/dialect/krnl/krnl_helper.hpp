#pragma once

#include <queue>

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/StandardTypes.h"

namespace onnf {

class KrnlDialectOperandParser {
 public:
  explicit KrnlDialectOperandParser(mlir::OpAsmParser& parser)
      : _parser(parser), _builder(parser.getBuilder()){};

  // Parse an optional operand.
  mlir::ParseResult ParseOptionalOperand(
      const mlir::Type& operandType, mlir::Value*& operand);

  // Parse an optional operand and push it to an operand list.
  mlir::ParseResult ParseOptionalOperand(const mlir::Type& operandType,
      llvm::SmallVectorImpl<mlir::Value*>& operandList);

  // Parse a required operand.
  mlir::ParseResult ParseOperand(
      const mlir::Type& operandType, mlir::Value*& operand);

  // Parse a required operand and push it to an operand list.
  mlir::ParseResult ParseOperand(const mlir::Type& operandType,
      llvm::SmallVectorImpl<mlir::Value*>& operandList);

  // Do we have more operands to parse?
  bool hasOperandLeft() { return !_operandRefQueue.empty(); }

 private:
  mlir::OpAsmParser& _parser;

  mlir::Builder& _builder;

  // A queue storing the parsed SSA id references.
  std::queue<mlir::OpAsmParser::OperandType> _operandRefQueue;
};

// Adapted from:
// https://github.com/tensorflow/mlir/blob/6a150d70c7e06fb37cddd7188fa48cde9a90fe59/lib/Dialect/StandardOps/Ops.cpp#L197
// Main difference is that it advances the iterator `begin` as it consumes
// dimension and symbol operands.
void printDimAndSymbolList(mlir::Operation::operand_iterator& begin,
    unsigned numDims, unsigned numSymbols, mlir::OpAsmPrinter& p);

// Adapted from:
// https://github.com/tensorflow/mlir/blob/5cb42c914fed14cebbbe5c170b4e2784d2628304/lib/Dialect/AffineOps/AffineOps.cpp#L1272
// Main difference is that it advances the iterator `boundOperandsBeg` as it
// prints bound.
void printBound(mlir::AffineMapAttr boundMap,
    mlir::Operation::operand_iterator& boundOperandsBeg, const char* prefix,
    mlir::OpAsmPrinter& p);
}  // namespace onnf

namespace mlir {

struct KrnlIterateOperandPack {
  KrnlIterateOperandPack(mlir::Builder& builder,
      llvm::ArrayRef<mlir::Value*> inputLoops,
      llvm::ArrayRef<mlir::Value*> optimizedLoops)
      : builder(builder),
        inputLoops(inputLoops),
        optimizedLoops(optimizedLoops) {
    _operands.insert(
        _operands.end(), optimizedLoops.begin(), optimizedLoops.end());
  }

  void pushConstantBound(int64_t bound);

  void pushOperandBound(mlir::Value* operand);

  llvm::SmallVector<mlir::Value*, 8> getOperands() const { return _operands; }

  mlir::ArrayAttr getAttributes() const {
    return builder.getArrayAttr(boundMaps);
  }

  size_t getNumOptimizedLoops() const { return optimizedLoops.size(); }

  size_t getNumInputLoops() const { return inputLoops.size(); }

 private:
  int _boundIdx = 0;

  llvm::SmallVector<mlir::Value*, 8> _operands;

  llvm::SmallVector<mlir::Attribute, 8> boundMaps;

  llvm::ArrayRef<mlir::Value*> inputLoops, optimizedLoops;

  mlir::Builder& builder;
};

}  // namespace mlir
