//===--------------------- krnl_ops.cpp - MLIR Operations -----------------===//
//
// Copyright 2019 The IBM Research Authors.
//
// =============================================================================
//
//===----------------------------------------------------------------------===//

#include <iostream>
#include <queue>

#include "src/compiler/dialect/krnl/parser_helper.hpp"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallBitVector.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Function.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/Module.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"

#include "krnl_ops.hpp"

using namespace mlir;

namespace mlir {
KrnlOpsDialect::KrnlOpsDialect(MLIRContext* context)
    : Dialect(getDialectNamespace(), context) {
  addOperations<
#define GET_OP_LIST
#include "src/compiler/krnl.cpp.inc"
      >();
  addTypes<LoopType>();
}

//===----------------------------------------------------------------------===//
// KrnlDefineLoopsOp
//===----------------------------------------------------------------------===//

void KrnlDefineLoopsOp::build(
    Builder* builder, OperationState& result, int64_t num_loops) {
  // Create the same number of dimension handlers as the number of
  // dimensions in the associated integer set.
  result.types.append(num_loops, LoopType::get(builder->getContext()));
  result.addAttribute(
      getNumLoopsAttrName(), builder->getI32IntegerAttr(num_loops));
}

void print(OpAsmPrinter& p, KrnlDefineLoopsOp& op) {
  auto num_loop_attr = op.getAttrOfType<IntegerAttr>(op.getNumLoopsAttrName());
  p << "krnl.define_loops " << num_loop_attr.getValue().getSExtValue();
}

ParseResult parseKrnlDefineLoopsOp(
    OpAsmParser& parser, OperationState& result) {
  // Parse the attribute indicating number of loops defined.
  IntegerAttr num_loops;
  auto& builder = parser.getBuilder();
  auto int32_type = builder.getIntegerType(64);
  if (parser.parseAttribute(num_loops, int32_type,
          KrnlDefineLoopsOp::getNumLoopsAttrName(), result.attributes))
    return failure();

  auto loop_types = llvm::SmallVector<Type, 4>(
      num_loops.getValue().getSExtValue(), LoopType::get(builder.getContext()));
  if (parser.addTypesToList(loop_types, result.types))
    return failure();

  return success();
}

//===----------------------------------------------------------------------===//
// KrnlOptimizeLoopsOp
//===----------------------------------------------------------------------===//

void KrnlOptimizeLoopsOp::build(
    Builder* builder, OperationState& result, int num_optimized_loops) {
  result.types.append(
      num_optimized_loops, LoopType::get(builder->getContext()));
  // Create a region and a block for the body.
  // Schedule intrinsics will be placed into this region.
  Region* region = result.addRegion();
  auto* body = new Block();
  region->push_back(body);
}

void print(OpAsmPrinter& p, KrnlOptimizeLoopsOp& op) {
  p << "krnl.optimize_loops ";
  p.printRegion(op.region(), /*printEntryBlockArgs=*/false,
      /*printBlockTerminators=*/true);
  p << " : ";
  p.printFunctionalType(op);
}

ParseResult parseKrnlOptimizeLoopsOp(
    OpAsmParser& parser, OperationState& result) {
  // Parse the schedule body region.
  Region* region = result.addRegion();
  if (parser.parseRegion(*region, llvm::None, llvm::None))
    return failure();

  // Parse the function type for the schedule operation.
  // Then following the hint of this parsed function type, parse the
  // returned timestamp space dimension handlers.
  FunctionType schedule_func_type;
  if (parser.parseColonType(schedule_func_type) ||
      parser.addTypesToList(schedule_func_type.getResults(), result.types)) {
    failure();
  }

  return success();
}

//===----------------------------------------------------------------------===//
// KrnlIterateOp
//===----------------------------------------------------------------------===//

/*!
 * Build a Krnl Dialect iterate operation.
 * input_loops: a collection of input krnl.loops being optimized.
 * optimized_loops: a collection of optimized (scheduled) krnl.loops.
 * operand_bounds: a collection of SSA value bounds.
 * const_bounds: a collection of constant bounds.
 * bound_types: a collection of integer values indicating how bounds are given.
 *   0 : bound is given as an integer in const_bounds.
 *   1 : bound is given as an operand in operand_bounds.
 *   2 : bound is given as an affine map. (TODO).
 *
 * The following example illustrates how induction variable bounds are parsed
 * from builder function inputs:
 *
 * - operand_bounds = [N, M]
 * - const_bounds = [10, 20]
 * - bound_types = [0, 1, 1, 0]
 *
 * Then the bounds will be parsed as:
 *   %i0 = 10 to N : %i1 = M to 20
 */
void KrnlIterateOp::build(Builder* builder, OperationState& result,
    ArrayRef<Value*> input_loops, ArrayRef<Value*> optimized_loops,
    ArrayRef<Value*> operand_bounds, ArrayRef<int64_t> const_bounds,
    ArrayRef<int> bound_types) {
  // Record optimized loops and the number of such loops.
  result.addOperands(optimized_loops);
  result.addAttribute(getNumOptimizedLoopsAttrName(),
      builder->getI64IntegerAttr(optimized_loops.size()));

  // Record input loops and the number of such loops.
  result.addOperands(input_loops);
  result.addAttribute(getNumInputLoopsAttrName(),
      builder->getI64IntegerAttr(input_loops.size()));

  // Record bound either as attribute or from operand list.
  auto next_operand_bound = operand_bounds.begin();
  auto next_const_bound = const_bounds.begin();
  for (size_t i = 0; i < bound_types.size(); i++) {
    auto bound_type = bound_types[i];
    if (bound_type == 0) {
      // Constant bound.
      result.addAttribute(getBoundAttrName(i / 2, i % 2),
          builder->getI64IntegerAttr(*next_const_bound));
      next_const_bound = std::next(next_const_bound);
    } else {
      // Operand bound.
      result.addOperands(*next_operand_bound);
      next_operand_bound = std::next(next_operand_bound);
    }
  }

  // Record bound types as attribute:
  result.addAttribute(KrnlIterateOp::getBoundTypesAttrName(),
      builder->getI32ArrayAttr(bound_types));

  // Create a region and a block for the body. The arguments of the region are
  // the loop induction variables; there can be multiple induction variables
  // associated with the same krnl.iterate operation.
  Region* bodyRegion = result.addRegion();
  auto* body = new Block();
  auto body_args = llvm::SmallVector<Type, 4>(
      input_loops.size(), IndexType::get(builder->getContext()));
  body->addArguments(body_args);
  bodyRegion->push_back(body);

  ensureTerminator(*bodyRegion, *builder, result.location);
}

void print(OpAsmPrinter& p, KrnlIterateOp& op) {
  p << "krnl.iterate(";
  // Print optimized loops:
  auto num_optimized_loops = op.getNumOptimizedLoops();
  p.printOperands(op.operand_begin(), op.operand_begin() + num_optimized_loops);
  p << ") with (";

  // Set up iterator to input loops:
  auto num_input_loops = op.getNumInputLoops();
  auto input_loop_begin = op.operand_begin() + num_optimized_loops;

  // Set up iterators to operand bounds.
  auto next_operand_bound = input_loop_begin + num_input_loops;

  // Function to print a lower or upper bound.
  auto print_bound = [&](ArrayRef<Attribute> bound_types, size_t idx) {
    IntegerAttr type = bound_types[idx].dyn_cast<IntegerAttr>();
    if (type.getValue().getSExtValue() == 0) {
      // Bound is an integer attribute.
      auto bound_idx = idx / 2;
      auto is_ub = idx % 2;
      IntegerAttr bound = op.getAttrOfType<IntegerAttr>(
          KrnlIterateOp::getBoundAttrName(bound_idx, is_ub));
      p << bound.getValue().getSExtValue();
    } else {
      // Bound is an operand.
      p.printOperand(*next_operand_bound);
      next_operand_bound = std::next(next_operand_bound);
    }
  };

  auto induction_variables = op.bodyRegion().front().getArguments();
  ArrayRef<Attribute> bound_types =
      op.getAttrOfType<ArrayAttr>(KrnlIterateOp::getBoundTypesAttrName())
          .getValue();

  // Print input loop operands, induction variables and their ranges.
  for (size_t i = 0; i < num_input_loops; i++) {
    if (i != 0)
      p << ", ";

    p.printOperand(*std::next(input_loop_begin, i));
    p << " -> ";

    // Print induction variable block argument.
    p.printOperand(induction_variables[i]);
    p << " = ";

    print_bound(bound_types, 2 * i);  // Print lower bound.
    p << " to ";
    print_bound(bound_types, 2 * i + 1);  // Print upper bound.
  }
  p << ")";

  p.printRegion(op.bodyRegion(), /*printEntryBlockArgs=*/false,
      /*printBlockTerminators=*/false);
}

ParseResult parseKrnlIterateOp(OpAsmParser& parser, OperationState& result) {
  auto builder = parser.getBuilder();
  auto context = builder.getContext();
  onnf::KrnlDialectOperandParser operand_parser(parser);

  // Parse optimized loops:
  SmallVector<OpAsmParser::OperandType, 4> num_optimized_loops;
  if (parser.parseOperandList(
          num_optimized_loops, OpAsmParser::Delimiter::Paren) ||
      parser.resolveOperands(num_optimized_loops,
          LoopType::get(result.getContext()), result.operands))
    return failure();

  // Record how many optimized loops did we parse.
  result.addAttribute(KrnlIterateOp::getNumOptimizedLoopsAttrName(),
      builder.getI64IntegerAttr(num_optimized_loops.size()));

  // Parse input loops and their lower and upper bounds.
  SmallVector<OpAsmParser::OperandType, 4> in_loop_refs, induction_var_refs;
  SmallVector<Value*, 4> in_loop_operands, operand_bounds;
  SmallVector<Attribute, 4> bound_types;
  SmallVector<IntegerAttr, 4> const_bounds;

  if (parser.parseKeyword("with") || parser.parseLParen())
    return failure();

  // A function to parse a lower or upper bound.
  auto parse_bound = [&result, &builder, &operand_parser, &parser, &bound_types,
                         &operand_bounds, &const_bounds](
                         bool is_ub, size_t bound_pair_count) -> ParseResult {
    // Try parse an SSA operand.
    Value* bound;
    operand_parser.ParseOptionalOperand(builder.getIndexType(), bound);

    if (bound != nullptr) {
      // Parsed an SSA id as bound.
      operand_bounds.emplace_back(bound);
      // Record bound_type as an operand type.
      bound_types.emplace_back(builder.getI32IntegerAttr(0));
    } else {
      // Bound is not an SSA id, then it must be an integer.
      // Parse an integer constant attribute.
      IntegerAttr boundAttr;
      if (parser.parseAttribute(boundAttr, builder.getIndexType(),
              KrnlIterateOp::getBoundAttrName(bound_pair_count, is_ub),
              result.attributes))
        return failure();
      const_bounds.emplace_back(
          builder.getIntegerAttr(builder.getIndexType(), boundAttr.getValue()));

      // Record that the bound_type is a constant integer attribute.
      bound_types.emplace_back(builder.getI32IntegerAttr(1));
    }
  };

  bool keep_parsing;            // Do we keep parsing loops/bounds?
  size_t bound_pair_count = 0;  // Record the number of bound pairs parsed.
  do {
    // Parse an input loop operand;
    Value* in_loop_operand;
    operand_parser.ParseOperand(LoopType::get(context), in_loop_operand);
    in_loop_operands.emplace_back(in_loop_operand);

    parser.parseArrow();

    // Parse induction variable.
    OpAsmParser::OperandType induction_var;
    if (parser.parseRegionArgument(induction_var) || parser.parseEqual())
      return failure();
    induction_var_refs.emplace_back(induction_var);

    // Parse bound par (min to max).
    if (parse_bound(false, bound_pair_count) || parser.parseKeyword("to") ||
        parse_bound(true, bound_pair_count))
      return failure();

    bound_pair_count++;
    // We may fail to parse a comma if an operand bound is followed by
    // a comma and the next input loop operand, in which case
    // the entire "{operand bound}, {input_loop_operand}" sequence will
    // be parsed as an operand list.
    parser.parseOptionalComma();

    // If we don't see a RParen token, we keep parsing.
    keep_parsing = failed(parser.parseOptionalRParen());
  } while (keep_parsing);

  // At this point, there shouldn't be any operands left to parse.
  if (operand_parser.has_operand_left())
    return parser.emitError(parser.getCurrentLocation());

  // Record how many input loops did we parse.
  result.addOperands(in_loop_operands);
  result.addAttribute(KrnlIterateOp::getNumInputLoopsAttrName(),
      builder.getI64IntegerAttr(in_loop_operands.size()));

  // Add operand bounds to the list of operands of current operation.
  result.addOperands(operand_bounds);

  // A list of 2N elements where the (2n) and (2n+1) th element specifies
  // whether the lower and upper bound of the n'th induction variable is stored
  // as an operand or as an attribute. N being the number of input loops
  // specified in this krnl.iterate operation.
  result.addAttribute(KrnlIterateOp::getBoundTypesAttrName(),
      builder.getArrayAttr(bound_types));

  // Parse the schedule body region.
  Region* region = result.addRegion();
  SmallVector<Type, 4> induction_var_types(
      induction_var_refs.size(), builder.getIndexType());
  if (parser.parseRegion(*region, induction_var_refs, induction_var_types))
    return failure();

  // Ensure iterate region is closed off with krnl.terminate.
  KrnlIterateOp::ensureTerminator(
      *region, parser.getBuilder(), result.location);

  return success();
}

static LogicalResult verify(KrnlIterateOp op) {
  // TODO: Verify number of induction variable bounds matches the number of
  // input loops.
}

//===----------------------------------------------------------------------===//
// KrnlReturnLoopsOp
//===----------------------------------------------------------------------===//

void print(OpAsmPrinter& p, KrnlReturnLoopsOp& op) {
  p << "krnl.return_loops ";
  p.printOperands(op.operand_begin(), op.operand_end());
}

ParseResult parseKrnlReturnLoopsOp(
    OpAsmParser& parser, OperationState& result) {
  // Parse the loops to return.
  SmallVector<OpAsmParser::OperandType, 4> timestamp_dim_handlers;
  if (parser.parseOperandList(timestamp_dim_handlers) ||
      parser.resolveOperands(timestamp_dim_handlers,
          LoopType::get(result.getContext()), result.operands))
    return failure();

  return success();
}

#define GET_OP_CLASSES
#include "src/compiler/krnl.cpp.inc"
}  // namespace mlir
