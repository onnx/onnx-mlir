/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------ CategoryMapper.cpp - Lowering CategoryMapper Op ---------===//
//
// Copyright 2021-2022 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers the ONNX CategoryMapper Operator to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Conversion/ONNXToKrnl/PerfectHash.hpp"
#include "src/Dialect/ONNX/ShapeInference/ONNXShapeHelper.hpp"

#include "mlir/Dialect/SCF/SCF.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace onnx_mlir;
using llvm::dbgs;

#define DEBUG_TYPE "category_mapper_onnx_to_krnl"

struct ONNXCategoryMapperOpLowering : public ConversionPattern {
  using PerfectHashTable = struct {
    Value G;
    Value V;
    Value len;
  };

  // When true causes injection of print stmts in the generated code.
  static const bool emitPrintStmts = false;

  ONNXCategoryMapperOpLowering(TypeConverter &typeConverter, MLIRContext *ctx)
      : ConversionPattern(
            typeConverter, ONNXCategoryMapperOp::getOperationName(), 1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    auto categoryMapperOp = cast<ONNXCategoryMapperOp>(op);
    ONNXCategoryMapperOpAdaptor operandAdaptor(operands);

    ONNXCategoryMapperOpShapeHelper shapeHelper(&categoryMapperOp, &rewriter,
        getDenseElementAttributeFromKrnlValue,
        loadDenseElementArrayValueAtIndex);
    LogicalResult shapeComputed = shapeHelper.computeShape(operandAdaptor);
    (void)shapeComputed;
    assert(succeeded(shapeComputed) && "Could not compute output shape");

    // Operands and attributes.
    Location loc = categoryMapperOp.getLoc();
    Value X = operandAdaptor.X();
    ArrayAttr cats_int64sAttr = categoryMapperOp.cats_int64sAttr();
    ArrayAttr cats_stringsAttr = categoryMapperOp.cats_stringsAttr();

    DenseElementsAttr cats_int64s = mlir::DenseElementsAttr::get(
        RankedTensorType::get(
            cats_int64sAttr.size(), rewriter.getIntegerType(64)),
        cats_int64sAttr.getValue());
    DenseElementsAttr cats_strings = mlir::DenseElementsAttr::get(
        RankedTensorType::get(cats_stringsAttr.size(),
            krnl::StringType::get(rewriter.getContext())),
        cats_stringsAttr.getValue());

    IntegerAttr default_int64 = categoryMapperOp.default_int64Attr();
    DenseElementsAttr default_string =
        (categoryMapperOp.default_stringAttr())
            ? mlir::DenseElementsAttr::get(
                  RankedTensorType::get(
                      {}, krnl::StringType::get(rewriter.getContext())),
                  categoryMapperOp.default_stringAttr().getValue())
            : nullptr;

    // Basic information.
    auto memRefType = convertToMemRefType(*op->result_type_begin());
    int64_t rank = memRefType.getShape().size();
    ShapedType inputType = X.getType().cast<ShapedType>();
    Type elementType = inputType.getElementType();

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = insertAllocAndDeallocSimple(
        rewriter, op, memRefType, loc, shapeHelper.dimsForOutput(0));

    MultiDialectBuilder<KrnlBuilder, MathBuilder> create(
        rewriter, op->getLoc());

    // Generate a perfect hash table. The hash table will be used to lookup the
    // index of the input values.
    PerfectHashTable perfectHashTable = createPerfectHashTable(cats_int64s,
        cats_strings, cats_int64sAttr, cats_stringsAttr, elementType, create);

    // Create loop invariant values.
    Value constantForCatsInt64s = create.krnl.constant(
        convertToMemRefType(cats_int64s.getType()), "cats_int64s", cats_int64s);

    Value constantForCatsStrings =
        create.krnl.constant(convertToMemRefType(cats_strings.getType()),
            "cats_strings", cats_strings);

    Value defaultInt64 = (default_int64)
                             ? create.math.constant(rewriter.getIntegerType(64),
                                   default_int64.getSInt())
                             : nullptr;
    Value defaultString =
        (default_string) ? create.krnl.constant(
                               MemRefType::get({}, krnl::StringType::get(
                                                       rewriter.getContext())),
                               "default_string", default_string)
                         : nullptr;

    // Lookup the index in the perfect hash table corresponding to
    // each input value.
    MemRefBoundsIndexCapture inputBounds(X);
    LiteralIndexExpr zeroIE(0);
    SmallVector<IndexExpr, 4> lbs(rank, zeroIE);
    SmallVector<IndexExpr, 4> ubs;
    inputBounds.getDimList(ubs);

    if (emitPrintStmts)
      create.krnl.printTensor("Input tensor:\n", X);

    ValueRange loopDef = create.krnl.defineLoops(rank);
    create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
        [&](KrnlBuilder &createKrnl, ValueRange loopInd) {
          // Determine the index of 'inputElem' in the perfect hash table
          // 'pHash'. Note: the index might not be valid (this happens
          // when the 'inputElem' is not present in the perfect hash
          // table).
          Value inputElem =
              loadElement(X, loopInd, elementType, rank, createKrnl);
          if (emitPrintStmts)
            create.krnl.printf("inputElem: ", inputElem, elementType);

          Value index, isIndexValid;
          std::tie(index, isIndexValid) =
              emitFindIndex(inputElem, elementType, perfectHashTable,
                  constantForCatsInt64s, constantForCatsStrings, create);

          if (emitPrintStmts)
            create.krnl.printf("index: ", index, index.getType());

          // Store the final result.
          scf::IfOp ifOp = rewriter.create<scf::IfOp>(
              loc, isIndexValid, /*withElseRegion=*/true);
          storeResult(index, elementType, ifOp, constantForCatsInt64s,
              constantForCatsStrings, defaultInt64, defaultString, alloc,
              loopInd, createKrnl, rewriter);
        });

    rewriter.replaceOp(op, alloc);

    LLVM_DEBUG({
      FuncOp function = getContainingFunction(op);
      assert(function && "Could not find parent function");
      dbgs() << "function:\n" << function << "\n";
    });

    return success();
  }

private:
  Attribute getElemAttr(ArrayAttr arr, int32_t idx) const {
    return arr.getValue()[idx];
  }

  // Generate a perfect hash table for the input dictionary.
  // Depending on the runtime type 'elementType' (the type of the element of
  // the input tensor) this function created a perfect hash table for:
  //  - cats_int64s if elementType is a int64_t
  //  - cats_strings if elementType is a string
  PerfectHashTable createPerfectHashTable(DenseElementsAttr cats_int64s,
      DenseElementsAttr cats_strings, ArrayAttr cats_int64s_ArrayAttr,
      ArrayAttr cats_strings_ArrayAttr, Type elementType,
      const MultiDialectBuilder<KrnlBuilder, MathBuilder> &create) const {
    OpBuilder builder = create.krnl.getBuilder();
    PerfectHashTable res;

    // Create constants to hold the arrays 'G' and 'V'.
    auto createConstants = [&](const SmallVector<int32_t> &G,
                               const SmallVector<int32_t> &V) {
      assert(V.size() == G.size() && "V and G should have the same size");

      MemRefType type = MemRefType::get(
          {static_cast<int64_t>(V.size())}, builder.getIntegerType(32));
      res.G = create.krnl.constant(type, "G", builder.getI32TensorAttr(G));
      res.V = create.krnl.constant(type, "V", builder.getI32TensorAttr(V));
      res.len = create.math.constant(builder.getIntegerType(32), G.size());
      return res;
    };

    TypeSwitch<Type>(elementType)
        .Case<IntegerType>([&](IntegerType type) {
          // Populate the dictionary.
          assert(type.getWidth() == 64 && "Unexpected integer type");
          std::map<int64_t, int32_t> dict;
          int32_t size = cats_int64s.size();
          for (int32_t idx = 0; idx < size; ++idx) {
            Attribute elemAttr = getElemAttr(cats_int64s_ArrayAttr, idx);
            int64_t key = elemAttr.cast<IntegerAttr>().getInt();
            dict[key] = idx;
          }

          // Create the perfect hash (i.e. G and V), store them into
          // constants.
          PerfectHash<int64_t, int32_t> pHash(dict);
          res = createConstants(pHash.getG(), pHash.getV());
        })
        .Case<krnl::StringType>([&](krnl::StringType type) {
          // Populate the dictionary.
          std::map<StringRef, int32_t> dict;
          int32_t size = cats_strings.size();
          for (int32_t idx = 0; idx < size; ++idx) {
            Attribute elemAttr = getElemAttr(cats_strings_ArrayAttr, idx);
            StringRef key = elemAttr.cast<StringAttr>().getValue();
            dict[key] = idx;
          }

          // Create the perfect hash (i.e. G and V), store them into
          // constants.
          PerfectHash<StringRef, int32_t> pHash(dict);
          res = createConstants(pHash.getG(), pHash.getV());
        })
        .Default([&](Type type) {
          llvm::errs() << "type: " << type << "\n";
          llvm_unreachable("Illegal KeyTy");
        });

    return res;
  }

  Value loadElement(Value memref, ValueRange loopInd, Type elementType,
      int64_t rank, KrnlBuilder &createKrnl) const {
    Value inputElem;
    TypeSwitch<Type>(elementType)
        .Case<IntegerType>(
            [&](IntegerType) { inputElem = createKrnl.load(memref, loopInd); })
        .Case<krnl::StringType>([&](krnl::StringType stringType) {
          MathBuilder createMath(createKrnl);
          Value zero = createMath.constant(
              createMath.getBuilder().getIntegerType(64), 0);
          auto memRefType = MemRefType::get(
              {rank}, krnl::StringType::get(elementType.getContext()));
          Value stringMemRef = createKrnl.getRef(memRefType, memref, zero);
          inputElem = createKrnl.load(stringMemRef, loopInd);
        })
        .Default([&](Type type) {
          llvm::errs() << "type: " << type << "\n";
          llvm_unreachable("Unexpected elementType");
        });

    return inputElem;
  }

  // Determine the index of 'inputElem' in the perfect hash table 'pHash'.
  // Return the index and a true/false value depending on whether the index is
  // valid or not.
  std::tuple<Value, Value> emitFindIndex(Value inputElem, Type elementType,
      const PerfectHashTable &pHash, Value constantForCatsInt64s,
      Value constantForCatsStrings,
      const MultiDialectBuilder<KrnlBuilder, MathBuilder> &create) const {
    OpBuilder builder = create.krnl.getBuilder();
    Value index = create.krnl.findIndex(inputElem, pHash.G, pHash.V, pHash.len);

    std::tuple<Value, Value> res;
    TypeSwitch<Type>(elementType)
        .Case<IntegerType>([&](IntegerType type) {
          // Determine whether the index returned is valid.
          // The index is valid if 'inputElem' compares equal to the string in
          // 'constantForCatsInt64s'.
          Value compareVal = create.krnl.load(constantForCatsInt64s, {index});
          Value isIndexValid = create.math.eq(inputElem, compareVal);
          res = std::make_tuple(index, isIndexValid);
        })
        .Case<krnl::StringType>([&](krnl::StringType type) {
          // Determine whether the index returned is valid.
          // The index is valid if 'inputElem' compares equal to the string in
          // 'constantForCatsStrings'.
          Value compareVal = create.krnl.load(constantForCatsStrings, {index});
          Value strlenRes = create.krnl.strlen(compareVal);
          Value strncmpRes =
              create.krnl.strncmp(inputElem, compareVal, strlenRes);
          Value zeroVal = create.math.constant(builder.getIntegerType(32), 0);
          Value isIndexValid = create.math.eq(strncmpRes, zeroVal);
          res = std::make_tuple(index, isIndexValid);
        })
        .Default([&](Type type) {
          llvm::errs() << "type: " << type << "\n";
          llvm_unreachable("Illegal KeyTy");
        });

    return res;
  }

  // Store the result in the 'alloc' buffer.
  // Given the 'index' of the input element and an 'ifOp' operation, this
  // function generates code in the 'then' and 'else' basic blocks to determines
  // whether the index is valid. If the index is valid it is used to retrieve
  // the mapped value corresponding to the input element, otherwise the default
  // value is used.
  void storeResult(Value index, Type elementType, scf::IfOp ifOp,
      Value constantForCatsInt64s, Value constantForCatsStrings,
      Value defaultInt64, Value defaultString, Value alloc, ValueRange loopInd,
      const KrnlBuilder &createKrnl,
      ConversionPatternRewriter &rewriter) const {
    TypeSwitch<Type>(elementType)
        .Case<IntegerType>([&](IntegerType type) {
          // index is valid: retrieve the value from 'cat_strings'.
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
          Value loadData = createKrnl.load(constantForCatsStrings, {index});
          createKrnl.store(loadData, alloc, loopInd);

          // index is not valid: store the default value.
          rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
          Value loadDefault = createKrnl.load(defaultString);
          createKrnl.store(loadDefault, alloc, loopInd);
        })
        .Case<krnl::StringType>([&](krnl::StringType type) {
          // index is valid: retrieve the value from 'cat_int64s'.
          rewriter.setInsertionPointToStart(&ifOp.getThenRegion().front());
          Value loadData = createKrnl.load(constantForCatsInt64s, {index});
          createKrnl.store(loadData, alloc, loopInd);

          // index is not valid: store the default value.
          rewriter.setInsertionPointToStart(&ifOp.getElseRegion().front());
          createKrnl.store(defaultInt64, alloc, loopInd);
        })
        .Default([&](Type type) {
          llvm::errs() << "type: " << type << "\n";
          llvm_unreachable("Illegal KeyTy");
        });
  }
};

void populateLoweringONNXCategoryMapperOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx) {
  patterns.insert<ONNXCategoryMapperOpLowering>(typeConverter, ctx);
}
