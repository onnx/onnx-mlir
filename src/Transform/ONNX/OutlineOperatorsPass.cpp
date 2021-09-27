/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------- OutlineOperatorsPass.cpp - Operator Outlining ---------------------===//
//
// Copyright 2021 The IBM Research Authors.
//
// =============================================================================
//
// This file implements a pass to outline each operator for debugging purposes
//
//===----------------------------------------------------------------------===//

#include <regex>

#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/BlockAndValueMapping.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Rewrite/PatternApplicator.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/SCF.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Support/raw_ostream.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Interface/ShapeInferenceOpInterface.hpp"
#include "src/Pass/Passes.hpp"
#include <iostream>

using BlockListType = llvm::iplist<mlir::Block>;
using namespace mlir;


#include <stdio.h>
#include <stdlib.h>
#include <execinfo.h>
#include <cxxabi.h>
/* Obtain a backtrace and print it to stdout. */
void
print_trace (void)
{
  void *array[50];
  char **symbollist;
  int size, i;

  size = backtrace (array, 50);
  symbollist = backtrace_symbols (array, size);
  /*if (strings != NULL)
  {

    printf ("Obtained %d stack frames.\n", size);
    for (i = 0; i < size; i++)
      printf ("%s\n", strings[i]);
  }*/
 // allocate string which will be filled with the demangled function name
    size_t funcnamesize = 256;
    char* funcname = (char*)malloc(funcnamesize);

    // iterate over the returned symbol lines. skip the first, it is the
    // address of this function.
    for (int i = 1; i < size; i++)
    {
	char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

	// find parentheses and +address offset surrounding the mangled name:
	// ./module(function+0x15c) [0x8048a6d]
	for (char *p = symbollist[i]; *p; ++p)
	{
	    if (*p == '(')
		begin_name = p;
	    else if (*p == '+')
		begin_offset = p;
	    else if (*p == ')' && begin_offset) {
		end_offset = p;
		break;
	    }
	}

	if (begin_name && begin_offset && end_offset
	    && begin_name < begin_offset)
	{
	    *begin_name++ = '\0';
	    *begin_offset++ = '\0';
	    *end_offset = '\0';

	    // mangled name is now in [begin_name, begin_offset) and caller
	    // offset in [begin_offset, end_offset). now apply
	    // __cxa_demangle():

	    int status;
	    char* ret = abi::__cxa_demangle(begin_name,
					    funcname, &funcnamesize, &status);
	    if (status == 0) {
		funcname = ret; // use possibly realloc()-ed string
		printf("  %s : %s+%s\n",
			symbollist[i], funcname, begin_offset);
	    }
	    else {
		// demangling failed. Output function name as a C function with
		// no arguments.
		printf("  %s : %s()+%s\n",
			symbollist[i], begin_name, begin_offset);
	    }
	}
	else
	{
	    // couldn't parse the line? print the whole line.
	    printf("  %s\n", symbollist[i]);
	}
    }

    free(funcname);
    free(symbollist);
}


namespace {

  std::string getOperationName(Operation *op) {
    auto symbolAttr =
        op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    if (symbolAttr)
      return std::string(symbolAttr.getValue());
    return (op->getName().getStringRef().str());
  }

static int64_t outlineCount=0;
/*!
 *  Pass that puts each operator in a separate function called from the
 *  main graph
 *  
 */
class OutlineOperatorsPass : public mlir::PassWrapper<OutlineOperatorsPass,
                               OperationPass<mlir::ModuleOp>> {
private:

public:
  OutlineOperatorsPass() {}

  std::string getOpName(Operation *op) {
    auto symbolAttr =
        op->getAttrOfType<StringAttr>(SymbolTable::getSymbolAttrName());
    if (symbolAttr)
      return std::string(symbolAttr.getValue());
    return (op->getName().getStringRef().str());
  }

class OutlinePattern : public RewritePattern {
public:
  //OutlinePattern(PatternBenefit benefit, MLIRContext *context)
  //    : RewritePattern(OnnxGemmOp::getOperationName(), benefit, context) {}

  OutlinePattern(StringRef opName,PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(opName, benefit, context) {}

  OutlinePattern(PatternBenefit benefit, MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), benefit, context) {}

  OutlinePattern(MLIRContext *context)
      : RewritePattern(MatchAnyOpTypeTag(), PatternBenefit(1), context) {}

  LogicalResult matchAndRewrite(Operation *op, PatternRewriter &rewriter) const {
      auto opName = getOperationName(op);
    std::cout << "**** RewritePattern - inside match and rewrite: " << "Operation is " << opName << std::endl;
    genOutlinedFunction(rewriter, op);
    return success();
  }    

};

/// Populate the pattern list.
void collectOutlinePatterns(RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<OutlinePattern>("onnx.Gemm",/*benefit=*/1, ctx);
  //patterns.add<OutlinePattern>("onnx.Sigmoid",/*benefit=*/1, ctx);
  //patterns.add<OutlinePattern>("onnx.Flatten",/*benefit=*/1, ctx);
  //patterns.add<OutlinePattern>(/*benefit=*/1, ctx);
}

/// Define a custom PatternRewriter for use by the driver.
class OutlinePatternRewriter : public PatternRewriter {
public:
  OutlinePatternRewriter(MLIRContext *ctx) : PatternRewriter(ctx) {}

  /// Override the necessary PatternRewriter hooks here.
};

/// Apply the custom driver to `op`.
void applyOutlinePatternDriver(Operation *op,
                          RewritePatternSet &patterns) {
  // Initialize the custom PatternRewriter.
  OutlinePatternRewriter rewriter(op->getContext());
  
  std::cout << "   entered pattern driver" << std::endl;
/*for (std::unique_ptr<RewritePattern> &pat : patterns.getNativePatterns()) {
  std::cout << "found pattern " << std::endl;
}*/
    // Create the applicator and apply our cost model.
  auto frozen = FrozenRewritePatternSet(std::move(patterns));
  PatternApplicator applicator(frozen);
  applicator.applyDefaultCostModel();

  std::cout << "   built cost model " << std::endl;
  /*applicator.walkAllPatterns([](const Pattern &pat){
      std::cout << "walking ..." <<std::endl;
  });*/
  // Try to match and apply a pattern.
op->walk([&](Operation *op){

  LogicalResult result = applicator.matchAndRewrite(op, rewriter, [op](const Pattern &pat) -> bool {
      std::cout << "matching ..." <<std::endl;
      //print_trace();
      if (op->getDialect()->getNamespace() == "onnx") {
          std::cout << "   found an onnx op" << std::endl;
          return true;
      }
      else return false;
  }, [](const Pattern &pat) {
      std::cout << "failing ..." <<std::endl;
  }, [](const Pattern &pat) -> LogicalResult {
      std::cout << "succeeding ..." <<std::endl;
      return success();
  });

  std::cout << "   applied pattern " << std::endl;

  if (failed(result)) {
    // ... No patterns were applied.
  }
  // ... A pattern was successfully applied.
  });
}
  static void genOutlinedFunction(PatternRewriter &rewriter, Operation *op) {
      MLIRContext *context=op->getContext();
      Location loc=op->getLoc();
      auto opName = getOperationName(op);

      const std::string funcName=opName+std::to_string(outlineCount);
      //const llvm::StringRef name=funcName;
      ++outlineCount;
      std::cout << "   entered genOutlinedFunction: funcName=" << funcName << std::endl;

      PatternRewriter::InsertionGuard insertGuard(rewriter);
      auto inputVals = op->getOperands();
      auto results = op->getResults();
      std::cout << "      number of results " << results.size() << "  number of inputs " << inputVals.size() << std::endl;
      TypeRange sgTRange(results);
      auto sgOp = rewriter.create<ONNXSubgraphOp>(loc,sgTRange,inputVals);
      Region sgRegion(sgOp.getOperation());
      Block *sgBlock = new Block();
      sgRegion.push_back(sgBlock);
      //      auto sgBlock = sgRegion.emplaceBlock();
      //sgOp.body().push_back(sgBlock);

      rewriter.setInsertionPointToStart(op->getParentOfType<ModuleOp>().getBody());
      FunctionType ftype = FunctionType::get(context, inputVals.getTypes(),sgTRange);
      auto outlinedFunc = rewriter.create<FuncOp>(loc, funcName, /*type=*/ftype);

      // Create and set insertion point to entry block.
      Block *funcBlock = new Block();
      outlinedFunc.body().push_back(funcBlock);
      for (auto arg : inputVals.getTypes())
         funcBlock->addArgument(arg);
      rewriter.setInsertionPointToStart(&outlinedFunc.body().back());
     BlockAndValueMapping bvm;
     for (auto it : llvm::zip(inputVals, outlinedFunc.getArguments()))
       bvm.map(std::get<0>(it), std::get<1>(it));
     auto bodyOp=rewriter.clone(*op,bvm);

    // for (auto op : term->getOperands())
    //   terminatorOperands.push_back(bvm.lookup(op));
     rewriter.create<ONNXReturnOp>(loc,bodyOp->getResults());
 
     //ifOrElseRegion.front().clear();
     //b.setInsertionPointToEnd(&ifOrElseRegion.front());
     //Operation *call = b.create<CallOp>(loc, outlinedFunc, values);
     //b.create<scf::YieldOp>(loc, call->getResults());
     
      rewriter.setInsertionPointToStart(sgBlock);
      Operation *call = rewriter.create<CallOp>(loc, outlinedFunc, inputVals);
      rewriter.create<scf::YieldOp>(loc, call->getResults());

       std::cout << "number of results for sgOp is " << sgOp.getNumResults() << std::endl;

      op->replaceAllUsesWith(sgOp);
      rewriter.eraseOp(op);
  }

  void processOp(Operation *op) {
      //auto onnxtype = ONNXOpsDialect.getTypeID();
      auto opName = getOpName(op);
      std::cout << "Operation is " << opName << std::endl;
      if (op->getDialect()->getNamespace() == "onnx") {
          std::cout << "   is an onnx op" << std::endl;
          if (opName == "onnx.Gemm") {
          std::cout << "   --- outline Gemm" << std::endl;
          }
      }
      for (Region &region : op->getRegions())
         processRegion(region);
  }

 void processRegion(Region &region) {
     std::cout << "   -- entering region" << std::endl;
     for (mlir::Block &block : region.getBlocks())
        processBlock(block);
  } 

  void processBlock(mlir::Block &block) {
    std::cout << "   -- entering block" << std::endl;
    for (mlir::Operation &op : block.getOperations())
        processOp(&op);
  }


  void runOnOperation() override {
    //processOp(getOperation());
    RewritePatternSet patterns(&getContext());
    collectOutlinePatterns(patterns,&getContext());
    std::cout << "---------------------------------------------" << std::endl;
    std::cout << "       apply patterns " << std::endl;
    applyOutlinePatternDriver(getOperation(),patterns);

   }

};
} // end anonymous namespace

/*!
 * Create an Outline Operators pass.
 */
std::unique_ptr<mlir::Pass> mlir::createOutlineOperatorsPass() {
  return std::make_unique<OutlineOperatorsPass>();
}
