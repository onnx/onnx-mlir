/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===---------------- Elementwise.cpp - Elementwise Ops -------------------===//
//
// Copyright 2019-2024 The IBM Research Authors.
//
// =============================================================================
//
// This file lowers ONNX element-wise operators to Krnl dialect.
//
//===----------------------------------------------------------------------===//

#define _USE_MATH_DEFINES
#include <cmath>

#include "llvm/Support/Debug.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Conversion/ONNXToKrnl/ONNXToKrnlCommon.hpp"
#include "src/Dialect/Krnl/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/ShapeHelper.hpp"

#define DEBUG_TYPE "lowering-to-krnl"

using namespace mlir;

// All code specific to generate a specific operation goes here.
#include "src/Conversion/ONNXToKrnl/Math/Elementwise.cpp.inc"

namespace onnx_mlir {

// Check the input, x, can be reused as the output buffer
bool isBufferReusable(Value x, MemRefType outputType) {
  if (!x.hasOneUse())
    return false;

  Type xType = x.getType();
  auto inputType = dyn_cast<ShapedType>(xType);
  if (!inputType)
    return false;
  // Currently, only static shape could be reused.
  // ToFix: use DimAnalysis to handle dynamic shape.
  if (!hasStaticShape(inputType))
    return false;
  if (!hasStaticShape(outputType))
    return false;

  // Currently reuse requires that the shape has to be the same.
  // ToFix: If the shape is not the same, memref.cast can be used.
  if (getRank(inputType) != getRank(outputType))
    return false;
  for (int64_t i = 0; i < getRank(inputType); i++) {
    if (inputType.getShape()[i] != outputType.getShape()[i])
      return false;
  }

  // ToFix: The simd padding is not checked
  // We did not record whether the memref is padded or not.
  // The padding added to the memref the as an attribute, or not needed.
  return true;
}

// Traverse the operands to find the candidate for buffer reuse.
// Return -1, if no candidate is found.
int whichBufferToReuse(ValueRange values, MemRefType outputType) {
  for (size_t i = 0; i < values.size(); i++) {
    if (isBufferReusable(values[i], outputType))
      return i;
  }
  return -1;
}

// Allocate memref (as before) if no input buffer can be reused.
// Default VL=0 is used for non SIMD allocation
Value allocOrReuse(MemRefBuilder &create, Operation *op,
    ValueRange generatedOperands, MemRefType outputMemRefType, DimsExprRef dims,
    int64_t alignment, int64_t VL = 0);

Value allocOrReuse(MemRefBuilder &create, Operation *op,
    ValueRange generatedOperands, MemRefType outputMemRefType, DimsExprRef dims,
    int64_t alignment, int64_t VL) {

  int indexToReuse = -1;
  // By default, enableKrnlBufferReuse is false. Simply allocate a memref.
  if (enableKrnlBufferReuse) {
    // Be aware to use the op->getOperands() to check the number of uses.
    // After buffer reuse, the number of uses of the transformed Value,
    // generatedOperands, will increase.
    indexToReuse = whichBufferToReuse(op->getOperands(), outputMemRefType);
  }

  if (indexToReuse != -1) {
    int size = getSizeInBytes(outputMemRefType);
    LLVM_DEBUG({
      llvm::dbgs() << "  malloc_size " << size << "\n";
      op->dump();
    });
    return generatedOperands[indexToReuse];
  } else {
    if (VL == 0)
      return create.alignedAlloc(outputMemRefType, dims, alignment);
    else
      return create.alignedAllocWithSimdPadding(
          outputMemRefType, dims, VL, alignment);
  }
}

//===----------------------------------------------------------------------===//
// SIMD code gen for kernels where data can be fully flattened.
//===----------------------------------------------------------------------===//

using MDBuilder = MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder,
    MemRefBuilder, VectorBuilder>;

//===----------------------------------------------------------------------===//
// SIMD code gen for kernels where data can be partially or fully flattened.
//===----------------------------------------------------------------------===//

template <typename OP_TYPE>
static LogicalResult getPartiallyFlattenedSimdCode(
    ConversionPatternRewriter &rewriter, MDBuilder &create,
    ONNXBroadcastOpShapeHelper *shapeHelper, Operation *op,
    MemRefType outputMemRefType, ValueRange operands, int64_t alignment,
    int64_t VL, int64_t collapsedInnermostLoops, bool ruledOutBroadcast,
    bool isUnaryOp, bool simdOnly, bool enableParallel) {
  Type outputElementType = outputMemRefType.getElementType();
  unsigned numArgs = op->getNumOperands();
  LLVM_DEBUG(llvm::dbgs() << "  partial SIMD code for elementwise op "
                          << op->getName() << " flattening "
                          << collapsedInnermostLoops << " inner dims\n");
  // If fully collapse the loop, then we can allocate more data and we don't
  // care if we compute a few more values... set simdOnly to true then
  // regardless of whether the dims allow us to do so or not.
  if (collapsedInnermostLoops == (int64_t)outputMemRefType.getRank()) {
    LLVM_DEBUG(llvm::dbgs() << "  fully flattened, set simdOnly to true\n");
    simdOnly = true;
  }
  // Generate SIMD code of VL elements per vector.
  IndexExprScope allocScope(create.vec, shapeHelper->getScope());
  DimsExpr outputDims;
  getIndexExprList<SymbolIndexExpr>(shapeHelper->getOutputDims(), outputDims);
  // Reuse the buffer from the input, or Alloc memory with padding for SIMD.
  // For the moment, its ok to go here; if we truly have partial flattening of
  // the simd code, then we only do it with static memref size that are
  // multiples of VL * unrollVL, so there should be no padding anyway. This
  // will change if we do partial flattening with non-multiple of VL *
  // unrollVL.
  Value alloc = allocOrReuse(
      create.mem, op, operands, outputMemRefType, outputDims, alignment, VL);
  // Create flat inputs in the last innerDinNum dims.
  llvm::SmallVector<Value, 4> flatOperands;
  for (Value oper : operands) {
    if (isNoneValue(oper) || hasOneElement(oper)) {
      // If its a none / scalar, it is not meant to be flattened.
      flatOperands.emplace_back(oper);
      continue;
    }
    DimsExpr operDims, flattenOperDims;
    create.krnlIE.getShapeAsSymbols(oper, operDims);
    // Because we fully fuse 1x1x128xf32 and 128xf32, the
    // collapsedInnermostLoops may be higher than the rank of this input.
    // Adjust collapsedInnermostLoops accordingly for the flatten below.
    int64_t currRank = operDims.size();
    int64_t currCollapsedNum = std::min(collapsedInnermostLoops, currRank);
    Value flatOper = create.mem.reshapeToFlatInnermost(
        oper, operDims, flattenOperDims, currCollapsedNum);
    flatOperands.emplace_back(flatOper);
  }

  // Create flat output.
  int64_t rank = outputDims.size() - collapsedInnermostLoops + 1;
  LLVM_DEBUG(
      llvm::dbgs() << "SIMD partial flatten with loop rank " << rank << "\n");
  SmallVector<IndexExpr, 4> flattenedOutputDims;
  Value flatAlloc = create.mem.reshapeToFlatInnermost(
      alloc, outputDims, flattenedOutputDims, collapsedInnermostLoops);

  // Create loop iteration, rank-1, all but the flattened innermost [simd]
  // loop.
  int64_t outerLoopRank = rank - 1;
  ValueRange loopDef = create.krnl.defineLoops(outerLoopRank);
  // Iterate only over the blocks.
  IndexExpr zero = LitIE(0);
  DimsExpr lbs(outerLoopRank, zero);
  DimsExpr ubs = flattenedOutputDims;
  IndexExpr simdUb = ubs.pop_back_val(); // Remove flattened ub.
  bool useParallelInSimdLoop = false;
  if (enableParallel) {
    if (outerLoopRank > 1) {
      // Outer loop parallelism.
      tryCreateKrnlParallel(create.krnl, op,
          "outer-loop of elementwise simd partially flattened", loopDef, lbs,
          ubs);
    } else {
      // SIMD loop parallelism.
      if (tryCreateKrnlParallel(create.krnl, op,
              "inner-loop of elementwise simd partially flattened", loopDef,
              {zero}, {simdUb}, 0, 1, {}, VL * 32,
              /*createKrnlParallel=*/false) != -1)
        useParallelInSimdLoop = true;
    }
  }
  create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
      [&](const KrnlBuilder &ck, ValueRange loopInd) {
        MultiDialectBuilder<KrnlBuilder> create(ck);
        // LoopInd has the current indices for all but the innermost dim.
        // Since we expect here the entire innermost loop iteration in one go,
        // the innermost loop starts at zero. Add here to the list of Dim
        // symbols.
        SmallVector<IndexExpr, 4> outputAccessExprs = DimListIE(loopInd);
        outputAccessExprs.emplace_back(zero);

        // Have to produce the list of input values and their access
        // functions.
        llvm::SmallVector<Value, 4> inputs = flatOperands;
        llvm::SmallVector<DimsExpr, 4> inputAFs;
        for (int64_t i = 0; i < (int64_t)inputs.size(); ++i) {
          Value input = inputs[i];
          // Define the access function for each of the inputs.
          DimsExpr inputAF;
          // Check if we have a none value.
          if (isNoneValue(input)) {
            // Have one, pass the none value with empty AF.
            inputAFs.emplace_back(inputAF);
            continue;
          }
          // We have a memref, analyze which kind.
          MemRefType inputType = mlir::dyn_cast<MemRefType>(input.getType());
          assert(inputType && "expected memref");
          // Check if we have a scalar.
          if (hasOneElement(input)) {
            // Not flattened, with only 1 dims, just put zeros as needed.
            int64_t inputRank = inputType.getRank();
            for (int r = 0; r < inputRank; ++r)
              inputAF.emplace_back(zero);
            inputAFs.emplace_back(inputAF);
            continue;
          }
          // We have a regular access.
          LogicalResult res =
              shapeHelper->getAccessExprs(input, i, outputAccessExprs, inputAF,
                  /*flattened*/ true, ruledOutBroadcast);
          assert(succeeded(res) && "Could not compute access indices");
          inputAFs.emplace_back(inputAF);
        }
        // Produce the list of outputs and output AFs
        Value output = flatAlloc;
        DimsExpr outputAF = outputAccessExprs;

        create.krnl.simdIterateIE(zero, SymIE(simdUb), VL, simdOnly,
            useParallelInSimdLoop, inputs, inputAFs, {output}, {outputAF},
            {[&](const KrnlBuilder &kb, ArrayRef<Value> inputVals, int64_t VL) {
              MultiDialectBuilder<MathBuilder> create(kb);
              Type currElementType = outputElementType;
              if (VL > 1)
                currElementType = VectorType::get({VL}, outputElementType);
              Value res;
              if (isUnaryOp) {
                // For unary op, we through all operands at once as the other
                // ones are scalars / none values.
                res = emitScalarOpFor<OP_TYPE>(
                    rewriter, create.getLoc(), op, currElementType, inputVals);
              } else {
                // For non-unary ops, each op is a flattened array that need
                // to be processed; process the two first ones, and then
                // "accumulate" one value at a time. Use the first operand as
                // temporary result.
                Value accumulated = inputVals[0];
                // Iterate over the remaining operands.
                for (unsigned i = 1; i < numArgs; ++i) {
                  Value next = inputVals[i];
                  // Fold.
                  accumulated =
                      emitScalarOpFor<OP_TYPE>(rewriter, create.getLoc(), op,
                          currElementType, {accumulated, next});
                }
                // Postprocessing (dummy op if none).
                res = emitPostProcessingFor<OP_TYPE>(rewriter, create.getLoc(),
                    op, currElementType, accumulated);
              }
              return res;
            }}); // SIMD kernel.
      });        // Outer loops.

  rewriter.replaceOp(op, alloc);
  return success();
}

//===----------------------------------------------------------------------===//
// Utilities for Op fusion at lowering
//===----------------------------------------------------------------------===//

// Function pointer type for the emitScalarOpFor<T> of elementwise Ops.
typedef Value (*EmitScalarFunc)(ConversionPatternRewriter &rewriter,
    Location loc, mlir::Operation *op, Type elementType,
    mlir::ArrayRef<Value> scalarOperands);

// Utility class for Op fusion.
// Start from the root op, which is being lowered as an Elementwise Op.
// Following the def-use chain from the root op, a line of fusible ops are
// identified. The fusible Ops have to be elementwise Op, and satisfy shape
// and dependence requirement.
// The scalar operations of these fusible elementwise ops are fused into the
// loop nest generated for the root Op.
// Finally the last op is replaced with the allocated memref and the other ops
// are deleted.
// ToFix: fusion for a graph structure, not just line, could be added in
// future.
class OpFusionHelper {
public:
  // Constructor
  OpFusionHelper(mlir::ConversionPatternRewriter &rewriter,
      mlir::Operation *rootOp, DimAnalysis *dimAnalysis)
      : rootOp(rootOp), rewriter(rewriter), dimAnalysis(dimAnalysis),
        fusibleOps(), fuseEmitFuctions() {}

  // Fusion should not break any control dependence
  static bool isControlFlowValidForFusion(Operation *useOp, Operation *defOp);

  // Check whether the inputs of the useOp are valid for useOp to be fused
  // with the defOp. The defOp defines one of useOp's inputs.
  static bool areInputsValidForFusion(
      Operation *useOp, Operation *defOp, DimAnalysis *dimAnalysis);

  // Check whether the op is fusible along the use-def chain from the defOp.
  // If true, record the op and its scalar op.
  bool checkFusibleOp(Operation *useOp, Operation *defOp,
      SmallVector<Operation *, 2> &fusibleOps,
      SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions);

  // Find the fusible op chain from the root op
  void findFusibleOps();

  bool isFusibleListEmpty() { return fusibleOps.size() == 0; }

  // The final output type after fusion.
  // The element type of an elementwise op may be different from its inputs.
  // For example, comparison ops, and cast Op.
  MemRefType getOutputType(MemRefType outputType);

  // Generate the code for the ops to be fused
  // procedureResult is the scalar value from producer
  // alloc is used to get the tensor for the producer, which is required by
  // by the shape helper.
  Value emitFuseOps(
      Value producerResult, const Value alloc, ValueRange loopInd = {});

  void replaceOrEraseONNXOps(Value alloc);

private:
  mlir::Operation *rootOp;
  mlir::ConversionPatternRewriter &rewriter;
  DimAnalysis *dimAnalysis;
  llvm::SmallVector<mlir::Operation *, 2> fusibleOps;
  llvm::SmallVector<EmitScalarFunc, 2> fuseEmitFuctions;
}; // End of OpFusionHelper Declaration

// Check a node with type T is fusible or not.
// If true, record the op to data structure
template <typename T>
bool enqueueFusibleOpImpl(Operation *useOp, Operation *defOp,
    SmallVector<Operation *, 2> &fusibleOps,
    SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions,
    DimAnalysis *dimAnalysis) {
  if (isa<T>(useOp)) {
    if (OpFusionHelper::isControlFlowValidForFusion(useOp, defOp) &&
        OpFusionHelper::areInputsValidForFusion(useOp, defOp, dimAnalysis)) {
      fusibleOps.emplace_back(useOp);
      fuseEmitFunctions.emplace_back(emitScalarOpFor<T>);
      return true;
    }
  }
  return false;
}

// Variadic template to iterate all the Elementwise Ops
template <typename T = void, class... Ts>
bool enqueueFusibleOp(Operation *useOp, Operation *defOp,
    SmallVector<Operation *, 2> &fusibleOps,
    SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions,
    DimAnalysis *dimAnalysis);

template <typename T, class... Ts>
bool enqueueFusibleOp(Operation *useOp, Operation *defOp,
    SmallVector<Operation *, 2> &fusibleOps,
    SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions,
    DimAnalysis *dimAnalysis) {
  if (enqueueFusibleOpImpl<T>(
          useOp, defOp, fusibleOps, fuseEmitFunctions, dimAnalysis))
    return true;
  return enqueueFusibleOp<Ts...>(
      useOp, defOp, fusibleOps, fuseEmitFunctions, dimAnalysis);
}

template <>
bool enqueueFusibleOp(Operation *useOp, Operation *defOp,
    SmallVector<Operation *, 2> &fusibleOps,
    SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions,
    DimAnalysis *dimAnalysis) {
  return false;
}

// Give a list of Elementwise ONNX Op for the template enqueueFusibleOp
// to iterate.
bool OpFusionHelper::checkFusibleOp(Operation *useOp, Operation *defOp,
    SmallVector<Operation *, 2> &fusibleOps,
    SmallVector<EmitScalarFunc, 2> &fuseEmitFunctions) {

  // Notice: Though ClipOp is classified as unary element op in this file,
  // ClipOp requires one required input and two optional input

  return enqueueFusibleOp<
      // Unary Op
      mlir::ONNXAbsOp, mlir::ONNXAtanOp, mlir::ONNXBinarizerOp,
      mlir::ONNXCastOp, mlir::ONNXCeilOp, mlir::ONNXCosOp, mlir::ONNXCoshOp,
      mlir::ONNXDequantizeLinearOp, mlir::ONNXCeluOp, mlir::ONNXEluOp,
      mlir::ONNXErfOp, mlir::ONNXAcosOp, mlir::ONNXAcoshOp, mlir::ONNXAsinOp,
      mlir::ONNXAsinhOp, mlir::ONNXAtanhOp, mlir::ONNXExpOp, mlir::ONNXFloorOp,
      mlir::ONNXGeluOp, mlir::ONNXBitwiseNotOp, mlir::ONNXHardSigmoidOp,
      mlir::ONNXHardSwishOp, mlir::ONNXIsInfOp, mlir::ONNXIsNaNOp,
      mlir::ONNXLeakyReluOp, mlir::ONNXLogOp, mlir::ONNXMishOp, mlir::ONNXNegOp,
      mlir::ONNXNotOp, mlir::ONNXReciprocalOp, mlir::ONNXReluOp,
      mlir::ONNXRoundOp, mlir::ONNXSeluOp, mlir::ONNXShrinkOp,
      mlir::ONNXSigmoidOp, mlir::ONNXSignOp, mlir::ONNXSinOp, mlir::ONNXSinhOp,
      mlir::ONNXSoftplusOp, mlir::ONNXSoftsignOp, mlir::ONNXSqrtOp,
      mlir::ONNXTanOp, mlir::ONNXTanhOp, mlir::ONNXThresholdedReluOp,
      // Binary Op
      mlir::ONNXBitShiftOp, mlir::ONNXEqualOp, mlir::ONNXGreaterOp,
      mlir::ONNXGreaterOrEqualOp, mlir::ONNXLessOp, mlir::ONNXLessOrEqualOp,
      mlir::ONNXModOp, mlir::ONNXPowOp,
      // Variadic Op
      mlir::ONNXAddOp, mlir::ONNXAndOp, mlir::ONNXDivOp, mlir::ONNXMaxOp,
      mlir::ONNXMeanOp, mlir::ONNXMinOp, mlir::ONNXMulOp, mlir::ONNXOrOp,
      mlir::ONNXSubOp, mlir::ONNXSumOp, mlir::ONNXXorOp>(
      useOp, defOp, fusibleOps, fuseEmitFunctions, dimAnalysis);
}

// Only operations are in the same block are allowed to fuse.
// ToFix: This requirement may be too conservative.
bool OpFusionHelper::isControlFlowValidForFusion(
    Operation *useOp, Operation *defOp) {
  if (useOp->getBlock() != defOp->getBlock())
    return false;
  return true;
}

// Check whether the inputs of the useOp are valid for useOp to be fused
// with the defOp. The defOp defines one of useOp's inputs.
// If fused, the two ops will use the same loop nests and the iteration space
// be the same. Fusion is not allowed along the d-u chain that is broadcasted
// in the useOp.
// clang-format off
// Some discussion can be found at https://github.com/onnx/onnx-mlir/issues/2199
// Example of shape valid for fusion:
// %1 = "onnx.Sqrt"(%0) : (tensor<16x24xf32>) -> tensor<16x24xf32>
// %2 = "onnx.Add"(%1, %3) : (tensor<16x24xf32>, tensor<24xf32>) -> tensor<16x24xf32>
// Example of shape not valid for fusion
// %1 = "onnx.Sqrt"(%0) : (tensor<24xf32>) -> tensor<24xf32>
// %2 = "onnx.Add"(%1, %3) : (tensor<24xf32>, tensor<16x24xf32>) -> tensor<16x24xf32>
// clang-format on

// Since the fusion will move the read of the inputs of useOp
// to the root Op, the other inputs of the useOp (not the one from the defOp)
// should be located before the root op to make the SSA correct.
// clang-format off
// Example:
// %3 = ...
// %1 = "onnx.Sqrt"(%0) : (tensor<16x24xf32>) -> tensor<16x24xf32>
// %2 = "onnx.Add"(%1, %3) : (tensor<16x24xf32>, tensor<24xf32>) -> tensor<16x24xf32>
// Example of invalid fusion without code motion:
// %1 = "onnx.Sqrt"(%0) : (tensor<16x24xf32>) -> tensor<16x24xf32>
// %3 = ...
// %2 = "onnx.Add"(%1, %3) : (tensor<16x24xf32>, tensor<24xf32>) -> tensor<16x24xf32>
// clang-format on
// In this implementation, no data dependence analysis and
// code motion for fusion is implemented yet. The only other inputs allowed
// are block argument and constant to guarantee they are before the root op.
// It is assumed the canonicalization has hoisted all constant to the
// beginning of the function by fold function.
bool OpFusionHelper::areInputsValidForFusion(
    Operation *useOp, Operation *defOp, DimAnalysis *dimAnalysis) {
  // Do not fuse ops with scalar tensors.
  if (llvm::all_of(
          useOp->getOperands(), [](Value v) { return isScalarTensor(v); }))
    return false;

  // Elementwise unary operation is always fusible
  if (useOp->getOperands().size() == 1)
    return true;

  Type defOutputType = defOp->getResultTypes()[0];
  Type useOutputType = useOp->getResultTypes()[0];
  ArrayRef<int64_t> defShape = getShape(defOutputType);
  ArrayRef<int64_t> useShape = getShape(useOutputType);
  if (defShape != useShape) {
    return false;
  }

  // Check the inputs in the useOp
  for (size_t i = 0; i < useOp->getOperands().size(); i++) {
    // Only input from block argument and constant is allowed,
    // if the input does not come from the defining Op
    if (!isa<BlockArgument>(useOp->getOperand(i))) {
      Operation *input = useOp->getOperand(i).getDefiningOp();
      if (input == defOp)
        continue;
      if (!isa<ONNXConstantOp>(useOp->getOperand(i).getDefiningOp())) {
        return false;
      }
    }
  }

  // Check whether this shape of the defOp is the same as the shape of
  // the output of use op. If true, the iteration space from the defOp is
  // sufficient for the element-wise operation for the useOp, even if
  // MDBroadcast occurs in the useOp.
  // Otherwise, the loop nest should be defined according to the tensor with
  // larger space.

  // First check the rank
  if (getRank(defOutputType) != getRank(useOutputType))
    return false;

  if (dimAnalysis) {
    if (!dimAnalysis->sameShape(defOp->getResult(0), useOp->getResult(0)))
      return false;
  } else {
    // If there is no dimAnalysis, check the simplest case.
    // Static and the same shape
    if (!hasStaticShape(useOutputType))
      return false;

    ArrayRef<int64_t> inputShape = getShape(useOutputType);
    if (inputShape != defShape)
      return false;
  }

  return true;
}

// The seach for fusible ops starts from the rootOp, an elementwise operation.
// A successor op (user) is fusible if it is the only user, it is in the
// fusible elementwise op list, and its inputs are valid for fusion.
void OpFusionHelper::findFusibleOps() {
  // Direct return if fusion is disabled
  if (disableKrnlOpFusion)
    return;
  Operation *defOp = rootOp;
  while (defOp->hasOneUse()) {
    // the possible ONNX Ops.
    Operation *useOp = *defOp->getUsers().begin();
    if (!checkFusibleOp(useOp, defOp, fusibleOps, fuseEmitFuctions))
      break;

    // Current useOp becomes the defOp for the next Op
    defOp = useOp;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "op fusion: fusible ops " << fusibleOps.size() << "\n";
    rootOp->dump();
    llvm::dbgs() << "begin fusible op list\n";
    for (auto op : fusibleOps)
      op->dump();
    llvm::dbgs() << "end fusible op list\n";
  });
}

// After fusion, the only store is for the last Op.
// Therefore, the allocation should be the output of the last Op
MemRefType OpFusionHelper::getOutputType(MemRefType outputType) {
  if (!isFusibleListEmpty()) {
    Operation *lastOp = fusibleOps[fusibleOps.size() - 1];
    return MemRefType::get(outputType.getShape(),
        getElementType(lastOp->getResults()[0].getType()));
  }
  return outputType;
}

// Emit fusion Ops
Value OpFusionHelper::emitFuseOps(
    Value defOpResult, const Value alloc, ValueRange loopInd) {
  if (isFusibleListEmpty())
    return defOpResult;

  // Handle the fused Ops
  Operation *defOp = rootOp;
  for (size_t i = 0; i < fusibleOps.size(); i++) {
    Operation *useOp = fusibleOps[i];
    auto emitScalar = fuseEmitFuctions[i];
    // ToFix: use the ONNX location(ONNXLoc) of each Op.
    // The current obstacle is that no easy way to know the type of Op,
    // which is needed by ONNXLoc<T>(op).
    Location loc = useOp->getLoc();
    MDBuilder create(rewriter, loc);
    Type currentElementType = getElementType(useOp->getResults()[0].getType());

    // useOperands is used for ShapeHelper and load op.
    // getRemappedValue is needed for load op.
    SmallVector<Value, 4> useOperands;
    for (auto oper : useOp->getOperands()) {
      if (oper.getDefiningOp() != defOp) {
        useOperands.emplace_back(rewriter.getRemappedValue(oper));
        Operation *constOp = oper.getDefiningOp<ONNXConstantOp>();
        if (constOp && defOp->isBeforeInBlock(constOp)) {
          // Move the constant op to be before the root op so that the IR is
          // valid.
          constOp->moveBefore(defOp);
        }
      } else {
        // Due to the op fusion, we will not generate a tensor for the current
        // oper, but only the scalar result from defOp.
        // This scalar value cannot be used to initialize ShapeHelper.
        // Instead, alloc is used because it has the same shape as the oper.
        // This is one of the prerequisites for fusion.
        // However, they may have different element type for some ops, such as
        // comparison and cast.
        // ONNXBroadcastOpShapeHelper cares only the shape of the operands,
        // not the element type.
        // In a previous implementation, the original output of defOp is used
        // with 'alloc = defOp->getResult(0)' at the end of the loop.
        // But ONNXBroadcastOpShapeHelper.computeShape() unexpectedly used
        // this parameter to generate some code (memref.dim) that is not
        // really needed. Due to this live user, the original op can not be
        // erased. This error occurred when there were more than one op with
        // dynamic dim to be fused in the previous implementation. Therefore,
        // alloc is used for all the fused op.
        useOperands.emplace_back(alloc);
      }
    }
    // Use shape helper to generate load index
    ONNXBroadcastOpShapeHelper shapeHelper(
        useOp, useOperands, &create.krnlIE, nullptr, false);
    shapeHelper.computeShapeAndAssertOnFailure();

    // Prepare Values for EmitScalarOpFor<T>
    SmallVector<Value, 2> inputValues;
    for (size_t i = 0; i < useOp->getOperands().size(); i++) {
      Value inputValue = useOp->getOperand(i);
      Operation *inputOp = inputValue.getDefiningOp();
      if (inputOp == defOp) {
        inputValues.emplace_back(defOpResult);
      } else {
        IndexExprScope innerScope(create.krnl, shapeHelper.getScope());
        SmallVector<IndexExpr, 4> outputAccessExprs;
        getIndexExprList<DimIndexExpr>(loopInd, outputAccessExprs);
        SmallVector<IndexExpr, 4> loadAccessExprs;
        LogicalResult res = shapeHelper.getAccessExprs(
            inputValue, i, outputAccessExprs, loadAccessExprs, true);
        assert(succeeded(res) && "Could not compute access indices");
        Value load = create.krnl.loadIE(useOperands[i], loadAccessExprs);
        inputValues.emplace_back(load);
      }
    }
    defOpResult =
        emitScalar(rewriter, loc, useOp, currentElementType, inputValues);
    defOp = useOp;
  }
  return defOpResult;
}

// Replace the last Op with allocated memref and erase the other Ops.
// When the fusible list is empty, the starting Op is the last.
void OpFusionHelper::replaceOrEraseONNXOps(Value alloc) {
  if (isFusibleListEmpty()) {
    rewriter.replaceOp(rootOp, alloc);
    return;
  }

  Operation *previous = rootOp;
  for (Operation *fusedOp : fusibleOps) {
    rewriter.eraseOp(previous);
    previous = fusedOp;
  }
  rewriter.replaceOp(previous, alloc);
}

//===----------------------------------------------------------------------===//
// End of Op Fusion Support
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Element-wise unary ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//

template <typename ElementwiseUnaryOp>
struct ONNXElementwiseUnaryOpLowering
    : public OpConversionPattern<ElementwiseUnaryOp> {
  using OpAdaptor = typename ElementwiseUnaryOp::Adaptor;
  DimAnalysis *dimAnalysis;
  bool enableSIMD = false;
  bool enableParallel = false;

  ONNXElementwiseUnaryOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      DimAnalysis *dimAnalysis, bool enableSIMD, bool enableParallel)
      : OpConversionPattern<ElementwiseUnaryOp>(typeConverter, ctx),
        dimAnalysis(dimAnalysis), enableSIMD(enableSIMD) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ElementwiseUnaryOp::getOperationName());
  }

  LogicalResult matchAndRewrite(ElementwiseUnaryOp elmsOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = elmsOp.getOperation();
    ValueRange operands = adaptor.getOperands();

    Location loc = ONNXLoc<ElementwiseUnaryOp>(op);
    Value X = operands[0];

    // If type is scalar or vector, there is no need to allocate a buffer.
    // Just call scalar computation and return the result. This is efficient
    // when elementwise ops are used as activations for ops like LSTM/GRU/RNN.
    if (!mlir::isa<TensorType>(X.getType()) &&
        !mlir::isa<MemRefType>(X.getType())) {
      SmallVector<Value> args;
      args.emplace_back(X);
      // Load the remaining (scalar) values.
      for (uint64_t i = 1; i < operands.size(); i++) {
        if (isNoneValue(operands[i])) {
          args.emplace_back(operands[i]);
          continue;
        }
        assert(!mlir::isa<TensorType>(operands[i].getType()) &&
               !mlir::isa<MemRefType>(operands[i].getType()) &&
               "unary expected scalar additional values");
        args.emplace_back(operands[i]);
      }
      Value res = emitScalarOpFor<ElementwiseUnaryOp>(
          rewriter, loc, op, X.getType(), args);
      rewriter.replaceOp(op, res);
      return success();
    }

    // Convert the output type to MemRefType.
    Type outputTensorType = elmsOp.getResult().getType();
    Type convertedType = this->typeConverter->convertType(outputTensorType);
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    int64_t outputRank = outputMemRefType.getRank();
    Type outputElementType = outputMemRefType.getElementType();

    // In unary, we don't have any broadcast, and thus our target is to fully
    // collapse the loop to a 1D loop.
    int64_t collapsedInnermostLoops = outputRank;

    // Shape helper.
    MDBuilder create(rewriter, loc);
    ONNXUnaryOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    LLVM_DEBUG({
      llvm::dbgs() << "Look at unary elementwise op: " << op->getName() << "\n";
      op->dump();
    });
    bool isScalar = hasAllScalarValues(operands);
    // SIMD is enabled for this operation, test if desired and feasible
    if (enableSIMD && !isScalar && !hasNonIdentityLayout(operands)) {
      int64_t simdLoopStaticTripCount;
      bool simdOnly, canOverCompute = true;
      GenOpMix mix = getGenOpMix<ElementwiseUnaryOp>(outputElementType, op);
      int64_t totVL = computeSuitableSimdUnrollFactor(outputMemRefType,
          collapsedInnermostLoops, mix, canOverCompute, simdLoopStaticTripCount,
          simdOnly);
      if (totVL > 1) {
        onnxToKrnlSimdReport(op, /*successful*/ true, totVL,
            simdLoopStaticTripCount, "unary fully flattened");
        return getPartiallyFlattenedSimdCode<ElementwiseUnaryOp>(rewriter,
            create, &shapeHelper, op, outputMemRefType, operands, alignment,
            totVL, collapsedInnermostLoops, /*ruleOutBroadcast*/ true,
            /*unary*/ true, simdOnly, enableParallel);
      }
      onnxToKrnlSimdReport(op, /*successful*/ false, 0, simdLoopStaticTripCount,
          "no simd in unary because could not find beneficial VL");

    } else {
      onnxToKrnlSimdReport(op, /*successful*/ false, 0, 0,
          "no simd in unary because scalar/layouts");
    }
    LLVM_DEBUG(llvm::dbgs() << "  scalar execution\n");

    // Try to fuse the unary elementwise consumers
    OpFusionHelper opFusionHelper(rewriter, op, dimAnalysis);
    opFusionHelper.findFusibleOps();
    outputMemRefType = opFusionHelper.getOutputType(outputMemRefType);

    // Insert an allocation for the result of this operation.
    Value alloc = allocOrReuse(create.mem, op, operands, outputMemRefType,
        shapeHelper.getOutputDims(), alignment);
    ;

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!isScalar) {
      ValueRange loopDef = create.krnl.defineLoops(outputRank);
      SmallVector<IndexExpr, 4> lbs(outputRank, LitIE(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(X, ubs);
      if (enableParallel)
        tryCreateKrnlParallel(create.krnl, op, "elementwise unary not simdized",
            loopDef, lbs, ubs);
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
            SmallVector<Value> args;
            Value loadedVal = createKrnl.load(X, loopInd);
            args.emplace_back(loadedVal);
            // Load the remaining (scalar) values.
            for (uint64_t i = 1; i < operands.size(); i++) {
              if (isNoneValue(operands[i])) {
                args.emplace_back(operands[i]);
                continue;
              }
              assert(isScalarValue(operands[i]) &&
                     "unary expected scalar additional values");
              Value loadedVal = create.krnl.load(operands[i]);
              args.emplace_back(loadedVal);
            }
            auto loweredOpResult = emitScalarOpFor<ElementwiseUnaryOp>(
                rewriter, loc, op, outputElementType, args);
            loweredOpResult =
                opFusionHelper.emitFuseOps(loweredOpResult, alloc, loopInd);
            // Store result in the resulting array.
            createKrnl.store(loweredOpResult, alloc, loopInd);
          });
    } else {
      Value loadedVal = create.krnl.load(X);
      SmallVector<Value> args;
      args.emplace_back(loadedVal);
      // Load the remaining (scalar) values.
      for (uint64_t i = 1; i < operands.size(); i++) {
        if (isNoneValue(operands[i])) {
          args.emplace_back(operands[i]);
          continue;
        }
        assert(isScalarValue(operands[i]) &&
               "unary expected scalar additional values");
        Value loadedVal = create.krnl.load(operands[i]);
        args.emplace_back(loadedVal);
      }
      auto loweredOpResult = emitScalarOpFor<ElementwiseUnaryOp>(
          rewriter, loc, op, outputElementType, args);
      loweredOpResult = opFusionHelper.emitFuseOps(loweredOpResult, alloc);
      // Store result in the resulting array.
      create.krnl.store(loweredOpResult, alloc);
    }

    // Replace the last Op with alloc and delete the other Ops
    opFusionHelper.replaceOrEraseONNXOps(alloc);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Element-wise binary ops lowering to Krnl dialect.
// This template can be used for binary ops that return a result whose type is
// different from the input type.
//===----------------------------------------------------------------------===//

template <typename ElementwiseBinaryOp>
struct ONNXElementwiseBinaryOpLowering
    : public OpConversionPattern<ElementwiseBinaryOp> {
  using OpAdaptor = typename ElementwiseBinaryOp::Adaptor;
  DimAnalysis *dimAnalysis;
  bool enableSIMD = false;
  bool isUniBroadcasting = false;
  bool enableParallel = false;

  ONNXElementwiseBinaryOpLowering(TypeConverter &typeConverter,
      MLIRContext *ctx, DimAnalysis *dimAnalysis, bool enableSIMD,
      bool isUniBroadcasting = false, bool enableParallel = false)
      : OpConversionPattern<ElementwiseBinaryOp>(typeConverter, ctx),
        dimAnalysis(dimAnalysis), enableSIMD(enableSIMD),
        isUniBroadcasting(isUniBroadcasting) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ElementwiseBinaryOp::getOperationName());
  }

  LogicalResult matchAndRewrite(ElementwiseBinaryOp elmsOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = elmsOp.getOperation();
    ValueRange operands = adaptor.getOperands();
    Location loc = ONNXLoc<ElementwiseBinaryOp>(op);

    // Convert the output type to MemRefType.
    Type outputTensorType = elmsOp.getResult().getType();
    Type convertedType = this->typeConverter->convertType(outputTensorType);
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    Type outputElementType = outputMemRefType.getElementType();
    int64_t outputRank = outputMemRefType.getRank();

    // Shape helper.
    MDBuilder create(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(
        op, operands, &create.krnlIE, nullptr, isUniBroadcasting);
    shapeHelper.computeShapeAndAssertOnFailure();

    LLVM_DEBUG({
      llvm::dbgs() << "Look at binary elementwise op: " << op->getName()
                   << "\n";
      op->dump();
    });

    int64_t collapsedInnermostLoops, collapsedLiteralSize;
    IndexExpr collapsedDynamicSize;
    bool isScalar = hasAllScalarValues(operands);
    bool hasNoBroadcast = shapeHelper.hasNoBroadcast(dimAnalysis);
    bool hasManageableBroadcast =
        shapeHelper.hasManageableBroadcastForInnerDims(collapsedInnermostLoops,
            collapsedLiteralSize, collapsedDynamicSize, dimAnalysis);
    LLVM_DEBUG({
      if (hasManageableBroadcast)
        llvm::dbgs() << "  simd with manageable broadcast: inner dims "
                     << collapsedInnermostLoops << " of lit size "
                     << collapsedLiteralSize << "\n";
      else
        llvm::dbgs() << "  simd not possible, unmanageable broadcast\n";
      if (hasNoBroadcast)
        llvm::dbgs() << "  simd possible: hasNoBroadcast\n";
    });

    // SIMD is enabled for this operation, test if desired and feasible
    if (enableSIMD && !isScalar && hasManageableBroadcast &&
        !hasNonIdentityLayout(operands)) {
      int64_t simdLoopStaticTripCount;
      bool simdOnly, canOverCompute = collapsedInnermostLoops == outputRank;
      GenOpMix mix = getGenOpMix<ElementwiseBinaryOp>(outputElementType, op);
      int64_t totVL = computeSuitableSimdUnrollFactor(outputMemRefType,
          collapsedInnermostLoops, mix, canOverCompute, simdLoopStaticTripCount,
          simdOnly);
      if (totVL > 1) {
        if (collapsedInnermostLoops == outputRank)
          onnxToKrnlSimdReport(op, /*successful*/ true, totVL,
              simdLoopStaticTripCount, "binary fully flattened");
        else
          onnxToKrnlSimdReport(op, /*successful*/ true, totVL,
              simdLoopStaticTripCount, "binary with manageable broadcast");
        return getPartiallyFlattenedSimdCode<ElementwiseBinaryOp>(rewriter,
            create, &shapeHelper, op, outputMemRefType, operands, alignment,
            totVL, collapsedInnermostLoops, hasNoBroadcast,
            /*unary*/ false, simdOnly, enableParallel);
      }
      onnxToKrnlSimdReport(op, /*successful*/ false, 0, simdLoopStaticTripCount,
          "no simd in binary because no beneficial VL");
    } else {
      onnxToKrnlSimdReport(op, /*successful*/ false, 0, 0,
          "no simd in binary because no manageable broadcast/layout ");
    }
    LLVM_DEBUG(llvm::dbgs() << "  scalar execution\n");

    // Try to fuse the unary elementwise consumers
    OpFusionHelper opFusionHelper(rewriter, op, dimAnalysis);
    opFusionHelper.findFusibleOps();
    outputMemRefType = opFusionHelper.getOutputType(outputMemRefType);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = allocOrReuse(create.mem, op, operands, outputMemRefType,
        shapeHelper.getOutputDims(), alignment);
    ;

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!isScalar) {
      ValueRange loopDef = create.krnl.defineLoops(outputRank);
      SmallVector<IndexExpr, 4> lbs(outputRank, LitIE(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(alloc, ubs);
      // TODO adjust in the future
      if (enableParallel)
        tryCreateKrnlParallel(create.krnl, op,
            "elementwise binary not simdized", loopDef, lbs, ubs);
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
            IndexExprScope innerScope(createKrnl, shapeHelper.getScope());
            SmallVector<IndexExpr, 4> outputAccessExprs;
            getIndexExprList<DimIndexExpr>(loopInd, outputAccessExprs);

            // Load the first value.
            SmallVector<IndexExpr, 4> lhsAccessExprs;
            LogicalResult res =
                shapeHelper.getAccessExprs(operands[0], 0, outputAccessExprs,
                    lhsAccessExprs, /*flattened dims*/ false, hasNoBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value lhs = createKrnl.loadIE(operands[0], lhsAccessExprs);

            // Load the second value.
            SmallVector<IndexExpr, 4> rhsAccessExprs;
            res = shapeHelper.getAccessExprs(operands[1], 1, outputAccessExprs,
                rhsAccessExprs, /*flattened dims*/ false, hasNoBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value rhs = createKrnl.loadIE(operands[1], rhsAccessExprs);

            // Apply the element-wise function.
            Value result = emitScalarOpFor<ElementwiseBinaryOp>(
                rewriter, loc, op, outputElementType, {lhs, rhs});

            result = opFusionHelper.emitFuseOps(result, alloc, loopInd);
            // Store result in the resulting array.
            createKrnl.store(result, alloc, loopInd);
          });
    } else {
      Value lhs = create.krnl.load(operands[0]);
      Value rhs = create.krnl.load(operands[1]);

      // Apply the element-wise function.
      Value result = emitScalarOpFor<ElementwiseBinaryOp>(
          rewriter, loc, op, outputElementType, {lhs, rhs});

      result = opFusionHelper.emitFuseOps(result, alloc);
      // Store result in the resulting array.
      create.krnl.store(result, alloc);
    }

    // Replace the last Op with alloc and delete the other Ops
    opFusionHelper.replaceOrEraseONNXOps(alloc);

    return success();
  }
}; // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// Element-wise variadic ops lowering to Krnl dialect.
//===----------------------------------------------------------------------===//

template <typename ElementwiseVariadicOp>
struct ONNXElementwiseVariadicOpLowering
    : public OpConversionPattern<ElementwiseVariadicOp> {
  using OpAdaptor = typename ElementwiseVariadicOp::Adaptor;
  DimAnalysis *dimAnalysis;
  bool enableSIMD = false;
  bool enableParallel = false;

  ONNXElementwiseVariadicOpLowering(TypeConverter &typeConverter,
      MLIRContext *ctx, DimAnalysis *dimAnalysis, bool enableSIMD,
      bool enableParallel)
      : OpConversionPattern<ElementwiseVariadicOp>(typeConverter, ctx),
        dimAnalysis(dimAnalysis), enableSIMD(enableSIMD) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ElementwiseVariadicOp::getOperationName());
  }

  LogicalResult matchAndRewrite(ElementwiseVariadicOp elmsOp, OpAdaptor adaptor,
      ConversionPatternRewriter &rewriter) const final {
    Operation *op = elmsOp.getOperation();
    Location loc = ONNXLoc<ElementwiseVariadicOp>(op);
    ValueRange operands = adaptor.getOperands();
    unsigned numArgs = elmsOp.getNumOperands();

    // Convert the output type to MemRefType.
    Type outputTensorType = elmsOp.getResult().getType();
    Type convertedType = this->typeConverter->convertType(outputTensorType);
    int64_t alignment =
        KrnlTypeConverter::getDefaultAllocAlignment(outputTensorType);
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    Type outputElementType = outputMemRefType.getElementType();
    int64_t outputRank = outputMemRefType.getRank();

    // Shape helper.
    MDBuilder create(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    LLVM_DEBUG({
      llvm::dbgs() << "Look at variadic elementwise op: " << op->getName()
                   << "\n";
      op->dump();
    });

    int64_t collapsedInnermostLoops, collapsedLiteralSize;
    IndexExpr collapsedDynamicSize;
    bool isScalar = hasAllScalarValues(operands);
    bool hasNoBroadcast = shapeHelper.hasNoBroadcast(dimAnalysis);
    bool hasManageableBroadcast =
        shapeHelper.hasManageableBroadcastForInnerDims(collapsedInnermostLoops,
            collapsedLiteralSize, collapsedDynamicSize, dimAnalysis);
    LLVM_DEBUG({
      if (hasManageableBroadcast)
        llvm::dbgs() << "  simd with manageable broadcast: inner dims "
                     << collapsedInnermostLoops << " of lit size "
                     << collapsedLiteralSize << "\n";
      else
        llvm::dbgs() << "  simd not possible, unmanageable broadcast\n";
      if (hasNoBroadcast)
        llvm::dbgs() << "  simd possible: hasNoBroadcast\n";
    });

    if (enableSIMD && !isScalar && hasManageableBroadcast &&
        !hasNonIdentityLayout(operands)) {
      // SIMD is enabled for this operation, test if desired and feasible
      int64_t simdLoopStaticTripCount;
      bool simdOnly, canOverCompute = collapsedInnermostLoops == outputRank;
      GenOpMix mix = getGenOpMix<ElementwiseVariadicOp>(outputElementType, op);
      int64_t totVL = computeSuitableSimdUnrollFactor(outputMemRefType,
          collapsedInnermostLoops, mix, canOverCompute, simdLoopStaticTripCount,
          simdOnly);
      if (totVL > 1) {
        if (collapsedInnermostLoops == outputRank)
          onnxToKrnlSimdReport(op, /*successful*/ true, totVL,
              simdLoopStaticTripCount, "variadic fully flattened");
        else
          onnxToKrnlSimdReport(op, /*successful*/ true, totVL,
              simdLoopStaticTripCount, "variadic with manageable broadcast");
        return getPartiallyFlattenedSimdCode<ElementwiseVariadicOp>(rewriter,
            create, &shapeHelper, op, outputMemRefType, operands, alignment,
            totVL, collapsedInnermostLoops, hasNoBroadcast,
            /*unary*/ false, simdOnly, enableParallel);
      }
      onnxToKrnlSimdReport(op, /*successful*/ false, 0, simdLoopStaticTripCount,
          "no simd in variadic because no beneficial VL");
    } else {
      onnxToKrnlSimdReport(op, /*successful*/ false, 0, 0,
          "no simd in variadic because no manageable broadcast/layout ");
    }
    LLVM_DEBUG(llvm::dbgs() << "  scalar execution\n");

    // Try to fuse the unary elementwise consumers
    OpFusionHelper opFusionHelper(rewriter, op, dimAnalysis);
    opFusionHelper.findFusibleOps();
    outputMemRefType = opFusionHelper.getOutputType(outputMemRefType);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc = allocOrReuse(create.mem, op, operands, outputMemRefType,
        shapeHelper.getOutputDims(), alignment);
    ;

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!isScalar) {
      ValueRange loopDef = create.krnl.defineLoops(outputRank);
      SmallVector<IndexExpr, 4> lbs(outputRank, LitIE(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(alloc, ubs);

      if (enableParallel)
        tryCreateKrnlParallel(create.krnl, op,
            "elementwise variable not simdized", loopDef, lbs, ubs);

      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
            IndexExprScope innerScope(createKrnl, shapeHelper.getScope());
            SmallVector<IndexExpr, 4> outputAccessExprs;
            getIndexExprList<DimIndexExpr>(loopInd, outputAccessExprs);

            // Fold over operands for each of their scalar values.
            // Obtain the first operand.
            SmallVector<IndexExpr, 4> oprdAccessExprs;
            LogicalResult res =
                shapeHelper.getAccessExprs(operands[0], 0, outputAccessExprs,
                    oprdAccessExprs, /*flattened dims*/ false, hasNoBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value accumulated = createKrnl.loadIE(operands[0], oprdAccessExprs);

            // Iterate over the remaining operands.
            for (unsigned i = 1; i < numArgs; ++i) {
              // Obtain the next operand.
              SmallVector<IndexExpr, 4> oprdAccessExprs;
              LogicalResult res = shapeHelper.getAccessExprs(operands[i], i,
                  outputAccessExprs, oprdAccessExprs,
                  /*flattened dims*/ false, hasNoBroadcast);
              assert(succeeded(res) && "Could not compute access indices");
              Value next = createKrnl.loadIE(operands[i], oprdAccessExprs);
              // Fold.
              accumulated = emitScalarOpFor<ElementwiseVariadicOp>(
                  rewriter, loc, op, outputElementType, {accumulated, next});
            }

            Value finalResult = emitPostProcessingFor<ElementwiseVariadicOp>(
                rewriter, loc, op, outputElementType, accumulated);
            finalResult =
                opFusionHelper.emitFuseOps(finalResult, alloc, loopInd);
            // Store result in the resulting array.
            createKrnl.storeIE(finalResult, alloc, outputAccessExprs);
          });
    } else {
      Value accumulated = create.krnl.load(operands[0]);

      // Iterate over the remaining operands.
      for (unsigned i = 1; i < numArgs; i++) {
        // Obtain the next operand.
        Value next = create.krnl.load(operands[i]);
        // Fold.
        accumulated = emitScalarOpFor<ElementwiseVariadicOp>(
            rewriter, loc, op, outputElementType, {accumulated, next});
      }
      Value finalResult = emitPostProcessingFor<ElementwiseVariadicOp>(
          rewriter, loc, op, outputElementType, accumulated);
      finalResult = opFusionHelper.emitFuseOps(finalResult, alloc);
      // Store result in the resulting array.
      create.krnl.store(finalResult, alloc);
    }

    // Replace the last Op with alloc and delete the other Ops
    opFusionHelper.replaceOrEraseONNXOps(alloc);
    return success();
  }
}; // namespace onnx_mlir

//===----------------------------------------------------------------------===//
// where op lowering to Krnl dialect.
//===----------------------------------------------------------------------===//

struct ONNXWhereOpLowering : public ConversionPattern {
  DimAnalysis *dimAnalysis;
  bool enableSIMD = false;
  bool enableParallel = false;

  ONNXWhereOpLowering(TypeConverter &typeConverter, MLIRContext *ctx,
      DimAnalysis *dimAnalysis, bool enableSIMD, bool enableParallel)
      : ConversionPattern(
            typeConverter, ONNXWhereOp::getOperationName(), 1, ctx),
        dimAnalysis(dimAnalysis), enableSIMD(enableSIMD) {
    this->enableParallel =
        enableParallel &&
        OnnxToKrnlLoweringConfiguration::enableSpecificParallelOps.isEnabled(
            ONNXWhereOp::getOperationName());
  }

  LogicalResult matchAndRewrite(Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const final {
    Location loc = NameLoc::get(
        StringAttr::get(op->getContext(), ONNXWhereOp::getOperationName()),
        op->getLoc());

    // Convert the output type to MemRefType.
    Type convertedType = typeConverter->convertType(*op->result_type_begin());
    assert(convertedType && mlir::isa<MemRefType>(convertedType) &&
           "Failed to convert type to MemRefType");
    MemRefType outputMemRefType = mlir::cast<MemRefType>(convertedType);
    int64_t outputRank = outputMemRefType.getRank();
    ONNXWhereOpAdaptor operandAdaptor(operands);

    // Shape helper.
    MultiDialectBuilder<IndexExprBuilderForKrnl, KrnlBuilder, MemRefBuilder>
        create(rewriter, loc);
    ONNXBroadcastOpShapeHelper shapeHelper(op, operands, &create.krnlIE);
    shapeHelper.computeShapeAndAssertOnFailure();
    bool hasNoBroadcast = shapeHelper.hasNoBroadcast(dimAnalysis);

    // Insert an allocation and deallocation for the result of this operation.
    Value alloc =
        create.mem.alignedAlloc(outputMemRefType, shapeHelper.getOutputDims());

    // Only create krnl.iterate if one of the operands is not scalar tensor.
    if (!hasAllScalarValues(operands)) {
      ValueRange loopDef = create.krnl.defineLoops(outputRank);
      SmallVector<IndexExpr, 4> lbs(outputRank, LitIE(0));
      SmallVector<IndexExpr, 4> ubs;
      create.krnlIE.getShapeAsDims(alloc, ubs);
      if (enableParallel)
        tryCreateKrnlParallel(
            create.krnl, op, "where op not simdized", loopDef, lbs, ubs);
      create.krnl.iterateIE(loopDef, loopDef, lbs, ubs,
          [&](const KrnlBuilder &createKrnl, ValueRange loopInd) {
            IndexExprScope innerScope(&rewriter, shapeHelper.getScope());
            SmallVector<IndexExpr, 4> outputAccessExprs;
            getIndexExprList<DimIndexExpr>(loopInd, outputAccessExprs);

            // Load the condition value.
            SmallVector<IndexExpr, 4> condAccessExprs;
            LogicalResult res = shapeHelper.getAccessExprs(
                operandAdaptor.getCondition(), 0, outputAccessExprs,
                condAccessExprs, /*flattened dims*/ false, hasNoBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value cond = createKrnl.loadIE(
                operandAdaptor.getCondition(), condAccessExprs);

            // Load the first value.
            SmallVector<IndexExpr, 4> lhsAccessExprs;
            res = shapeHelper.getAccessExprs(operandAdaptor.getX(), 1,
                outputAccessExprs, lhsAccessExprs, /*flattened dims*/ false,
                hasNoBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value lhs =
                createKrnl.loadIE(operandAdaptor.getX(), lhsAccessExprs);

            // Load the second value.
            SmallVector<IndexExpr, 4> rhsAccessExprs;
            res = shapeHelper.getAccessExprs(operandAdaptor.getY(), 2,
                outputAccessExprs, rhsAccessExprs, /*flattened dims*/ false,
                hasNoBroadcast);
            assert(succeeded(res) && "Could not compute access indices");
            Value rhs =
                createKrnl.loadIE(operandAdaptor.getY(), rhsAccessExprs);

            // Return lhs if cond is true else rhs.
            Value result =
                rewriter.create<arith::SelectOp>(loc, cond, lhs, rhs);

            // Store result in the resulting array.
            createKrnl.storeIE(result, alloc, outputAccessExprs);
          });
    } else {
      // Load the condition value.
      Value cond = create.krnl.load(operandAdaptor.getCondition());

      // Load the first value.
      Value lhs = create.krnl.load(operandAdaptor.getX());

      // Load the second value.
      Value rhs = create.krnl.load(operandAdaptor.getY());

      // Return lhs if cond is true else rhs.
      Value result = rewriter.create<arith::SelectOp>(loc, cond, lhs, rhs);

      // Store result in the resulting array.
      create.krnl.store(result, alloc);
    }

    rewriter.replaceOp(op, alloc);

    return success();
  }
};

void populateLoweringONNXElementwiseOpPattern(RewritePatternSet &patterns,
    TypeConverter &typeConverter, MLIRContext *ctx, DimAnalysis *dimAnalysis,
    bool enableSIMD, bool enableParallel) {
  patterns.insert<ONNXElementwiseUnaryOpLowering<mlir::ONNXAbsOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXAddOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXAndOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXAtanOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXBinarizerOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXBitShiftOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXBitwiseAndOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXBitwiseOrOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXBitwiseXorOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXBitwiseNotOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCastOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCeilOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCeluOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCosOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXCoshOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXDequantizeLinearOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXDivOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXEluOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXErfOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXAcosOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXAcoshOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXAsinOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXAsinhOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXAtanhOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXEqualOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXExpOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXFloorOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXGeluOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXGreaterOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXGreaterOrEqualOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXHardSigmoidOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXHardSwishOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXIsInfOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXIsNaNOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXLeakyReluOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXLessOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXLessOrEqualOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXLogOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXMaxOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXMeanOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXMinOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXMishOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXModOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXMulOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXNegOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXNotOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXOrOp>,
      ONNXElementwiseBinaryOpLowering<mlir::ONNXPowOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXReciprocalOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXReluOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXRoundOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXClipOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSeluOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXShrinkOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSigmoidOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSignOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSinOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSinhOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSoftplusOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSoftsignOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXSqrtOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXSubOp>,
      ONNXElementwiseVariadicOpLowering<mlir::ONNXSumOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXTanOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXTanhOp>,
      ONNXElementwiseUnaryOpLowering<mlir::ONNXThresholdedReluOp>,
      ONNXWhereOpLowering, ONNXElementwiseVariadicOpLowering<mlir::ONNXXorOp>>(
      typeConverter, ctx, dimAnalysis, enableSIMD, enableParallel);
  patterns.insert<ONNXElementwiseBinaryOpLowering<mlir::ONNXPReluOp>>(
      typeConverter, ctx, dimAnalysis, enableSIMD, /*isUniBroadcasting=*/true,
      enableParallel);
}

} // namespace onnx_mlir
