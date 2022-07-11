/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- DecomposeEinsum.cpp - Decompose Einsum op ----------------===//
//
// This file implements the decomposition of ONNXEinsumOp to simpler ops.
//
// WIP: so far only handrolled decompositions of the equations used in
// test/backend/inference_backend.py
// TODO: implement all case, leveraging einsum::inferSignature()
//
//===----------------------------------------------------------------------===//

#include "src/Transform/ONNX/DecomposeEinsum.hpp"
#include "src/Dialect/ONNX/ONNXEinsumOpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

using namespace mlir;

namespace onnx_mlir {

// This decomposition requires:
//
// 1. "high precision numeric types" (integer types of width >= 32 and all
// floating point types) because it uses ReduceSum and MatMul which only work
// for those types.
//
// 2. ranks and dims of all input tensors must be known at compile time to
// make many easy compile time decisions: constant indices for
// GatherElements, constant shape for Reshape, Transpose of ellipses,
// and Squeeze and Unsqueeze dim 1 axes.
//
// These requirements could be lifted at the cost of implementation complexity.

namespace {

bool isDecomposableElementType(Type elementType) {
  if (elementType.isa<FloatType>())
    return true;
  if (IntegerType intType = elementType.dyn_cast<IntegerType>())
    return intType.getWidth() >= 32;
  return false;
}

bool isDecomposableType(Type type) {
  ShapedType s = type.cast<ShapedType>();
  return isDecomposableElementType(s.getElementType()) && s.hasStaticShape();
}

bool isDecomposableOp(ONNXEinsumOp einsumOp) {
  return llvm::all_of(einsumOp.Inputs().getTypes(), isDecomposableType);
}

Attribute zero(Type elementType) {
  if (elementType.isa<FloatType>())
    return FloatAttr::get(elementType, 0);
  assert(elementType.isa<IntegerType>()
      && "elementType must be IntegerType if not FloatType");
  return IntegerAttr::get(elementType, 0);
}

typedef SmallVector<int64_t, 4> Axes;

struct Output : public einsum::Parameter {
  Output(const einsum::Parameter& parameter, Value value)
    : einsum::Parameter(parameter),
      value(value) {}
  Value value;

  size_t size() const { return shape.size(); }

  RankedTensorType type(Type elementType) const {
    return RankedTensorType::get(shape, elementType);
  }

  void eraseAxis(int64_t a) {
    assert(0 <= a); // for simplicity axes must be non-negative
    assert(a < (int64_t)size());
    shape.erase(shape.begin() + a);
    subscripts.erase(subscripts.begin() + a);
  }

  void eraseAxes(const Axes &axes) {
    for (auto it = axes.rbegin(); it != axes.rend(); ++it) {
      eraseAxis(*it);
    }
  }
};

class Decomposer {
public:
  Decomposer(OpBuilder &builder, Location loc,
      const einsum::Signature& signature, ValueRange values)
      : builder(builder), loc(loc) {
    assert(values.size() >= 1);
    elementType = values[0].getType().cast<ShapedType>().getElementType();
    result = signature.output;
    assert(values.size() == signature.inputs.size());
    for (size_t i = 0; i < values.size(); ++i) {
      outputs.emplace_back(signature.inputs[i], values[i]);
    }
  }

  void squeeze(Output& output, const Axes& axes) {
    if (axes.empty())
      return;
    assert(llvm::all_of(axes, [&output](int64_t a) {
      return output.shape[a] == 1;
    }));
    output.eraseAxes(axes);
    output.value = builder.create<ONNXSqueezeOp>(
        loc, output.type(elementType), output.value, tensor1D(axes))
        .getResult();
  }

  void squeezeNonResults(Output& output) {
    Axes axes;
    for (size_t a = 0; a < output.subscripts.size(); ++a) {
      char subscript = output.subscripts[a];
      if (output.shape[a] == 1 && result.subscripts.count(subscript) == 0) {
        axes.push_back(a);
      }
    }
    squeeze(output, axes);
  }

  Value decompose(OpBuilder &builder, Location loc) {
    if (ShapedType::getNumElements(result.shape) == 0 ||
        llvm::any_of(outputs, [](const Output& output) {
          return ShapedType::getNumElements(output.shape) == 0;
        })) {
      // result is empty, or all zeros because there's a zero dim
      // in an input (ReduceSum of the zero dim makes everything zero)
      return zeros(result.shape, elementType);
    }

    for (auto& output : outputs) {
      // Squeeze axes that don't appear in the result.
      // This avoids broadcast of reducible axes in matmul later on.
      // Must be run for all outputs before (diagonalize and) reduce pass
      // because it may enable more axes to reduce.
      squeezeNonResults(output);
    }

    // TODO: cover all the other cases
    return nullptr;
  }

private:
  Value zeros(ArrayRef<int64_t> shape, Type elementType) {
    RankedTensorType tensorType = RankedTensorType::get(shape, elementType);
    SmallVector<Attribute> values(tensorType.getNumElements(), zero(elementType));
    return createONNXConstantOpWithDenseAttr(builder, loc,
        DenseElementsAttr::get(tensorType, makeArrayRef(values)));
  }

  Value tensor1D(ArrayRef<int64_t> values) {
    return createONNXConstantOpWithDenseAttr(builder, loc,
        builder.getI64TensorAttr(values));
  }

  OpBuilder& builder;
  Location loc;
  Type elementType;
  einsum::Parameter result;
  std::vector<Output> outputs;
};

} // namespace

LogicalResult DecomposeEinsumPattern::matchAndRewrite(ONNXEinsumOp einsumOp, PatternRewriter &rewriter) const {
  auto loc = einsumOp.getLoc();
  ONNXEinsumOpAdaptor operandAdaptor(einsumOp);
  einsum::ErrorFn errorFn = [&einsumOp]() {
    return einsumOp.emitOpError()
        << "equation '" << einsumOp.equation() << "': ";
  };
  auto signature = einsum::inferSignature(operandAdaptor, errorFn);
  assert(succeeded(signature) && "any failure should be caught in verify()");
  Decomposer decomposer(rewriter, loc, *signature, operandAdaptor.Inputs());
  // TODO: use signature to decompose einsumOp

  IntegerType typeI64 = rewriter.getI64Type();
  auto zeroScalar = [&rewriter, loc](Type elementType) {
    return createONNXConstantOpWithDenseAttr(rewriter, loc,
        rewriter.getZeroAttr(RankedTensorType::get({}, elementType)));
  };
  auto scalar = [&rewriter, loc, typeI64](int64_t v) {
    return createONNXConstantOpWithDenseAttr(rewriter, loc,
        DenseElementsAttr::get(
            RankedTensorType::get({}, typeI64),
            IntegerAttr::get(typeI64, v)));
  };
  auto tensor1D = [&rewriter, loc](ArrayRef<int64_t> vs) {
    return createONNXConstantOpWithDenseAttr(rewriter, loc,
        rewriter.getI64TensorAttr(vs));
  };
  auto equation = einsumOp.equation();
  size_t commas = std::count(equation.begin(), equation.end(), ',');
  if (einsumOp.Inputs().size() != commas + 1) {
    return einsumOp->emitError("Einsum equation, Inputs size mismatch");
  }
  ValueRange inputs = operandAdaptor.Inputs();
  ShapedType input0Type = inputs[0].getType().cast<ShapedType>();
  Type elementType = input0Type.getElementType();
  Value result;
  // TODO: remove these special cases when the general logic is implemented
  if (equation == "ij->ji") { // transpose
    ONNXTransposeOp transposeOp = rewriter.create<ONNXTransposeOp>(
        loc, UnrankedTensorType::get(elementType), inputs[0], /*perm=*/nullptr);
    (void)transposeOp.inferShapes([](Region &region) {});
    result = transposeOp.getResult();
  } else if (equation == "ij->i") { // sum
    if (!isDecomposableElementType(elementType)) {
      return einsumOp->emitError("unsupported element type prevents Einsum decomposition");
    }
    Value axes = tensor1D({1});
    ONNXReduceSumOp reduceSumOp = rewriter.create<ONNXReduceSumOp>(
        loc, UnrankedTensorType::get(elementType), inputs[0], axes, /*keepdims=*/0);
    (void)reduceSumOp.inferShapes([](Region &region) {});
    result = reduceSumOp.getResult();
  } else if (equation == "i,i" || equation == "bij, bjk -> bik") { // inner_prod, batch_matmul
    if (!isDecomposableElementType(elementType)) {
      return einsumOp->emitError("unsupported element type prevents Einsum decomposition");
    }
    ONNXMatMulOp matMulOp = rewriter.create<ONNXMatMulOp>(
        loc, UnrankedTensorType::get(elementType), inputs[0], inputs[1]);
    (void)matMulOp.inferShapes([](Region &region) {});
    if (equation == "i,i") {
      // The shape is wrong in this case: [1] instead of [], i.e. 1D instead of scalar.
      // The Squeeze op below works around the problem.
      Value axes = tensor1D({0});
      ONNXSqueezeOp squeezeOp = rewriter.create<ONNXSqueezeOp>(
          loc, UnrankedTensorType::get(elementType), matMulOp.getResult(), axes);
      (void)squeezeOp.inferShapes([](Region &region) {});
      result = squeezeOp.getResult();
    } else {
      result = matMulOp.getResult();
    }
  } else if (equation == "...ii ->...i") { // batch_diagonal, IdMatrix+Where+ReduceSum implementation
    // diagDim = input[0].shape[-1]
    // diagRange = Range(0,diagDim,1)
    // unsqueezedDiagRange = Unsqueeze(diagRange,[1])
    // mask = Equal(diagRange,unsqueezedDiagRange)
    // masked = Where(mask,input[0],scalar(0))
    // result = ReduceSum(diagonal,[-1],keepdims=0)
    if (!isDecomposableElementType(elementType)) {
      return einsumOp->emitError("unsupported element type prevents Einsum decomposition");
    }
    if (!input0Type.hasStaticShape()) {
      return einsumOp->emitError("unknown shape prevents Einsum decomposition");
    }
    auto inputShape = input0Type.getShape();
    auto inputRank = inputShape.size();
    assert(inputRank >= 2);
    auto resultShape = llvm::makeArrayRef(inputShape.begin(), inputShape.end() - 1);
    int64_t i = inputShape[inputRank - 1];
    Value diagDim = scalar(i);
    Value zero = scalar(0);
    Value one = scalar(1);
    Type diagRangeType = RankedTensorType::get({i}, typeI64);
    Value diagRange = rewriter.create<ONNXRangeOp>(
        loc, diagRangeType, /*start=*/zero, /*limit=*/diagDim, /*delta=*/one)
        .getResult();
    Type unsqueezedDiagRangeType = RankedTensorType::get({i, 1}, typeI64);
    Value one1D = tensor1D({1});
    Value unsqueezedDiagRange = rewriter.create<ONNXUnsqueezeOp>(
        loc, unsqueezedDiagRangeType, diagRange, one1D)
        .getResult();
    Value mask = rewriter.create<ONNXEqualOp>(
        loc, RankedTensorType::get({i, i}, rewriter.getI1Type()), diagRange, unsqueezedDiagRange)
        .getResult();
    Value masked = rewriter.create<ONNXWhereOp>(
        loc, input0Type, mask, inputs[0], zeroScalar(elementType));
    Value axes = tensor1D({-1});
    ONNXReduceSumOp reduceSumOp = rewriter.create<ONNXReduceSumOp>(
        loc, RankedTensorType::get(resultShape, elementType), masked, axes, /*keepdims=*/0);
    result = reduceSumOp.getResult();
  } else if (equation == "...ii ->...i") { // batch_diagonal, GatherElements implementation
    // resultShape = shape of ...i = Shape(input[0])[:-1]
    // diagDim = last dim
    // range = Range(0,diagDim,1)
    // squeezedIndices = Expand(range,resultShape)
    // indices = Unsqueeze(indices,[-1])
    // unsqueezedResult = GatherElements(input[0],indices,axis=-1)
    // result = squeeze(unsqueezedResult,-1)
    Value resultShape;
    Value diagDim;
    Type diagRangeType = RankedTensorType::get({ShapedType::kDynamicSize}, typeI64);
    Type squeezedIndicesType = UnrankedTensorType::get(typeI64);
    Type indicesType = UnrankedTensorType::get(typeI64);
    Type unsqueezedResultType = UnrankedTensorType::get(elementType);
    Type resultType = UnrankedTensorType::get(elementType);
    Value negativeOne1D = tensor1D({-1});
    if (input0Type.hasStaticShape()) {
      auto inputShape = input0Type.getShape();
      auto inputRank = inputShape.size();
      assert(inputRank >= 2);
      resultShape = tensor1D(llvm::makeArrayRef(inputShape.begin(), inputShape.end() - 1));
      int64_t i = inputShape[inputRank - 1];
      diagDim = scalar(i);
      diagRangeType = RankedTensorType::get({i}, typeI64);
      SmallVector<int64_t, 5> indicesShape(inputShape.begin(), inputShape.end());
      indicesShape.back() = 1;
      SmallVector<int64_t, 4> resultShape(inputShape.begin(), inputShape.end() - 1);
      squeezedIndicesType = RankedTensorType::get(resultShape, typeI64);
      indicesType = RankedTensorType::get(indicesShape, typeI64);
      unsqueezedResultType = RankedTensorType::get(indicesShape, elementType);
      resultType = RankedTensorType::get(resultShape, elementType);
    } else {
      ONNXShapeOp shapeOp = rewriter.create<ONNXShapeOp>(
          loc, RankedTensorType::get({ShapedType::kDynamicSize}, typeI64), inputs[0]);
      Value inputShape = shapeOp.getResult();

      Value zero1D = tensor1D({0});
      Value one1D = tensor1D({1});
      ONNXSliceOp sliceOp = rewriter.create<ONNXSliceOp>(
          loc, RankedTensorType::get({ShapedType::kDynamicSize}, typeI64), inputShape,
          /*starts=*/zero1D, /*ends=*/negativeOne1D, /*axes=*/zero1D, /*steps=*/one1D);
      resultShape = sliceOp.getResult();

      Value negativeOne = scalar(-1);
      ONNXGatherOp diagGatherOp = rewriter.create<ONNXGatherOp>(
          loc, RankedTensorType::get({}, typeI64), inputShape, /*indices=*/negativeOne);
      diagDim = diagGatherOp.getResult();
    }

    Value zero = scalar(0);
    Value one = scalar(1);
    ONNXRangeOp rangeOp = rewriter.create<ONNXRangeOp>(
        loc, diagRangeType, /*start=*/zero, /*limit=*/diagDim, /*delta=*/one);
    Value diagRange = rangeOp.getResult();

    ONNXExpandOp expandOp = rewriter.create<ONNXExpandOp>(
        loc, squeezedIndicesType, diagRange, resultShape);
    Value squeezedIndices = expandOp.getResult();

    ONNXUnsqueezeOp unsqueezeOp = rewriter.create<ONNXUnsqueezeOp>(
        loc, indicesType, squeezedIndices, negativeOne1D);
    Value indices = unsqueezeOp.getResult();

    ONNXGatherElementsOp gatherElementsOp = rewriter.create<ONNXGatherElementsOp>(
        loc, unsqueezedResultType, inputs[0], indices, /*axis=*/-1);
    Value unsqueezedResult = gatherElementsOp.getResult();

    ONNXSqueezeOp squeezeOp = rewriter.create<ONNXSqueezeOp>(
        loc, resultType, unsqueezedResult, negativeOne1D);
    result = squeezeOp.getResult();
  } else {
    if (!isDecomposableOp(einsumOp)) {
      return einsumOp->emitError("unsupported element type or unknown shapes prevent Einsum decomposition");
    }
    return einsumOp->emitError("Einsum decomposition unimplemented"); // TODO: implement all cases
  }
  rewriter.replaceOp(einsumOp, result);
  return success();
}

} // namespace onnx_mlir
