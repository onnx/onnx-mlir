/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- DecomposeEinsum.cpp - Decompose Einsum op ----------------===//
//
// This file implements the decomposition of ONNXEinsumOp to simpler ops.
//
//===----------------------------------------------------------------------===//

#include "src/Transform/ONNX/DecomposeEinsum.hpp"
#include "src/Dialect/ONNX/ONNXEinsumOpHelper.hpp"
#include "src/Dialect/ONNX/ONNXOpsHelper.hpp"

#include <unordered_map>
#include <unordered_set>

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

using einsum::Shape;
using einsum::Subscripts;

typedef SmallVector<int64_t, 4> Axes;
typedef ArrayRef<int64_t> AxesRef;

// axes must be nonnegative and sorted
Shape shapeExpandDims(const Shape& shape, AxesRef axes) {
  Shape expanded = shape;
  for (auto a : axes) {
    expanded.insert(expanded.begin() + a, 1);
  }
  assert(expanded.size() == shape.size() + axes.size());
  return expanded;
}

Shape shapeBroadcast(const Shape &shape1, const Shape &shape2) {
  Shape shape;
  bool success = OpTrait::util::getBroadcastedShape(shape1, shape2, shape);
  assert(success);
  return shape;
}

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

  void eraseAxes(AxesRef axes) {
    for (auto it = axes.rbegin(); it != axes.rend(); ++it) {
      eraseAxis(*it);
    }
  }

  Subscripts duplicates() const {
    Subscripts dups;
    std::unordered_map<char, int> counts;
    for (char x : subscripts) {
      counts[x] += 1; // counts[x] initializes to 0 if not yet mapped
    }
    for (const auto& entry : counts) { // entry == pair (x, count)
      if (entry.second > 1) // multiple occurrences
        dups.push_back(entry.first);
    }
    return dups;
  }

  std::unordered_set<char> subscriptsSet() const {
    std::unordered_set<char> set;
    for (char x : subscripts) {
      set.insert(x);
    }
    return set;
  }
};

Attribute zero(Type elementType) {
  if (elementType.isa<FloatType>())
    return FloatAttr::get(elementType, 0);
  assert(elementType.isa<IntegerType>()
      && "elementType must be IntegerType if not FloatType");
  return IntegerAttr::get(elementType, 0);
}

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

  void squeeze(Output& output, AxesRef axes) {
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

  void sum(Output& output, AxesRef axes) {
    if (axes.empty())
      return;
    if (llvm::all_of(axes, [&output](int64_t a) {
          return output.shape[a] == 1;
        })) {
      squeeze(output, axes);
      return;
    }
    output.eraseAxes(axes);
    output.value = builder.create<ONNXReduceSumOp>(
        loc, output.type(elementType), output.value, tensor1D(axes), /*keepdims=*/0)
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

  void diagonal(Output& output, AxesRef axes) {
    char subscript = output.subscripts[axes[0]];
    assert(output.subscripts.count(subscript) == axes.size());
    int64_t d = output.shape[axes[0]];
    assert(llvm::all_of(axes, [&](int64_t a) { return output.shape[a] == d; }));
    if (d == 1) {
      squeeze(output, axes.drop_front());
      return;
    }
    // Create a boolean mask with the shape of the diagonal axes, "unsqueezed"
    // with dim 1 for all the other axes. For instance, if output.shape is
    // (2,4,4,5,4) and axes is [1,2,4] then maskShape is (1,4,4,1,4).
    // The maskValues has true on the "diagonal", i.e.
    // maskShape[_,i,i,_,i]==true for i in {0,1,2,3} and false elsewhere.
    Shape maskShape;
    for (size_t i = 0; i < output.size(); ++i) {
      maskShape.push_back(output.subscripts[i] == subscript ? d : 1);
    }
    int64_t size = ShapedType::getNumElements(maskShape);
    SmallVector<bool> maskValues(size, false);
    // In the flat maskValues representation of mask the true values are evenly
    // spaced out between maskValues[0]==true,...,maskValues[size-1]==true.
    assert((size - 1) % (d - 1) == 0);
    auto distance = (size - 1) / (d - 1);
    for (int64_t i = 0; i < d; ++i) {
      maskValues[i * distance] = true;
    }
    Value mask = tensor<bool>(maskShape, maskValues, builder.getI1Type());
    output.value = builder.create<ONNXWhereOp>(
        loc, output.type(elementType), mask, output.value, zeroScalar(elementType))
        .getResult();
    sum(output, axes.drop_front());
  }

  void diagonalize(Output& output) {
    auto dups = output.duplicates();
    for (char x : dups) {
      Axes axes;
      for (size_t a = 0; a < output.size(); ++a) {
        if (output.subscripts[a] == x)
          axes.push_back(a);
      }
      diagonal(output, axes);
    }
  }

  void reduce(Output& output, const std::unordered_set<char> &keep) {
    Axes axes;
    for (size_t a = 0; a < output.size(); ++a) {
      if (keep.count(output.subscripts[a]) == 0)
        axes.push_back(a);
    }
    sum(output, axes);
  }

  std::unordered_set<char> otherSubscripts(const std::vector<einsum::Parameter*>& ignore) const {
    std::unordered_set<char> subscriptsSet;
    for (const Output &output : outputs) {
      if (std::find(ignore.begin(), ignore.end(), &output) == ignore.end()) {
        for (char x : output.subscripts)
          subscriptsSet.insert(x);
      }
    }
    if (std::find(ignore.begin(), ignore.end(), &result) == ignore.end()) {
      for (char x : result.subscripts)
        subscriptsSet.insert(x);
    }
    return subscriptsSet;
  }

  Axes transposePerm(const Subscripts &original, const Subscripts &transposed) const {
    Axes axes;
    for (char x : transposed) {
      axes.push_back(original.find(x));
    }
    return axes;
  }

  Shape permuteShape(const Shape& shape, AxesRef perm) {
    Shape permuted;
    for (size_t a = 0; a < shape.size(); ++a) {
      permuted.push_back(shape[perm[a]]);
    }
    return permuted;
  }

  void transpose(Output& output, const Subscripts &transposedSubscripts) {
    assert(output.subscripts.size() == transposedSubscripts.size());
    if (output.subscripts == transposedSubscripts)
      return;

    Axes perm = transposePerm(output.subscripts, transposedSubscripts);
    output.subscripts = transposedSubscripts;
    output.shape = permuteShape(output.shape, perm);
    output.value = builder.create<ONNXTransposeOp>(
        loc, output.type(elementType), output.value, builder.getI64ArrayAttr(perm))
        .getResult();
  }

  void unsqueeze(Output& output, const Subscripts &unsqueezedSubscripts) {
    Axes axes;
    std::unordered_set<char> in = output.subscriptsSet();
    for (size_t a = 0; a < unsqueezedSubscripts.size(); ++a) {
      char x = unsqueezedSubscripts[a];
      if (in.count(x) == 0)
        axes.push_back(a);
    }
    if (axes.empty())
      return;
    output.subscripts = unsqueezedSubscripts;
    output.shape = shapeExpandDims(output.shape, axes);
    output.value = builder.create<ONNXUnsqueezeOp>(
        loc, output.type(elementType), output.value, tensor1D(axes))
        .getResult();
  }

  void mul(Output &output1, Output &output2, bool reduceAtEnd = false) {
    std::unordered_set<char> in1 = output1.subscriptsSet();
    std::unordered_set<char> in2 = output2.subscriptsSet();
    Subscripts sharedSubscripts;
    for (char x : output2.subscripts) {
      if (in1.count(x) != 0)
        sharedSubscripts.push_back(x);
    }
    Subscripts subscripts1unshared;
    for (char x : output1.subscripts) {
      if (in2.count(x) == 0)
        subscripts1unshared.push_back(x);
    }
    Subscripts subscripts1transposed = subscripts1unshared;
    subscripts1transposed += sharedSubscripts;
    transpose(output1, subscripts1transposed);
    Subscripts subscripts = subscripts1unshared;
    subscripts += output2.subscripts;
    unsqueeze(output1, subscripts);
    output1.subscripts = subscripts;
    output1.shape = shapeBroadcast(output1.shape, output2.shape);
    output1.value = builder.create<ONNXMulOp>(
        loc, output1.type(elementType), output1.value, output2.value)
        .getResult();
    if (reduceAtEnd) {
      auto keep = otherSubscripts({&output1, &output2});
      reduce(output1, keep);
    }
    remove(output2);
  }

  void matmul(Output &output1, Output &output2, const std::unordered_set<char> &reducible) {
    // TODO: implement with ONNXMatMulOp
    mul(output1, output2, /*reduceAtEnd=*/true);
  }

  void remove(Output& output) {
    for (auto iter = outputs.begin(); iter != outputs.end(); ++iter) {
      if (&*iter == &output) {
        outputs.erase(iter);
        return;
      }
    }
    assert(false); // should have returned from the loop
  }

  void contract(Output& output1, Output& output2) {
    auto keep = otherSubscripts({&output1, &output2});
    // we populate reducible with the subscripts in the intersection
    // of output1 and output2 subscripts, minus keep
    std::unordered_set<char> reducible;
    std::unordered_set<char> in1 = output1.subscriptsSet();
    for (char x : output2.subscripts) {
      if (in1.count(x) != 0) {
        // x is in intersection of output1 and output2 subscripts
        if (keep.count(x) == 0) {
          reducible.insert(x);
        }
      }
    }
    if (reducible.empty()) {
      mul(output1, output2);
    } else {
      matmul(output1, output2, reducible);
    }
  }

  void finalize() {
    assert(outputs.size() == 1);
    Output &output = outputs[0];
    transpose(output, result.subscripts);
  }

  Value decompose() {
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

    for (auto& output : outputs) {
      diagonalize(output);
      auto keep = otherSubscripts({&output});
      reduce(output, keep);
    }

    while (outputs.size() > 1) {
      contract(outputs[0], outputs[1]);
    }

    finalize();
    return outputs[0].value;
  }

private:
  template <typename T>
  Value tensor(ArrayRef<int64_t> shape, ArrayRef<T> values, Type elementType) {
    RankedTensorType tensorType = RankedTensorType::get(shape, elementType);
    return createONNXConstantOpWithDenseAttr(builder, loc,
        DenseElementsAttr::get(tensorType, values));
  }

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

  Value zeroScalar(Type elementType) {
    return createONNXConstantOpWithDenseAttr(builder, loc,
        builder.getZeroAttr(RankedTensorType::get({}, elementType)));
  }

  OpBuilder& builder;
  Location loc;
  Type elementType;
  einsum::Parameter result;
  std::vector<Output> outputs;
};

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

} // namespace

LogicalResult DecomposeEinsumPattern::matchAndRewrite(ONNXEinsumOp einsumOp, PatternRewriter &rewriter) const {
  if (!isDecomposableOp(einsumOp)) {
    return einsumOp->emitError("unsupported element type or unknown shapes prevent Einsum decomposition");
  }

  auto loc = einsumOp.getLoc();
  ONNXEinsumOpAdaptor operandAdaptor(einsumOp);
  einsum::ErrorFn errorFn = [&einsumOp]() {
    return einsumOp.emitOpError()
        << "equation '" << einsumOp.equation() << "': ";
  };
  auto signature = einsum::inferSignature(operandAdaptor, errorFn);
  assert(succeeded(signature) && "any failure should be caught in verify()");
  Decomposer decomposer(rewriter, loc, *signature, operandAdaptor.Inputs());
  Value result = decomposer.decompose();
  rewriter.replaceOp(einsumOp, result);
  return success();
}

} // namespace onnx_mlir
