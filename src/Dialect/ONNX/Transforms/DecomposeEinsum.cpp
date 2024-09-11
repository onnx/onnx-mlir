/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------- DecomposeEinsum.cpp - Decompose Einsum op ----------------===//
//
// This file implements the decomposition of ONNXEinsumOp to simpler ops.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/Transforms/DecomposeEinsum.hpp"
#include "src/Dialect/Mlir/DialectBuilder.hpp"
#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps/Math/EinsumHelper.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <tuple>
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
typedef ArrayRef<int64_t> ShapeRef;

using einsum::Subscripts;
typedef std::unordered_set<char> SubscriptsSet;

typedef SmallVector<int64_t, 4> Axes;
typedef ArrayRef<int64_t> AxesRef;

// axes must be nonnegative and sorted
Shape shapeExpandDims(const Shape &shape, AxesRef axes) {
  Shape expanded = shape;
  for (auto a : axes) {
    expanded.insert(expanded.begin() + a, 1);
  }
  return expanded;
}

Shape shapeBroadcast(ShapeRef shape1, ShapeRef shape2) {
  Shape shape;
  bool success = OpTrait::util::getBroadcastedShape(shape1, shape2, shape);
  assert(success && "shapes should be broadcast compatible");
  return shape;
}

Shape shapePermute(const Shape &shape, AxesRef perm) {
  Shape permuted;
  for (size_t a = 0; a < shape.size(); ++a) {
    permuted.push_back(shape[perm[a]]);
  }
  return permuted;
}

Axes transposePerm(const Subscripts &original, const Subscripts &transposed) {
  Axes axes;
  for (char x : transposed) {
    axes.push_back(original.find(x));
  }
  return axes;
}

Shape shapeConcat(ShapeRef fst, ShapeRef snd, ShapeRef trd = {}) {
  Shape shape;
  shape.reserve(fst.size() + snd.size() + trd.size());
  shape.append(fst.begin(), fst.end());
  shape.append(snd.begin(), snd.end());
  shape.append(trd.begin(), trd.end());
  return shape;
}

template <typename T>
std::tuple<ArrayRef<T>, ArrayRef<T>, ArrayRef<T>> split3(
    ArrayRef<T> aref, size_t len1, size_t len2, size_t len3) {
  assert(aref.size() == len1 + len2 + len3 && "split3 sizes mismatch");
  return {aref.take_front(len1), aref.slice(len1, len2), aref.take_back(len3)};
}

struct Output : public einsum::Parameter {
  Output(const einsum::Parameter &parameter, Value value)
      : einsum::Parameter(parameter), value(value) {}
  Value value;

  size_t size() const { return shape.size(); }

  RankedTensorType type(Type elementType) const {
    return RankedTensorType::get(shape, elementType);
  }

  void eraseAxis(int64_t a) {
    assert(0 <= a && a < static_cast<int64_t>(size()) &&
           "axis a should be nonnegative and within range");
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
    for (const auto &entry : counts) { // entry == pair (x, count)
      if (entry.second > 1)            // multiple occurrences
        dups.push_back(entry.first);
    }
    return dups;
  }

  SubscriptsSet subscriptsSet() const {
    return SubscriptsSet(subscripts.begin(), subscripts.end());
  }
};

class Decomposer {
public:
  Decomposer(OpBuilder &builder, Location loc,
      const einsum::Signature &signature, ValueRange values)
      : builder(builder), loc(loc), create(builder, loc) {
    assert(values.size() >= 1 && "Einsum must have >= 1 inputs");
    elementType = mlir::cast<ShapedType>(values[0].getType()).getElementType();
    result = signature.output;
    assert(values.size() == signature.inputs.size() &&
           "Einsum signature inputs (from equation) must match actual inputs");
    for (size_t i = 0; i < values.size(); ++i) {
      outputs.emplace_back(signature.inputs[i], values[i]);
    }
  }

  void squeeze(Output &output, AxesRef axes) {
    if (axes.empty())
      return;
    assert(llvm::all_of(axes, [&output](int64_t a) {
      return output.shape[a] == 1;
    }) && "only squeeze axes with dim 1");
    output.eraseAxes(axes);
    output.value = create.onnx.squeeze(
        output.type(elementType), output.value, tensor1D(axes));
  }

  void sum(Output &output, AxesRef axes) {
    if (axes.empty())
      return;
    if (llvm::all_of(
            axes, [&output](int64_t a) { return output.shape[a] == 1; })) {
      squeeze(output, axes);
      return;
    }
    output.eraseAxes(axes);
    output.value = create.onnx.reduceSum(output.type(elementType), output.value,
        tensor1D(axes), /*keepdims=*/false);
  }

  void squeezeNonResults(Output &output) {
    Axes axes;
    for (size_t a = 0; a < output.subscripts.size(); ++a) {
      char subscript = output.subscripts[a];
      if (output.shape[a] == 1 && result.subscripts.count(subscript) == 0) {
        axes.push_back(a);
      }
    }
    squeeze(output, axes);
  }

  void diagonal(Output &output, char subscript) {
    Axes axes;
    for (size_t a = 0; a < output.size(); ++a) {
      if (output.subscripts[a] == subscript)
        axes.push_back(a);
    }
    size_t n = axes.size();
    assert(n >= 2 && "only take diagonal of multiple axes");
    int64_t d = output.shape[axes[0]];
    assert(
        llvm::all_of(axes, [&](int64_t a) { return output.shape[a] == d; }) &&
        "all axes with same subscript in same input must have same dimension, "
        "checked in verify()");
    if (d == 1) {
      squeeze(output, AxesRef(axes).drop_front());
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
    int64_t size = ShapedType::getNumElements(maskShape); // size == d**n
    SmallVector<bool> maskValues(size, false);
    // In the flat maskValues representation of mask the true values are evenly
    // spaced out between maskValues[0]==true,...,maskValues[size-1]==true.
    int64_t distance = (size - 1) / (d - 1); // d-1 always divides d**n-1
    for (int64_t i = 0; i < d; ++i) {
      maskValues[i * distance] = true;
    }
    Value mask = tensor<bool>(maskShape, maskValues, builder.getI1Type());
    output.value = create.onnx.where(
        output.type(elementType), mask, output.value, zeros({}, elementType));
    sum(output, AxesRef(axes).drop_front());
  }

  void diagonalize(Output &output) {
    Subscripts dups = output.duplicates();
    for (char x : dups) {
      diagonal(output, x);
    }
  }

  void reduce(Output &output, const SubscriptsSet &keep) {
    Axes axes;
    for (size_t a = 0; a < output.size(); ++a) {
      if (keep.count(output.subscripts[a]) == 0)
        axes.push_back(a);
    }
    sum(output, axes);
  }

  SubscriptsSet otherSubscripts(
      const std::vector<einsum::Parameter *> &ignore) const {
    SubscriptsSet subscriptsSet;
    for (const Output &output : outputs) {
      if (std::find(ignore.begin(), ignore.end(), &output) == ignore.end())
        subscriptsSet.insert(
            output.subscripts.begin(), output.subscripts.end());
    }
    if (std::find(ignore.begin(), ignore.end(), &result) == ignore.end())
      subscriptsSet.insert(result.subscripts.begin(), result.subscripts.end());
    return subscriptsSet;
  }

  void transpose(Output &output, const Subscripts &transposedSubscripts) {
    assert(output.subscripts.size() == transposedSubscripts.size() &&
           "transposed subscripts must be permutation of existing subscripts");
    if (output.subscripts == transposedSubscripts)
      return;

    Axes perm = transposePerm(output.subscripts, transposedSubscripts);
    output.subscripts = transposedSubscripts;
    output.shape = shapePermute(output.shape, perm);
    output.value = create.onnx.transpose(
        output.type(elementType), output.value, builder.getI64ArrayAttr(perm));
  }

  void unsqueeze(Output &output, const Subscripts &unsqueezedSubscripts) {
    Axes axes;
    SubscriptsSet in = output.subscriptsSet();
    for (size_t a = 0; a < unsqueezedSubscripts.size(); ++a) {
      char x = unsqueezedSubscripts[a];
      if (in.count(x) == 0)
        axes.push_back(a);
    }
    if (axes.empty())
      return;
    output.subscripts = unsqueezedSubscripts;
    output.shape = shapeExpandDims(output.shape, axes);
    output.value = create.onnx.unsqueeze(
        output.type(elementType), output.value, tensor1D(axes));
  }

  void reshape(
      Output &output, const Shape &reShape, const Subscripts &reSubscripts) {
    assert(reShape.size() == reSubscripts.size() &&
           "maintain shape.size() == subscripts.size() invariant");
    if (output.shape == reShape) {
      output.subscripts = reSubscripts;
      return;
    }
    assert(ShapedType::getNumElements(output.shape) ==
               ShapedType::getNumElements(reShape) &&
           "only reshape to shape with same #elements");
    // no zero dims (dispatched at the beginning in decompose()) so
    // we can ignore ONNXReshapeOp allowzero
    assert(ShapedType::getNumElements(reShape) > 0 &&
           "there should be no zero dims");
    output.subscripts = reSubscripts;
    output.shape = reShape;
    output.value = create.onnx.reshape(
        output.type(elementType), output.value, tensor1D(reShape));
  }

  void mul(Output &output1, Output &output2, bool reduceAtEnd = false) {
    SubscriptsSet in1 = output1.subscriptsSet();
    SubscriptsSet in2 = output2.subscriptsSet();
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
    Subscripts subscripts1transposed{subscripts1unshared, sharedSubscripts};
    transpose(output1, subscripts1transposed);
    Subscripts subscripts{subscripts1unshared, output2.subscripts};
    unsqueeze(output1, subscripts);
    output1.subscripts = subscripts;
    output1.shape = shapeBroadcast(output1.shape, output2.shape);
    output1.value = create.onnx.mul(output1.value, output2.value);
    if (reduceAtEnd) {
      SubscriptsSet keep = otherSubscripts({&output1, &output2});
      reduce(output1, keep);
    }
    remove(output2);
  }

  void matmul(
      Output &output1, Output &output2, const SubscriptsSet &reducible) {
    assert(!reducible.empty() && "should call mul() if reducible is empty");

    // Possible alternative implementation without ONNXMatMulOp:
    //
    //   mul(output1, output2, /*reduceAtEnd=*/true);
    //
    // which could be useful to implement Einsum decomposition for types that
    // MatMul doesn't support

    // transpose output1, output2 to put their subscripts in the order:
    //
    // output1: sharedKeepSubscripts + subscripts1unshared + reducibleSubscripts
    // output2: sharedKeepSubscripts + reducibleSubscripts + subscripts2unshared
    //
    // with sharedKeepSubscripts, reducibleSubscripts in the order they appear
    // in output1
    SubscriptsSet in1 = output1.subscriptsSet();
    SubscriptsSet in2 = output2.subscriptsSet();
    Subscripts sharedKeepSubscripts;
    Subscripts reducibleSubscripts;
    Subscripts subscripts1unshared;
    for (char x : output1.subscripts) {
      if (in2.count(x) == 0) {
        subscripts1unshared.push_back(x);
      } else if (reducible.count(x) != 0) {
        reducibleSubscripts.push_back(x);
      } else {
        sharedKeepSubscripts.push_back(x);
      }
    }
    assert(reducible.size() == reducibleSubscripts.size() &&
           "reducible subscripts should appear in both outputs");
    Subscripts subscripts2unshared;
    for (char x : output2.subscripts) {
      if (in1.count(x) == 0) {
        subscripts2unshared.push_back(x);
      }
    }
    Subscripts subscripts1transposed{
        sharedKeepSubscripts, subscripts1unshared, reducibleSubscripts};
    Subscripts subscripts2transposed{
        sharedKeepSubscripts, reducibleSubscripts, subscripts2unshared};
    transpose(output1, subscripts1transposed);
    transpose(output2, subscripts2transposed);

    // copy shapes, the ShapeRefs below will point into these copies,
    // they cannot point into output1.shape and output2.shape because
    // they are reshaped before we're done with the ShapeRefs
    Shape shape1 = output1.shape;
    Shape shape2 = output2.shape;
    // read off the shapes corresponding to the transposed subscripts
    ShapeRef sharedKeep1Shape, unshared1Shape, reducibleShape;
    std::tie(sharedKeep1Shape, unshared1Shape, reducibleShape) =
        split3(ArrayRef(shape1), sharedKeepSubscripts.size(),
            subscripts1unshared.size(), reducibleSubscripts.size());
    ShapeRef sharedKeep2Shape, reducible2Shape, unshared2Shape;
    std::tie(sharedKeep2Shape, reducible2Shape, unshared2Shape) =
        split3(ArrayRef(shape2), sharedKeepSubscripts.size(),
            reducibleSubscripts.size(), subscripts2unshared.size());
    // broadcast not needed because non-result 1-dim axes were squeezed at
    // outset
    assert(
        reducibleShape == reducible2Shape && "broadcast should not be needed");

    // reshape unshared and reducible dims into one dim each
    int64_t unshared1Size = ShapedType::getNumElements(unshared1Shape);
    int64_t reducibleSize = ShapedType::getNumElements(reducibleShape);
    int64_t unshared2Size = ShapedType::getNumElements(unshared2Shape);
    Shape reShape1 =
        shapeConcat(sharedKeep1Shape, {unshared1Size, reducibleSize});
    Shape reShape2 =
        shapeConcat(sharedKeep2Shape, {reducibleSize, unshared2Size});
    // left, right are out-of-band subscripts representing
    // unshared1, unshared2 dims
    const char *left = "(";
    const char *right = ")";
    // red (1st reducible subscript) represents the reshaped reducible dims
    StringRef red = reducibleSubscripts.substr(0, 1);
    Subscripts reshaped1subscripts{sharedKeepSubscripts, left, red};
    Subscripts reshaped2subscripts{sharedKeepSubscripts, red, right};
    reshape(output1, reShape1, reshaped1subscripts);
    reshape(output2, reShape2, reshaped2subscripts);

    // matmul
    Shape sharedKeepShape = shapeBroadcast(sharedKeep1Shape, sharedKeep2Shape);
    output1.subscripts = {sharedKeepSubscripts, left, right};
    output1.shape =
        shapeConcat(sharedKeepShape, {unshared1Size, unshared2Size});
    output1.value = create.onnx.matmul(
        output1.type(elementType), output1.value, output2.value);

    // reshape to get unshared dims back
    Shape shape = shapeConcat(sharedKeepShape, unshared1Shape, unshared2Shape);
    Subscripts subscripts{
        sharedKeepSubscripts, subscripts1unshared, subscripts2unshared};
    reshape(output1, shape, subscripts);

    remove(output2);
  }

  void contract(Output &output1, Output &output2) {
    SubscriptsSet keep = otherSubscripts({&output1, &output2});
    // we populate reducible with the subscripts in the intersection
    // of output1 and output2 subscripts, minus keep
    SubscriptsSet reducible;
    SubscriptsSet in1 = output1.subscriptsSet();
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
    assert(outputs.size() == 1 && "only finalize after all contractions");
    Output &output = outputs[0];
    transpose(output, result.subscripts);
  }

  Value decompose() {
    if (ShapedType::getNumElements(result.shape) == 0 ||
        llvm::any_of(outputs, [](const Output &output) {
          return ShapedType::getNumElements(output.shape) == 0;
        })) {
      // result is empty, or all zeros because there's a zero dim
      // in an input (ReduceSum of the zero dim makes everything zero)
      return zeros(result.shape, elementType);
    }

    for (auto &output : outputs) {
      // Squeeze axes that don't appear in the result.
      // This avoids broadcast of reducible axes in matmul later on.
      // Must be run for all outputs before (diagonalize and) reduce pass
      // because it may enable more axes to reduce.
      squeezeNonResults(output);
    }

    for (auto &output : outputs) {
      diagonalize(output);
      SubscriptsSet keep = otherSubscripts({&output});
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
    return create.onnx.constant(DenseElementsAttr::get(tensorType, values));
  }

  Value zeros(ArrayRef<int64_t> shape, Type elementType) {
    SmallVector<Attribute> values(
        ShapedType::getNumElements(shape), builder.getZeroAttr(elementType));
    return tensor<Attribute>(shape, values, elementType);
  }

  Value tensor1D(ArrayRef<int64_t> values) {
    return create.onnx.constant(builder.getI64TensorAttr(values));
  }

  void remove(Output &output) {
    for (auto iter = outputs.begin(); iter != outputs.end(); ++iter) {
      if (&*iter == &output) {
        outputs.erase(iter);
        return;
      }
    }
    assert(false && "output should be found in outputs");
  }

  OpBuilder &builder;
  Location loc;
  MultiDialectBuilder<OnnxBuilder> create;
  Type elementType;
  einsum::Parameter result;
  std::vector<Output> outputs;
};

// currently limited to the types supported by ReduceSum and MatMul (which
// we decompose to in most cases) which exclude integers with width < 32
bool isDecomposableElementType(Type elementType) {
  if (mlir::isa<FloatType>(elementType))
    return true;
  if (IntegerType intType = mlir::dyn_cast<IntegerType>(elementType))
    return intType.getWidth() >= 32;
  return false;
}

} // namespace

LogicalResult DecomposeEinsumPattern::matchAndRewrite(
    ONNXEinsumOp einsumOp, PatternRewriter &rewriter) const {
  // verify() checked #inputs > 0 and all have same element type, here we check
  // that the element type is one that our decomposition can handle
  //
  // TODO: detect when we don't decompose to ReduceSum or MatMul and
  // accept all types in those cases
  Location loc = einsumOp.getLoc();
  ValueRange inputs = einsumOp.getInputs();

  Type elementType =
      mlir::cast<ShapedType>(inputs[0].getType()).getElementType();
  if (!isDecomposableElementType(elementType))
    return rewriter.notifyMatchFailure(
        loc, "unsupported element type prevents Einsum decomposition");

  if (!llvm::all_of(inputs.getTypes(), hasStaticShape))
    return rewriter.notifyMatchFailure(
        loc, "unknown shapes prevent Einsum decomposition");

  // Wait for shape inference to assign static result shape because otherwise
  // rewriter.replaceOp(einsumOp, result) will fail with
  //
  //   failed to materialize conversion for result #0 of operation
  //   'onnx.Einsum' that remained live after conversion
  //
  // because of the shape mismatch between einsumOp and result.
  if (!hasStaticShape(einsumOp.getOutput().getType()))
    return rewriter.notifyMatchFailure(
        loc, "unknown result shape prevent Einsum decomposition");

  ONNXEinsumOpAdaptor operandAdaptor(einsumOp);
  auto errorFn = [&einsumOp]() {
    return einsumOp.emitOpError()
           << "equation '" << einsumOp.getEquation() << "': ";
  };
  FailureOr<einsum::Signature> signature =
      einsum::inferSignature(operandAdaptor, errorFn);
  assert(succeeded(signature) && "any failure should be caught in verify()");
  Decomposer decomposer(rewriter, loc, *signature, operandAdaptor.getInputs());
  Value result = decomposer.decompose();
  rewriter.replaceOp(einsumOp, result);
  return success();
}

/*static*/
bool DecomposeEinsumPattern::isDecomposable(mlir::ONNXEinsumOp einsumOp) {
  // TODO: deduplicate repeated logic from matchAndRewrite()
  ValueRange inputs = einsumOp.getInputs();
  Type elementType =
      mlir::cast<ShapedType>(inputs[0].getType()).getElementType();
  return isDecomposableElementType(elementType) &&
         llvm::all_of(inputs.getTypes(), hasStaticShape) &&
         hasStaticShape(einsumOp.getOutput().getType());
}

} // namespace onnx_mlir
