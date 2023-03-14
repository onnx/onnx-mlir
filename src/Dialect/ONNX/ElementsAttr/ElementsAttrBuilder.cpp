/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ElementsAttrBuilder.cpp ----------------------===//
//
// Builds DisposableElementsAttr instances.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrBuilder.hpp"

#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposablePool.hpp"
#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrHelper.hpp"
#include "src/Dialect/ONNX/ElementsAttr/Strides.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <numeric>

using namespace mlir;

namespace onnx_mlir {

namespace {
std::unique_ptr<llvm::MemoryBuffer> getMemoryBuffer(DenseElementsAttr dense) {
  if (dense.getElementType().isInteger(1)) {
    // Don't use dense.rawData() which is bit packed, whereas
    // DisposableElementsAttr represents bools with one byte per bool value.
    if (dense.isSplat()) {
      char b = dense.getSplatValue<bool>();
      StringRef s(&b, 1);
      return llvm::MemoryBuffer::getMemBufferCopy(s);
    } else {
      std::unique_ptr<llvm::WritableMemoryBuffer> writeBuffer =
          llvm::WritableMemoryBuffer::getNewUninitMemBuffer(dense.size());
      std::copy_n(dense.value_begin<bool>(), dense.size(),
          writeBuffer->getBuffer().begin());
      return std::move(writeBuffer);
    }
  } else {
    ArrayRef<char> bytes = dense.getRawData();
    int64_t size = bytes.size();
    if (dense.isSplat())
      assert(size == getEltSizeInBytes(dense.getType()) && "size mismatch");
    else
      assert(size == getSizeInBytes(dense.getType()) && "size mismatch");
    return llvm::MemoryBuffer::getMemBuffer(asStringRef(bytes),
        /*BufferName=*/"", /*RequiresNullTerminator=*/false);
  }
}
} // namespace

struct ElementsAttrBuilder::ElementsProperties {
  BType bufferBType;
  SmallVector<int64_t, 4> strides;
  std::shared_ptr<llvm::MemoryBuffer> buffer;
  const Transformer &transformer;
};

ElementsAttrBuilder::ElementsAttrBuilder(DisposablePool &disposablePool)
    : disposablePool(disposablePool) {}

ElementsAttr ElementsAttrBuilder::fromMemoryBuffer(
    ShapedType type, std::unique_ptr<llvm::MemoryBuffer> membuf) {
  BType btype = btypeOfMlirType(type.getElementType());
  return createWithDefaultStrides(type, btype, std::move(membuf));
}

DisposableElementsAttr ElementsAttrBuilder::toDisposableElementsAttr(
    ElementsAttr elements) {
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>())
    return disposable;
  if (auto dense = elements.dyn_cast<DenseElementsAttr>()) {
    if (!disposablePool.isActive())
      return nullptr;
    ElementsProperties props = getElementsProperties(dense);
    ElementsAttr created = create(dense.getType(), props.bufferBType,
        props.strides, props.buffer, props.transformer);
    // Check for race condition where disposablePool became inactive since we
    // checked, in which case it returns a DenseElementsAttr which we don't
    // want.
    if (auto disposable = created.dyn_cast<DisposableElementsAttr>())
      return disposable;
    else
      return nullptr;
  }
  // TODO: consider supporting more ElementsAttr types
  llvm_unreachable("unexpected ElementsAttr instance");
}

/*static*/
DenseElementsAttr ElementsAttrBuilder::toDenseElementsAttr(
    ElementsAttr elements) {
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>())
    return disposable.toDenseElementsAttr();
  if (auto dense = elements.dyn_cast<DenseElementsAttr>())
    return dense;
  // TODO: consider supporting more ElementsAttr types
  llvm_unreachable("unexpected ElementsAttr instance");
}

ElementsAttr ElementsAttrBuilder::fromWideNums(
    ShapedType type, const Filler<WideNum> &wideDataFiller) {
  BType bufferBType = wideBTypeOfBType(btypeOfMlirType(type.getElementType()));
  return fromRawBytes(
      type, bufferBType, [&wideDataFiller](MutableArrayRef<char> bytes) {
        wideDataFiller(castMutableArrayRef<WideNum>(bytes));
      });
}

// TODO: Inline this implementation to help the compiler inline combiner into
//       the closures constructed in expandAndTransform, if benchmarking
//       demonstrates a speedup.
ElementsAttr ElementsAttrBuilder::combine(ElementsAttr lhs, ElementsAttr rhs,
    ShapedType combinedType, WideNum (*combiner)(WideNum, WideNum)) {
  if (lhs.isSplat()) {
    WideNum lhsNum = getElementsSplatWideNum(lhs);
    return expandAndTransform(rhs, combinedType,
        functionTransformer(
            [lhsNum, combiner](WideNum n) { return combiner(lhsNum, n); }));
  }

  if (rhs.isSplat()) {
    WideNum rhsNum = getElementsSplatWideNum(rhs);
    return expandAndTransform(lhs, combinedType,
        functionTransformer(
            [rhsNum, combiner](WideNum n) { return combiner(n, rhsNum); }));
  }

  auto combinedShape = combinedType.getShape();

  SmallVector<int64_t, 4> xpLhsStrides;
  ArrayBuffer<WideNum> lhsNums =
      getWideNumsAndExpandedStrides(lhs, combinedShape, xpLhsStrides);
  StridedArrayRef<WideNum> stridedLhs(lhsNums.get(), xpLhsStrides);

  SmallVector<int64_t, 4> xpRhsStrides;
  ArrayBuffer<WideNum> rhsNums =
      getWideNumsAndExpandedStrides(rhs, combinedShape, xpRhsStrides);
  StridedArrayRef<WideNum> stridedRhs(rhsNums.get(), xpRhsStrides);

  return fromWideNums(combinedType, [&](MutableArrayRef<WideNum> dstNums) {
    mapStrides<WideNum, WideNum, WideNum>(
        combinedShape, dstNums, stridedLhs, stridedRhs, combiner);
  });
}

ElementsAttr ElementsAttrBuilder::where(ElementsAttr cond, ElementsAttr lhs,
    ElementsAttr rhs, ShapedType combinedType) {
  assert(cond.getElementType().isInteger(1));
  assert(lhs.getElementType() == rhs.getElementType());
  assert(lhs.getElementType() == combinedType.getElementType());

  if (cond.isSplat()) {
    bool condBool = getElementsSplatWideNum(cond).u64;
    return expand(condBool ? lhs : rhs, combinedType.getShape());
  }

  if (lhs.isSplat() && rhs.isSplat()) {
    WideNum lhsNum = getElementsSplatWideNum(lhs);
    WideNum rhsNum = getElementsSplatWideNum(rhs);
    return expandAndTransform(cond, combinedType,
        functionTransformer(
            [lhsNum, rhsNum](WideNum n) { return n.u64 ? lhsNum : rhsNum; }));
  }

  auto combinedShape = combinedType.getShape();

  SmallVector<int64_t, 4> xpCondStrides;
  ArrayBuffer<WideNum> condNums =
      getWideNumsAndExpandedStrides(cond, combinedShape, xpCondStrides);

  SmallVector<int64_t, 4> xpLhsStrides;
  ArrayBuffer<WideNum> lhsNums =
      getWideNumsAndExpandedStrides(lhs, combinedShape, xpLhsStrides);
  StridedArrayRef<WideNum> stridedLhs(lhsNums.get(), xpLhsStrides);

  SmallVector<int64_t, 4> xpRhsStrides;
  ArrayBuffer<WideNum> rhsNums =
      getWideNumsAndExpandedStrides(rhs, combinedShape, xpRhsStrides);
  StridedArrayRef<WideNum> stridedRhs(rhsNums.get(), xpRhsStrides);

  return fromWideNums(combinedType, [&](MutableArrayRef<WideNum> dstNums) {
    // Copy cond into dstNums with broadcast.
    restrideArray<WideNum>(
        combinedShape, xpCondStrides, condNums.get(), dstNums);

    WideNum *end = traverseStrides<WideNum *, WideNum, WideNum>(combinedShape,
        dstNums.begin(), stridedLhs, stridedRhs,
        [](WideNum *res, const WideNum *x, const WideNum *y) {
          *res = res->u64 ? *x : *y;
        });
    assert(end == dstNums.end() && "traverses every dstNums element");
  });
}

namespace {
using ElementsTransformer = std::function<void(llvm::MutableArrayRef<WideNum>)>;

ElementsTransformer composeTransforms(
    ElementsTransformer first, ElementsTransformer second) {
  if (first == nullptr)
    return second;
  else
    return [fst = std::move(first), snd = std::move(second)](
               MutableArrayRef<WideNum> dst) {
      fst(dst);
      snd(dst);
    };
}

template <typename SrcT, typename DstT>
struct Caster {
  static inline constexpr DstT eval(SrcT src) { return static_cast<DstT>(src); }
};

template <typename SrcT, typename DstT>
using WideCaster = WideNumWrappedFunction<Caster<SrcT, DstT>>;

auto wideCaster(BType src, BType dst) -> WideNum (*)(WideNum) {
  constexpr BType DBL = BType::DOUBLE, I64 = BType::INT64, U64 = BType::UINT64;
  // clang-format off
  if (src == DBL && dst == I64) return WideCaster<double, int64_t>::eval;
  if (src == DBL && dst == U64) return WideCaster<double, uint64_t>::eval;
  if (src == I64 && dst == DBL) return WideCaster<int64_t, double>::eval;
  if (src == I64 && dst == U64) return WideCaster<int64_t, uint64_t>::eval;
  if (src == U64 && dst == DBL) return WideCaster<uint64_t, double>::eval;
  if (src == U64 && dst == I64) return WideCaster<uint64_t, int64_t>::eval;
  // clang-format on
  llvm_unreachable("wideCaster must be called with 2 different wide types");
}
} // namespace

ElementsAttr ElementsAttrBuilder::castElementType(
    ElementsAttr elms, Type newElementType) {
  Type oldElementType = elms.getElementType();
  if (newElementType == oldElementType)
    return elms;

  ElementsProperties props = getElementsProperties(elms);

  ShapedType newType = elms.getType().clone(newElementType);
  BType newBType = btypeOfMlirType(newElementType);
  BType oldBType = btypeOfMlirType(oldElementType);
  BType newWideType = wideBTypeOfBType(newBType);
  BType oldWideType = wideBTypeOfBType(oldBType);

  auto transformer =
      oldWideType == newWideType
          ? props.transformer
          : composeTransforms(props.transformer,
                functionTransformer(wideCaster(oldWideType, newWideType)));
  return create(newType, props.bufferBType, props.strides, props.buffer,
      std::move(transformer));
}

namespace {
bool isIdentityPermutation(ArrayRef<uint64_t> perm) {
  for (size_t i = 0; i < perm.size(); ++i) {
    if (perm[i] != i)
      return false;
  }
  return true;
}
} // namespace

ElementsAttr ElementsAttrBuilder::transpose(
    ElementsAttr elms, ArrayRef<uint64_t> perm) {
  if (isIdentityPermutation(perm))
    return elms;

  ElementsProperties props = getElementsProperties(elms);

  ShapedType type = elms.getType();
  auto transposedShape = transposeDims(type.getShape(), perm);
  ShapedType transposedType = type.clone(transposedShape);
  auto transposedStrides = transposeDims(props.strides, perm);
  return create(transposedType, props.bufferBType, transposedStrides,
      props.buffer, props.transformer);
}

ElementsAttr ElementsAttrBuilder::reshape(
    ElementsAttr elms, ArrayRef<int64_t> reshapedShape) {
  ShapedType type = elms.getType();
  auto shape = type.getShape();
  if (reshapedShape == shape)
    return elms;

  ElementsProperties props = getElementsProperties(elms);

  ShapedType reshapedType = type.clone(reshapedShape);
  if (auto reshapedStrides =
          reshapeStrides(shape, props.strides, reshapedShape))
    return create(reshapedType, props.bufferBType, *reshapedStrides,
        props.buffer, props.transformer);

  auto disp = elms.dyn_cast<DisposableElementsAttr>();
  assert(disp && "reshapeStrides() always succeeds for non-Disposable "
                 "ElementsAttr as strides are always default or splat");

  if (!disp.isTransformed()) // Skip WideNums absent element-wise transform.
    return fromRawBytes(
        reshapedType, disp.getBufferBType(), [disp](MutableArrayRef<char> dst) {
          auto src = disp.getBufferBytes();
          restrideArray(disp.getBufferElementBytewidth(), disp.getShape(),
              disp.getStrides(), src, dst);
        });

  return fromWideNums(reshapedType, [disp](MutableArrayRef<WideNum> wideData) {
    disp.readWideNums(wideData);
  });
}

ElementsAttr ElementsAttrBuilder::expand(
    ElementsAttr elms, ArrayRef<int64_t> expandedShape) {
  ShapedType type = elms.getType();
  if (expandedShape == type.getShape())
    return elms;

  ElementsProperties props = getElementsProperties(elms);

  ShapedType expandedType = type.clone(expandedShape);
  auto expandedStrides = expandStrides(props.strides, expandedShape);
  return create(expandedType, props.bufferBType, expandedStrides, props.buffer,
      props.transformer);
}

namespace {
void splitImpl(ArrayRef<WideNum> data, size_t start, size_t len, size_t stride,
    MutableArrayRef<WideNum> splitData) {
  auto in = data.begin();
  auto out = splitData.begin();
  for (size_t offset = start; offset < data.size(); offset += stride)
    out = std::copy_n(in + offset, len, out);
  assert(out == splitData.end() && "result num elements mismatch");
}
} // namespace

std::vector<ElementsAttr> ElementsAttrBuilder::split(
    ElementsAttr elms, unsigned axis, ArrayRef<int64_t> sizes) {
  auto type = elms.getType();
  auto shape = type.getShape();
  assert(axis < shape.size());
  auto axisSize = shape[axis];
  assert(std::accumulate(sizes.begin(), sizes.end(), 0) == axisSize);
  if (sizes.empty()) {
    return {};
  }
  std::vector<ElementsAttr> results;
  results.reserve(sizes.size());
  if (sizes.size() == 1) {
    results.push_back(elms);
    return results;
  }

  ArrayBuffer<WideNum> data = getElementsWideNums(elms);
  size_t stride = ShapedType::getNumElements(shape.drop_front(axis));
  size_t substride = stride / axisSize;
  size_t offset = 0;
  SmallVector<int64_t, 4> splitShape(shape.begin(), shape.end());
  splitShape[axis] = 0; // Is set in every iteration.
  for (size_t i = 0; i < sizes.size(); ++i) {
    splitShape[axis] = sizes[i];
    ShapedType splitType = type.clone(splitShape);
    size_t len = sizes[i] * substride;
    ElementsAttr splitElms =
        fromWideNums(splitType, [&](MutableArrayRef<WideNum> splitData) {
          splitImpl(data.get(), offset, len, stride, splitData);
        });
    results.push_back(splitElms);
    offset += len;
  }
  return results;
}

ElementsAttr ElementsAttrBuilder::reduce(ElementsAttr elms,
    ArrayRef<unsigned> axes, bool keepdims,
    WideNum (*reducer)(WideNum, WideNum)) {
  if (axes.empty())
    return elms;

  SmallVector<unsigned, 4> sortedAxes(axes);
  std::sort(sortedAxes.begin(), sortedAxes.end());
  assert(
      std::unique(sortedAxes.begin(), sortedAxes.end()) == sortedAxes.end() &&
      "axes must be unique");

  ShapedType type = elms.getType();
  auto shape = type.getShape();

  SmallVector<int64_t, 4> strides;
  ArrayBuffer<WideNum> srcNums = getWideNumsAndStrides(elms, strides);
  SmallVector<int64_t, 4> axesShape, reducedShape;
  SmallVector<int64_t, 4> axesStrides, reducedStrides;
  auto it = sortedAxes.begin();
  for (unsigned axis = 0; axis < shape.size(); ++axis) {
    if (it != sortedAxes.end() && *it == axis) {
      axesShape.push_back(shape[axis]);
      axesStrides.push_back(strides[axis]);
      if (keepdims) {
        reducedShape.push_back(1);
        reducedStrides.push_back(0);
      }
      ++it;
    } else {
      reducedShape.push_back(shape[axis]);
      reducedStrides.push_back(strides[axis]);
    }
  }

  // StridedArrayRef and traverseStrides are used in an unusual way below.
  // (axesShape, axesStrides) on one hand and (reducedShape, reducedStrides)
  // on the other hand partition (shape, strides) into two partial mappings
  // of srcNums to the tensor shape.
  //
  // (axesShape, axesStrides) describes all the elements to reduce for each
  // result element, namely count == ShapedType::getNumElements(axesShape).
  // The outer traverseStrides "loop" runs count many times, each time
  // calculating the offset of the next element to reduce.
  // (Note that offsets be repeated if there are any zeros in axesStrides.)
  //
  // The inner traverseStrides loop reduces the next reduce element into the
  // result for each each element in the resulting reduced tensor (dstNums).
  StridedArrayRef<WideNum> axesStrided(srcNums.get(), axesStrides);
  StridedArrayRef<WideNum> reducedStrided(srcNums.get(), reducedStrides);
  ShapedType reducedType = type.clone(reducedShape);
  return fromWideNums(reducedType, [&](MutableArrayRef<WideNum> dstNums) {
    // First copy the 1st element to reduce to each result element in dstNums.
    WideNum *end = traverseStrides<WideNum *, WideNum>(reducedShape,
        dstNums.begin(), reducedStrided,
        [](WideNum *dst, const WideNum *src) { *dst = *src; });
    assert(end == dstNums.end() && "traverses every dstNums element");
    int64_t count = traverseStrides<int64_t, WideNum>(
        axesShape, 0, axesStrided, [&](int64_t counter, const WideNum *iter) {
          // Skip if counter == 0 as 1st element is already copied to dstNums.
          if (counter > 0) {
            ptrdiff_t offset = iter - axesStrided.begin();
            WideNum *end = traverseStrides<WideNum *, WideNum>(reducedShape,
                dstNums.begin(), reducedStrided,
                [&reducer, offset](WideNum *dst, const WideNum *src) {
                  *dst = reducer(*dst, *(src + offset));
                });
            assert(end == dstNums.end() && "traverses every dstNums element");
          }
        });
    assert(count == ShapedType::getNumElements(axesShape) &&
           "traverses all reduce axes");
  });
}

auto ElementsAttrBuilder::getElementsProperties(ElementsAttr elements) const
    -> ElementsProperties {
  static Transformer nullTransformer = nullptr;
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>()) {
    ArrayRef<int64_t> strides = disposable.getStrides();
    return {/*.bufferBType=*/disposable.getBufferBType(),
        /*.strides=*/{strides.begin(), strides.end()},
        /*.buffer=*/disposable.getBuffer(),
        /*.transformer=*/disposable.getTransformer()};
  } else if (auto dense = elements.dyn_cast<DenseElementsAttr>()) {
    ShapedType type = dense.getType();
    SmallVector<int64_t, 4> strides;
    if (dense.isSplat()) {
      strides = getSplatStrides(type.getShape());
    } else {
      strides = getDefaultStrides(type.getShape());
    }
    return {/*.bufferBType=*/btypeOfMlirType(type.getElementType()),
        /*.strides=*/{strides.begin(), strides.end()},
        /*.buffer=*/getMemoryBuffer(dense),
        /*.transformer=*/nullTransformer};
  }
  // TODO: consider supporting more ElementsAttr types
  llvm_unreachable("unexpected ElementsAttr instance");
}

ArrayBuffer<WideNum> ElementsAttrBuilder::getWideNumsAndExpandedStrides(
    ElementsAttr elms, llvm::ArrayRef<int64_t> expandedShape,
    llvm::SmallVectorImpl<int64_t> &expandedStrides) const {
  if (auto disposable = elms.dyn_cast<DisposableElementsAttr>()) {
    expandedStrides = expandStrides(disposable.getStrides(), expandedShape);
    return disposable.getBufferAsWideNums();
  } else if (elms.isSplat()) {
    expandedStrides = getSplatStrides(expandedShape);
    return ArrayBuffer<WideNum>::Vector(1, getElementsSplatWideNum(elms));
  } else {
    auto strides = getDefaultStrides(elms.getType().getShape());
    expandedStrides = expandStrides(strides, expandedShape);
    return getElementsWideNums(elms);
  };
}

ElementsAttr ElementsAttrBuilder::doTransform(
    ElementsAttr elms, Type transformedElementType, Transformer transformer) {
  ShapedType transformedType = elms.getType().clone(transformedElementType);

  ElementsProperties props = getElementsProperties(elms);

  return create(transformedType, props.bufferBType, props.strides, props.buffer,
      composeTransforms(props.transformer, std::move(transformer)));
}

ElementsAttr ElementsAttrBuilder::expandAndTransform(ElementsAttr elms,
    ShapedType expandedTransformedType, Transformer transformer) {
  ElementsProperties props = getElementsProperties(elms);

  auto expandedStrides =
      expandStrides(props.strides, expandedTransformedType.getShape());

  return create(expandedTransformedType, props.bufferBType, expandedStrides,
      props.buffer,
      composeTransforms(props.transformer, std::move(transformer)));
}

ElementsAttr ElementsAttrBuilder::fromRawBytes(
    ShapedType type, BType bufferBType, const Filler<char> &bytesFiller) {
  size_t size = type.getNumElements() * bytewidthOfBType(bufferBType);
  std::unique_ptr<llvm::WritableMemoryBuffer> writeBuffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(size);
  bytesFiller(writeBuffer->getBuffer());
  // We trust bytesFiller and skip testRawBytesValidityAndSplatness()
  return createWithDefaultStrides(type, bufferBType, std::move(writeBuffer));
}

ElementsAttr ElementsAttrBuilder::createWithDefaultStrides(ShapedType type,
    BType bufferBType, std::unique_ptr<llvm::MemoryBuffer> membuf) {
  auto strides = getDefaultStrides(type.getShape());
  return create(type, bufferBType, strides, std::move(membuf));
}

ElementsAttr ElementsAttrBuilder::create(ShapedType type, BType bufferBType,
    ArrayRef<int64_t> strides,
    const std::shared_ptr<llvm::MemoryBuffer> &buffer,
    Transformer transformer) {
  return disposablePool.createElementsAttr(
      type, bufferBType, strides, buffer, std::move(transformer));
}

} // namespace onnx_mlir
