/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ElementsAttrBuilder.cpp ----------------------===//
//
// Builds DisposableElementsAttr instances.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrBuilder.hpp"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Threading.h"
#include "llvm/ADT/STLExtras.h"

#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposablePool.hpp"
#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrHelper.hpp"
#include "src/Dialect/ONNX/ElementsAttr/Strides.hpp"
#include "src/Dialect/ONNX/ElementsAttr/StridesRange.hpp"
#include "src/Support/TypeUtilities.hpp"

#include <algorithm>
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
  if (auto disposable = mlir::dyn_cast<DisposableElementsAttr>(elements))
    return disposable;
  if (auto dense = mlir::dyn_cast<DenseElementsAttr>(elements)) {
    if (!disposablePool.isActive())
      return nullptr;
    ElementsProperties props = getElementsProperties(dense);
    ElementsAttr created = create(dense.getType(), props.bufferBType,
        props.strides, props.buffer, props.transformer);
    // Check for race condition where disposablePool became inactive since we
    // checked, in which case it returns a DenseElementsAttr which we don't
    // want.
    if (auto disposable = mlir::dyn_cast<DisposableElementsAttr>(created))
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
  if (auto disposable = mlir::dyn_cast<DisposableElementsAttr>(elements))
    return disposable.toDenseElementsAttr();
  if (auto dense = mlir::dyn_cast<DenseElementsAttr>(elements))
    return dense;
  // TODO: consider supporting more ElementsAttr types
  llvm_unreachable("unexpected ElementsAttr instance");
}

/*static*/
bool ElementsAttrBuilder::equal(ElementsAttr lhs, ElementsAttr rhs) {
  auto lhsType = lhs.getShapedType();
  auto rhsType = rhs.getShapedType();
  auto elementType = lhsType.getElementType();
  assert(elementType == rhsType.getElementType() &&
         "equal() requires identical element types");

  SmallVector<int64_t> combinedShape;
  if (!OpTrait::util::getBroadcastedShape(
          lhsType.getShape(), rhsType.getShape(), combinedShape))
    llvm_unreachable("equal() requires broadcast compatible shapes");

  SmallVector<int64_t, 4> xpLhsStrides;
  ArrayBuffer<WideNum> lhsNums =
      getWideNumsAndExpandedStrides(lhs, combinedShape, xpLhsStrides);

  SmallVector<int64_t, 4> xpRhsStrides;
  ArrayBuffer<WideNum> rhsNums =
      getWideNumsAndExpandedStrides(rhs, combinedShape, xpRhsStrides);

  StridesRange<2> range(combinedShape, {xpLhsStrides, xpRhsStrides});
  return dispatchByBType(btypeOfMlirType(elementType), [&](auto btype) {
    using cpptype = CppType<btype>;
    return llvm::all_of(range, [&](StridesIndexOffsets<2> idxoffs) {
      constexpr BType TAG = toBType<cpptype>;
      return lhsNums.get()[idxoffs[0]].narrow<TAG>() ==
             rhsNums.get()[idxoffs[1]].narrow<TAG>();
    });
  });
}

/*static*/
bool ElementsAttrBuilder::allEqual(
    ElementsAttr lhs, WideNum broadcastedRhsValue) {
  WideNum n = broadcastedRhsValue;
  return dispatchByBType(
      btypeOfMlirType(lhs.getElementType()), [lhs, n](auto btype) {
        using cpptype = CppType<btype>;
        auto nEquals = [n](cpptype x) {
          constexpr BType TAG = toBType<cpptype>;
          return n.narrow<TAG>() == x;
        };
        if (auto disposable = mlir::dyn_cast<DisposableElementsAttr>(lhs)) {
          if (disposable.isTransformedOrCast()) {
            ArrayBuffer<WideNum> nums = disposable.getBufferAsWideNums();
            return llvm::all_of(nums.get(), [n](WideNum m) {
              constexpr BType TAG = toBType<cpptype>;
              return n.narrow<TAG>() == m.narrow<TAG>();
            });
          } else {
            auto values = castArrayRef<cpptype>(disposable.getBufferBytes());
            return llvm::all_of(values, nEquals);
          }
        } else if (auto dense = mlir::dyn_cast<DenseElementsAttr>(lhs)) {
          if (dense.isSplat()) {
            cpptype x = dense.getSplatValue<cpptype>();
            return nEquals(x);
          } else {
            auto values = dense.getValues<cpptype>();
            return llvm::all_of(values, nEquals);
          }
        }
        // TODO: consider supporting more ElementsAttr types
        llvm_unreachable("unexpected ElementsAttr instance");
      });
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

  SmallVector<int64_t, 4> xpRhsStrides;
  ArrayBuffer<WideNum> rhsNums =
      getWideNumsAndExpandedStrides(rhs, combinedShape, xpRhsStrides);

  return fromWideNums(combinedType, [&](MutableArrayRef<WideNum> dstNums) {
    // dstNums, lhsNums, rhsNums are accessed via raw pointers dst, lhs/rhsSrc
    // because otherwise the ArrayRef range checks slow down the inner loop.
    WideNum *dst = dstNums.data();
    const WideNum *lhsSrc = lhsNums.get().data();
    const WideNum *rhsSrc = rhsNums.get().data();
    for (auto &idxoffs :
        StridesRange<2>(combinedShape, {xpLhsStrides, xpRhsStrides})) {
      dst[idxoffs.flattenedIndex] =
          combiner(lhsSrc[idxoffs[0]], rhsSrc[idxoffs[1]]);
    }
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

  SmallVector<int64_t, 4> xpRhsStrides;
  ArrayBuffer<WideNum> rhsNums =
      getWideNumsAndExpandedStrides(rhs, combinedShape, xpRhsStrides);

  return fromWideNums(combinedType, [&](MutableArrayRef<WideNum> dstNums) {
    // Copy cond into dstNums with broadcast.
    restrideArray<WideNum>(
        combinedShape, xpCondStrides, condNums.get(), dstNums);

    // dstNums, lhsNums, rhsNums are accessed via raw pointers dst, lhs/rhsSrc
    // because otherwise the ArrayRef range checks slow down the inner loop.
    WideNum *dst = dstNums.data();
    const WideNum *lhsSrc = lhsNums.get().data();
    const WideNum *rhsSrc = rhsNums.get().data();
    for (auto &idxoffs :
        StridesRange<2>(combinedShape, {xpLhsStrides, xpRhsStrides})) {
      WideNum &res = dst[idxoffs.flattenedIndex];
      res = res.u64 ? lhsSrc[idxoffs[0]] : rhsSrc[idxoffs[1]];
    }
  });
}

ElementsAttr ElementsAttrBuilder::castElementType(
    ElementsAttr elms, Type newElementType) {
  if (auto ftype = dyn_cast<FloatType>(newElementType)) {
    // TODO: Consider saturating when ftype has no infinity:
    //       saturate=APFloat::getInf(ftype.getFloatSemantics()).isNaN()
    return castToFPElementType(elms, ftype);
  }
  if (auto itype = dyn_cast<IntegerType>(newElementType)) {
    return castToIntElementType(elms, itype);
  }
  llvm_unreachable("unsupported newElementType");
}

namespace {

// Rounds (ties to even) and saturates (out of range numbers become MIN or MAX).
// Returns zero if from is NaN, like llvm::APFloat::convertToInteger().
// From must be a floating point type (double, float, float_16, float_8e5m2).
// To must be an integer type with size <= size(long), i.e., bitwidth <= 64.
//
// TODO: consider making it configurable whether to convert NaN to
//       number farthest from zero (like X86 SSE)
//       or just highest bit set (like CUDA) or zero
//
// TODO: optimize w/X86 SSE instructions https://stackoverflow.com/a/47347224
//
template <bool TRUNCATE, typename TO>
TO convertIntFromDouble(double from, TO min, TO max) {
  if (std::isnan(from))
    return 0;
  if (from < static_cast<double>(min))
    return min;
  // static_cast<double>(max)) can round to a larger number
  // so return max if from is greater or equal, not just if greater
  if (from >= max)
    return max;

  if (TRUNCATE)
    return static_cast<TO>(from);

  // llrint recommendation: https://stackoverflow.com/a/47347224
  // rounds to nearest, ties to even, in the default rounding mode
  using llrintType = decltype(llrint(from));
  if constexpr (std::is_same_v<TO, uint64_t>) {
    static_assert(
        sizeof(llrintType) >= sizeof(TO), "insufficient llrint range");
    // llrintType is int64_t which doesn't cover the numeric range of uint64_t
    // so we work around this by breaking the range into 2 as follows:
    uint64_t mid = uint64_t(1) << 63; // middle of uint64_t numeric range
    if (from < mid) {
      // from is inside llrint's numerical range [-2^63, 2^63)
      return llrint(from);
    } else {
      // subtract and add to translate into and llrint's numeric range and back
      return mid + llrint(from - mid);
    }
  } else {
    // llrintType covers the numeric range of TO, namely llrintType is int64_t
    // and TO is int64_t or a narrower signed or unsigned type
    static_assert(sizeof(llrintType) > sizeof(TO) ||
                      (sizeof(llrintType) == sizeof(TO) &&
                          std::numeric_limits<TO>::is_signed),
        "insufficient llrint range");
    return llrint(from);
  }
}

template <bool TRUNCATE, typename TO>
auto convertIntFromFP(TO min, TO max) {
  return [min, max](WideNum n) -> WideNum {
    double from = n.narrow<BType::DOUBLE>();
    TO to = convertIntFromDouble<TRUNCATE, TO>(from, min, max);
    return WideNum::widen<toBType<TO>>(to);
  };
}

template <typename FROM>
WideNum isWideNonZero(WideNum n) {
  return WideNum::widen<BType::BOOL>(n.narrow<toBType<FROM>>() != 0);
}

template <typename TO, typename FROM>
WideNum wideCast(WideNum n) {
  return WideNum::widen<toBType<TO>>(
      static_cast<TO>(n.narrow<toBType<FROM>>()));
};

template <typename FROM>
double wideToDouble(WideNum n) {
  return static_cast<double>(n.narrow<toBType<FROM>>());
};

} // namespace

ElementsAttr ElementsAttrBuilder::castToIntElementType(
    ElementsAttr elms, IntegerType newElementType, bool round) {
  Type oldElementType = elms.getElementType();
  if (newElementType == oldElementType)
    return elms;

  Transformer transformer;
  if (newElementType.isInteger(1)) {
    // Bool: +/-zero cast to 0, everything else including NaN cast to 1.
    transformer = wideZeroDispatchNonBool(oldElementType, [&](auto wideZero) {
      using cpptype = decltype(wideZero);
      return functionTransformer(isWideNonZero<cpptype>);
    });
  } else if (isa<FloatType>(oldElementType)) {
    constexpr bool ROUND = false, TRUNCATE = true;
    unsigned width = newElementType.getWidth();
    if (newElementType.isUnsigned()) {
      uint64_t min = 0;
      uint64_t max = std::numeric_limits<uint64_t>::max() >> (64 - width);
      transformer = round ? functionTransformer(
                                convertIntFromFP<ROUND, uint64_t>(min, max))
                          : functionTransformer(
                                convertIntFromFP<TRUNCATE, uint64_t>(min, max));
    } else {
      int64_t min = std::numeric_limits<int64_t>::min() >> (64 - width);
      int64_t max = std::numeric_limits<int64_t>::max() >> (64 - width);
      transformer = round ? functionTransformer(
                                convertIntFromFP<ROUND, int64_t>(min, max))
                          : functionTransformer(
                                convertIntFromFP<TRUNCATE, int64_t>(min, max));
    }
  } else if (isa<IntegerType>(oldElementType)) {
    // We assume that casts to other integer types don't intend to truncate the
    // numeric range and we delay any truncation until the data is read and
    // allow the untruncated numbers as inputs to any further transformations.
    //
    // TODO: Add configuration options to support other behaviors.
    //       See https://github.com/onnx/onnx-mlir/issues/2209
    if (newElementType.isUnsigned() != oldElementType.isUnsignedInteger()) {
      // DisposableElementsAttr requires transformation between integers with
      // different signs.
      // TODO: Consider relaxing the requirement and omit this transformation.
      transformer = newElementType.isUnsigned()
                        ? functionTransformer(wideCast<uint64_t, int64_t>)
                        : functionTransformer(wideCast<int64_t, uint64_t>);
    } else {
      ElementsProperties props = getElementsProperties(elms);
      ShapedType newType = elms.getShapedType().clone(newElementType);
      return create(newType, props.bufferBType, props.strides, props.buffer,
          props.transformer);
    }
  } else {
    llvm_unreachable("unsupported element type");
  }
  return doTransform(elms, newElementType, transformer);
}

ElementsAttr ElementsAttrBuilder::castToFPElementType(
    ElementsAttr elms, FloatType newElementType, bool saturate) {
  Type oldElementType = elms.getElementType();
  if (newElementType == oldElementType)
    return elms;

  return wideZeroDispatchNonBool(oldElementType, [&](auto wideZero) {
    using cpptype = decltype(wideZero);
    Transformer transformer;
    if (saturate) {
      // Smallest is -max for all ONNX fp types.
      const double max = APFloat::getLargest(newElementType.getFloatSemantics())
                             .convertToDouble();
      // Note that we saturate by clipping which isn't 100% faithful to the
      // onnx spec here: https://onnx.ai/onnx/technical/float8.html
      // and here: https://github.com/onnx/onnx/blob/main/docs/Operators.md#Cast
      // which, in the case of E4M3FNUZ and E5M2FNUZ, requires infinite values
      // to saturate to NaN, whereas we saturate them to smallest/largest with
      // clipping. Our clipping implementation matches the reference
      // implementation in onnx/reference/ops/op_cast.py.
      // See https://github.com/onnx/onnx-mlir/issues/2369
      //
      // TODO: Change implementation to match the spec, or change the spec.
      transformer = functionTransformer([max](WideNum n) {
        double d = wideToDouble<cpptype>(n);
        return WideNum::widen<BType::DOUBLE>(
            // Order of operations is important to ensure NaN stays NaN:
            d <= -max ? -max : (d >= max ? max : d));
      });
    } else if constexpr (std::is_integral_v<cpptype>) {
      transformer = functionTransformer([](WideNum n) {
        return WideNum::widen<BType::DOUBLE>(wideToDouble<cpptype>(n));
      });
    } else {
      ElementsProperties props = getElementsProperties(elms);
      ShapedType newType = elms.getShapedType().clone(newElementType);
      return create(newType, props.bufferBType, props.strides, props.buffer,
          props.transformer);
    }
    return doTransform(elms, newElementType, transformer);
  });
}

ElementsAttr ElementsAttrBuilder::clip(
    ElementsAttr elms, WideNum min, WideNum max) {
  return wideZeroDispatchNonBool(elms.getElementType(), [&](auto wideZero) {
    using cpptype = decltype(wideZero);
    return transform(elms, elms.getElementType(), [min, max](WideNum n) {
      constexpr BType TAG = toBType<cpptype>;
      cpptype x = n.narrow<TAG>();
      if (x < min.narrow<TAG>())
        return min;
      if (x > max.narrow<TAG>())
        return max;
      return n;
    });
  });
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

  ShapedType type = elms.getShapedType();
  auto transposedShape = transposeDims(type.getShape(), perm);
  ShapedType transposedType = type.clone(transposedShape);
  auto transposedStrides = transposeDims(props.strides, perm);
  return create(transposedType, props.bufferBType, transposedStrides,
      props.buffer, props.transformer);
}

ElementsAttr ElementsAttrBuilder::reshape(
    ElementsAttr elms, ArrayRef<int64_t> reshapedShape) {
  ShapedType type = elms.getShapedType();
  auto shape = type.getShape();
  if (reshapedShape == shape)
    return elms;

  ElementsProperties props = getElementsProperties(elms);

  ShapedType reshapedType = type.clone(reshapedShape);
  if (auto reshapedStrides =
          reshapeStrides(shape, props.strides, reshapedShape))
    return create(reshapedType, props.bufferBType, *reshapedStrides,
        props.buffer, props.transformer);

  auto disp = mlir::dyn_cast<DisposableElementsAttr>(elms);
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
  ShapedType type = elms.getShapedType();
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
  auto type = elms.getShapedType();
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

ElementsAttr ElementsAttrBuilder::concat(
    ArrayRef<ElementsAttr> elms, unsigned axis) {
  assert(elms.size() >= 1 && "concat tensors must be non-empty");

  ElementsAttr first = elms.front();
  ArrayRef<int64_t> firstShape = first.getShapedType().getShape();
  size_t rank = firstShape.size();
  assert(axis < rank && "concat axis out of range");
  if (elms.size() == 1)
    return first;

  // Check elms are compatible and construct outShape:
  SmallVector<int64_t> outShape(firstShape);
  for (size_t i = 1; i < elms.size(); ++i) {
    ElementsAttr next = elms[i];
    assert(next.getElementType() == first.getElementType() &&
           "concat tensors element types must agree");
    ArrayRef<int64_t> nextShape = next.getShapedType().getShape();
    assert(nextShape.size() == rank && "concat tensors ranks must agree");
    for (unsigned a = 0; a < rank; ++a) {
      assert((a == axis || nextShape[a] == firstShape[a]) &&
             "concat tensors shapes must agree except for the concat axis");
    }
    outShape[axis] += nextShape[axis];
  }

  ShapedType outType = first.getShapedType().clone(outShape);
  return fromWideNums(outType, [&](MutableArrayRef<WideNum> dst) {
    auto postAxisShape = ArrayRef(outShape).drop_front(axis + 1);
    size_t postAxisNumElements = ShapedType::getNumElements(postAxisShape);
    // A "block" fixes the axes before axis and iterates over the others.
    size_t outBlockLen = outShape[axis] * postAxisNumElements;
    size_t start = 0;
    for (ElementsAttr input : elms) {
      ArrayRef<int64_t> inputShape = input.getShapedType().getShape();
      size_t inputBlockLen = inputShape[axis] * postAxisNumElements;
      SmallVector<int64_t> strides;
      ArrayBuffer<WideNum> src = getWideNumsAndStrides(input, strides);
      StridesRange<1> range(inputShape, {strides});
      auto it = range.begin();
      for (size_t offset = start; offset < dst.size(); offset += outBlockLen) {
        for (size_t pos = 0; pos < inputBlockLen; ++pos) {
          dst[offset + pos] = src.get()[it->at(0)];
          ++it;
        }
      }
      assert(it == range.end() && "input num elements mismatch");
      start += inputBlockLen;
    }
  });
}

ElementsAttr ElementsAttrBuilder::slice(ElementsAttr elms,
    ArrayRef<int64_t> shape, ArrayRef<int64_t> starts,
    ArrayRef<int64_t> steps) {
  ShapedType outType = elms.getShapedType().clone(shape);
  return fromWideNums(outType, [&](MutableArrayRef<WideNum> dst) {
    SmallVector<int64_t> strides;
    ArrayBuffer<WideNum> src = getWideNumsAndStrides(elms, strides);
    const WideNum *start = src.get().begin();
    for (size_t axis = 0; axis < shape.size(); ++axis) {
      start += starts[axis] * strides[axis];
      strides[axis] *= steps[axis];
    }
    for (auto &idxoffs : StridesRange<1>(shape, {strides}))
      dst[idxoffs.flattenedIndex] = *(start + idxoffs[0]);
  });
}

namespace {
ElementsAttr splat(ShapedType type, WideNum num) {
  Type elemType = type.getElementType();
  BType btype = btypeOfMlirType(elemType);
  if (isFloatBType(btype))
    return DenseElementsAttr::get(type, num.toAPFloat(btype));
  if (isIntBType(btype))
    return DenseElementsAttr::get(type, num.toAPInt(btype));
  llvm_unreachable("unsupported element type");
}
} // namespace

ElementsAttr ElementsAttrBuilder::pad(
    ElementsAttr elms, ArrayRef<int64_t> pads, WideNum padValue) {
  ArrayRef<int64_t> inputShape = elms.getShapedType().getShape();
  size_t rank = inputShape.size();
  if (rank == 0)
    return elms;
  ArrayRef<int64_t> leftPads = pads.take_front(rank);
  ArrayRef<int64_t> rightPads = pads.take_back(rank);
  SmallVector<int64_t> outShape(inputShape);
  for (size_t axis = 0; axis < rank; ++axis)
    outShape[axis] += leftPads[axis] + rightPads[axis];
  ShapedType outType = elms.getShapedType().clone(outShape);
  if (elms.empty())
    return splat(outType, padValue);
  return fromWideNums(outType, [&](MutableArrayRef<WideNum> dst) {
    // The following unrolls the recurse(dst.begin(), elms.getValues())
    // recursive algorithm described by the pseudo code:
    //
    //   def recurse(DstIter &out, SrcIter &inp, int axis = 0):
    //     auto subSize = numElements(outShape.drop_front(axis + 1))
    //     out = std::fill_n(out, leftPads[axis] * subSize)
    //     if axis == rank - 1:
    //       nunCols = inputShape.back()
    //       out = std::copy(inp, inp += numCols, out)
    //     else:
    //       for i in range(inputShape[axis]):
    //         recurse(out, inp, axis + 1)
    //     out = std::fill_n(out, rightPads[axis] * subSize)

    SmallVector<int64_t> betweenRowsPads(rank, 0);
    int64_t beginPad = 0, endPad = 0;
    int64_t multiplier = 1;
    for (int64_t axis = rank - 1; axis >= 0; --axis) {
      beginPad += leftPads[axis] * multiplier;
      endPad += rightPads[axis] * multiplier;
      betweenRowsPads[axis] = beginPad + endPad;
      multiplier *= outShape[axis];
    }

    auto out = dst.begin();
    out = std::fill_n(out, beginPad, padValue);

    SmallVector<int64_t> strides;
    ArrayBuffer<WideNum> src = getWideNumsAndStrides(elms, strides);
    StridesRange<1> range(inputShape, {strides});
    auto it = range.begin(), end = range.end();
    assert(it != end && "elms must be non-empty");
    assert(!inputShape.empty() && "elms must have rank > 0");
    const int numCols = inputShape.back();
    for (;;) {
      // Copy next row from elms to dst.
      for (int64_t col = 0; col < numCols; ++col, ++out, ++it) {
        *out = src.get()[it->at(0)];
      }

      if (it == end)
        break;

      assert(it->index.back() == 0 && "it is at the start of a row");
      int lastZero = rank - 1;
      while (lastZero > 0 && it->index[lastZero - 1] == 0)
        --lastZero;
      out = std::fill_n(out, betweenRowsPads[lastZero], padValue);
    }

    out = std::fill_n(out, endPad, padValue);
    assert(out == dst.end());
  });
}

ElementsAttr ElementsAttrBuilder::gather(
    ElementsAttr input, ElementsAttr indices, unsigned axis) {
  ShapedType inputType = input.getShapedType();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  assert(axis < inputShape.size() && "gather axis out of range");
  auto postAxisShape = inputShape.drop_front(axis + 1);
  ShapedType indicesType = indices.getShapedType();
  assert(indicesType.getElementType().isSignlessInteger() &&
         "gather indices must be i32 or i64");
  ArrayRef<int64_t> indicesShape = indicesType.getShape();
  SmallVector<int64_t> outShape(inputShape.take_front(axis));
  outShape.append(indicesShape.begin(), indicesShape.end());
  outShape.append(postAxisShape.begin(), postAxisShape.end());
  auto outType = inputType.clone(outShape);
  return fromWideNums(outType, [&](MutableArrayRef<WideNum> dst) {
    size_t postAxisNumElements = ShapedType::getNumElements(postAxisShape);
    ArrayBuffer<WideNum> src = getElementsWideNums(input);
    // Convert indices of any signed int element type to int64 by
    // first promoting to WideNum and then casting to int64.
    // In practice we support both int32 and int64 in this way.
    ArrayBuffer<WideNum> indicesWideNums = getElementsWideNums(indices);
    ArrayRef<int64_t> indicesArray =
        castArrayRef<int64_t, WideNum>(indicesWideNums.get());
    size_t axisInputSize = inputShape[axis];
    size_t inputBlockLen = axisInputSize * postAxisNumElements;
    size_t outBlockLen = indicesArray.size() * postAxisNumElements;
    size_t start = 0;
    WideNum *out = dst.begin();
    for (int64_t idx : indicesArray) {
      int64_t adjustedIdx = idx < 0 ? idx + axisInputSize : idx;
      const WideNum *in = src.get().begin() + adjustedIdx * postAxisNumElements;
      for (size_t offset = start; offset < dst.size(); offset += outBlockLen) {
        std::copy_n(in, postAxisNumElements, out + offset);
        in += inputBlockLen;
      }
      start += postAxisNumElements;
    }
  });
}

ElementsAttr ElementsAttrBuilder::scatterND(
    ElementsAttr input, ElementsAttr indices, ElementsAttr updates) {
  return fromWideNums(input.getShapedType(), [&](MutableArrayRef<WideNum> dst) {
    // numpy implementation:
    //
    //   dst = np.copy(input)
    //   outer = indices.shape[:-1]
    //   for idx in np.ndindex(outer):
    //     dst[indices[idx]] = updates[idx]

    ArrayRef<int64_t> inputShape = input.getShapedType().getShape();
    ArrayRef<int64_t> indicesShape = indices.getShapedType().getShape();
    ArrayRef<int64_t> updatesShape = updates.getShapedType().getShape();

    int64_t indices_nd = indicesShape.back();
    auto outer = indicesShape.drop_back();
    int64_t n_slices = ShapedType::getNumElements(outer);
    int64_t slice_size =
        ShapedType::getNumElements(updatesShape.drop_front(outer.size()));
    SmallVector<int64_t, 4> inputStrides = getDefaultStrides(inputShape);
    auto sliceStrides = llvm::ArrayRef(inputStrides).take_front(indices_nd);

    readElementsWideNums(input, dst);
    SmallVector<int64_t> updatesStrides;
    ArrayBuffer<WideNum> updatesData =
        getWideNumsAndStrides(updates, updatesStrides);
    StridesRange<1> updatesRange(updatesShape, {updatesStrides});
    auto updatesIter = updatesRange.begin();
    ArrayBuffer<int64_t> indicesBuffer = getElementsArray<int64_t>(indices);
    const int64_t *indicesIter = indicesBuffer.get().begin();
    for (int64_t i = 0; i < n_slices; ++i) {
      ArrayRef<uint64_t> idxs =
          castArrayRef<uint64_t>(ArrayRef(indicesIter, indices_nd));
      int64_t pos = getStridesPosition(idxs, sliceStrides);
      for (int64_t j = 0; j < slice_size; ++j) {
        dst[pos] = updatesData.get()[updatesIter->at(0)];
        ++pos;
        ++updatesIter;
      }
      indicesIter += indices_nd;
    }
    assert(
        updatesIter == updatesRange.end() && "updates num elements mismatch");
  });
}

ElementsAttr ElementsAttrBuilder::reduce(ElementsAttr elms,
    ArrayRef<unsigned> axes, bool keepdims,
    WideNum (*reducer)(WideNum, WideNum)) {
  assert(!elms.empty());
  if (axes.empty())
    return elms;

  Type elementType = elms.getElementType();
  MLIRContext *ctx = elementType.getContext();
  SmallVector<unsigned, 4> sortedAxes(axes);
  std::sort(sortedAxes.begin(), sortedAxes.end());
  assert(
      std::unique(sortedAxes.begin(), sortedAxes.end()) == sortedAxes.end() &&
      "axes must be unique");

  ShapedType type = elms.getShapedType();
  auto shape = type.getShape();

  SmallVector<int64_t, 4> strides;
  ArrayBuffer<WideNum> srcNums = getWideNumsAndStrides(elms, strides);

  // axesShape and axesStrides describe the src elements that reduce together
  // into one dst element.
  // reducedShape and reducedStrides describe the mapping from src to dst
  // for the first src element that reduces to each dst element.
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

  ShapedType reducedType = type.clone(reducedShape);
  return fromWideNums(reducedType, [&](MutableArrayRef<WideNum> dstNums) {
    StridesRange<1> sRange(reducedShape, {reducedStrides});
    StridesRange<1> axesRange(axesShape, {axesStrides});
    SmallVector<std::pair<int64_t, uint64_t>, 4> batch;
    for (auto &idxoffs : sRange)
      batch.emplace_back(std::make_pair(idxoffs.flattenedIndex, idxoffs[0]));

    auto fetchBatch = [&](size_t threadNumber, bool parallel) {
      // retrun all data without spliting for sequential execution.
      if (!parallel)
        return llvm::make_range(batch.begin(), batch.end());
      // Each thread fetches the same batch size. The leftovers are set in the
      // threads with small thread number.
      size_t tileSize = floor(batch.size() / ctx->getNumThreads());
      size_t leftovers = batch.size() % ctx->getNumThreads();
      int beginOffset;
      if (threadNumber < leftovers) {
        // for the first few threads, it is as if the block size is larger by 1.
        tileSize++;
        beginOffset = threadNumber * tileSize;
      } else {
        // for the last threads, its as we shift the start by leftovers.
        beginOffset = threadNumber * tileSize + leftovers;
      }
      int endOffset = beginOffset + tileSize;
      return llvm::make_range(
          batch.begin() + beginOffset, batch.begin() + endOffset);
    };

    auto work = [&](size_t threadNumber, bool parallel = true) {
      auto tile = fetchBatch(threadNumber, parallel);
      // Traverse and populate each element d in dstNums.
      for (auto b : tile) {
        WideNum &d = dstNums[b.first];
        int64_t srcPos = b.second;
        // Traverse all the elements that reduce together into d.
        // srcNums elements may be repeated if there are zeros in axesStrides.
        auto axesIter = axesRange.begin();
        auto axesEnd = axesRange.end();
        assert(axesIter->at(0) == 0 && "initial src offset must be zero");
        d = srcNums.get()[srcPos];
        while (++axesIter != axesEnd) {
          int64_t srcOffset = axesIter->at(0);
          d = reducer(d, srcNums.get()[srcPos + srcOffset]);
        }
      }
    };
    // Using 'parallelFor()' introduces large overhead. Followings are actual
    // measurement results on IBM z16 to decide the 'minCount'. We measured
    // 'onnx.ReduceSum()' in 'test/mlir/onnx/onnx_constprop_parallel.mlir' using
    // several input size. From these results, we decided to use 2000 as the
    // 'minCount'.
    //
    // inputCounts|Sequential  | Parallel with 2 threads
    //            | (work())   | (parallelFor())
    //            | (msec)     | (msec)
    // --------------------------------------------------
    //    400     |   0.065    |   0.153
    //    800     |   0.115    |   0.164
    //    1200    |   0.175    |   0.201
    //    1600    |   0.226    |   0.228
    //    2000    |   0.282    |   0.258
    //    2400    |   0.336    |   0.284
    constexpr size_t minCount = 2000;
    size_t inputCount = batch.size() * axesRange.size();
    if (inputCount < minCount)
      work(0, /*parallel*/ false);
    else
      parallelFor(ctx, 0, ctx->getNumThreads(), work);
  });
}

ElementsAttr ElementsAttrBuilder::matMul(ElementsAttr lhs, ElementsAttr rhs) {
  ShapedType lhsType = lhs.getShapedType();
  ShapedType rhsType = rhs.getShapedType();
  Type elementType = lhsType.getElementType();
  assert(elementType == rhsType.getElementType() &&
         "matMul() requires identical element types");
  assert(!elementType.isInteger(1) && "matMul() elements must be numbers");

  ArrayRef<int64_t> lhsShape = lhsType.getShape();
  size_t lhsRank = lhsShape.size();

  ArrayRef<int64_t> rhsShape = rhsType.getShape();
  size_t rhsRank = rhsShape.size();

  assert(lhsRank >= 1);
  assert(rhsRank >= 1);
  // If lhs is 1-D with dim size K then it's treated as a 1xK matrix,
  // otherwise we refer to the last two dimension sizes of lhs as MxK.
  // If rhs is 1-D with dim size K then it's treated as a Kx1 matrix
  // otherwise we refer to the last two dimension sizes of rhs as KxN.
  int64_t M = lhsRank == 1 ? 1 : lhsShape[lhsRank - 2];
  int64_t N = rhsRank == 1 ? 1 : rhsShape[rhsRank - 1];
  int64_t K = lhsShape[lhsRank - 1];
  assert(K == rhsShape[rhsRank == 1 ? 0 : (rhsRank - 2)]);

  // MatMul is similar to Reduce, because a MatMul is the same as broadcasts
  // followed by element wise multiplication and then ReduceSum on the
  // reduction axis. The implementation is is similar to Reduce:
  // An outer loop runs over the elements of the result tensor and, for
  // each result element, the dot product of a LHS row and a RHS column is
  // computed in an inner loop.
  //
  // matMulShape is the result shape before the M or N axes are collapsed in
  // the cases where lhs or rhs is a vector (has rank 1), it is used to
  // drive the iteration in the outer loop.
  //
  // lhsRedStrides/rhsRedStrides are LHS/RHS strides for the outer loop
  // ("Red" is short for "Reduced").
  //
  // matMulAxisLhsStride/matMulAxisRhsStride are the inner loop strides.

  ArrayRef<int64_t> lhsBatchShape = lhsShape.drop_back(lhsRank == 1 ? 1 : 2);
  ArrayRef<int64_t> rhsBatchShape = rhsShape.drop_back(rhsRank == 1 ? 1 : 2);
  SmallVector<int64_t> combinedBatchShape;
  if (!OpTrait::util::getBroadcastedShape(
          lhsBatchShape, rhsBatchShape, combinedBatchShape))
    llvm_unreachable("matMul() requires broadcast compatible batch shapes");
  SmallVector<int64_t> matMulShape = combinedBatchShape;
  matMulShape.push_back(M);
  matMulShape.push_back(N);
  size_t matMulRank = matMulShape.size();

  SmallVector<int64_t> xpLhsShape = combinedBatchShape;
  if (lhsRank != 1)
    xpLhsShape.push_back(M);
  xpLhsShape.push_back(K);
  SmallVector<int64_t> lhsRedStrides;
  ArrayBuffer<WideNum> lhsNums =
      getWideNumsAndExpandedStrides(lhs, xpLhsShape, lhsRedStrides);
  if (lhsRank == 1)
    lhsRedStrides.insert(lhsRedStrides.end() - 1, 0);
  assert(lhsRedStrides.size() == matMulRank);
  // Record the LHS stride on the MatMul reduction axis and then clear it in
  // the strides so it is ignored during the outer reduction loop below.
  // (The MatMul reduction axis stride is used in the inner reduction loop.)
  int64_t matMulAxisLhsStride = lhsRedStrides[matMulRank - 1];
  lhsRedStrides[matMulRank - 1] = 0;

  SmallVector<int64_t> xpRhsShape = combinedBatchShape;
  xpRhsShape.push_back(K);
  if (rhsRank != 1)
    xpRhsShape.push_back(N);
  SmallVector<int64_t> rhsRedStrides;
  ArrayBuffer<WideNum> rhsNums =
      getWideNumsAndExpandedStrides(rhs, xpRhsShape, rhsRedStrides);
  if (rhsRank == 1)
    rhsRedStrides.push_back(0);
  assert(rhsRedStrides.size() == matMulRank);
  // Record the RHS stride on the MatMul reduction axis and then clear it in
  // the strides so it is ignored during the outer reduction loop below.
  // (The MatMul reduction axis stride is used in the inner reduction loop.)
  int64_t matMulAxisRhsStride = rhsRedStrides[matMulRank - 2];
  rhsRedStrides[matMulRank - 2] = 0;

  SmallVector<int64_t> resultShape = combinedBatchShape;
  if (lhsRank != 1)
    resultShape.push_back(M);
  if (rhsRank != 1)
    resultShape.push_back(N);
  ShapedType resultType = lhsType.clone(resultShape);
  return fromWideNums(resultType, [&](MutableArrayRef<WideNum> dstNums) {
    wideZeroDispatchNonBool(elementType, [&](auto wideZero) {
      using cpptype = decltype(wideZero);
      constexpr BType TAG = toBType<cpptype>;
      // Traverse and populate each element d in dstNums.
      for (auto &idxoffs :
          StridesRange<2>(matMulShape, {lhsRedStrides, rhsRedStrides})) {
        // Traverse all the elements that reduce together into d.
        // srcNums elements may be repeated if there are zeros in axesStrides.
        cpptype accumulator = 0;
        int64_t lhsPos = idxoffs[0];
        int64_t rhsPos = idxoffs[1];
        for (int64_t i = 0; i < K; ++i) {
          accumulator += lhsNums.get()[lhsPos].narrow<TAG>() *
                         rhsNums.get()[rhsPos].narrow<TAG>();
          lhsPos += matMulAxisLhsStride;
          rhsPos += matMulAxisRhsStride;
        }
        dstNums[idxoffs.flattenedIndex] = WideNum::widen<TAG>(accumulator);
      }
    });
  });
}

ElementsAttr ElementsAttrBuilder::range(
    ShapedType resultType, WideNum start, WideNum delta) {
  return fromWideNums(resultType, [&](MutableArrayRef<WideNum> dstNums) {
    wideZeroDispatchNonBool(resultType.getElementType(), [&](auto wideZero) {
      using cpptype = decltype(wideZero);
      constexpr BType TAG = toBType<cpptype>;
      // Traverse and populate each element d in dstNums.
      cpptype x = start.narrow<TAG>();
      for (auto &d : dstNums) {
        d = WideNum::widen<TAG>(x);
        x += delta.narrow<TAG>();
      }
    });
  });
}

namespace {
// Returns indices with non-zero values. The indices are placed back to back,
// lexicographically sorted. Can be viewed as a [count, rank] shaped matrix
// linearized in row-major order, where rank is the rank of elms' shape and
// count is the number of non-zero values. AP_TYPE should be APFloat or APInt.
// TODO: If this is too slow then reimplement in the style of allEqual()
//       with WideNum instead of APFloat/APInt.
template <typename AP_TYPE>
SmallVector<int64_t> nonZeroIndices(ElementsAttr elms) {
  SmallVector<int64_t> indices;
  auto values = elms.getValues<AP_TYPE>();
  for (const auto &idxpos :
      StridesRange<0>(elms.getShapedType().getShape(), {})) {
    if (!values[idxpos.flattenedIndex].isZero())
      indices.append(idxpos.index.begin(), idxpos.index.end());
  }
  return indices;
}
} // namespace

ElementsAttr ElementsAttrBuilder::nonZero(ElementsAttr elms) {
  SmallVector<int64_t> indices = isa<FloatType>(elms.getElementType())
                                     ? nonZeroIndices<APFloat>(elms)
                                     : nonZeroIndices<APInt>(elms);
  int64_t rank = elms.getShapedType().getRank();
  assert(indices.size() % rank == 0);
  int64_t count = indices.size() / rank;
  Type I64 = IntegerType::get(elms.getContext(), 64);
  // Return transposition from shape [count, rank] to [rank, count].
  auto nonZeroType = RankedTensorType::get({rank, count}, I64);
  return fromArray<int64_t>(nonZeroType, [&](MutableArrayRef<int64_t> dst) {
    for (int64_t i = 0; i < count; ++i) {
      for (int64_t j = 0; j < rank; ++j)
        dst[j * count + i] = indices[i * rank + j];
    }
  });
}

/*static*/
auto ElementsAttrBuilder::getElementsProperties(ElementsAttr elements)
    -> ElementsProperties {
  static Transformer nullTransformer = nullptr;
  if (auto disposable = mlir::dyn_cast<DisposableElementsAttr>(elements)) {
    ArrayRef<int64_t> strides = disposable.getStrides();
    return {/*.bufferBType=*/disposable.getBufferBType(),
        /*.strides=*/{strides.begin(), strides.end()},
        /*.buffer=*/disposable.getBuffer(),
        /*.transformer=*/disposable.getTransformer()};
  } else if (auto dense = mlir::dyn_cast<DenseElementsAttr>(elements)) {
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

/*static*/
ArrayBuffer<WideNum> ElementsAttrBuilder::getWideNumsAndExpandedStrides(
    ElementsAttr elms, llvm::ArrayRef<int64_t> expandedShape,
    llvm::SmallVectorImpl<int64_t> &expandedStrides) {
  if (auto disposable = mlir::dyn_cast<DisposableElementsAttr>(elms)) {
    expandedStrides = expandStrides(disposable.getStrides(), expandedShape);
    return disposable.getBufferAsWideNums();
  } else if (elms.isSplat()) {
    expandedStrides = getSplatStrides(expandedShape);
    return ArrayBuffer<WideNum>::Vector(1, getElementsSplatWideNum(elms));
  } else {
    auto strides = getDefaultStrides(elms.getShapedType().getShape());
    expandedStrides = expandStrides(strides, expandedShape);
    return getElementsWideNums(elms);
  };
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
} // namespace

ElementsAttr ElementsAttrBuilder::doTransform(
    ElementsAttr elms, Type transformedElementType, Transformer transformer) {
  ShapedType transformedType =
      elms.getShapedType().clone(transformedElementType);

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
