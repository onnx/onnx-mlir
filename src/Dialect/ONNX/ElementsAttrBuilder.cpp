/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===----------------------- ElementsAttrBuilder.hpp ----------------------===//
//
// Builds DisposableElementsAttr instances.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttrBuilder.hpp"

#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/DisposablePool.hpp"
#include "src/Support/Strides.hpp"
#include "src/Support/TypeUtilities.hpp"

using namespace mlir;

namespace onnx_mlir {

namespace {

// Returns whether isSplat. Fails assert or llvm_unreachable if invalid.
bool testBoolsValidityAndSplatness(ArrayRef<char> bytes) {
  for (char c : bytes)
    assert(c == 0 || c == 1);
  return bytes.size() == 1;
}

// Returns whether isSplat. Fails assert or llvm_unreachable if invalid.
bool testRawBytesValidityAndSplatness(
    ShapedType type, BType bufferBType, ArrayRef<char> bytes) {
  BType btype = btypeOfMlirType(type.getElementType());
  assert(wideBTypeOfBType(btype) == wideBTypeOfBType(bufferBType));
  if (bufferBType == BType::BOOL) {
    size_t numElements = type.getNumElements();
    assert(bytes.size() == numElements || bytes.size() == 1);
    return testBoolsValidityAndSplatness(bytes);
  }
  ShapedType bufferType =
      btype == bufferBType
          ? type
          : type.clone(mlirTypeOfBType(bufferBType, type.getContext()));
  bool isSplat;
  if (!DenseElementsAttr::isValidRawBuffer(bufferType, bytes, isSplat))
    llvm_unreachable("invalid dense int or fps raw buffer");
  return isSplat;
}

std::unique_ptr<llvm::MemoryBuffer> getMemoryBuffer(DenseElementsAttr dense) {
  ShapedType type = dense.getType();
  if (type.isInteger(1)) {
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
    StringRef s = asStringRef(dense.getRawData());
    int64_t size = s.size();
    if (dense.isSplat())
      assert(size == getEltSizeInBytes(type) && "size mismatch");
    else
      assert(size == getSizeInBytes(type) && "size mismatch");
    return llvm::MemoryBuffer::getMemBuffer(
        s, /*BufferName=*/"", /*RequiresNullTerminator=*/false);
  }
}

} // namespace

struct ElementsAttrBuilder::ElementsProperties {
  BType bufferBType;
  SmallVector<int64_t, 4> strides;
  std::shared_ptr<llvm::MemoryBuffer> buffer;
  const Transformer &transformer;
};

ElementsAttrBuilder::ElementsAttrBuilder(MLIRContext *context)
    : disposablePool(*DisposablePool::get(context)) {}

DisposableElementsAttr ElementsAttrBuilder::fromMemoryBuffer(
    ShapedType type, std::unique_ptr<llvm::MemoryBuffer> membuf) {
  BType btype = btypeOfMlirType(type.getElementType());
  return createWithDefaultStrides(type, btype, std::move(membuf));
}

DisposableElementsAttr ElementsAttrBuilder::toDisposableElementsAttr(
    ElementsAttr elements) {
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>())
    return disposable;
  if (auto dense = elements.dyn_cast<DenseElementsAttr>()) {
    ElementsProperties props = getElementsProperties(dense);
    return create(dense.getType(), props.bufferBType, props.strides,
        props.buffer, props.transformer);
  }
  // TODO: consider supporting more ElementsAttr types
  llvm_unreachable("unexpected ElementsAttr instance");
}

DisposableElementsAttr ElementsAttrBuilder::fromWideNums(
    ShapedType type, const Filler<WideNum> &wideDataFiller) {
  BType bufferBType = wideBTypeOfBType(btypeOfMlirType(type.getElementType()));
  return fromRawBytes(
      type, bufferBType, [&wideDataFiller](MutableArrayRef<char> bytes) {
        wideDataFiller(castMutableArrayRef<WideNum>(bytes));
      });
}

/*static*/
ElementsAttrBuilder::Transformer ElementsAttrBuilder::functionTransformer(
    WideNum (*fun)(WideNum)) {
  return [fun = std::move(fun)](MutableArrayRef<WideNum> data) -> void {
    for (WideNum &n : data)
      n = fun(n);
  };
}

namespace {
ElementsAttrBuilder::Transformer composeTransforms(
    ElementsAttrBuilder::Transformer first,
    ElementsAttrBuilder::Transformer second) {
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

ElementsAttr ElementsAttrBuilder::transform(
    ElementsAttr elms, Type transformedElementType, Transformer transformer) {
  ShapedType transformedType = elms.getType().clone(transformedElementType);

  ElementsProperties props = getElementsProperties(elms);

  return create(transformedType, props.bufferBType, props.strides, props.buffer,
      composeTransforms(props.transformer, std::move(transformer)));
}

ElementsAttr ElementsAttrBuilder::combine(ElementsAttr lhsElms,
    ElementsAttr rhsElms, ShapedType combinedType,
    WideNum (*combiner)(WideNum, WideNum)) {
  // TODO: Use getElementsProperties to avoid the construction of a
  //       DisposableElementsAttr if lhs or rhs is a DenseElementsAttr.
  DisposableElementsAttr lhs = toDisposableElementsAttr(lhsElms);
  DisposableElementsAttr rhs = toDisposableElementsAttr(rhsElms);

  auto combinedShape = combinedType.getShape();
  auto lhsStrides = expandStrides(lhs.getStrides(), combinedShape);
  auto rhsStrides = expandStrides(rhs.getStrides(), combinedShape);

  using Transformer = ElementsAttrBuilder::Transformer;
  auto composeTransformerFunction = [](Transformer transformer,
                                        auto fun) -> Transformer {
    return [transformer = std::move(transformer), fun = std::move(fun)](
               MutableArrayRef<WideNum> data) -> void {
      if (transformer != nullptr)
        transformer(data);
      for (WideNum &n : data)
        n = fun(n);
    };
  };

  if (lhs.isSplat()) {
    WideNum lhsNum = lhs.atFlatIndex(0);
    return create(combinedType, rhs.getBufferBType(), rhsStrides,
        rhs.getBuffer(),
        composeTransformerFunction(rhs.getTransformer(),
            [lhsNum, combiner](WideNum n) { return combiner(lhsNum, n); }));
  }

  if (rhs.isSplat()) {
    WideNum rhsNum = rhs.atFlatIndex(0);
    return create(combinedType, lhs.getBufferBType(), lhsStrides,
        lhs.getBuffer(),
        composeTransformerFunction(lhs.getTransformer(),
            [rhsNum, combiner](WideNum n) { return combiner(n, rhsNum); }));
  }

  ArrayBuffer<WideNum> lhsNums = lhs.getBufferAsWideNums();
  ArrayBuffer<WideNum> rhsNums = rhs.getBufferAsWideNums();
  StridedArrayRef<WideNum> stridedLhs(lhsNums.get(), lhsStrides);
  StridedArrayRef<WideNum> stridedRhs(rhsNums.get(), rhsStrides);
  return fromWideNums(combinedType, [&](MutableArrayRef<WideNum> dstNums) {
    mapStrides<WideNum, WideNum, WideNum>(
        combinedShape, dstNums, stridedLhs, stridedRhs, combiner);
  });
}

namespace {
template <typename SrcT, typename DstT>
struct Caster {
  static inline constexpr DstT eval(SrcT src) { return static_cast<DstT>(src); }
};

template <typename SrcT, typename DstT>
void wideCaster(MutableArrayRef<WideNum> nums) {
  for (WideNum &n : nums)
    n = WideNum::WrappedFunction<Caster<SrcT, DstT>>::eval(n);
}

ElementsAttrBuilder::Transformer wideCaster(BType src, BType dst) {
  constexpr BType DBL = BType::DOUBLE, I64 = BType::INT64, U64 = BType::UINT64;
  // clang-format off
  if (src == DBL && dst == I64) return wideCaster<double, int64_t>;
  if (src == DBL && dst == U64) return wideCaster<double, uint64_t>;
  if (src == I64 && dst == DBL) return wideCaster<int64_t, double>;
  if (src == I64 && dst == U64) return wideCaster<int64_t, uint64_t>;
  if (src == U64 && dst == DBL) return wideCaster<uint64_t, double>;
  if (src == U64 && dst == I64) return wideCaster<uint64_t, int64_t>;
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

  auto transformer = oldWideType == newWideType
                         ? props.transformer
                         : composeTransforms(props.transformer,
                               wideCaster(oldWideType, newWideType));
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
          reshapeStrides(shape, props.strides, reshapedShape)) {
    return create(reshapedType, props.bufferBType, *reshapedStrides,
        props.buffer, props.transformer);
  } else {
    assert(!elms.isa<DenseElementsAttr>() &&
           "reshapeStrides() always succeeds for DenseElementsAttr default or "
           "splat strides");
  }

  auto disp = elms.cast<DisposableElementsAttr>();

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
      strides.assign(type.getRank(), 0);
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

DisposableElementsAttr ElementsAttrBuilder::fromRawBytes(
    ShapedType type, BType bufferBType, ArrayRef<char> bytes) {
  std::unique_ptr<llvm::MemoryBuffer> membuf =
      llvm::MemoryBuffer::getMemBufferCopy(asStringRef(bytes));
  return testRawBytesValidityAndSplatness(type, bufferBType, bytes)
             ? createSplat(type, bufferBType, std::move(membuf))
             : createWithDefaultStrides(type, bufferBType, std::move(membuf));
}

DisposableElementsAttr ElementsAttrBuilder::fromRawBytes(
    ShapedType type, BType bufferBType, const Filler<char> &bytesFiller) {
  size_t size = type.getNumElements() * bytewidthOfBType(bufferBType);
  std::unique_ptr<llvm::WritableMemoryBuffer> writeBuffer =
      llvm::WritableMemoryBuffer::getNewUninitMemBuffer(size);
  bytesFiller(writeBuffer->getBuffer());
  // We trust bytesFiller and skip testRawBytesValidityAndSplatness()
  return createWithDefaultStrides(type, bufferBType, std::move(writeBuffer));
}

DisposableElementsAttr ElementsAttrBuilder::createWithDefaultStrides(
    ShapedType type, BType bufferBType,
    std::unique_ptr<llvm::MemoryBuffer> membuf) {
  auto strides = getDefaultStrides(type.getShape());
  return create(type, bufferBType, strides, std::move(membuf));
}

DisposableElementsAttr ElementsAttrBuilder::createSplat(ShapedType type,
    BType bufferBType, std::unique_ptr<llvm::MemoryBuffer> membuf) {
  SmallVector<int64_t, 4> zerosStrides(type.getRank(), 0);
  return create(type, bufferBType, zerosStrides, std::move(membuf));
}

DisposableElementsAttr ElementsAttrBuilder::create(ShapedType type,
    BType bufferBType, ArrayRef<int64_t> strides,
    const std::shared_ptr<llvm::MemoryBuffer> &buffer,
    Transformer transformer) {
  return disposablePool.createDisposableElementsAttr(
      type, bufferBType, strides, buffer, std::move(transformer));
}

} // namespace onnx_mlir
