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

} // namespace

ElementsAttrBuilder::ElementsAttrBuilder(MLIRContext *context)
    : disposablePool(*DisposablePool::get(context)) {}

DisposableElementsAttr ElementsAttrBuilder::fromMemoryBuffer(
    ShapedType type, std::unique_ptr<llvm::MemoryBuffer> membuf) {
  BType btype = btypeOfMlirType(type.getElementType());
  return createWithDefaultStrides(type, btype, std::move(membuf));
}

DisposableElementsAttr ElementsAttrBuilder::fromElementsAttr(
    ElementsAttr elements) {
  if (auto disposable = elements.dyn_cast<DisposableElementsAttr>())
    return disposable;
  if (auto dense = elements.dyn_cast<DenseElementsAttr>()) {
    ShapedType type = dense.getType();
    BType btype = btypeOfMlirType(type.getElementType());
    if (btype == BType::BOOL) {
      // Don't use dense.rawData() which is bit packed, whereas
      // DisposableElementsAttr represents bools with one byte per bool value.
      if (dense.isSplat()) {
        char b = dense.getSplatValue<bool>();
        return fromRawBytes(type, btype, llvm::makeArrayRef(b));
      } else {
        return fromRawBytes(type, btype, [dense](MutableArrayRef<char> dst) {
          std::copy_n(
              dense.value_begin<bool>(), dense.getNumElements(), dst.begin());
        });
      }
    } else {
      StringRef s = asStringRef(dense.getRawData());
      std::unique_ptr<llvm::MemoryBuffer> membuf =
          llvm::MemoryBuffer::getMemBuffer(
              s, /*BufferName=*/"", /*RequiresNullTerminator=*/false);
      return dense.isSplat()
                 ? createSplat(type, btype, std::move(membuf))
                 : createWithDefaultStrides(type, btype, std::move(membuf));
    }
  }
  // TODO: consider supporting more ElementsAttr types
  llvm_unreachable("unexpected ElementsAttr instance");
}

DisposableElementsAttr ElementsAttrBuilder::fromWideNums(
    ShapedType type, const Filler<WideNum> &wideDataFiller) {
  BType bufferBType = wideBTypeOfBType(btypeOfMlirType(type.getElementType()));
  return fromRawBytes(
      type, bufferBType, [&wideDataFiller](llvm::MutableArrayRef<char> bytes) {
        wideDataFiller(castMutableArrayRef<WideNum>(bytes));
      });
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

DisposableElementsAttr ElementsAttrBuilder::transform(
    DisposableElementsAttr elms, Type transformedElementType,
    Transformer transformer) {
  ShapedType transformedType = elms.getType().clone(transformedElementType);
  return create(transformedType, elms.getBufferBType(), elms.getStrides(),
      elms.getBuffer(),
      composeTransforms(elms.getTransformer(), std::move(transformer)));
}

namespace {
template <BType SRC_TAG, BType DST_TAG>
void wideCaster(MutableArrayRef<WideNum> nums) {
  using S = WideBType<SRC_TAG>;
  using D = WideBType<DST_TAG>;
  for (WideNum &n : nums)
    n = D::pack(static_cast<typename D::type>(S::unpack(n)));
}

ElementsAttrBuilder::Transformer wideCaster(BType src, BType dst) {
  constexpr BType DBL = BType::DOUBLE, I64 = BType::INT64, U64 = BType::UINT64;
  // clang-format off
  if (src == DBL && dst == I64) return wideCaster<DBL, I64>;
  if (src == DBL && dst == U64) return wideCaster<DBL, U64>;
  if (src == I64 && dst == DBL) return wideCaster<I64, DBL>;
  if (src == I64 && dst == U64) return wideCaster<I64, U64>;
  if (src == U64 && dst == DBL) return wideCaster<U64, DBL>;
  if (src == U64 && dst == I64) return wideCaster<U64, I64>;
  // clang-format on
  llvm_unreachable("wideCaster must be called with 2 different wide types");
}
} // namespace

DisposableElementsAttr ElementsAttrBuilder::castElementType(
    DisposableElementsAttr elms, Type newElementType) {
  if (newElementType == elms.getElementType())
    return elms;

  ShapedType newType = elms.getType().clone(newElementType);
  BType newBType = btypeOfMlirType(newElementType);
  BType newWideType = wideBTypeOfBType(newBType);
  BType oldWideType = wideBTypeOfBType(elms.getBType());

  auto transformer = oldWideType == newWideType
                         ? elms.getTransformer()
                         : composeTransforms(elms.getTransformer(),
                               wideCaster(oldWideType, newWideType));
  return create(newType, elms.getBufferBType(), elms.getStrides(),
      elms.getBuffer(), std::move(transformer));
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

DisposableElementsAttr ElementsAttrBuilder::transpose(
    DisposableElementsAttr elms, ArrayRef<uint64_t> perm) {
  if (isIdentityPermutation(perm))
    return elms;

  auto transposedShape = transposeDims(elms.getShape(), perm);
  ShapedType transposedType = elms.getType().clone(transposedShape);
  auto transposedStrides = transposeDims(elms.getStrides(), perm);
  return create(transposedType, elms.getBufferBType(), transposedStrides,
      elms.getBuffer(), elms.getTransformer());
}

DisposableElementsAttr ElementsAttrBuilder::reshape(
    DisposableElementsAttr elms, ArrayRef<int64_t> reshapedShape) {
  if (reshapedShape == elms.getShape())
    return elms;

  ShapedType reshapedType = elms.getType().clone(reshapedShape);
  if (auto reshapedStrides =
          reshapeStrides(elms.getShape(), elms.getStrides(), reshapedShape)) {
    return create(reshapedType, elms.getBufferBType(), *reshapedStrides,
        elms.getBuffer(), elms.getTransformer());
  }

  if (!elms.isTransformed()) { // Skip WideNums absent element-wise transform.
    return fromRawBytes(
        reshapedType, elms.getBufferBType(), [elms](MutableArrayRef<char> dst) {
          auto src = elms.getBufferBytes();
          restrideArray(elms.getBufferElementBytewidth(), elms.getShape(),
              elms.getStrides(), src, dst);
        });
  }

  return fromWideNums(reshapedType, [elms](MutableArrayRef<WideNum> wideData) {
    elms.readWideNums(wideData);
  });
}

DisposableElementsAttr ElementsAttrBuilder::expand(
    DisposableElementsAttr elms, ArrayRef<int64_t> expandedShape) {
  if (expandedShape == elms.getShape())
    return elms;

  ShapedType expandedType = elms.getType().clone(expandedShape);
  auto expandedStrides = expandStrides(elms.getStrides(), expandedShape);
  return create(expandedType, elms.getBufferBType(), expandedStrides,
      elms.getBuffer(), elms.getTransformer());
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

mlir::DisposableElementsAttr ElementsAttrBuilder::createSplat(ShapedType type,
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
