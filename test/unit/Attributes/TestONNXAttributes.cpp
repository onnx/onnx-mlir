/*
 * SPDX-License-Identifier: Apache-2.0
 */

//============-- TestONNXAttributes.cpp - ONNXAttributes tests --=============//
//
// Tests DisposableElementsAttr.
//
// NOTE: AVERT YOUR EYES, THIS FILE IS A GARBAGE DUMP IN ITS CURRENT STATE
//
// TODO: CLEAN THIS UP
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/AttributesHelper.hpp"
#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"
#include "src/Support/DType.hpp"

#include "mlir/IR/Builders.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/SwapByteOrder.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

using namespace mlir;
using namespace onnx_mlir;

namespace {

typedef llvm::SmallVector<int64_t> Shape;

std::ostream &operator<<(std::ostream &os, const ArrayRef<int64_t> &v) {
  os << "(";
  for (auto i : v)
    os << i << ",";
  os << ")";
  return os;
}
inline raw_ostream &operator<<(raw_ostream &os, float_16 f16) {
  return os << "F16(" << f16.toFloat() << ")";
}
inline raw_ostream &operator<<(raw_ostream &os, bfloat_16 bf16) {
  return os << "BF16(" << bf16.toFloat() << ")";
}
inline raw_ostream &operator<<(raw_ostream &os, APFloat af) {
  return os << "APFloat(" << af.convertToDouble() << ")";
}
inline raw_ostream &operator<<(raw_ostream &os, onnx_mlir::WideNum n) {
  return os << "WideNum(i=" << n.i64 << ",u=" << n.u64 << ",f=" << n.dbl << ")";
}
inline raw_ostream &operator<<(raw_ostream &os, onnx_mlir::DType dtype) {
  return os << "DType(" << static_cast<int>(dtype) << ")";
}

MLIRContext *createCtx() {
  MLIRContext *ctx = new MLIRContext();
  ctx->loadDialect<ONNXDialect>();
  return ctx;
}

template <typename Dst = char>
ArrayRef<Dst> asArrayRef(StringRef s) {
  return llvm::makeArrayRef(
      reinterpret_cast<const Dst *>(s.data()), s.size() / sizeof(Dst));
}

template <typename Dst = char>
ArrayRef<Dst> asArrayRef(const llvm::MemoryBuffer &b) {
  return asArrayRef<Dst>(b.getBuffer());
}

template <typename T>
std::shared_ptr<llvm::MemoryBuffer> buffer(ArrayRef<T> data) {
  return std::shared_ptr<llvm::MemoryBuffer>(
      llvm::MemoryBuffer::getMemBufferCopy(asStringRef(data)));
}

class Test {
  MLIRContext *ctx;
  Location loc;
  OpBuilder builder;
  ElementsAttrBuilder elmsBuilder;
  Type F32;
  Type I32;
  Type I64;

public:
  Test()
      : ctx(createCtx()), loc(UnknownLoc::get(ctx)), builder(ctx),
        elmsBuilder(DisposablePool::create(ctx)) {
    F32 = builder.getF32Type();
    I32 = builder.getI32Type();
    I64 = builder.getI64Type();
  }
  ~Test() { delete ctx; }

  IntegerType getUInt(unsigned width) const {
    return IntegerType::get(ctx, width, IntegerType::Unsigned);
  }

  int test_dispatchByDType() {
    llvm::errs() << "test_dispatch_DTypeToken:\n";
    for (DType d = static_cast<DType>(0); d <= DType::MAX_DTYPE;
         d = static_cast<DType>(static_cast<int>(d) + 1)) {
      if (d == DType::UNDEFINED || d == DType::STRING ||
               d == DType::COMPLEX64 || d == DType::COMPLEX128)
        continue;
      if (isIntOrFloatDType(d)) {
        if (isFloatDType(d)) {
          assert(!isSignedIntDType(d));
          assert(!isUnsignedIntDType(d));
        } else {
          assert(isSignedIntDType(d) ^ isUnsignedIntDType(d));
        }
      }
      auto dty = dispatchByDType(d, [d](auto dtype) -> DType {
        using Q = DTypeTrait<dtype>;
        assert(isFloatDType(dtype) == Q::isFloat);
        assert(isIntOrFloatDType(dtype) == Q::isIntOrFloat);
        assert(isSignedIntDType(dtype) == Q::isSignedInt);
        assert(isUnsignedIntDType(dtype) == Q::isUnsignedInt);
        assert(sizeof(CppType<dtype>) == Q::bytewidth);
        assert(d == dtype);
        return dtype;
      });
      assert(dty == d);
    }
    return 0;
  }

  int test_float_16() {
    llvm::errs() << "test_float_16:\n";
    using cpptype = float_16;
    constexpr WideNum n = WideNum::from(toDType<cpptype>, true);
    assert(n.dbl == 1.0);
    float_16 f9984(9984);
    bfloat_16 fminus1(-1);
    float_16 bfminus1(fminus1);
    bfloat_16 bf9984(f9984);
    llvm::errs() << "float16 " << f9984.toFloat() << " as uint "
                 << f9984.bitcastToU16() << "\n";
    llvm::errs() << "float16 " << bfminus1.toFloat() << " as uint "
                 << bfminus1.bitcastToU16() << "\n";
    llvm::errs() << "bfloat16 " << bf9984.toFloat() << " as uint "
                 << bf9984.bitcastToU16() << "\n";
    llvm::errs() << "bfloat16 " << fminus1.toFloat() << " as uint "
                 << fminus1.bitcastToU16() << "\n";
    // assert(static_cast<bfloat_16>(bfminus1) == bf9984); // fails, == not
    // defined
    assert(bfminus1.toFloat() == fminus1.toFloat());
    assert(static_cast<float_16>(bf9984).toFloat() ==
           static_cast<bfloat_16>(f9984).toFloat());
    constexpr float_16 f16z = float_16();
    constexpr bfloat_16 bf16z = bfloat_16();
    constexpr float_16 f16z2 = f16z;
    constexpr bfloat_16 bf16z2 = bf16z;
    constexpr uint16_t f16zu = f16z2.bitcastToU16();
    constexpr uint16_t bf16zu = bf16z2.bitcastToU16();
    constexpr DType df16 = toDType<decltype(f16z)>;
    constexpr DType dbf16 = toDType<decltype(bf16z)>;
    assert(df16 == toDType<float_16>);
    assert(dbf16 == toDType<bfloat_16>);
    assert((std::is_same_v<CppType<df16>, float_16>));
    assert((std::is_same_v<CppType<dbf16>, bfloat_16>));
    assert((std::is_same_v<CppType<toDType<float>>, float>));
    llvm::errs() << "float16 " << f16z.toFloat() << " as uint " << f16zu
                 << ", dtype=" << df16 << "\n";
    llvm::errs() << "bfloat16 " << bf16z.toFloat() << " as uint " << bf16zu
                 << ", dtype=" << dbf16 << "\n";
    return 0;
  }

  int test_DType() {
    llvm::errs() << "test_DType:\n";
    uint64_t u;
    int8_t i = -128;
    u = i;
    llvm::errs() << "-128i8 as u64 " << u << "\n";
    llvm::errs() << "static_cast<u64>(-128i8) " << static_cast<uint64_t>(i)
                 << "\n";
    assert(CppTypeTrait<float>::isFloat);
    return 0;
  }

  int test_WideNum() {
    llvm::errs() << "test_WideNum:\n";
    constexpr WideNum nf = WideNum::from(DType::DOUBLE, 42.0);
    llvm::errs() << "nf " << nf << "\n";
    llvm::errs() << "nf as APFloat " << nf.toAPFloat(DType::DOUBLE) << "\n";
    constexpr int64_t i = -42;
    constexpr WideNum ni = WideNum::from(DType::INT64, i);
    llvm::errs() << "ni " << ni << "\n";
    llvm::errs() << "ni as APInt " << ni.toAPInt(DType::INT64) << "\n";
    constexpr uint64_t u = 1ULL << 63;
    constexpr WideNum nu = WideNum::from(DType::UINT64, u);
    llvm::errs() << "nu " << nu << "\n";
    llvm::errs() << "nu as APInt " << nu.toAPInt(DType::UINT64) << "\n";
    constexpr bool b = true;
    constexpr WideNum nb = WideNum::from(DType::UINT64, b);
    constexpr bool b3 = nb.to<bool>(DType::BOOL);
    llvm::errs() << "b3 " << b3 << "\n";
    return 0;
  }

  int test_ElementsAttrBuilder() {
    llvm::errs() << "test_ElementsAttrBuilder:\n";
    ShapedType type = RankedTensorType::get({1}, getUInt(1));
    auto dispo = elmsBuilder.create(type, buffer<bool>({true}));
    assert(dispo.isSplat());
    return 0;
  }

  int test_makeDense() {
    llvm::errs() << "test_makeDense:\n";
#if 0
    ShapedType type = RankedTensorType::get({2}, builder.getF32Type());
    auto b = buffer<float>({42.0f, 42.0f});

    auto eCopy =
        makeElementsAttrFromRawBytes(type, asArrayRef(*b), /*mustCopy=*/true);
    llvm::errs() << "eCopy " << eCopy << "\n";
    assert(eCopy.isa<DisposableElementsAttr>());
    // auto dCopy = eCopy.cast<DisposableElementsAttr>();
    // assert(dCopy.getBuffer()->getBuffer().data() != b->getBuffer().data());

    auto e =
        makeElementsAttrFromRawBytes(type, asArrayRef(*b), /*mustCopy=*/false);
    assert(e.isa<DisposableElementsAttr>());
    // auto d = e.cast<DisposableElementsAttr>();
    // assert(d.getBuffer()->getBuffer().data() == b->getBuffer().data());
#endif
    return 0;
  }

  int test_splat() {
    llvm::errs() << "test_splat:\n";
    ShapedType type = RankedTensorType::get({1}, builder.getF32Type());
    Attribute a = elmsBuilder.create(type, buffer<float>({4.2}));
    assert(a);
    assert(a.isa<ElementsAttr>());
    ElementsAttr e = a.cast<ElementsAttr>();
    assert(a.isa<DisposableElementsAttr>());
    DisposableElementsAttr i = a.cast<DisposableElementsAttr>();
    llvm::errs() << "as DisposableElementsAttr " << i << "\n";
    llvm::errs() << "as ElementsAttr " << e << "\n";
    llvm::errs() << "as Attribute " << a << "\n";
    assert(e.isSplat());
    llvm::errs() << "splat value " << i.getSplatValue<float>() << "\n";
    assert(fabs(i.getSplatValue<float>() - 4.2) < 1e-6);
    auto b = i.value_begin<float>();
    auto x = *b;
    llvm::errs() << "x " << x << "\n";
    auto f = i.getSplatValue<APFloat>();
    assert(fabs(f.convertToDouble() - 4.2) < 1e-6);
    auto d = toDenseElementsAttr(i);
    d = toDenseElementsAttr(i);
    llvm::errs() << "as DenseElementsAttr " << d << "\n";
    return 0;
  }

  int test_f16() {
    llvm::errs() << "test_f16:\n";
    assert(fabs(float_16::fromFloat(4.2).toFloat() - 4.2) < 1e-3);
    ShapedType type = RankedTensorType::get({1}, builder.getF16Type());
    Attribute a =
        elmsBuilder.create(type, buffer<float_16>({float_16::fromFloat(4.2)}));
    assert(a);
    assert(a.isa<ElementsAttr>());
    ElementsAttr e = a.cast<ElementsAttr>();
    assert(a.isa<DisposableElementsAttr>());
    DisposableElementsAttr i = a.cast<DisposableElementsAttr>();
    assert(e.isSplat());
    llvm::errs() << "splat value " << i.getSplatValue<float>() << "\n";
    assert(fabs(i.getSplatValue<float>() - 4.2) < 1e-3);
    auto b = i.value_begin<float>();
    auto x = *b;
    llvm::errs() << "x " << x << "\n";
    auto d = toDenseElementsAttr(i);
    d = toDenseElementsAttr(i);
    llvm::errs() << "as DenseElementsAttr " << d << "\n";
    return 0;
  }

  int test_bool() {
    llvm::errs() << "test_bool:\n";
    ShapedType type = RankedTensorType::get({1}, getUInt(1));
    Attribute a = elmsBuilder.create(type, buffer<bool>({true}));
    assert(a);
    assert(a.isa<ElementsAttr>());
    ElementsAttr e = a.cast<ElementsAttr>();
    assert(a.isa<DisposableElementsAttr>());
    DisposableElementsAttr i = a.cast<DisposableElementsAttr>();
    assert(e.isSplat());
    llvm::errs() << "splat value " << i.getSplatValue<bool>() << "\n";
    assert(i.getSplatValue<bool>());
    auto b = i.value_begin<bool>();
    auto x = *b;
    llvm::errs() << "x " << x << "\n";
    auto d = toDenseElementsAttr(i);
    d = toDenseElementsAttr(i);
    llvm::errs() << "as DenseElementsAttr " << d << "\n";
    return 0;
  }

  int test_attributes() {
    llvm::errs() << "test_attributes:\n";
    ShapedType type = RankedTensorType::get({2}, getUInt(64));
    Attribute a;
    a = elmsBuilder.create(type, buffer<uint64_t>({7, 9}));
    assert(a);
    assert(a.isa<DisposableElementsAttr>());
    DisposableElementsAttr i = a.cast<DisposableElementsAttr>();
    auto d = toDenseElementsAttr(i);
    d = toDenseElementsAttr(a.cast<DisposableElementsAttr>());
    llvm::errs() << "as DisposableElementsAttr " << i << "\n";
    llvm::errs() << "as DenseElementsAttr " << d << "\n";
    llvm::errs() << "as Attribute " << a << "\n";

    ShapedType t = i.getType();
    llvm::errs() << "type:" << t << "\n";
    std::cerr << "shape:" << t.getShape() << "\n";
    assert(i.isa<ElementsAttr>());
    assert(!i.isSplat());
    assert(succeeded(i.getValuesImpl(TypeID::get<uint64_t>())));
    // assert(i.try_value_begin<uint64_t>());
    auto begin = i.value_begin<uint64_t>();
    auto end = i.value_end<uint64_t>();
    assert(begin != end);
    assert(begin == i.getValues<uint64_t>().begin());
    assert(end == i.getValues<uint64_t>().end());
    auto x = *begin;
    llvm::errs() << "x " << x << "\n";
    assert(*begin == 7);
    assert(*(end - 1) == 9);
    assert(*--end == 9);
    std::cerr << "next:" << *++begin << "\n";
    // assert(succeeded(i.tryGetValues<uint64_t>()));
    for (auto v : i.getValues<uint64_t>())
      std::cerr << "ivalue:" << v << "\n";
    assert(i.cast<ElementsAttr>().try_value_begin<uint64_t>());
    std::cerr << "empty:" << i.empty() << "\n";

    auto apbegin = i.value_begin<APInt>();
    auto api = *apbegin;
    assert(api.getZExtValue() == 7);

    // crashes because ints refuse to cast to APFloat
    // auto apfbegin = i.value_begin<APFloat>();

    // iteration by attributes is not supported
    // auto atbegin = i.value_begin<IntegerAttr>(); // crashes

    ElementsAttr e = i; // i.cast<ElementsAttr>();
    t = e.getType();
    assert(!e.isSplat());
    assert(t);
    llvm::errs() << "type:" << t << "\n";
    assert(succeeded(e.getValuesImpl(TypeID::get<uint64_t>())));
    assert(e.try_value_begin<uint64_t>());
    std::cerr << "*e.try_value_begin():" << (**e.try_value_begin<uint64_t>())
              << "\n";
    auto it = *e.try_value_begin<uint64_t>();
    std::cerr << "++*e.try_value_begin():" << *++it << "\n";
    for (auto it = e.tryGetValues<uint64_t>()->begin(),
              en = e.tryGetValues<uint64_t>()->end();
         it != en; ++it)
      std::cerr << "evalue:" << *it << "\n";
    auto vs = e.tryGetValues<uint64_t>();
    for (auto v : *vs) // we crash here, why?
      std::cerr << "evalue:" << v << "\n";
    // for (auto v : *e.tryGetValues<uint64_t>()) // we crash here, why?
    //   std::cerr << "evalue:" << v << "\n";

    return 0;
  }

  template <typename T, T... ints>
  std::vector<T> nums(std::integer_sequence<T, ints...> int_seq) {
    std::vector<T> v;
    (v.push_back(ints), ...);
    return v;
  }

  int test_transpose() {
    llvm::errs() << "test_transpose:\n";
    ShapedType type = RankedTensorType::get({2, 3, 5}, getUInt(8));
    auto elms = nums<uint8_t>(std::make_integer_sequence<uint8_t, 30>{});
    auto e = elmsBuilder.create(type, buffer<uint8_t>(elms));
    std::cerr << "before transpose " << e.getShape();
    for (auto x : e.getValues<uint8_t>())
      std::cerr << " " << unsigned(x);
    std::cerr << "\n";
    auto et = elmsBuilder.transpose(e, {1, 2, 0});
    std::cerr << "after  transpose " << et.getShape();
    for (auto x : et.getValues<uint8_t>())
      std::cerr << " " << unsigned(x);
    std::cerr << "\n";
    return 0;
  }

  int test_cast() {
    llvm::errs() << "test_cast:\n";
    ShapedType type = RankedTensorType::get({1}, I64);
    auto e = elmsBuilder.create(type, buffer<int64_t>({256}));
    std::cerr << "before cast " << e.getShape();
    for (auto x : e.getValues<int64_t>())
      std::cerr << " " << x;
    std::cerr << "\n";
    auto ec = elmsBuilder.castElementType(e, F32);
    std::cerr << "after  cast " << ec.getShape();
    for (auto x : ec.getValues<float>())
      std::cerr << " " << x;
    std::cerr << "\n";
    return 0;
  }
};

} // namespace

int main(int argc, char *argv[]) {
  Test test;
  int failures = 0;
  failures += test.test_dispatchByDType();
  failures += test.test_float_16();
  failures += test.test_DType();
  failures += test.test_WideNum();
  failures += test.test_ElementsAttrBuilder();
  failures += test.test_makeDense();
  failures += test.test_splat();
  failures += test.test_f16();
  failures += test.test_bool();
  failures += test.test_attributes();
  failures += test.test_transpose();
  failures += test.test_cast();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
