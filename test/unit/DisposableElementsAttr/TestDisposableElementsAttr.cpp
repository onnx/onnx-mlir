/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===================-- TestDisposableElementsAttr.cpp --=====================//
//
// Tests DisposableElementsAttr.
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

bool near(double a, double b) { return fabs(a - b) < 1e-6; }

template <typename CPPTY>
bool eq(CPPTY a, CPPTY b) {
  if constexpr (isFP16Type<CPPTY>)
    return a.toFloat() == b.toFloat();
  else
    return a == b;
}

bool forAllDTypes(std::function<bool(DType)> predicate) {
  bool result = true;
  for (DType d = static_cast<DType>(0); d <= DType::MAX_DTYPE;
       d = static_cast<DType>(static_cast<int>(d) + 1)) {
    if (d == DType::UNDEFINED || d == DType::STRING || d == DType::COMPLEX64 ||
        d == DType::COMPLEX128)
      continue;
    result &= predicate(d);
  }
  return result;
}

MLIRContext *createCtx() {
  MLIRContext *ctx = new MLIRContext();
  ctx->loadDialect<ONNXDialect>();
  return ctx;
}

template <typename T>
std::shared_ptr<llvm::MemoryBuffer> buffer(ArrayRef<T> data) {
  return std::shared_ptr<llvm::MemoryBuffer>(
      llvm::MemoryBuffer::getMemBufferCopy(asStringRef(data)));
}

template <typename Dst = char>
ArrayRef<Dst> asArrayRef(const llvm::MemoryBuffer &b) {
  return asArrayRef<Dst>(b.getBuffer());
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

  int test_splat() {
    std::cout << "test_splat:\n";

    bool all = forAllDTypes([this](DType d) {
      return dispatchByDType(d, [this](auto dtype) {
        using cpptype = CppType<dtype>;

        ShapedType type =
            RankedTensorType::get({1}, mlirTypeOfDType(dtype, ctx));
        cpptype one(1);
        Attribute a = elmsBuilder.create(type, buffer<cpptype>({one}));
        ElementsAttr e = a.cast<ElementsAttr>();
        assert(e.isSplat());
        DisposableElementsAttr i = e.cast<DisposableElementsAttr>();
        assert(i.isSplat());

        assert(eq<cpptype>(i.getSplatValue<cpptype>(), one));
        auto b = i.value_begin<cpptype>();
        assert(eq<cpptype>(*b, one));

        if (isFloatDType(dtype)) {
          auto apf = i.getSplatValue<APFloat>();
          assert(near(apf.convertToDouble(), static_cast<double>(one)));
        } else {
          auto api = i.getSplatValue<APInt>();
          auto x = WideNum::fromAPInt(dtype, api).template to<cpptype>(dtype);
          assert(eq<cpptype>(x, one));
        }

        auto d = toDenseElementsAttr(i);
        assert(eq<cpptype>(d.getSplatValue<cpptype>(), one));

        return true;
      });
    });
    assert(all);

    return 0;
  }

  int test_attributes() {
    std::cout << "test_attributes:\n";
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
    std::cout << "test_transpose:\n";
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
    std::cout << "test_cast:\n";
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
  failures += test.test_splat();
  failures += test.test_attributes();
  failures += test.test_transpose();
  failures += test.test_cast();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
