/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===================-- TestDisposableElementsAttr.cpp --=====================//
//
// Tests DisposableElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/AttributesHelper.hpp"
#include "src/Dialect/ONNX/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Support/BType.hpp"

#include "mlir/IR/Builders.h"
#include "llvm/Support/MemoryBuffer.h"

#include <iostream>
#include <memory>
#include <vector>

using namespace mlir;
using namespace onnx_mlir;

namespace {

bool near(double a, double b) { return fabs(a - b) < 1e-6; }

template <typename CPPTY>
bool eq(CPPTY a, CPPTY b) {
  if constexpr (isFP16Type<CPPTY>)
    return a.toFloat() == b.toFloat();
  else
    return a == b;
}

bool forAllBTypes(std::function<bool(BType)> predicate) {
  bool result = true;
  for (BType d = static_cast<BType>(0); d <= BType::MAX_DTYPE;
       d = static_cast<BType>(static_cast<int>(d) + 1)) {
    if (d == BType::UNDEFINED || d == BType::STRING || d == BType::COMPLEX64 ||
        d == BType::COMPLEX128)
      continue;
    result &= predicate(d);
  }
  return result;
}

template <typename T, T... ints>
std::vector<T> nums(std::integer_sequence<T, ints...> int_seq) {
  std::vector<T> v;
  (v.push_back(ints), ...);
  return v;
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
    std::cout << "test_splat:" << std::endl;

    bool all = forAllBTypes([this](BType d) {
      return dispatchByBType(d, [this](auto btype) {
        using cpptype = CppType<btype>;

        Type elementType = mlirTypeOfBType(btype, ctx);
        ShapedType type = RankedTensorType::get({2, 1}, elementType);
        cpptype one(1);
        std::vector<int64_t> zerosStrides(2, 0);
        Attribute a = elmsBuilder.create(
            type, buffer<cpptype>({one}), llvm::makeArrayRef(zerosStrides));
        ElementsAttr e = a.cast<ElementsAttr>();
        assert(e.isSplat());
        DisposableElementsAttr i = e.cast<DisposableElementsAttr>();
        assert(i.isSplat());

        assert(eq<cpptype>(i.getSplatValue<cpptype>(), one));

        auto b = i.value_begin<cpptype>();
        assert(eq<cpptype>(*b, one));

        if (isFloatBType(btype)) {
          auto apf = i.getSplatValue<APFloat>();
          assert(near(apf.convertToDouble(), static_cast<double>(one)));
        } else {
          auto api = i.getSplatValue<APInt>();
          auto x = WideNum::fromAPInt(btype, api).template to<cpptype>(btype);
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

  int test_transpose() {
    std::cout << "test_transpose:" << std::endl;

    ShapedType type = RankedTensorType::get({2, 3, 5}, getUInt(8));
    auto elms = nums<uint8_t>(std::make_integer_sequence<uint8_t, 30>{});
    auto e = elmsBuilder.create(type, buffer<uint8_t>(elms));
    assert(e.getValues<uint8_t>()[0] == 0);
    assert(e.getValues<uint8_t>()[1] == 1);
    assert(e.getValues<uint8_t>()[28] == 28);
    assert(e.getValues<uint8_t>()[29] == 29);

    auto t = elmsBuilder.transpose(e, {1, 2, 0});
    assert(t.getValues<uint8_t>()[0] == 0);
    assert(t.getValues<uint8_t>()[1] == 15);
    assert(t.getValues<uint8_t>()[28] == 14);
    assert(t.getValues<uint8_t>()[29] == 29);

    return 0;
  }

  int test_cast() {
    std::cout << "test_cast:" << std::endl;

    ShapedType type = RankedTensorType::get({1}, I64);
    auto e = elmsBuilder.create(type, buffer<int64_t>({256}));
    assert(e.getSplatValue<int64_t>() == 256);

    auto c = elmsBuilder.castElementType(e, F32);
    assert(c.getSplatValue<float>() == 256.0);

    return 0;
  }
};

} // namespace

int main(int argc, char *argv[]) {
  Test test;
  int failures = 0;
  failures += test.test_splat();
  failures += test.test_transpose();
  failures += test.test_cast();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
