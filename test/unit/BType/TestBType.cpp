/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===============================-- TestBType.cpp ---=========================//
//
// Tests BType.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttr/BType.hpp"
#include "src/Dialect/ONNX/ElementsAttr/WideNum.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"

#include "mlir/IR/Builders.h"

#include <iostream>

using namespace mlir;
using namespace onnx_mlir;

namespace {

MLIRContext *createCtx() {
  MLIRContext *ctx = new MLIRContext();
  ctx->loadDialect<ONNXDialect>();
  return ctx;
}

class Test {
  MLIRContext *ctx;
  OpBuilder builder;
  Type F32;
  Type I32;
  Type I64;

public:
  Test() : ctx(createCtx()), builder(ctx) {
    F32 = builder.getF32Type();
    I32 = builder.getI32Type();
    I64 = builder.getI64Type();
  }
  ~Test() { delete ctx; }

  IntegerType getUInt(unsigned width) const {
    return IntegerType::get(ctx, width, IntegerType::Unsigned);
  }

  int test_dispatchByBType() {
    std::cout << "test_dispatchByBType:" << std::endl;

    for (BType d = static_cast<BType>(0); d <= BType::MAX_BTYPE;
         d = static_cast<BType>(static_cast<int>(d) + 1)) {
      if (d == BType::UNDEFINED || d == BType::STRING ||
          d == BType::COMPLEX64 || d == BType::COMPLEX128)
        continue;

      if (isIntOrFloatBType(d)) {
        if (isFloatBType(d)) {
          assert(!isSignedIntBType(d));
          assert(!isUnsignedIntBType(d));
        } else {
          assert(isSignedIntBType(d) ^ isUnsignedIntBType(d));
        }
      }

      auto dty = dispatchByBType(d, [d, this](auto btype) -> BType {
        assert(d == btype);

        using Q = BTypeTrait<btype>;
        assert(isFloatBType(btype) == Q::isFloat);
        assert(isIntOrFloatBType(btype) == Q::isIntOrFloat);
        assert(isSignedIntBType(btype) == Q::isSignedInt);
        assert(isUnsignedIntBType(btype) == Q::isUnsignedInt);

        using cpptype = CppType<btype>;
        assert(sizeof(cpptype) == Q::bytewidth);
        assert(d == toBType<cpptype>);
        Type t = toMlirType<cpptype>(ctx);
        assert(d == btypeOfMlirType(t));

        return btype;
      });
      assert(dty == d);

      Type t = mlirTypeOfBType(d, ctx);
      assert(d == btypeOfMlirType(t));
    }

    return 0;
  }

  int test_FloatingPoint16() {
    std::cout << "test_FloatingPoint16:" << std::endl;

    // Test that constexpr works for all these:
    constexpr float_16 f16z{};
    constexpr bfloat_16 bf16z{};
    constexpr BType df16 = toBType<decltype(f16z)>;
    constexpr BType dbf16 = toBType<decltype(bf16z)>;
    assert(df16 == toBType<float_16>);
    assert(dbf16 == toBType<bfloat_16>);
    assert((std::is_same_v<CppType<df16>, float_16>));
    assert((std::is_same_v<CppType<dbf16>, bfloat_16>));
    assert((std::is_same_v<CppType<toBType<float>>, float>));

    constexpr WideNum n = WideNum::from(toBType<float_16>, true);
    assert(n.dbl == 1.0);

    return 0;
  }

  int test_FloatingPoint8() {
    std::cout << "test_FloatingPoint8:" << std::endl;

    // Test that constexpr works for all these:
    constexpr float_8e4m3fn f8e4m3fn{};
    constexpr float_8e4m3fnuz f8e4m3fnuz{};
    constexpr float_8e5m2 f8e5m2{};
    constexpr float_8e5m2fnuz f8e5m2fnuz{};
    constexpr BType df8e4m3fn = toBType<decltype(f8e4m3fn)>;
    constexpr BType df8e4m3fnuz = toBType<decltype(f8e4m3fnuz)>;
    constexpr BType df8e5m2 = toBType<decltype(f8e5m2)>;
    constexpr BType df8e5m2fnuz = toBType<decltype(f8e5m2fnuz)>;
    assert(df8e4m3fn == toBType<float_8e4m3fn>);
    assert(df8e4m3fnuz == toBType<float_8e4m3fnuz>);
    assert(df8e5m2 == toBType<float_8e5m2>);
    assert(df8e5m2fnuz == toBType<float_8e5m2fnuz>);
    assert((std::is_same_v<CppType<df8e4m3fn>, float_8e4m3fn>));
    assert((std::is_same_v<CppType<df8e4m3fnuz>, float_8e4m3fnuz>));
    assert((std::is_same_v<CppType<df8e5m2>, float_8e5m2>));
    assert((std::is_same_v<CppType<df8e5m2fnuz>, float_8e5m2fnuz>));

    constexpr WideNum n = WideNum::from(toBType<float_8e4m3fn>, true);
    assert(n.dbl == 1.0);

    return 0;
  }
};

} // namespace

int main(int argc, char *argv[]) {
  Test test;
  int failures = 0;
  failures += test.test_dispatchByBType();
  failures += test.test_FloatingPoint16();
  failures += test.test_FloatingPoint8();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
