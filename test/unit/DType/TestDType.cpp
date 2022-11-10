/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===============================-- TestDType.cpp ---=========================//
//
// Tests DType.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Support/DType.hpp"
#include "src/Support/WideNum.hpp"

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

  int test_dispatchByDType() {
    std::cout << "test_dispatchByDType:" << std::endl;

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

      auto dty = dispatchByDType(d, [d, this](auto dtype) -> DType {
        assert(d == dtype);

        using Q = DTypeTrait<dtype>;
        assert(isFloatDType(dtype) == Q::isFloat);
        assert(isIntOrFloatDType(dtype) == Q::isIntOrFloat);
        assert(isSignedIntDType(dtype) == Q::isSignedInt);
        assert(isUnsignedIntDType(dtype) == Q::isUnsignedInt);

        using cpptype = CppType<dtype>;
        assert(sizeof(cpptype) == Q::bytewidth);
        assert(d == toDType<cpptype>);
        Type t = toMlirType<cpptype>(ctx);
        assert(d == dtypeOfMlirType(t));

        return dtype;
      });
      assert(dty == d);

      Type t = mlirTypeOfDType(d, ctx);
      assert(d == dtypeOfMlirType(t));
    }

    return 0;
  }

  int test_FloatingPoint16() {
    std::cout << "test_FloatingPoint16:" << std::endl;

    float_16 f9984(9984);
    bfloat_16 fminus1(-1);
    float_16 bfminus1(fminus1);
    bfloat_16 bf9984(f9984);
    assert(bfminus1.toFloat() == fminus1.toFloat());
    assert(static_cast<float_16>(bf9984).toFloat() ==
           static_cast<bfloat_16>(f9984).toFloat());

    // Test that constexpr works for all these:
    constexpr float_16 f16z = float_16();
    constexpr bfloat_16 bf16z = bfloat_16();
    constexpr DType df16 = toDType<decltype(f16z)>;
    constexpr DType dbf16 = toDType<decltype(bf16z)>;
    assert(df16 == toDType<float_16>);
    assert(dbf16 == toDType<bfloat_16>);
    assert((std::is_same_v<CppType<df16>, float_16>));
    assert((std::is_same_v<CppType<dbf16>, bfloat_16>));
    assert((std::is_same_v<CppType<toDType<float>>, float>));

    constexpr WideNum n = WideNum::from(toDType<float_16>, true);
    assert(n.dbl == 1.0);

    return 0;
  }
};

} // namespace

int main(int argc, char *argv[]) {
  Test test;
  int failures = 0;
  failures += test.test_dispatchByDType();
  failures += test.test_FloatingPoint16();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
