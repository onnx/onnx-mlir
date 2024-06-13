/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===================-- TestDisposableElementsAttr.cpp --=====================//
//
// Tests DisposableElementsAttr.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/ElementsAttr/BType.hpp"
#include "src/Dialect/ONNX/ElementsAttr/DisposableElementsAttr.hpp"
#include "src/Dialect/ONNX/ElementsAttr/ElementsAttrBuilder.hpp"
#include "src/Dialect/ONNX/ONNXDialect.hpp"
#include "src/Dialect/ONNX/OnnxElementsAttrBuilder.hpp"
#include "src/Support/Arrays.hpp"

#include "mlir/IR/Builders.h"
#include "llvm/Support/MemoryBuffer.h"

#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

using namespace mlir;
using namespace onnx_mlir;

namespace {

bool near(double a, double b) { return fabs(a - b) < 1e-6; }

template <typename CPPTY>
bool eq(CPPTY a, CPPTY b) {
  if constexpr (isSmallFPType<CPPTY>)
    return a.toFloat() == b.toFloat();
  else
    return a == b;
}

bool forAllBTypes(std::function<bool(BType)> predicate) {
  bool result = true;
  for (BType d = static_cast<BType>(0); d <= BType::MAX_BTYPE;
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
std::unique_ptr<llvm::MemoryBuffer> buffer(ArrayRef<T> data) {
  return llvm::MemoryBuffer::getMemBufferCopy(asStringRef(data));
}

class Test {
  MLIRContext *ctx;
  Location loc;
  OpBuilder builder;
  OnnxElementsAttrBuilder elmsBuilder;
  Type F32;
  Type F16;
  Type U32;
  Type U8;
  Type I32;
  Type I64;
  Type I8;
  Type I1;

public:
  Test()
      : ctx(createCtx()), loc(UnknownLoc::get(ctx)), builder(ctx),
        elmsBuilder(ctx) {
    F32 = builder.getF32Type();
    F16 = builder.getF16Type();
    U32 = builder.getIntegerType(32, /*isSigned=*/false);
    U8 = builder.getIntegerType(8, /*isSigned=*/false);
    I32 = builder.getI32Type();
    I64 = builder.getI64Type();
    I8 = builder.getI8Type();
    I1 = builder.getI1Type();
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
        Attribute a = elmsBuilder.toDisposableElementsAttr(
            DenseElementsAttr::get(type, one));
        ElementsAttr e = mlir::cast<ElementsAttr>(a);
        assert(e.isSplat());
        DisposableElementsAttr i = mlir::cast<DisposableElementsAttr>(e);
        assert(i.isSplat());

        assert(eq<cpptype>(i.getSplatValue<cpptype>(), one));

        auto b = i.value_begin<cpptype>();
        assert(eq<cpptype>(*b, one));

        if (isFloatBType(btype)) {
          auto apf = i.getSplatValue<APFloat>();
          assert(near(apf.convertToDouble(), static_cast<double>(one)));
        } else {
          bool isSigned = isSignedIntBType(btype);
          auto api = i.getSplatValue<APInt>();
          auto x =
              WideNum::fromAPInt(api, isSigned).template to<cpptype>(btype);
          assert(eq<cpptype>(x, one));
        }

        auto d = i.toDenseElementsAttr();
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
    auto e = elmsBuilder.fromMemoryBuffer(type, buffer<uint8_t>(elms));
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
    auto e = elmsBuilder.fromMemoryBuffer(type, buffer<int64_t>({256}));
    auto c = elmsBuilder.castElementType(e, F32);
    assert(c.getSplatValue<float>() == 256.0);

    return 0;
  }

  int test_equal_ints() {
    std::cout << "test_equal_ints:" << std::endl;

    ShapedType type2xi64 = RankedTensorType::get({2}, I64);
    auto e2s_i64 =
        elmsBuilder.fromMemoryBuffer(type2xi64, buffer<int64_t>({-2, 2}));
    auto e3s_i64 =
        elmsBuilder.fromMemoryBuffer(type2xi64, buffer<int64_t>({-3, 3}));

    assert(ElementsAttrBuilder::equal(e2s_i64, e2s_i64));
    assert(!ElementsAttrBuilder::equal(e3s_i64, e2s_i64));

    ShapedType type2xu8 = RankedTensorType::get({2}, U8);
    auto e2s_u8 =
        elmsBuilder.fromMemoryBuffer(type2xu8, buffer<uint8_t>({0xfe, 2}));
    auto e2s_i64_u8 = elmsBuilder.castElementType(e2s_i64, U8);
    auto e3s_i64_u8 = elmsBuilder.castElementType(e3s_i64, U8);

    assert(!ElementsAttrBuilder::equal(e3s_i64_u8, e2s_u8));
    assert(ElementsAttrBuilder::equal(e2s_i64_u8, e2s_u8));

    uint8_t u8_0xfe = 0xfe, u8_2 = 2;
    auto d2s_u8 = DenseElementsAttr::get(type2xu8, {u8_0xfe, u8_2});

    assert(ElementsAttrBuilder::equal(d2s_u8, e2s_u8));
    assert(ElementsAttrBuilder::equal(d2s_u8, e2s_i64_u8));
    assert(!ElementsAttrBuilder::equal(d2s_u8, e3s_i64_u8));

    return 0;
  }

  int test_equal_fps() {
    std::cout << "test_equal_fps:" << std::endl;

    ShapedType type2xf32 = RankedTensorType::get({2}, F32);
    float zero = 0.0f;
    auto d0s_f32_splat = DenseElementsAttr::get(type2xf32, {zero});
    auto d0s_f32 = DenseElementsAttr::get(type2xf32, {zero, -zero});

    assert(d0s_f32_splat != d0s_f32);
    assert(ElementsAttrBuilder::equal(d0s_f32_splat, d0s_f32));

    float nan = std::nanf("");
    assert(std::isnan(nan));
    auto dnans = DenseElementsAttr::get(type2xf32, {nan});

    // float NaN != NaN and the same goes for ElementsAttr::equal
    assert(nan != nan);
    assert(!ElementsAttrBuilder::equal(dnans, dnans));

    // one+delta can be expressed with f32 precision but not f16
    float one = 1.0f, delta = 0.00001f;

    ShapedType type1xf32 = RankedTensorType::get({1}, F32);
    auto d_one_f32 = DenseElementsAttr::get(type1xf32, {one});
    auto d_oneplus_f32 = DenseElementsAttr::get(type1xf32, {one + delta});
    assert(!ElementsAttrBuilder::equal(d_one_f32, d_oneplus_f32));

    auto d_one_f16 = elmsBuilder.castElementType(d_one_f32, F16);
    auto d_oneplus_f16 = elmsBuilder.castElementType(d_oneplus_f32, F16);
    assert(ElementsAttrBuilder::equal(d_one_f16, d_oneplus_f16));

    return 0;
  }

  int test_equal_bools() {
    std::cout << "test_equal_bools:" << std::endl;

    ShapedType type2xu32 = RankedTensorType::get({2}, U32);
    uint32_t u32_0 = 0, u32_2 = 2;
    auto d0_2_u32 = DenseElementsAttr::get(type2xu32, {u32_0, u32_2});
    auto e0_2_i1 = elmsBuilder.castElementType(d0_2_u32, I1);

    ShapedType type2xi1 = RankedTensorType::get({2}, I1);
    auto dF_T_i1 = DenseElementsAttr::get(type2xi1, {false, true});

    assert(e0_2_i1 != dF_T_i1);
    assert(ElementsAttrBuilder::equal(e0_2_i1, dF_T_i1));

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
  failures += test.test_equal_ints();
  failures += test.test_equal_fps();
  failures += test.test_equal_bools();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
