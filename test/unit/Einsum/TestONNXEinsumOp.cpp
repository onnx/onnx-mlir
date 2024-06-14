/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===============-- TestONNXEinsumOp.cpp - ONNXEinsumOp tests -===============//
//
// Tests ONNXEinsumOp verify() and inferShapes() methods.
//
// NOTE: it might be more idiomatic to do all these as .mlir tests,
// e.g. like test/mlir/onnx/onnx_shape_inference_error.mlir,
// but the approach in this test lets us concisely describe many test cases
// with one line for each, e.g. in test_verify_inferShapes_success() the line
//
//   {"ij,jk", {{2,3}, {3,4}}, {2,4}}
//
// says that from equation "ij,jk" and input shapes {2,3} and {3,4}
// the output shape {2,4} should be inferred.
//
//===----------------------------------------------------------------------===//

#include "src/Dialect/ONNX/DialectBuilder.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Dialect/ONNX/ONNXOps/OpHelper.hpp"

#include <algorithm>
#include <iostream>
#include <iterator>
#include <string>
#include <vector>

using namespace mlir;
using namespace onnx_mlir;

namespace {

std::string removeSpaces(StringRef s) {
  std::string r;
  std::copy_if(s.begin(), s.end(), std::back_inserter(r),
      [](char x) { return x != ' '; });
  return r;
}

typedef llvm::SmallVector<int64_t> Shape;

std::ostream &operator<<(std::ostream &os, const ArrayRef<int64_t> &v) {
  os << "(";
  for (auto i : v)
    os << i << ",";
  os << ")";
  return os;
}

MLIRContext *createCtx() {
  MLIRContext *ctx = new MLIRContext();
  ctx->loadDialect<ONNXDialect>();
  return ctx;
}

class Test {
  std::unique_ptr<MLIRContext> ctx;
  Location loc;
  OpBuilder builder;
  Type F32;
  Type I32;

  Attribute zero(Type t) {
    if (mlir::isa<FloatType>(t))
      return FloatAttr::get(t, 0);
    assert(mlir::isa<IntegerType>(t) && "must be IntegerType if not FloatType");
    return IntegerAttr::get(t, 0);
  }

  Value zeros(ArrayRef<int64_t> shape, Type t) {
    RankedTensorType tensorType = RankedTensorType::get(shape, t);
    SmallVector<Attribute> values(tensorType.getNumElements(), zero(t));
    return OnnxBuilder(builder, loc)
        .constant(DenseElementsAttr::get(tensorType, ArrayRef(values)));
  }

  ONNXEinsumOp einsumOp(
      StringRef equation, const std::vector<Value> &inputs, Type elementType) {
    return builder.create<ONNXEinsumOp>(loc,
        UnrankedTensorType::get(elementType), llvm::ArrayRef(inputs), equation);
  }

  ONNXEinsumOp einsumOp(StringRef equation,
      const std::vector<Shape> &inputShapes, Type elementType) {
    std::vector<Value> inputs;
    for (const Shape &shape : inputShapes) {
      inputs.push_back(zeros(shape, elementType));
    }
    return einsumOp(equation, inputs, elementType);
  }

  std::vector<Shape> zeroShapes(StringRef equation) {
    std::string equationWithoutSpaces = removeSpaces(equation);
    equation = StringRef(equationWithoutSpaces);
    auto inputs_output = equation.split("->");
    llvm::SmallVector<StringRef> inputs;
    inputs_output.first.split(inputs, ',');
    std::vector<Shape> inputShapes;
    for (StringRef input : inputs) {
      Shape shape(input.size(), 0);
      inputShapes.emplace_back(input.size(), 7);
    }
    return inputShapes;
  }

public:
  Test() : ctx(createCtx()), loc(UnknownLoc::get(&*ctx)), builder(&*ctx) {
    F32 = builder.getF32Type();
    I32 = builder.getI32Type();
  }

  int verify(bool expectSuccess, const std::vector<StringRef> &equations) {
    int failures = 0;
    for (StringRef equation : equations) {
      auto inputShapes = zeroShapes(equation);
      ONNXEinsumOp op = einsumOp(equation, inputShapes, F32);
      auto outcome = op.verify();
      failures += expectSuccess != succeeded(outcome);
    }
    return failures;
  }

  int test_verify_success() {
    return verify(true, {
                            "",
                            " ",
                            ",",
                            "->",
                            " -  > ",
                            ",->",
                            "a",
                            "a,",
                            ",a",
                            "abc,ABa",
                            "ij,jk",
                            "ij,jk->ijk",
                            "...",
                            "...,...",
                            "...->...",
                            "->...",
                            "...a,b...->a...",
                            "a...b->...",
                        });
  }

  int test_verify_failure() {
    return verify(false, {
                             "-",
                             ">",
                             ":",
                             "->->",
                             "..",
                             "....",
                             "...a...",
                             "->a",
                             "a->b",
                             "aa->aa",
                             "a,b->a,b",
                         });
  }

  using EqnInputShapes = std::tuple<StringRef, std::vector<Shape>>;

  int verify_with_shapes(
      bool expectSuccess, const std::vector<EqnInputShapes> &eqnsInputShapes) {
    int failures = 0;
    for (const auto &eis : eqnsInputShapes) {
      StringRef equation;
      std::vector<Shape> inputShapes;
      std::tie(equation, inputShapes) = eis;
      ONNXEinsumOp op = einsumOp(equation, inputShapes, F32);
      auto outcome = op.verify();
      failures += expectSuccess != succeeded(outcome);
    }
    return failures;
  }

  int test_verify_incorrect_numInputs() {
    return verify_with_shapes(false, {
                                         {"a", {}},
                                         {"a", {{0}, {}}},
                                         {",", {{}}},
                                         {",", {{}, {}, {}}},
                                         {"a,b->a", {{0}}},
                                         {"a,b->a", {{0}, {0}, {0}}},
                                     });
  }

  int test_verify_shapes_failure() {
    return verify_with_shapes(
        false, {
                   {"i", {{}}},
                   {"i", {{2, 2}}},
                   {"ii", {{1, 2}}},
                   {"i,i->i", {{2}, {3}}},
                   {"...->", {{2}}},
                   {"a...b", {{2}}},
                   {"...,...", {{2}, {}}},
                   {"...", {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10}}},
               });
  }

  int test_verify_different_types() {
    int failures = 0;
    Value fst = zeros({2}, F32);
    Value snd = zeros({3}, I32);
    ONNXEinsumOp op = einsumOp("i,j", {fst, snd}, F32);
    failures += succeeded(op.verify());
    return failures;
  }

  using EqnShapes = std::tuple<StringRef, std::vector<Shape>, Shape>;

  int verify_inferShapes(
      bool expectSuccess, const std::vector<EqnShapes> &eqnsShapes) {
    int failures = 0;
    for (const auto &es : eqnsShapes) {
      StringRef equation;
      std::vector<Shape> inputShapes;
      Shape expectedOutputShape;
      std::tie(equation, inputShapes, expectedOutputShape) = es;

      ONNXEinsumOp op = einsumOp(equation, inputShapes, F32);
      if (failed(op.verify())) {
        failures += expectSuccess != false;
        continue;
      }
      bool inferenceSuccess = succeeded(op.inferShapes(nullptr));
      if (expectSuccess && inferenceSuccess) {
        auto outputShape =
            mlir::cast<mlir::ShapedType>(op.getResult().getType()).getShape();
        if (expectedOutputShape != outputShape) {
          std::cerr << "inferred output shape " << outputShape
                    << " != expected " << expectedOutputShape << "\n";
          failures += 1;
          continue;
        }
      }
      failures += expectSuccess != inferenceSuccess;
    }
    return failures;
  }

  int test_verify_inferShapes_success() {
    Shape shape10{0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
    return verify_inferShapes(true, {
                                        {"i", {{0}}, {0}},
                                        {"i", {{1}}, {1}},
                                        {"i", {{2}}, {2}},
                                        {"ii", {{2, 2}}, {}},
                                        {"i,i->i", {{1}, {1}}, {1}},
                                        {"i,i->i", {{2}, {2}}, {2}},
                                        {"i,i->i", {{1}, {2}}, {2}},
                                        {"i,i->i", {{2}, {1}}, {2}},
                                        {"i,i->i", {{1}, {0}}, {0}},
                                        {"i,i->i", {{0}, {1}}, {0}},
                                        {"...", {{}}, {}},
                                        {"...", {shape10}, shape10},
                                        {"...,...", {{}, {}}, {}},
                                        {"...,...", {{1}, {1}}, {1}},
                                        {"...,...", {{1}, {2}}, {2}},
                                        {"...,...", {{2}, {1}}, {2}},
                                        {"...,...", {{1, 2}, {2, 1}}, {2, 2}},
                                        {"ij,jk", {{2, 3}, {3, 4}}, {2, 4}},
                                        {"ij,jk->ik", {{2, 3}, {3, 4}}, {2, 4}},
                                        {"ij,jk->ki", {{2, 3}, {3, 4}}, {4, 2}},
                                        {"kj,ji", {{2, 3}, {3, 4}}, {4, 2}},
                                        {"aB", {{2, 3}}, {3, 2}},
                                    });
  }
};

} // namespace

int main(int argc, char *argv[]) {
  Test test;
  int failures = 0;
  failures += test.test_verify_incorrect_numInputs();
  failures += test.test_verify_success();
  failures += test.test_verify_failure();
  failures += test.test_verify_shapes_failure();
  failures += test.test_verify_different_types();
  failures += test.test_verify_inferShapes_success();
  if (failures != 0) {
    std::cerr << failures << " test failures\n";
    return 1;
  }
  return 0;
}
