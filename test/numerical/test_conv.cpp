#include <rapidcheck.h>

#include <rapidcheck.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include "mlir/IR/Module.h"
#include "llvm/Support/FileSystem.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/MainUtils.hpp"

#define NO_PYTHON
#include "src/Runtime/Runtime.hpp"

using namespace std;

template <typename T>
void printVector(std::vector<T> vec, std::string _delimiter = ",",
    std::ostream &stream = std::cout) {
  std::string delimiter;
  for (const auto &elem : vec) {
    stream << delimiter << elem;
    delimiter = _delimiter;
  }
}

DynMemRef *getRandomTensor(
    std::vector<int64_t> sizes, float lb = -1.0, float ub = 1.0) {
  // Will be used to obtain a seed for the random number engine
  std::random_device rd;
  // Standard mersenne_twister_engine seeded with rd()
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(lb, ub);
  auto dmr = DynMemRef::create<float>(sizes);
  auto ptr = (float *)dmr->data;
  std::generate(ptr, ptr + dmr->size(), [&]() { return dis(gen); });
  return dmr;
}

inline bool assertClose(
    float a, float b, float rtol = 1e-5, float atol = 1e-5) {
  auto absoluteDiff = std::abs(a - b);
  auto withinRtol = (absoluteDiff / a < rtol);
  auto withinAtol = (absoluteDiff < atol);
  return withinRtol && withinAtol;
}

template <typename T>
inline bool assertDmrClose(
    DynMemRef *a, DynMemRef *b, float rtol = 1e-5, float atol = 1e-5) {

  // Compare shape.
  auto aShape = std::vector<int64_t>(a->sizes, a->sizes + a->rank);
  auto bShape = std::vector<int64_t>(b->sizes, b->sizes + b->rank);
  if (aShape != bShape) {
    std::cerr << "Shape mismatch ";
    printVector(aShape, ",", std::cerr);
    std::cerr << " != ";
    printVector(bShape, ",", std::cerr);
    return false;
  }

  // Compute absolute difference, verify it's within tolerable range.
  std::vector<T> absoluteDiff(a->size());
  std::transform(a->typedPtr<T>(), a->typedPtr<T>() + a->size(),
      b->typedPtr<T>(), absoluteDiff.begin(), std::minus<>());
  std::transform(absoluteDiff.begin(), absoluteDiff.end(), absoluteDiff.begin(),
      static_cast<T (*)(T)>(&std::abs));
  bool atolSatisfied = std::all_of(
      absoluteDiff.begin(), absoluteDiff.end(), [&](T a) { return a < atol; });

  // Compute relative difference, verify it's within tolerable range.
  std::vector<T> relativeDiff(a->size());
  std::transform(absoluteDiff.begin(), absoluteDiff.end(), a->typedPtr<T>(),
      relativeDiff.begin(), std::divides<>());
  bool rtolSatisfied = std::all_of(
      relativeDiff.begin(), relativeDiff.end(), [&](T a) { return a < rtol; });

  if (atolSatisfied && rtolSatisfied) {
    return true;
  } else {
    for (const auto &idx : a->indexSet()) {
      T aElem = a->elem<T>(idx);
      T bElem = b->elem<T>(idx);
      auto elmAbsDiff = std::abs(aElem - bElem);
      auto withinRtol = (elmAbsDiff / aElem < rtol);
      auto withinAtol = (elmAbsDiff < atol);
      if (!withinRtol || !withinAtol) {
        std::cerr << "a[";
        printVector(idx, ",", std::cerr);
        std::cerr << "] = " << aElem << " != ";
        std::cerr << "b[";
        printVector(idx, ",", std::cerr);
        std::cerr << "] = " << bElem << std::endl;
      }
    }
    return false;
  }
}

int main() {
  rc::check("double reversal yields the original value", []() {
    const auto N = *rc::gen::inRange(1, 10);
    const auto C = *rc::gen::inRange(1, 10);
    const auto H = *rc::gen::inRange(5, 10);
    const auto W = *rc::gen::inRange(5, 10);

    const auto kH = *rc::gen::inRange(3, 10);
    const auto kW = *rc::gen::inRange(3, 10);

    // We don't want an entire window of padding.
    const auto pHBegin = *rc::gen::inRange(0, kH - 1);
    const auto pHEnd = *rc::gen::inRange(0, kH - 1);
    const auto pWBegin = *rc::gen::inRange(0, kW - 1);
    const auto pWEnd = *rc::gen::inRange(0, kW - 1);

    // Make sure we have at least 1 output per dimension.
    RC_PRE((H >= kH) && (W > kW));

    registerDialects();
    mlir::MLIRContext ctx;

    auto module = mlir::ModuleOp::create(mlir::UnknownLoc::get(&ctx));
    mlir::OpBuilder builder(&ctx);
    llvm::SmallVector<int64_t, 4> xShape = {N, C, H, W};
    llvm::SmallVector<int64_t, 1> bShape = {C};
    llvm::SmallVector<int64_t, 4> wShape = {C, C, kH, kW};
    auto xType = mlir::RankedTensorType::get(xShape, builder.getF32Type());
    auto wType = mlir::RankedTensorType::get(wShape, builder.getF32Type());
    auto yType = mlir::UnrankedTensorType::get(builder.getF32Type());

    llvm::SmallVector<mlir::Type, 2> inputsType{xType, wType};
    llvm::SmallVector<mlir::Type, 1> outputsType{yType};

    auto funcType = builder.getFunctionType(inputsType, outputsType);
    std::string funcName = "test_conv";
    llvm::SmallVector<mlir::NamedAttribute, 1> attrs;
    auto funcOp = builder.create<mlir::FuncOp>(
        mlir::UnknownLoc::get(&ctx), funcName, funcType, attrs);

    auto entryBlock = funcOp.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);

    auto xVal = entryBlock->getArgument(0);
    auto wVal = entryBlock->getArgument(1);
    auto bVal = builder
                    .create<mlir::ConstantOp>(
                        mlir::UnknownLoc::get(&ctx), builder.getUnitAttr())
                    .getResult();

    auto dilations = builder.getI64ArrayAttr({1, 1});
    auto kernel_shape = builder.getI64ArrayAttr({kH, kW});
    auto pads = builder.getI64ArrayAttr({pHBegin, pHBegin, pHBegin, pHBegin});
    //    auto pads = builder.getI64ArrayAttr({pHBegin, pWBegin, pHEnd, pWEnd});
    auto strides = builder.getI64ArrayAttr({1, 1});

    auto convOp = builder.create<mlir::ONNXConvOp>(mlir::UnknownLoc::get(&ctx),
        /*Y=*/yType,
        /*X=*/xVal, /*W=*/wVal, /*B=*/bVal,
        /*auto_pad=*/builder.getStringAttr("NOTSET"),
        /*dilations=*/dilations, /*group=*/builder.getI64IntegerAttr(1),
        /*kernel_shape=*/kernel_shape, /*pads=*/pads,
        /*strides=*/strides);

    // Use the convOp shape inference method to compute output shape, and unset
    // the shape so that we don't leave IR in a inconsistent state.
    convOp.inferShapes();
    auto outputShape =
        convOp.getResult().getType().cast<mlir::ShapedType>().getShape();
    auto NOut = outputShape[0];
    auto COut = outputShape[1];
    auto HOut = outputShape[2];
    auto WOut = outputShape[3];
    convOp.getResult().setType(yType);

    llvm::SmallVector<mlir::Value, 1> results = {convOp.getResult()};
    builder.create<mlir::ReturnOp>(mlir::UnknownLoc::get(&ctx), results);
    module.push_back(funcOp);

    // Emit the entry point operation which specifies the number of user
    // inputs and outputs.
    auto entryPoint =
        mlir::ONNXEntryPointOp::create(mlir::UnknownLoc::get(&ctx), funcOp,
            /*numInputs=*/2,
            /*numOutputs=*/1);
    module.push_back(entryPoint);

    mlir::OwningModuleRef moduleRef(module);

    llvm::SmallVector<char, 10> path;
    llvm::sys::fs::createTemporaryFile("_test_conv", "", path);
    std::string pathStr(path.begin(), path.end());
    llvm::FileRemover remover(path);

    compileModule(moduleRef, ctx, pathStr, EmitLib);
    ExecutionSession sess(pathStr + ".so", "_dyn_entry_point_test_conv");

    std::vector<std::unique_ptr<DynMemRef>> inputs;
    auto xDmr = std::unique_ptr<DynMemRef>(getRandomTensor({N, C, H, W}));
    inputs.emplace_back(std::move(xDmr));
    auto wDmr = std::unique_ptr<DynMemRef>(getRandomTensor({C, C, kH, kW}));
    inputs.emplace_back(std::move(wDmr));

    auto ref = DynMemRef::create<float>({NOut, COut, HOut, WOut});

    auto &img = inputs.at(0);
    auto &filter = inputs.at(1);
    for (int64_t n = 0; n < NOut; n++)
      for (int64_t c = 0; c < COut; c++)
        for (int64_t h = 0; h < HOut; h++)
          for (int64_t w = 0; w < WOut; w++) {
            ref->elem<float>({n, c, h, w}) = 0;
            for (int64_t ci = 0; ci < C; ci++)
              for (int64_t kh = 0; kh < kH; kh++)
                for (int64_t kw = 0; kw < kW; kw++)
                  if ((h + kh - pHBegin >= 0 && h + kh - pHBegin < H) &&
                      (w + kw - pWBegin >= 0 && w + kw - pWBegin < W))
                    ref->elem<float>({n, c, h, w}) +=
                        img->elem<float>(
                            {n, ci, h + kh - pHBegin, w + kw - pWBegin}) *
                        filter->elem<float>({c, ci, kh, kw});
          }

    auto outputs = sess.run(std::move(inputs));
    auto &conv = outputs.at(0);

    RC_ASSERT(assertDmrClose<float>(conv.get(), ref));
  });

  return 0;
}