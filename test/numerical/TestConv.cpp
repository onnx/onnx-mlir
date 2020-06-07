#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <rapidcheck.h>
#include <string>
#include <vector>

#include "mlir/IR/Module.h"
#include "llvm/Support/FileSystem.h"

#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/MainUtils.hpp"
#include "src/Runtime/ExecusionSession.hpp"

using namespace std;

// Returns whether onnx-mlir compiled convolution is producing the same results
// as a naive implementation of convolution for a specific set of convolution
// parameters/configuration.
bool isOMConvTheSameAsNaiveImplFor(const int N, const int C, const int H,
    const int W, const int kH, const int kW, const int pHBegin, const int pHEnd,
    const int pWBegin, const int pWEnd) {
  registerDialects();
  MLIRContext ctx;

  auto module = ModuleOp::create(UnknownLoc::get(&ctx));
  OpBuilder builder(&ctx);
  llvm::SmallVector<int64_t, 4> xShape = {N, C, H, W};
  llvm::SmallVector<int64_t, 1> bShape = {C};
  llvm::SmallVector<int64_t, 4> wShape = {C, C, kH, kW};
  auto xType = RankedTensorType::get(xShape, builder.getF32Type());
  auto wType = RankedTensorType::get(wShape, builder.getF32Type());
  auto yType = UnrankedTensorType::get(builder.getF32Type());

  llvm::SmallVector<Type, 2> inputsType{xType, wType};
  llvm::SmallVector<Type, 1> outputsType{yType};

  auto funcType = builder.getFunctionType(inputsType, outputsType);
  string funcName = "test_conv";
  llvm::SmallVector<NamedAttribute, 1> attrs;
  auto funcOp =
      builder.create<FuncOp>(UnknownLoc::get(&ctx), funcName, funcType, attrs);

  auto entryBlock = funcOp.addEntryBlock();
  builder.setInsertionPointToStart(entryBlock);

  auto xVal = entryBlock->getArgument(0);
  auto wVal = entryBlock->getArgument(1);
  auto bVal =
      builder.create<ConstantOp>(UnknownLoc::get(&ctx), builder.getUnitAttr())
          .getResult();

  auto dilations = builder.getI64ArrayAttr({1, 1});
  auto kernel_shape = builder.getI64ArrayAttr({kH, kW});
  auto pads = builder.getI64ArrayAttr({pHBegin, pWBegin, pHEnd, pWEnd});
  auto strides = builder.getI64ArrayAttr({1, 1});

  auto convOp = builder.create<ONNXConvOp>(UnknownLoc::get(&ctx),
      /*Y=*/yType,
      /*X=*/xVal, /*W=*/wVal, /*B=*/bVal,
      /*auto_pad=*/builder.getStringAttr("NOTSET"),
      /*dilations=*/dilations, /*group=*/builder.getI64IntegerAttr(1),
      /*kernel_shape=*/kernel_shape, /*pads=*/pads,
      /*strides=*/strides);

  // Use the convOp shape inference method to compute output shape, and unset
  // the shape so that we don't leave IR in a inconsistent state.
  convOp.inferShapes();
  auto outputShape = convOp.getResult().getType().cast<ShapedType>().getShape();
  auto NOut = outputShape[0];
  auto COut = outputShape[1];
  auto HOut = outputShape[2];
  auto WOut = outputShape[3];
  convOp.getResult().setType(yType);

  llvm::SmallVector<Value, 1> results = {convOp.getResult()};
  builder.create<ReturnOp>(UnknownLoc::get(&ctx), results);
  module.push_back(funcOp);

  // Emit the entry point operation which specifies the number of user
  // inputs and outputs.
  auto entryPoint = ONNXEntryPointOp::create(UnknownLoc::get(&ctx), funcOp,
      /*numInputs=*/2,
      /*numOutputs=*/1);
  module.push_back(entryPoint);

  OwningModuleRef moduleRef(module);

  llvm::SmallVector<char, 10> path;
  llvm::sys::fs::createTemporaryFile("_test_conv", "", path);
  string pathStr(path.begin(), path.end());
  llvm::FileRemover remover(path);

  compileModule(moduleRef, ctx, pathStr, EmitLib);
  onnx_mlir::ExecutionSession sess(
      pathStr + ".so", "_dyn_entry_point_test_conv");

  std::vector<unique_ptr<DynMemRef>> inputs;
  auto xDmr = unique_ptr<DynMemRef>(getRndRealDmr<float>({N, C, H, W}));
  inputs.emplace_back(move(xDmr));
  auto wDmr = unique_ptr<DynMemRef>(getRndRealDmr<float>({C, C, kH, kW}));
  inputs.emplace_back(move(wDmr));

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

  auto outputs = sess.run(move(inputs));
  auto &conv = outputs.at(0);

  return isDmrClose<float>(conv.get(), ref);
}

int main() {
  // Automatic checking - generating a bunch of possible configurations randomly
  // and test them for correctness.
  rc::check("convolution implementation correctness", []() {
    const auto N = *rc::gen::inRange(1, 10);
    const auto C = *rc::gen::inRange(1, 20);
    const auto H = *rc::gen::inRange(5, 20);
    const auto W = *rc::gen::inRange(5, 20);

    const auto kH = *rc::gen::inRange(1, 15);
    const auto kW = *rc::gen::inRange(1, 15);

    // We don't want an entire window of padding.
    const auto pHBegin = *rc::gen::inRange(0, kH - 1);
    const auto pHEnd = *rc::gen::inRange(0, kH - 1);
    const auto pWBegin = *rc::gen::inRange(0, kW - 1);
    const auto pWEnd = *rc::gen::inRange(0, kW - 1);

    // Make sure we have at least 1 output per dimension.
    RC_PRE((H >= kH) && (W > kW));

    RC_ASSERT(isOMConvTheSameAsNaiveImplFor(
        N, C, H, W, kH, kW, pHBegin, pHEnd, pWBegin, pWEnd));
  });

  // Exhaustive checking for all padding configuration for 3x3 convolution.
  for (int pHBegin = 0; pHBegin < 3; pHBegin++)
    for (int pHEnd = 0; pHEnd < 3; pHEnd++)
      for (int pWBegin = 0; pWBegin < 3; pWBegin++)
        for (int pWEnd = 0; pWEnd < 3; pWEnd++)
          assert(isOMConvTheSameAsNaiveImplFor(
              2, 4, 5, 5, 3, 3, pHBegin, pHEnd, pWBegin, pWEnd));
  return 0;
}
