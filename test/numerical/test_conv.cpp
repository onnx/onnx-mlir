#include <rapidcheck.h>

#include <rapidcheck.h>

#include <algorithm>
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

std::vector<float> getRandomTensor(
    int64_t size, float lb = -1.0, float ub = 1.0) {
  std::random_device
      rd; // Will be used to obtain a seed for the random number engine
  std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  std::uniform_real_distribution<> dis(lb, ub);
  std::vector<float> tensor(size);
  std::generate(tensor.begin(), tensor.end(), [&]() { return dis(gen); });
  return tensor;
}

inline bool assertClose(float a, float b) {
  return ((a - b) / a < 0.0001f);
}

int main() {
  rc::check("double reversal yields the original value", []() {
    const auto N = *rc::gen::inRange(1, 10);
    const auto C = *rc::gen::inRange(1, 10);
    const auto H = *rc::gen::inRange(5, 10);
    const auto W = *rc::gen::inRange(5, 10);

    const auto kH = *rc::gen::inRange(3, 10);
    const auto kW = *rc::gen::inRange(3, 10);

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
                    .getResult(); // entryBlock->getArgument(2);

    auto dilations = builder.getI64ArrayAttr({1, 1});
    auto kernel_shape = builder.getI64ArrayAttr({kH, kW});
    auto pads = builder.getI64ArrayAttr({0, 0, 0, 0});
    auto strides = builder.getI64ArrayAttr({1, 1});

    auto conv = builder.create<mlir::ONNXConvOp>(mlir::UnknownLoc::get(&ctx),
        /*Y=*/yType,
        /*X=*/xVal, /*W=*/wVal, /*B=*/bVal,
        /*auto_pad=*/builder.getStringAttr("NOTSET"),
        /*dilations=*/dilations, /*group=*/builder.getI64IntegerAttr(1),
        /*kernel_shape=*/kernel_shape, /*pads=*/pads,
        /*strides=*/strides);

    // Use the conv shape inference method to compute output shape, and unset
    // the shape so that we don't leave IR in a inconsistent state.
    conv.inferShapes();
    auto outputShape =
        conv.getResult().getType().cast<mlir::ShapedType>().getShape();
    auto NOut = outputShape[0];
    auto COut = outputShape[1];
    auto HOut = outputShape[2];
    auto WOut = outputShape[3];
    conv.getResult().setType(yType);

    llvm::SmallVector<mlir::Value, 1> results = {conv.getResult()};
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
    auto xDMR = std::unique_ptr<DynMemRef>(createDynMemRef(4));
    const auto xData = getRandomTensor(N * C * H * W);
    xDMR->data = (void *)&xData[0];
    std::vector<int64_t> xShapeVec = {N, C, H, W};
    xDMR->sizes = &xShapeVec[0];
    inputs.emplace_back(std::move(xDMR));

    auto wDMR = std::unique_ptr<DynMemRef>(createDynMemRef(4));
    auto wData = std::vector<float>(C * C * kH * kW, 1.0f);
    wDMR->data = (void *)&wData[0];
    std::vector<int64_t> wShapeVec = {C, C, kH, kW};
    wDMR->sizes = &wShapeVec[0];
    inputs.emplace_back(std::move(wDMR));

    std::vector<float> convOutputRefData(NOut * COut * HOut * WOut, 0.0f);
    std::vector<int64_t> sizes = {NOut, COut, HOut, WOut};
    auto refConv = *createDynMemRef(4);
    refConv.data = &convOutputRefData[0];
    refConv.sizes = &sizes[0];

    for (int64_t n = 0; n < NOut; n++)
      for (int64_t c = 0; c < COut; c++)
        for (int64_t h = 0; h < HOut; h++)
          for (int64_t w = 0; w < WOut; w++) {
            refConv.elem<float>({n, c, h, w}) = 0;
            for (int64_t ci = 0; ci < C; ci++)
              for (int64_t kh = 0; kh < kH; kh++)
                for (int64_t kw = 0; kw < kW; kw++)
                  refConv.elem<float>({n, c, h, w}) +=
                      inputs.at(0)->elem<float>({n, ci, h + kh, w + kw}) *
                      inputs.at(1)->elem<float>({c, ci, kh, kw});
          }

    auto outputs = sess.run(std::move(inputs));
    printf("%s runs okay!\n", pathStr.c_str());
    llvm::sys::fs::remove(pathStr);

    auto &convOutput = outputs.at(0);
    float *convOutputData = (float *)convOutput->data;

    auto refConvData = (float *)refConv.data;
    for (int64_t i = 0; i < NOut * COut * HOut * WOut; i++) {
      RC_ASSERT(assertClose(refConvData[i], convOutputData[i]));
    }
  });

  return 0;
}