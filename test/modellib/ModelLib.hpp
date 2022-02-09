/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====-- ModelLib.hpp - Building Models for numerical and benchmark tests -===//
//
// Copyright 2022 The IBM Research Authors.
//
// =============================================================================
//
// This file contains the declarations for all the models that can be built.
// The result of each function is a .so built using the modelName.
//
// For each model, the function that implements it is named "main_graph".
//
//===----------------------------------------------------------------------===//

#include <string>

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"

// Padding schemes
#define AUTO_PAD_NOTSET 0
#define AUTO_PAD_VALID 1
#define AUTO_PAD_LOWER 2
#define AUTO_PAD_UPPER 3
#define AUTO_PAD_UB 4
const std::string getAutoPadName(const int autoPad);

// Conv2D
bool genConv2DModelAndCompile(
    /* compile option */
    const std::string &modelName, const CompilerOptionList &options,
    /* conv param in*/
    const int N, const int C, const int H, const int W, const int kH,
    const int kW, const int autoPad, const int pHBegin, const int pHEnd,
    const int pWBegin, const int pWEnd, const int stride, const int dilation,
    const int isDynamic,
    /* conv param out */
    int &NOut, int &COut, int &HOut, int &WOut);

// GEMM
bool genGemmAndCompileModel(
    /* compile option */
    const std::string &modelName, const CompilerOptionList &options,
    /* GEMM param in*/
    const int I, const int J, const int K, const int aTrans, const int bTrans,
    const int cRank, const float alphaVal, const float betaVal,
    /* GEMM param out*/
    llvm::SmallVector<int64_t, 2> &aShape,
    llvm::SmallVector<int64_t, 2> &bShape,
    llvm::SmallVector<int64_t, 2> &cShape);

// MatMul
bool genMatMul2DModelAndCompile(
    /* compile option */
    const std::string &modelName, const CompilerOptionList &options,
    /* conv param in*/
    const int I, const int J, const int K);

// GRU
bool genGRUModelAndCompile(
    /* compile option */
    const std::string &modelName, const CompilerOptionList &options,
    /* GRU param in*/
    const int direction, const int S, const int B, const int I, const int H,
    const int LinearBeforeReset, const bool isDynamicS, const bool isDynamicB,
    /* GRU param out*/
    int &D, llvm::SmallVector<int64_t, 3> &xShape,
    llvm::SmallVector<int64_t, 3> &hShape, OMTensor *&wOmt, OMTensor *&rOmt,
    OMTensor *&bOmt);

// RNN
bool genRNNModelAndCompile(
    /* compile option */
    const std::string &modelName, const CompilerOptionList &options,
    /* RNN param in*/
    const int direction, const int S, const int B, const int I, const int H,
    const bool isDynamicS, const bool isDynamicB,
    /* RNN param out*/
    int &D, llvm::SmallVector<int64_t, 3> &xShape,
    llvm::SmallVector<int64_t, 3> &hShape, OMTensor *&wOmt, OMTensor *&rOmt,
    OMTensor *&bOmt);

// LSTM
bool genLSTMModelAndCompile(
    /* compile option */
    const std::string &modelName, const CompilerOptionList &options,
    /* LSTM param in*/
    const int direction, const int S, const int B, const int I, const int H,
    const bool isDynamicS, const bool isDynamicB,
    /* LTSM param out*/
    int &D, llvm::SmallVector<int64_t, 3> &xShape,
    llvm::SmallVector<int64_t, 3> &hShape,
    llvm::SmallVector<int64_t, 3> &cShape, OMTensor *&wOmt, OMTensor *&rOmt,
    OMTensor *&bOmt, OMTensor *&pOmt);


class ModelLibBuilder {
public:
  ModelLibBuilder(const std::string &sharedLibBaseName);
  // Build, subclass should add all parameters needed to determine the precise
  // parameter of the model being build.
  bool build();
  // Compile model.
  bool compile(const CompilerOptionList &options);
  // Run model. Subclasses must define the inputs.
  bool run();
  // Test model, subclass must define the reference and compare against it.
  bool test();

protected:
  // Create a function with an empty body.
  // This function will contain the model to be tested.
  mlir::FuncOp createEmptyTestFunction(
      const llvm::SmallVectorImpl<mlir::Type> &inputsType,
      const llvm::SmallVectorImpl<mlir::Type> &outputsType);

  // Create the entry point function (used to call the model test function).
  void createEntryPoint(mlir::FuncOp &funcOp);

  mlir::ONNXConstantOp buildONNXConstantOp(
      OMTensor *omt, mlir::RankedTensorType resultType);

  // Data (order matter).
  const std::string sharedLibBaseName;
  mlir::MLIRContext ctx;
  mlir::Location loc;
  mlir::ModuleOp module;
  mlir::OpBuilder builder;
};
