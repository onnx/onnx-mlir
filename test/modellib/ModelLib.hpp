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
#include "src/Runtime/ExecutionSession.hpp"

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
    const std::string &modelName,
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
    const std::string &modelName,
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
    const std::string &modelName,
    /* conv param in*/
    const int I, const int J, const int K);

// GRU
bool genGRUModelAndCompile(
    /* compile option */
    const std::string &modelName,
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
    const std::string &modelName,
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
    const std::string &modelName,
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
  // Define the model. Subclass should add to the builder all of the specific
  // parameters that uniquely define the model.
  ModelLibBuilder(const std::string &sharedLibBaseName);
  // Destructor needed to free the inputs/outputs data structures.
  ~ModelLibBuilder();
  // Build, subclass should generate a graph. If constant nodes are needed by
  // the model, they should be created here and saved in the subclass, as these
  // values will be needed to verify the accuracy of the model. The model is
  // saved in the model and ctx variable.
  bool build() { llvm_unreachable("subclass must implement build."); }
  // Compile model from the model and ctx variables. The output is an executable
  // dynamic library.
  bool compileAndLoad();
  // Prepare inputs for running model.
  bool prepareInputs() { llvm_unreachable("subclass must implement prepare."); }
  // Run model using prepared inputs, resulting in outputs.
  bool run();
  // Verify outputs with reference data.
  bool verifyOutputs() { llvm_unreachable("subclass must implement verify."); }
  // Get the dynamic library file name compiled here.
  static std::string getSharedLibName(const std::string &sharedLibBaseName);

protected:
  // Create a function with an empty body.
  // This function will contain the model to be tested.
  mlir::FuncOp createEmptyTestFunction(
      const llvm::SmallVectorImpl<mlir::Type> &inputsType,
      const llvm::SmallVectorImpl<mlir::Type> &outputsType);
  // Create the entry point function (used to call the model test function).
  void createEntryPoint(mlir::FuncOp &funcOp);
  // Create a onnx constant op loaded with values in the tensor omt.
  mlir::ONNXConstantOp buildONNXConstantOp(
      /* hi alex const*/ OMTensor *omt,
      const mlir::RankedTensorType resultType);
  // Compare results as float.
  bool areCloseFloat(/* hi alex const*/ OMTensor *res, /*const*/ OMTensor *ref);

  // Data for building and compiling the model.
  const std::string sharedLibBaseName; // Name for the library.
  mlir::MLIRContext ctx;   // Context for the model (used until compilation).
  mlir::Location loc;      // Location for the model (used during building).
  mlir::OpBuilder builder; // Builder (used during building)
  mlir::ModuleOp module;   // Code for the model (used until compilation)

  // Data for runing the model (freed in destructor).
  OMTensorList *inputs, *outputs;
  onnx_mlir::ExecutionSession *exec;
};

class GemmLibBuilder : public ModelLibBuilder {
public:
  GemmLibBuilder(const std::string &modelName, const int I, const int J,
      const int K, const int aTrans, const int bTrans, const int cRank,
      const float alphaVal, const float betaVal);
  bool build();
  bool prepareInputs();
  bool verifyOutputs();

private:
  // Data that defines model.
  const int I, J, K, aTrans, bTrans, cRank;
  const float alphaVal, betaVal;
  // Derived data that defines model.
  llvm::SmallVector<int64_t, 2> aShape, bShape, cShape;
};

class MatMul2DLibBuilder : public ModelLibBuilder {
public:
  MatMul2DLibBuilder(
      const std::string &modelName, const int I, const int J, const int K);
  bool build();
  bool prepareInputs();
  bool verifyOutputs();

private:
  // Data that defines model.
  const int I, J, K;
};

class Conv2DLibBuilder : public ModelLibBuilder {
public:
  Conv2DLibBuilder(const std::string &modelName, const int N, const int C,
      const int H, const int W, const int kH, const int kW, const int autoPad,
      const int pHBegin, const int pHEnd, const int pWBegin, const int pWEnd,
      const int stride, const int dilation, const int isDynamic);
  bool build();
  bool prepareInputs();
  bool verifyOutputs();

private:
  const std::string getAutoPadName(const int autoPad);
  bool verifyShapeAndComputeBeginEnd();

  // Data that defines model.
  const int N, C, H, W, kH, kW, autoPad;
  int pHBegin, pHEnd, pWBegin, pWEnd;
  const int stride, dilation, isDynamic;
  int NOut, COut, HOut, WOut;
};
