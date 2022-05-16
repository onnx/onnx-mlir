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

#pragma once

#include <string>
#include <type_traits>

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/SmallVector.h"

#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/ExecutionSession.hpp"

namespace onnx_mlir {
namespace test {

/*
   Superclass that defines a template to create models, creating an ONNX
   function programatically, then compiling, loading, runing and testing the
   validity of the results.
   The general flow of using a model is as follow:

     // Create a mat mul object with parameter I, J, K.
     MatMul2DLibBuilder matmul(SHARED_LIB_BASE.str(), I, J, K);
     bool success = matmul.build() &&   // Build onnx function.
            matmul.compileAndLoad() &&  // Compile and load.
            matmul.prepareInputs() &&   // Prepare inputs.
            matmul.run() &&             // Run compiled model.
            matmul.verifyOutputs();     // Optionally verify outputs.

  Each type of model has different parameters. The superclass maintains
  the inputs and outputs for testing. When object is deleted, defined
  tensors will be freed.

  The build function must execute first. The compileAndLoad and
  prepareInputs function can then execute in any orders. The run function
  must execute after compileAndLoad and prepareInputs. VerifyOutputs can only
  execute after the run function.

  A actual model will typically define its model-specific functions, namely
  constructor/destructor to handle custom model variables, the build() function,
  the prepareInputs() function, and the optional verifyOutputs() function.

  The compile function take into account the ONNX-MLIR compiler options.
  They can be set / read from the command line / environment variable
  using the code below (defined in full in the OnnxMlirCompiler.h interface).

    setCompilerOption(OptionKind::CompilerOptLevel, "3"); // Default is O3.
    llvm::cl::ParseCommandLineOptions( // Read from args and TEST_ARGS env var.
      argc, argv, "TestMatMul2D\n", nullptr, "TEST_ARGS");
  The compileAndLoad function can also process compiler options, which will
  remain in effect until changed again.
*/

class ModelLibBuilder {
public:
  // Define the model. Subclasses should add to the builder all of the specific
  // parameters that uniquely define the model.
  ModelLibBuilder(const std::string &sharedLibBaseName);
  // Destructor needed to free the inputs/outputs data structures.
  virtual ~ModelLibBuilder();
  // Default constructor removed.
  ModelLibBuilder() = delete;
  // Build, subclass should generate a graph. If constant nodes are needed by
  // the model, they should be created here and saved in the subclass, as these
  // values will be needed to verify the accuracy of the model. The model is
  // saved in the model and ctx variable. It must run first.
  virtual bool build() = 0;
  // Compile model from the model and ctx variables. The output is an executable
  // dynamic library. It can run second or third.
  bool compileAndLoad();
  bool compileAndLoad(const onnx_mlir::CompilerOptionList &list);
  // Prepare inputs for running model. Subclass may add arguments as necessary.
  // It can run second or third.
  virtual bool prepareInputs() = 0;
  // Run model using prepared inputs, resulting in outputs. It must run fourth.
  bool run();
  // Verify outputs from a run with reference data. It can run last.
  virtual bool verifyOutputs() = 0;

  // Helper functions.
  // Get the dynamic library file name compiled here.
  static std::string getSharedLibName(const std::string &sharedLibBaseName);
  // Set the random number generator seed to the value passed by the environment
  // variable; if not found, use a random seed. Optional call to enable
  // reproducible random numbers.
  static void setRandomNumberGeneratorSeed(const std::string &envVar);

protected:
  // Create a function with an empty body.
  // This function will contain the model to be tested.
  mlir::func::FuncOp createEmptyTestFunction(
      const llvm::SmallVectorImpl<mlir::Type> &inputsType,
      const llvm::SmallVectorImpl<mlir::Type> &outputsType);
  // Create the entry point function (used to call the model test function).
  void createEntryPoint(mlir::func::FuncOp &funcOp);
  // Create a onnx constant op loaded with values in the tensor omt.
  mlir::ONNXConstantOp buildONNXConstantOp(
      const OMTensor *omt, const mlir::RankedTensorType resultType);
  // Compare results as float.
  bool areCloseFloat(const OMTensor *res, const OMTensor *ref) const;

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

template <typename T1, typename T2>
class CategoryMapperLibBuilder : public ModelLibBuilder {
  // Ensure template is instatiated with expected types.
  static_assert((std::is_same<T1, int64_t>::value ||
                    std::is_same<T1, const char *>::value),
      "T1 must be int64_t or const char *");
  static_assert((std::is_same<T1, int64_t>::value &&
                    std::is_same<T2, const char *>::value) ||
                    (std::is_same<T1, const char *>::value &&
                        std::is_same<T2, int64_t>::value),
      "T1 and/or T2 are not correct");

public:
  // CategoryMapper attributes.
  struct CMAttributes {
    llvm::ArrayRef<int64_t> cat_int64s;
    llvm::ArrayRef<llvm::StringRef> cat_strings;
    int64_t default_int;
    llvm::StringRef default_string;
  };

  CategoryMapperLibBuilder(std::string name, const CMAttributes &attributes,
      llvm::ArrayRef<T1> input, llvm::ArrayRef<T2> expOutput)
      : ModelLibBuilder(name), attributes(attributes), input(input),
        expOutput(expOutput) {
    assert(input.size() == expOutput.size() &&
           "Expecting input/expOutput to have the same size");
  }

  bool build() final;
  bool prepareInputs() final;
  bool verifyOutputs() final;

private:
  // Create the function to test.
  void createTestFunction(mlir::Type inputType, mlir::Type outputType,
      const CMAttributes &attributes);

  // Create the category mapper operator, and insert it into the test function.
  void createCategoryMapper(mlir::Type outputType,
      const CMAttributes &attributes, mlir::func::FuncOp &funcOp);

  // Verify that the output tensor has the expected rank.
  bool verifyRank(const OMTensor &out, int64_t rank) const;

  // Verify that the output tensor has the expected number of elements.
  bool verifyNumElements(const OMTensor &out, int64_t numElems) const;

  // Verify that the output tensor contains the expected result.
  bool verifyResults(const OMTensor *out, const OMTensor *expected) const;

private:
  const CMAttributes &attributes;     // CategoryMapper attributes.
  const llvm::ArrayRef<T1> input;     // model input data.
  const llvm::ArrayRef<T2> expOutput; // expected result.
};

class GemmLibBuilder : public ModelLibBuilder {
public:
  GemmLibBuilder(const std::string &modelName, const int I, const int J,
      const int K, const int aTrans, const int bTrans, const int cRank,
      const float alphaVal, const float betaVal);
  bool build() final;
  bool prepareInputs() final;
  bool prepareInputs(float dataRange);
  bool verifyOutputs() final;

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
  bool build() final;
  bool prepareInputs() final;
  bool prepareInputs(float dataRange);
  bool verifyOutputs() final;

private:
  // Data that defines model.
  const int I, J, K;
};

// Padding schemes for Convolutions.
enum ConvAutoPad {
  NOTSET = 0,
  VALID = 1,
  LOWER = 2,
  UPPER = 3,
  UB = 4 // Always the last element.
};

class Conv2DLibBuilder : public ModelLibBuilder {
public:
  Conv2DLibBuilder(const std::string &modelName, const int N, const int C,
      const int H, const int W, const int kH, const int kW,
      const ConvAutoPad autoPad, const int pHBegin, const int pHEnd,
      const int pWBegin, const int pWEnd, const int stride, const int dilation,
      const int isDynamic);
  bool build() final;
  bool prepareInputs() final;
  bool verifyOutputs() final;

  static const std::string getAutoPadName(const ConvAutoPad autoPad);

private:
  bool verifyShapeAndComputeBeginEnd();

  // Data that defines model, where const define model, non-const are derived
  // paramters.
  const int N, C, H, W, kH, kW;
  const ConvAutoPad autoPad;
  int pHBegin, pHEnd, pWBegin, pWEnd;
  const int stride, dilation, isDynamic;
  int NOut, COut, HOut, WOut;
};

class LSTMLibBuilder : public ModelLibBuilder {
public:
  LSTMLibBuilder(const std::string &modelName, const int direction, const int S,
      const int B, const int I, const int H, const bool isDynamicS,
      const bool isDynamicB, const bool isNoneH = false,
      const bool isNoneC = false, const bool isNoneP = false);
  ~LSTMLibBuilder();
  bool build() final;
  bool prepareInputs() final;
  bool verifyOutputs() final;

private:
  // Data that defines model.
  const int direction, S, B, I, H;
  const bool isDynamicS, isDynamicB, isNoneH, isNoneC, isNoneP;
  // Computed parameters.
  int D;
  llvm::SmallVector<int64_t, 3> xShape, hShape, cShape;
  OMTensor *wOmt, *rOmt, *bOmt, *pOmt;
};

class GRULibBuilder : public ModelLibBuilder {
public:
  GRULibBuilder(const std::string &modelName, const int direction, const int S,
      const int B, const int I, const int H, const int linearBeforeReset,
      const bool isDynamicS, const bool isDynamicB);
  ~GRULibBuilder();
  bool build() final;
  bool prepareInputs() final;
  bool verifyOutputs() final;

private:
  // Data that defines model.
  const int direction, S, B, I, H, linearBeforeReset, isDynamicS, isDynamicB;
  // Computed parameters.
  int D;
  llvm::SmallVector<int64_t, 3> xShape, hShape;
  OMTensor *wOmt, *rOmt, *bOmt;
};

class RNNLibBuilder : public ModelLibBuilder {
public:
  RNNLibBuilder(const std::string &modelName, const int direction, const int S,
      const int B, const int I, const int H, const bool isDynamicS,
      const bool isDynamicB);
  ~RNNLibBuilder();
  bool build() final;
  bool prepareInputs() final;
  bool verifyOutputs() final;

private:
  // Data that defines model.
  const int direction, S, B, I, H, isDynamicS, isDynamicB;
  // Computed parameters.
  int D;
  llvm::SmallVector<int64_t, 3> xShape, hShape;
  OMTensor *wOmt, *rOmt, *bOmt;
};

} // namespace test
} // namespace onnx_mlir
