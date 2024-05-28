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
#include "llvm/Support/FileUtilities.h"

#include "src/Compiler/CompilerOptions.hpp"
#include "src/Compiler/CompilerUtils.hpp"
#include "src/Dialect/ONNX/ONNXOps.hpp"
#include "src/Runtime/ExecutionSession.hpp"

namespace onnx_mlir {
namespace test {

const static float omDefaultRangeBound = 1.0;

/*
   Superclass that defines a template to create models, creating an ONNX
   function programmatically, then compiling, loading, running and testing the
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

  ===========================================================================
  Advice for debugging models. If you have a model that fails.

  1 Isolate the failing model by changing the TestXXX.cpp to only generate that
    one.
  2 Execute the test while setting this env: TEST_ARGS="--preserveMLIR"
    Debug/bin/TestXXX .
  3 Then you should have a TestXXX_main_graph.input.mlir in the current
    directory.
  4 You should now be able to call "onnx-mlir -O3 -mlir-print-ir-after-all
    TestXXX_main_graph.input.mlir" with a older good compiler and the newer
    failing compiler, and hopefully find out where the issue might be.

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
  // Check whether a particular instruction extracted from environment variable
  // specified in the argument is included in the dynamic library file name
  // compiled here. If not found, return false.
  bool checkInstructionFromEnv(const std::string envCheckInstruction);
  // Check whether a particular instruction specified in the argument is
  // included in the dynamic library file name compiled here.
  // If not found, return false.
  // TODO: set multiple instructions
  bool checkInstruction(const std::string instructionName);
  // Prepare inputs for running model. Subclass may add arguments as necessary.
  // It can run second or third.
  virtual bool prepareInputs() = 0;
  // Run model using prepared inputs, resulting in outputs. It must run fourth.
  bool run();
  // Verify outputs from a run with reference data. It can run last.
  virtual bool verifyOutputs() = 0;

  // Helper functions.
  // Set the random number generator seed to the value passed by the environment
  // variable; if not found, use a random seed. Optional call to enable
  // reproducible random numbers.
  static void setRandomNumberGeneratorSeed(const std::string &envVar);

  static std::map<std::string, std::string> getTestConfigFromEnv(
      const std::string &envVar);

  static std::vector<float> getDataRangeFromEnv(const std::string &envVar);

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
  bool areCloseFloat(const OMTensor *res, const OMTensor *ref,
      float defaultRtol = 1e-5, float defaultAtol = 1e-5) const;
  // Print indices rank and values, for debugging.
  void printIndices(
      const std::string message, const std::vector<int64_t> &indices) const;
  // Print tensor, as a python numpy array if requested, for debugging.
  void printTensor(
      const std::string varName, const OMTensor *t, bool asNumpy = true) const;

  // Data for building and compiling the model.
  const std::string sharedLibBaseName; // Name for the library.
  mlir::MLIRContext ctx;   // Context for the model (used until compilation).
  mlir::Location loc;      // Location for the model (used during building).
  mlir::OpBuilder builder; // Builder (used during building)
  mlir::ModuleOp module;   // Code for the model (used until compilation)

  // Data for running the model (freed in destructor).
  OMTensorList *inputs, *outputs;
  onnx_mlir::ExecutionSession *exec;

private:
  // Helper recursive function to print tensors.
  void printTensor(const OMTensor *t, std::vector<int64_t> &indices,
      bool isLast = false) const;
};

#define MAX_INPUT_RANK_FOR_TEST 3
template <typename T1, typename T2>
class CategoryMapperLibBuilder : public ModelLibBuilder {
  // Ensure template is instantiated with expected types.
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
    llvm::SmallVector<int64_t> cat_int64s;
    llvm::SmallVector<llvm::StringRef> cat_strings;
    int64_t default_int;
    llvm::StringRef default_string;
  };

  CategoryMapperLibBuilder(std::string name, const CMAttributes &attributes,
      llvm::ArrayRef<T1> input, llvm::ArrayRef<T2> expOutput, int inputRank)
      : ModelLibBuilder(name), attributes(attributes), input(input),
        expOutput(expOutput), inputRank(inputRank) {
    assert(input.size() == expOutput.size() &&
           "Expecting input/expOutput to have the same size");
    inputShape[0] = inputShape[1] = inputShape[2] = 0;
    switch (inputRank) {
    case 1:
      inputShape[0] = static_cast<int64_t>(input.size());
      break;
    case 2:
      assert(((input.size() % 2) == 0) &&
             "CategoryMapperLibBuilder: invalid input size");
      inputShape[0] = static_cast<int64_t>(input.size() / 2);
      inputShape[1] = 2;
      break;
    case 3:
      assert(((input.size() % 6) == 0) &&
             "CategoryMapperLibBuilder: invalid input size");
      inputShape[0] = static_cast<int64_t>(input.size() / 6);
      inputShape[1] = 2;
      inputShape[2] = 3;
      break;
    default:
      llvm_unreachable("CategoryMapperLibBuilder: non supported rank");
    }
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
  const CMAttributes &attributes;              // CategoryMapper attributes.
  const llvm::ArrayRef<T1> input;              // model input data.
  const llvm::ArrayRef<T2> expOutput;          // expected result.
  const int inputRank;                         // rank of input
  int64_t inputShape[MAX_INPUT_RANK_FOR_TEST]; // shape of input
};

class GemmLibBuilder : public ModelLibBuilder {
public:
  GemmLibBuilder(const std::string &modelName, const int I, const int J,
      const int K, const int aTrans, const int bTrans, const int cRank,
      const float alphaVal, const float betaVal);
  bool build() final;
  bool prepareInputs() final;
  bool prepareInputs(float dataRangeLB, float dataRangeUB);
  bool prepareInputsFromEnv(const std::string envDataRange);
  bool verifyOutputs() final;

private:
  // Data that defines model.
  const int I, J, K, aTrans, bTrans, cRank;
  const float alphaVal, betaVal;
  // Derived data that defines model.
  llvm::SmallVector<int64_t, 2> aShape, bShape, cShape;
};

class ScanLibBuilder : public ModelLibBuilder {
public:
  ScanLibBuilder(const std::string &modelName, const int /*seq=*/S,
      const int /*inner-dim=*/I, const int /*batch=*/B, const bool is_v8);
  bool build() final;
  bool prepareInputs() final;
  bool prepareInputs(float dataRangeLB, float dataRangeUB);
  bool prepareInputsFromEnv(const std::string envDataRange);
  bool verifyOutputs() final;

private:
  // Data that defines model.
  const int S, I, B;
  const bool is_v8;
  // Derived data that defines model.
  llvm::SmallVector<int64_t, 2> initialShape, xShape;
  // model definition in std::string
  std::string moduleIR;
};

class LeakyReluLibBuilder : public ModelLibBuilder {
public:
  LeakyReluLibBuilder(
      const std::string &modelName, const int N, const float alpha);
  bool build() final;
  bool prepareInputs() final;
  bool prepareInputs(float dataRangeLB, float dataRangeUB);
  bool prepareInputsFromEnv(const std::string envDataRange);
  bool verifyOutputs() final;

private:
  // Data that defines model.
  const int N;
  const float alphaVal;
  // Derived data that defines model.
  llvm::SmallVector<int64_t, 2> xShape, yShape;
  // model definition in std::string
  std::string moduleIR;
};

// 2x2 matmul with no broadcast
class MatMul2DLibBuilder : public ModelLibBuilder {
public:
  MatMul2DLibBuilder(
      const std::string &modelName, const int I, const int J, const int K);
  bool build() final;
  bool prepareInputs() final;
  bool prepareInputs(float dataRangeLB, float dataRangeUB);
  bool prepareInputsFromEnv(const std::string envDataRange);
  bool verifyOutputs() final;

private:
  // Data that defines model.
  const int I, J, K;
};

// Matmul where there is broadcasting in either A or B, but not both.
// If broadcasting A, then A has a higher rank; if broadcasting B, then B has a
// higher rank.
class MatMulSingleBroadcastLibBuilder : public ModelLibBuilder {
public:
  // When broadcastingB is true, then the rank of B > rank of A=2. When
  // broadcastingB is false, then the rank of A > rank of B=2.
  // But when sameStaticBroadcast, then both A & B's rank >2, and they must have
  // the same static broadcasting ranks. The broadcasted dimensions are given by
  // broadcastDims, and the traditional 2D matrix multiplication dims are given
  // by I, J, and K.
  MatMulSingleBroadcastLibBuilder(const std::string &modelName,
      bool broadcastingB, bool sameStaticBroadcast,
      std::vector<int64_t> broadcastDims, const int I, const int J,
      const int K);
  bool build() final;
  bool prepareInputs() final;
  bool prepareInputs(float dataRange);
  bool verifyOutputs() final;

private:
  // Compute one matmul for a given broadcast
  void computeOneMatMul(OMTensor *a, OMTensor *b, OMTensor *c,
      std::vector<int64_t> &aIndexValues, std::vector<int64_t> &bIndexValues,
      std::vector<int64_t> &yIndexValues);
  // Data that defines model.
  bool broadcastingB;
  bool sameStaticBroadcast;
  std::vector<int64_t> broadcastDims;
  const int I, J, K;
  // Computed data from inputs.
  std::vector<int64_t> aShape, bShape, yShape;
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
  Conv2DLibBuilder(const std::string &modelName, const int N, const int Cin,
      const int Cout, const int H, const int W, const int kH, const int kW,
      const ConvAutoPad autoPad, const int pHBegin, const int pHEnd,
      const int pWBegin, const int pWEnd, const int stride, const int dilation,
      const int isDynamic);
  bool build() final;
  bool prepareInputs() final;
  bool prepareInputs(float dataRangeLB, float dataRangeUB);
  bool prepareInputsFromEnv(const std::string envDataRange);
  bool verifyOutputs() final;

  static const std::string getAutoPadName(const ConvAutoPad autoPad);

private:
  bool verifyShapeAndComputeBeginEnd();

  // Data that defines model, where const define model, non-const are derived
  // parameters.
  const int N, CIn, COut, H, W, kH, kW;
  const ConvAutoPad autoPad;
  int pHBegin, pHEnd, pWBegin, pWEnd;
  const int stride, dilation, isDynamic;
  int modelNOut, modelCOut, modelHOut, modelWOut;
};

class RNNModelLibBuilder : public ModelLibBuilder {
public:
  RNNModelLibBuilder(const std::string &sharedLibBaseName, int64_t layout);
  virtual ~RNNModelLibBuilder();

protected:
  // To transpose between [batch_size, seq_length/num_directions, size]
  //                  and [seq_length/num_directions, batch_size, size]
  // when layout == 1.
  llvm::SmallVector<int64_t, 3> perm3(int64_t a, int64_t b, int64_t c) const {
    if (layout == 0)
      return {a, b, c};
    else
      return {b, a, c};
  }

  // To transpose from [seq_length, num_directions, batch_size, hidden_size]
  //                to [batch_size, seq_length, num_directions, hidden_size]
  // when layout == 1.
  llvm::SmallVector<int64_t, 4> perm4(
      int64_t s, int64_t d, int64_t b, int64_t h) const {
    if (layout == 0)
      return {s, d, b, h};
    else
      return {b, s, d, h};
  }

  const int64_t layout;
};

class LSTMLibBuilder : public RNNModelLibBuilder {
public:
  LSTMLibBuilder(const std::string &modelName, const int direction, const int S,
      const int B, const int I, const int H, const bool isDynamicS,
      const bool isDynamicB, const bool isNoneH = false,
      const bool isNoneC = false, const bool isNoneP = false,
      const int layout = 0);
  virtual ~LSTMLibBuilder();
  bool build() final;
  bool prepareInputs() final;
  bool prepareInputs(float dataRangeLB, float dataRangeUB);
  bool prepareInputsFromEnv(const std::string envDataRange);
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

class GRULibBuilder : public RNNModelLibBuilder {
public:
  GRULibBuilder(const std::string &modelName, const int direction, const int S,
      const int B, const int I, const int H, const int linearBeforeReset,
      const bool isDynamicS, const bool isDynamicB, const int layout = 0);
  virtual ~GRULibBuilder();
  bool build() final;
  bool prepareInputs() final;
  bool prepareInputs(float dataRangeLB, float dataRangeUB);
  bool prepareInputsFromEnv(const std::string envDataRange);
  bool verifyOutputs() final;

private:
  // Data that defines model.
  const int direction, S, B, I, H, linearBeforeReset, isDynamicS, isDynamicB;
  // Computed parameters.
  int D;
  llvm::SmallVector<int64_t, 3> xShape, hShape;
  OMTensor *wOmt, *rOmt, *bOmt;
};

class RNNLibBuilder : public RNNModelLibBuilder {
public:
  RNNLibBuilder(const std::string &modelName, const int direction, const int S,
      const int B, const int I, const int H, const bool isDynamicS,
      const bool isDynamicB, const int layout = 0);
  virtual ~RNNLibBuilder();
  bool build() final;
  bool prepareInputs() final;
  bool prepareInputs(float dataRangeLB, float dataRangeUB);
  bool prepareInputsFromEnv(const std::string envDataRange);
  bool verifyOutputs() final;

private:
  // Data that defines model.
  const int direction, S, B, I, H, isDynamicS, isDynamicB;
  // Computed parameters.
  int D;
  llvm::SmallVector<int64_t, 3> xShape, hShape;
  OMTensor *wOmt, *rOmt, *bOmt;
};

// 2D elementwise with no broadcast
class Elementwise2DLibBuilder : public ModelLibBuilder {
public:
  Elementwise2DLibBuilder(const std::string &modelName,
      const std::string &onnxOpName, const int I, const int J);
  bool build() final;
  bool prepareInputs() final;
  bool prepareInputs(float dataRangeLB, float dataRangeUB);
  bool prepareInputsFromEnv(const std::string envDataRange);
  bool verifyOutputs() final;

private:
  // Data that defines model.
  std::string onnxOpName;
  const int I, J;
  const int inputNum;
};

class SoftplusLibBuilder : public ModelLibBuilder {
public:
  SoftplusLibBuilder(
      const std::string &modelName, const int N);
  bool build() final;
  bool prepareInputs() final;
  bool prepareInputs(float dataRangeLB, float dataRangeUB);
  bool prepareInputsFromEnv(const std::string envDataRange);
  bool verifyOutputs() final;

private:
  // Data that defines model.
  const int N;
  // Derived data that defines model.
  llvm::SmallVector<int64_t, 2> xShape, yShape;
  // model definition in std::string
  std::string moduleIR;
};

} // namespace test
} // namespace onnx_mlir
