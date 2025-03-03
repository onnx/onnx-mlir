/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===------------------------ CompilerOptions.hpp -------------------------===//
//
// Copyright 2022-2025 The IBM Research Authors.
//
// =============================================================================
//
// Functions for adding options.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_COMPILER_OPTIONS_H
#define ONNX_MLIR_COMPILER_OPTIONS_H

#include "onnx-mlir/Compiler/OMCompilerTypes.h"
#include "src/Accelerators/Accelerator.hpp"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include <map>
#include <set>
#include <string>
#include <vector>

#define DEFAULT_INSTRUMENTSTAGE_CL_ENUM clEnumVal(Onnx, "Profile for onnx ops.")

// Variable contains the name of the default environment variable that is used
// to find the default onnx-mlir options.
// Its default value is ONNX_MLIR_FLAGS, as defined in CompilerOptions.cpp.
// TODO: may want to do this constant set by a variable in CMakeFiles.
extern const std::string OnnxMlirEnvOptionName;

namespace onnx_mlir {

typedef enum {
  // clang-format off
  None,
  Onnx
  APPLY_TO_ACCELERATORS(ACCEL_INSTRUMENTSTAGE_ENUM)
  // clang-format on
} InstrumentStages;

using ProfileIRs = InstrumentStages;

typedef enum {
  // clang-format off
  small,
  medium,   // Reserve for future.
  large,
  huge      // Reserve for future.
  // clang-format on
} ModelSize;

typedef enum {
  // clang-format off
  NoReport,
  Parallel,  // Generates diagnostic reporting for parallel (krnl lowering).
  Simd       // Generates diagnostic reporting for SIMD (krnl lowering).
  APPLY_TO_ACCELERATORS(ACCEL_OPTREPORT_ENUM)
  // clang-format on
} OptReport;

extern const std::string modelSizeStr[];

// Common options shared between onnx-mlir and onnx-mlir-opt.
extern llvm::cl::OptionCategory OnnxMlirCommonOptions;
// Options for onnx-mlir only.
extern llvm::cl::OptionCategory OnnxMlirOptions;
// Options for onnx-mlir-opt only.
extern llvm::cl::OptionCategory OnnxMlirOptOptions;

// Options known to onnx-mlir and/or onnx-mlir-opt
extern std::string inputFilename;                             // common for both
extern std::string outputBaseName;                            // common for both
extern std::vector<accel::Accelerator::Kind> maccel;          // common for both
extern OptLevel OptimizationLevel;                            // common for both
extern std::string mtriple;                                   // common for both
extern std::string mcpu;                                      // common for both
extern float nnpaEpsilon;                                     // common for both
extern std::string march;                                     // common for both
extern InstrumentStages instrumentStage;                      // common for both
extern bool onnxConstPropRoundFPToInt;                        // common for both
extern int onnxConstPropExpansionBound;                       // common for both
extern std::vector<std::string> onnxConstPropDisablePatterns; // common for both
extern bool enableONNXHybridPass;                             // common for both
extern std::vector<std::string> functionsToDecompose;         // common for both
extern std::string opsForCall;                                // common for both
extern bool disableKrnlOpFusion;                              // common for both
extern bool disableQuantZeroPoint;                            // common for both
extern bool enableKrnlBufferReuse;                            // common for both
extern bool enableSafeCodeGen;                                // common for both
extern bool disableMemRefPrefetch;                            // common for both
extern uint64_t compilationNumThreads;                        // common for both
extern EmissionTargetType emissionTarget;                     // onnx-mlir only
extern bool invokeOnnxVersionConverter;                       // onnx-mlir only
extern bool preserveLocations;                                // onnx-mlir only
extern bool printIR;                                          // onnx-mlir only
extern bool preserveBitcode;                                  // onnx-mlir only
extern bool preserveLLVMIR;                                   // onnx-mlir only
extern bool preserveMLIR;                                     // onnx-mlir only
extern bool doNotEmitFullMLIRCode;                            // onnx-mlir only
extern bool useOnnxModelTypes;                                // onnx-mlir only
extern int repeatOnnxTransform;                               // onnx-mlir only
extern std::string shapeInformation;                          // onnx-mlir only
extern std::string dimParams;                                 // onnx-mlir only
extern ModelSize modelSize;                                   // onnx-mlir only
extern bool storeConstantsToFile;                             // onnx-mlir only
extern float constantsToFileTotalThreshold;                   // onnx-mlir only
extern float constantsToFileSingleThreshold;                  // onnx-mlir only
extern bool VerboseOutput;                                    // onnx-mlir only
extern std::vector<std::string> Xopt;                         // onnx-mlir only
extern std::vector<std::string> Xllc;                         // onnx-mlir only
extern std::string mllvm;                                     // onnx-mlir only
extern std::string instrumentOps;                             // onnx-mlir only
extern unsigned instrumentControlBits;                        // onnx-mlir only
extern std::string parallelizeOps;                            // onnx-mlir only
extern std::string instrumentSignatures;                      // onnx-mlir only
extern std::string instrumentOnnxNode;                        // onnx-mlir only
extern std::string ONNXOpStats;                               // onnx-mlir only
extern int onnxOpTransformThreshold;                          // onnx-mlir only
extern bool onnxOpTransformReport;                            // onnx-mlir only
extern bool enableParallel;                                   // onnx-mlir only
extern bool disableSimdOption;                                // onnx-mlir only
extern bool enableFastMathOption;                             // onnx-mlir only
extern bool disableRecomposeOption;                           // onnx-mlir only
extern bool enableSimdDataLayout;                             // onnx-mlir only
extern bool verifyInputTensors;                               // onnx-mlir only
extern bool allowSorting;                                     // onnx-mlir only
extern std::vector<std::string> reportHeapBefore;             // onnx-mlir only
extern std::vector<std::string> reportHeapAfter;              // onnx-mlir only
extern std::string modelTag;                                  // onnx-mlir only
extern bool enableConvOptPass;                                // onnx-mlir only
extern bool disableConstantProp;                              // onnx-mlir only
extern std::vector<std::string> extraLibPaths;                // onnx-mlir only
extern std::vector<std::string> extraLibs;                    // onnx-mlir only
extern ProfileIRs profileIR;                                  // onnx-mlir only
extern OptReport optReport;                                   // onnx-mlir only
extern bool useOldBufferization;                              // onnx-mlir only
extern bool enableTiming;                                     // onnx-mlir only
extern bool enableBoundCheck;                                 // onnx-mlir only
extern bool debugTestCompilerOpt;                             // onnx-mlir only

extern bool split_input_file;          // onnx-mlir-opt only
extern bool verify_diagnostics;        // onnx-mlir-opt only
extern bool verify_passes;             // onnx-mlir-opt only
extern bool allowUnregisteredDialects; // onnx-mlir-opt only

extern std::string customEnvFlags;

// The customEnvFlags must be scanned before the normal options.
bool parseCustomEnvFlagsCommandLineOption(int argc, const char *const *argv,
    llvm::raw_ostream *errs = (llvm::raw_ostream *)nullptr);

void setCustomEnvVar(const std::string &envVarName);
void clearCustomEnvVar();
std::string getCustomEnvVarOption();

void setTargetTriple(const std::string &triple);
void clearTargetTriple();
std::string getTargetTripleOption();

void setTargetArch(const std::string &arch);
void clearTargetArch();
int64_t getZArchNum(const std::string &arch, const std::string cpu);
std::string getTargetArchOption(bool forLLVMToolchain = false);

void setTargetCPU(const std::string &cpu);
void clearTargetCPU();
std::string getTargetCPUOption(
    bool forLLVMToolchain = false, bool cpuOnly = false);

int setTargetAccel(const std::string &str);
void setTargetAccel(const accel::Accelerator::Kind accel);
void clearTargetAccel();
std::string getTargetAccel();

void setOptLevel(const onnx_mlir::OptLevel level);
void clearOptLevel();
std::string getOptimizationLevelOption();

void setXoptOption(const std::vector<std::string> &flags);
void clearXoptOption();
std::vector<std::string> getXoptOption();

void setXllcOption(const std::vector<std::string> &flags);
void clearXllcOption();
std::vector<std::string> getXllcOption();

void setLLVMOption(const std::string &flag);
void clearLLVMOption();
std::string getLLVMOption();
// Break down the result of getLLVMOption into substrings
std::vector<std::string> getLLVMOptions();
std::vector<std::string> getLLVMOPTOptions();
std::vector<std::string> getLLVMLLCOptions();

// Options support for OMCompilerOptions.
using CompilerOptionList =
    llvm::SmallVector<std::pair<onnx_mlir::OptionKind, std::string>, 4>;

#define CCM_SHARED_LIB_DEPS "sharedLibDeps"
#define CCM_SHARED_LIB_PATH_DEPS "sharedLibPathDeps"
extern std::map<std::string, std::vector<std::string>> CompilerConfigMap;

// Return 0 on success. These functions are not thread-safe and should be called
// by a single program thread.
int setCompilerOption(const onnx_mlir::OptionKind kind, const std::string &val);
int setCompilerOptions(const CompilerOptionList &list);
void clearCompilerOption(const onnx_mlir::OptionKind kind);
std::string getCompilerOption(const onnx_mlir::OptionKind kind);

// The add and del functions are not thread-safe and should only be
// called from one thread.
std::vector<std::string> getCompilerConfig(std::string k);
void addCompilerConfig(
    std::string k, std::vector<std::string> v, bool head = false);
void delCompilerConfig(std::string k, std::vector<std::string> v);

// Functions related to initializing compiler configuration states based on
// parsed options
std::optional<std::string> getEnvVar(std::string name);
std::string getExecPath();
std::string getLibraryPath();
std::string getToolPath(const std::string &tool, bool flag = false);
void removeUnrelatedOptions(
    const std::vector<llvm::cl::OptionCategory *> Categories);
void initCompilerConfig();

} // namespace onnx_mlir
#endif
