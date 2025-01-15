/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- Accelerator.hpp ---------------------------===//
//
// Copyright 2022-2024 The IBM Research Authors.
//
// =============================================================================
//
// Accelerator base class
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_ACCELERATOR_H
#define ONNX_MLIR_ACCELERATOR_H

#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/SmallVector.h"

#include "include/onnx-mlir/Compiler/OMCompilerTypes.h"
#include "src/Accelerators/Accelerators.inc"

// TODO: Remove NNPA from this header
#include "src/Accelerators/NNPA/Compiler/NNPACompilerOptions.hpp"

// Define the macros used to generate various accelerators artifacts (via the
// use of the APPLY_TO_ACCELERATORS macro, which is defined in the cmake
// generated file Accelerators.inc).
#define CREATE_ACCEL_ENUM(name) name,
#define DECLARE_ACCEL_INIT_FUNCTION(name) extern Accelerator *create##name();
#define INVOKE_ACCEL_INIT_FUNCTION(name, kinds)                                \
  if (!(kinds).empty() &&                                                      \
      llvm::is_contained(kinds, accel::Accelerator::Kind::name))               \
    create##name()->setName(#name);
#define CREATE_ACCEL_CL_ENUM(name)                                             \
  clEnumValN(accel::Accelerator::Kind::name, #name, #name " accelerator"),
#define ACCEL_CL_ENUM_FROM_STRING(name, var, str)                              \
  if ((str).compare(std::string(#name)) == 0) {                                \
    (var) = accel::Accelerator::Kind::name;                                    \
    return true;                                                               \
  }
#define ACCEL_CL_ENUM_TO_STRING(name, map)                                     \
  map[accel::Accelerator::Kind::name] = #name;

#define ACCEL_INSTRUMENTSTAGE_ENUM(name) INSTRUMENTSTAGE_ENUM_##name

#define ACCEL_INSTRUMENTSTAGE_CL_ENUM(name) INSTRUMENTSTAGE_CL_ENUM_##name

#define ACCEL_PROFILEIR_CL_ENUM(name) PROFILEIR_CL_ENUM_##name

#define ACCEL_OPTREPORT_ENUM(name) OPTREPORT_ENUM_##name

#define ACCEL_OPTREPORT_CL_ENUM(name) OPTREPORT_CL_ENUM_##name

namespace onnx_mlir {
namespace accel {

class Accelerator {
public:
  /// Kinds of accelerators.
  enum class Kind {
    // clang-format off
    NONE,
    APPLY_TO_ACCELERATORS(CREATE_ACCEL_ENUM)
    // clang-format on
  };

  Accelerator(Kind kind) : kind(kind) {}
  virtual ~Accelerator() = default;

  /// Getter for the kind of this accelerator.
  Kind getKind() const { return kind; }

  /// Returns the set of accelerators available.
  static const llvm::SmallVectorImpl<Accelerator *> &getAccelerators();

  /// Getter for the name of this accelerator.
  std::string getName() { return name; }

  /// Setter for the name of this accelerator.
  void setName(std::string _name) { name = _name; }

  /// Returns the version number of the accelerator library.
  /// Version number format: 0x[major][minor][patch]
  virtual uint64_t getVersionNumber() const = 0;

  //===--------------------------------------------------------------------===//
  // Hooks for onnx-mlir driver
  //===--------------------------------------------------------------------===//

  /// Add the transformations necessary to support the accelerator.
  virtual void addPasses(mlir::OwningOpRef<mlir::ModuleOp> &module,
      mlir::PassManager &pm, onnx_mlir::EmissionTargetType &emissionTarget,
      std::string outputNameNoExt) const = 0;

  //===--------------------------------------------------------------------===//
  // Hooks for onnx-mlir-opt driver
  //===--------------------------------------------------------------------===//

  /// Register the MLIR dialects required to support an accelerator.
  virtual void registerDialects(mlir::DialectRegistry &registry) const = 0;

  /// Register accelerator transformation passes to make available as
  /// command line options.
  virtual void registerPasses(int optLevel) const = 0;

  //===--------------------------------------------------------------------===//
  // Hooks for both onnx-mlir and onnx-mlir-opt drivers
  //===--------------------------------------------------------------------===//

  /// Configure passes for the accelerator.
  virtual void configurePasses() const = 0;

  //===--------------------------------------------------------------------===//
  // Hooks for onnx-to-krnl pass
  //===--------------------------------------------------------------------===//

  /// Convert TensorType to MemRefType.
  /// Accelerators may have special versions of TensorType. If not, override
  /// this method and return nullptr.
  virtual mlir::MemRefType convertTensorTypeToMemRefType(
      const mlir::TensorType tensorType) const = 0;

  // Return the default alignment value used when allocating a MemRef buffer for
  // the given type. E.g. some special types for accelerators requires
  // 4K-aligned buffers.
  // Return -1 if there is no specific alignment.
  virtual int64_t getDefaultAllocAlignment(const mlir::TensorType type) const {
    return -1;
  }

  /// Define conversion target to be used with ONNXToKrnl.
  virtual void conversionTargetONNXToKrnl(
      mlir::ConversionTarget &target) const = 0;

  /// Define rewrite patterns to be used with ONNXToKrnl.
  virtual void rewritePatternONNXToKrnl(mlir::RewritePatternSet &patterns,
      mlir::TypeConverter &typeConverter, mlir::MLIRContext *ctx) const = 0;

  //===--------------------------------------------------------------------===//
  // Hooks for krnl-to-llvm pass
  //===--------------------------------------------------------------------===//

  /// Define conversion target to be used with KrnlToLLVM.
  virtual void conversionTargetKrnlToLLVM(
      mlir::ConversionTarget &target) const = 0;

  /// Define rewrite patterns to be used with KrnlToLLVM.
  virtual void rewritePatternKrnlToLLVM(mlir::RewritePatternSet &patterns,
      mlir::LLVMTypeConverter &typeConverter, mlir::MLIRContext *ctx) const = 0;

protected:
  static llvm::SmallVector<Accelerator *, 4> acceleratorTargets;
  /// Kind of accelerator.
  Kind kind;

private:
  /// Name of accelerator.
  std::string name = "";
};

// Help to print accelerator kinds.
std::ostream &operator<<(std::ostream &out, const Accelerator::Kind kind);
llvm::raw_ostream &operator<<(
    llvm::raw_ostream &out, const Accelerator::Kind kind);

extern void initAccelerators(llvm::ArrayRef<Accelerator::Kind> kinds);

} // namespace accel
} // namespace onnx_mlir
#endif
