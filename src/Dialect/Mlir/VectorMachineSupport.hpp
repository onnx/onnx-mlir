/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- VectorMachineSupport.hpp - Helper for what SIMD ops are supported -===//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================
//
// Support to determine which high level op is supported on a given target.
// Does not need to be exact (as LLVM backend can lower any vector code),
// however it is at time useful to have a rough idea of what is eventually
// supported by the target hardware to better direct high level optimizations.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_VECTOR_MACHINE_H
#define ONNX_MLIR_VECTOR_MACHINE_H

#include "mlir/IR/Types.h"
#include "llvm/ADT/SmallVector.h"

namespace onnx_mlir {

//===----------------------------------------------------------------------===//
// Generic ops to determine if operations are supported for lowering to vector
// SIMD operations. An additional operation's element type will be provided to
// further refine whether an operation can be effectively vectorized on the
// given target hardware. This list roughly corresponds to the operations
// supported by MathDialectBuilder, with some combining of similar operations
// (e.g. all the compares).

enum class GenericOps {
  /////////////////////////////////////
  // Generic ops.
  /////////////////////////////////////

  AbsGop,
  ArithmeticGop, /* Simple compute ops: add/sub/neg + ops of same complexity. */
  CeilDivGop,
  CeilGop,
  CompareGop, /* All compare operations, signed/unsigned fixed/float. */
  ConversionGop,
  CopySignGop,
  DivGop,
  ExpGop,
  ErfGop,
  FloorDivGop,
  FloorGop,
  FmaGop,
  LogGop,
  LogicalGop, /* All logical ops: and, or, xor, not, nor, nand,... */
  MinMaxGop,
  MinMaxAcrossGop, /* Compute min/max across vector. */
  MulGop,
  PowGop,
  RemGop,
  roundEvenGop,  /* FP to FP round to nearest even ONNX */
  ScalarOnlyGop, /* Any ops that are guaranteed to be scalar on any arch. */
  SelectGop,
  ShiftGop,   /* Shift operations: logical/arithmetic. */
  ShuffleGop, /* All bit/byte moving operations: shuffle, rotate, shift. */
  SqrtGop,
  SumAcrossGop,      /* Sum across vector. */
  TrigArcGop,        /* Arc trigonometry ops: asin, acos, atan. */
  TrigGop,           /* Trigonometry ops: sin, cos, tan. */
  TrigHyperbolicGop, /* Hyperbolic trig. */

  LastGop, /* Marker of the last op. Used to delineate from other metrics. */

  /////////////////////////////////////
  // Metrics others than operations.
  /////////////////////////////////////

  // Metric that provides an estimate of the maximum number of vector registers
  // used in a kernel. If none is provided, we estimate the pressure based on
  // the number of operations.
  EstimatedVectorRegisterPressure,
};

// Describe the mix of Generic operations in a given kernel. Each generic
// operation is associated with a number, which indicates the number of
// occurrence of that generic op in the given kernel.
using GenOpMix = llvm::SmallDenseMap<GenericOps, int64_t, 8>;

GenOpMix computeGenOpMixUnion(const GenOpMix &mix1, const GenOpMix &mix2);

//===----------------------------------------------------------------------===//
// Generic vector machine support class, which must be refined for each
// supported machine type.

class VectorMachineSupport {
protected:
  VectorMachineSupport() = default;
  virtual ~VectorMachineSupport() = default;

public:
  // Must call setGlobalVectorMachineSupport once before using any calls below.
  static void setGlobalVectorMachineSupport(
      const std::string &arch, const std::string &cpu, const std::string &attr);
  static void clearGlobalVectorMachineSupport();

  static std::string getArchName() { return vms()->computeArchName(); }

  // Determine if the machine has simd. Requires an initialized vector machine
  // support.
  static bool hasSimd() { return getArchVectorRegisterNum() > 0; }

  // Determine if custom asm is needed (aka operation not supported by llvm).
  static bool requireCustomASM(GenericOps gop, mlir::Type elementType) {
    return vms()->needCustomASM(gop, elementType);
  }

  // When querying Vector length for machines with unsupported simd, UNSUPPORTED
  // (aka 0) is returned.
  static const int64_t UNSUPPORTED = 1;

  // Number of vector registers available.
  static int64_t getArchVectorRegisterNum() {
    // Indirection to the object specific to a subclass.
    return vms()->computeArchVectorRegisterNum();
  }

  // Return the bit width of the SIMD unit regardless of the type/operation.
  // This is an upper bound and does not guarantee that an actual operation can
  // provide this VL. A value of zero means no SIMD available.
  static int64_t getArchVectorBitWidth() {
    // Indirection to the object specific to a subclass.
    return vms()->computeArchVectorBitWidth();
  }
  // Return the number of elements that can be processed in SIMD fashion
  // regardless of the operation. This is an upper bound and does not guarantee
  // that an actual operation can provide this VL. A value of zero means no SIMD
  // available.
  static int64_t getArchVectorLength(mlir::Type elementType) {
    // Indirection to the object specific to a subclass.
    return vms()->computeArchVectorLength(elementType);
  }

  // Return the number of elements that can be processed in SIMD fashion if
  // support exists. A value of zero means no SIMD available.
  static int64_t getArchVectorLength(GenericOps gop, mlir::Type elementType) {
    // Indirection to the object specific to a subclass.
    return vms()->computeArchVectorLength(gop, elementType);
  }

  // Analyze the benefits of using SIMD on a list of generic ops in an algorithm
  // where each op on the list occurs a given number of times. The function
  // returns the weighted average vector length among the operations listed in
  // the GenOps list, where each entry is a pair of generic operation and the
  // number of times that generic operation was found. Note that scalar
  // operation have a vector length of one in the weighted average as they still
  // contribute one result.
  // Max vector register pressure is also reported, either from an explicit
  // mention in the genOps, or estimated as one vector register per vector
  // operation.
  static double getAvgArchVectorLength(GenOpMix &genOps, mlir::Type elementType,
      int64_t &vectorizedOpNum, int64_t &scalarOpNum,
      int64_t &maxVectorRegisterPressure);

protected:
  // Virtual functions that do the actual work. Called by the "get" functions.
  virtual std::string computeArchName() = 0;
  virtual bool needCustomASM(GenericOps gop, mlir::Type elementType) = 0;
  virtual int64_t computeArchVectorRegisterNum() = 0;
  virtual int64_t computeArchVectorBitWidth() = 0;
  virtual int64_t computeArchVectorLength(mlir::Type elementType);
  virtual int64_t computeArchVectorLength(
      GenericOps gop, mlir::Type elementType) = 0;

private:
  static VectorMachineSupport *vms() {
    assert(globalVectorMachineSupport && "vector machine support undefined");
    return globalVectorMachineSupport;
  }

  static VectorMachineSupport *globalVectorMachineSupport; // Init to null.
};

// No support for SIMD.
class NoVectorMachineSupport : public VectorMachineSupport {
public:
  NoVectorMachineSupport() = default;
  virtual ~NoVectorMachineSupport() = default;

  std::string computeArchName() override { return "no_vector"; }
  bool needCustomASM(GenericOps gop, mlir::Type elementType) override {
    return false;
  }
  int64_t computeArchVectorRegisterNum() override { return 0; }
  int64_t computeArchVectorBitWidth() override { return 0; }
  int64_t computeArchVectorLength(mlir::Type elementType) override {
    return UNSUPPORTED;
  }
  int64_t computeArchVectorLength(
      GenericOps gop, mlir::Type elementType) override {
    return UNSUPPORTED;
  }
};

// Support for IBM Z servers.

class ZArch14VectorMachineSupport : public VectorMachineSupport {
public:
  ZArch14VectorMachineSupport() = default;
  virtual ~ZArch14VectorMachineSupport() = default;

  std::string computeArchName() override { return "z16/arch14 equivalent"; }
  bool needCustomASM(GenericOps gop, mlir::Type elementType) override;
  int64_t computeArchVectorRegisterNum() override { return 32; }
  int64_t computeArchVectorBitWidth() override { return 128; }
  int64_t computeArchVectorLength(
      GenericOps gop, mlir::Type elementType) override;
};

// TODO: create models for arch12, arch13, arch15.
using ZArch12VectorMachineSupport = ZArch14VectorMachineSupport;
using ZArch13VectorMachineSupport = ZArch14VectorMachineSupport;
using ZArch15VectorMachineSupport = ZArch14VectorMachineSupport;

// Support for x86 processors (SSE 4.2 and AVX2)
class SSE42x86VectorMachineSupport : public VectorMachineSupport {
public:
  SSE42x86VectorMachineSupport() = default;
  virtual ~SSE42x86VectorMachineSupport() = default;

  std::string computeArchName() override { return "x86-sse4.2"; }
  bool needCustomASM(GenericOps gop, mlir::Type elementType) override;
  int64_t computeArchVectorRegisterNum() override { return 16; }
  int64_t computeArchVectorBitWidth() override { return 128; }
  int64_t computeArchVectorLength(
      GenericOps gop, mlir::Type elementType) override;
};

class AVX2x86VectorMachineSupport : public SSE42x86VectorMachineSupport {
public:
  AVX2x86VectorMachineSupport() = default;
  virtual ~AVX2x86VectorMachineSupport() = default;

  std::string computeArchName() override { return "x86-avx2"; }
  int64_t computeArchVectorBitWidth() override { return 258; }
};

// Support for Arm 64

class NeonVectorMachineSupport : public VectorMachineSupport {
public:
  NeonVectorMachineSupport() = default;
  virtual ~NeonVectorMachineSupport() = default;

  std::string computeArchName() override { return "arm64-neon"; }
  bool needCustomASM(GenericOps gop, mlir::Type elementType) override;
  int64_t computeArchVectorRegisterNum() override { return 32; }
  int64_t computeArchVectorBitWidth() override { return 128; }
  int64_t computeArchVectorLength(
      GenericOps gop, mlir::Type elementType) override;
};

} // namespace onnx_mlir
#endif
