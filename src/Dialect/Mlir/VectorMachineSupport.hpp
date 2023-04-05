/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- VectorMachineSupport.hpp - Helper for what SIMD ops are supported -===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================
//
// Support to determine which high level op is supported on a given target.
// Does not need to be exact (as LLVM backend can lower any vector code),
// however it is at time useful to have a rough idea of what is eventually
// supported by the target hardware to better direct high level optimizations.
//
//===----------------------------------------------------------------------===//

#pragma once

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
  AbsGop,
  ArithmeticGop, /* Simple compute ops: add/sub/neg + ops of same complexity */
  CeilDivGop,
  CeilGop,
  CompareGop, /* All compare operations, signed/unsigned fixed/float. */
  ConversionGop,
  CopySignGop,
  DivGop,
  ExpGop,
  FloorDivGop,
  FloorGop,
  FmaGop,
  LogGop,
  LogicalGop, /* All logical ops: and, or, xor, not, nor, nand,... */
  MinMaxGop,
  MulGop,
  PowGop,
  RemGop,
  RoundGop,
  SelectGop,
  ShiftGop,   /* Shift operations: logical/arithmetic. */
  ShuffleGop, /* All bit/byte moving operations: shuffle, rotate, shift. */
  SqrtGop,
  SumAcrossGop,      /* Sum across vector. */
  TrigArcGop,        /* Arc trigonometry ops: asin, acos, atan. */
  TrigGop,           /* Trigonometry ops: sin, cos, tan. */
  TrigHyperbolicGop, /* Hyperbolic trig. */
};

//===----------------------------------------------------------------------===//
// Generic vector machine support class, which must be refined for each
// supported machine type.

class VectorMachineSupport {
protected:
  VectorMachineSupport() = default;
  virtual ~VectorMachineSupport() = default;

public:
  // The class encapsulate a single static vector machine support
  static VectorMachineSupport *getGlobalVectorMachineSupport() {
    assert(globalVectorMachineSupport && "expected nonnull ptr");
    return globalVectorMachineSupport;
  }
  static void setGlobalVectorMachineSupport(
      std::string arch, std::string cpu, std::string attr);
  static void clearGlobalVectorMachineSupport();
  static bool hasSimd();

  // Has simd (check num registers>0). Requires an initialized vector machine
  // support.
  static const int64_t UNSUPPORTED = 0;

  // Number of vector registers available.
  virtual int64_t VectorRegisterNum() = 0;

  // Return the bit width of the SIMD unit regardless of the type/operation.
  // This is an upper bound and does not guarantee that an actual operation can
  // provide this VL. A value of zero means no SIMD available.
  virtual int64_t getVectorBitWidth() = 0;
  // Return the number of elements that can be processed in SIMD fashion
  // regardless of the operation. This is an upper bound and does not guarantee
  // that an actual operation can provide this VL. A value of zero means no SIMD
  // available.
  virtual int64_t getVectorLength(mlir::Type elementType);
  // Return the number of elements that can be processed in SIMD fashion if
  // support exists. A value of zero means no SIMD available.
  virtual int64_t getVectorLength(GenericOps gop, mlir::Type elementType) = 0;

  // Analyze the benefits of using SIMD on a list of generic ops in an algorithm
  // where each op on the list occurs a given number of times. The function
  // returns the weighted average vector length among the operations listed in
  // the gops list, where each operation gops[i] occur exactly gopsNum[i] times
  // in the algorithm. Note that scalar operation have a vector length of
  // one in the weighted average as they still contribute one result. The opNums
  // are also weighted by the gopsNum to better represent the mix of
  // vectorized and scalar operations present in the algorithm.
  double getAvgVectorLength(mlir::ArrayRef<GenericOps> &gops,
      mlir::ArrayRef<int64_t> &gopsNum, mlir::Type elementType,
      int64_t &vectorizedOpNum, int64_t &scalarOpNum);

private:
  static VectorMachineSupport *globalVectorMachineSupport;
};

// No support for SIMD.
class NoVectorMachineSupport : public VectorMachineSupport {
public:
  NoVectorMachineSupport() = default;
  virtual ~NoVectorMachineSupport() = default;

  int64_t VectorRegisterNum() override { return 0; }
  int64_t getVectorBitWidth() override { return 0; }
  int64_t getVectorLength(mlir::Type elementType) override {
    return UNSUPPORTED;
  }
  int64_t getVectorLength(GenericOps gop, mlir::Type elementType) override {
    return UNSUPPORTED;
  }
};

// Support for IBM Z servers.

class Z16VectorMachineSupport : public VectorMachineSupport {
public:
  Z16VectorMachineSupport() = default;
  virtual ~Z16VectorMachineSupport() = default;

  int64_t VectorRegisterNum() override { return 32; }
  int64_t getVectorBitWidth() override { return 128; }
  int64_t getVectorLength(GenericOps gop, mlir::Type elementType) override;
};

// TODO: create models for z14 and z15.
using Z14VectorMachineSupport = Z16VectorMachineSupport;
using Z15VectorMachineSupport = Z16VectorMachineSupport;

// Support for x86 processors (SSE 4.2 and AVX2)
class SSE42x86VectorMachineSupport : public VectorMachineSupport {
public:
  SSE42x86VectorMachineSupport() = default;
  virtual ~SSE42x86VectorMachineSupport() = default;

  int64_t VectorRegisterNum() override { return 16; }
  int64_t getVectorBitWidth() override { return 128; }
  int64_t getVectorLength(GenericOps gop, mlir::Type elementType) override;
};

class AVX2x86VectorMachineSupport : public SSE42x86VectorMachineSupport {
public:
  AVX2x86VectorMachineSupport() = default;
  virtual ~AVX2x86VectorMachineSupport() = default;

  int64_t getVectorBitWidth() override { return 258; }
};

} // namespace onnx_mlir