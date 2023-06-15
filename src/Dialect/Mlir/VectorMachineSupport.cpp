/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- VectorMachineSupport.cpp - Helper for what SIMD ops are supported -===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================

#include "src/Dialect/Mlir/VectorMachineSupport.hpp"
#include "mlir/IR/BuiltinTypes.h"
#include <algorithm>

#define DEBUG_TYPE "dialect_builder"

using namespace mlir;

namespace onnx_mlir {

// =============================================================================
// Handling of global vector machine support pointer

/*static*/ VectorMachineSupport
    *VectorMachineSupport::globalVectorMachineSupport = nullptr;

/*static*/ void VectorMachineSupport::setGlobalVectorMachineSupport(
    std::string arch, std::string cpu, std::string attr) {
  // IBM Z servers use mcpu.
  if (cpu.compare("z14") == 0) {
    globalVectorMachineSupport = new Z14VectorMachineSupport();
  } else if (cpu.compare("z15") == 0) {
    globalVectorMachineSupport = new Z15VectorMachineSupport();
  } else if (cpu.compare("z16") == 0) {
    globalVectorMachineSupport = new Z16VectorMachineSupport();
  } else if (arch.compare("x86-64") == 0) {
    // Intel arch
    if (cpu.compare("skylake") == 0 && attr.compare("avx2") == 0)
      globalVectorMachineSupport = new AVX2x86VectorMachineSupport();
    else
      // Default seems to be SSE
      globalVectorMachineSupport = new SSE42x86VectorMachineSupport();
  } else {
    // Unknown: disable
    globalVectorMachineSupport = new NoVectorMachineSupport();
  }
  assert(globalVectorMachineSupport &&
         "failed to allocate vector machine support");
}

/*static*/ void VectorMachineSupport::clearGlobalVectorMachineSupport() {
  if (!globalVectorMachineSupport)
    return;
  delete globalVectorMachineSupport;
  globalVectorMachineSupport = nullptr;
}

/*static*/ bool VectorMachineSupport::hasSimd() {
  return getGlobalVectorMachineSupport()->VectorRegisterNum() > 0;
}
// =============================================================================
// Methods shared among all VectorMachineSupport classes and subclasses

int64_t VectorMachineSupport::getVectorLength(Type elementType) {
  int64_t simdBitSize = getVectorBitWidth();
  int64_t typeBitSize = elementType.getIntOrFloatBitWidth();
  assert(simdBitSize >= typeBitSize && simdBitSize % typeBitSize == 0 &&
         "bad machine vector length");
  return (simdBitSize / typeBitSize);
}

double VectorMachineSupport::getAvgVectorLength(ArrayRef<GenericOps> &gops,
    ArrayRef<int64_t> &gopsNum, Type elementType, int64_t &vectorizedOpNum,
    int64_t &scalarOpNum) {
  assert(gopsNum.size() == gops.size() && "expect same length for both lists");
  int64_t gopsSize = gops.size();
  int64_t totProcessedValues = 0.0;
  vectorizedOpNum = 0;
  scalarOpNum = 0;
  // Determine which operations support SIMD and accumulate their vector
  // lengths.
  for (int64_t i = 0; i < gopsSize; ++i) {
    int64_t vl = getVectorLength(gops[i], elementType);
    // If past last value, assume 1; otherwise use actual value.
    int64_t num = gopsNum[i];
    // Accumulate weighted scalar/vectorized num and vl length.
    if (vl > 0)
      vectorizedOpNum += num;
    else
      scalarOpNum += num;
    // For totVL, when an operation is scalar, it still process 1 element
    int64_t processedValues = std::max((int64_t)1, vl);
    totProcessedValues += processedValues * num;
  }
  // Compute final values
  int64_t totNum = vectorizedOpNum + scalarOpNum;
  scalarOpNum = gopsSize - vectorizedOpNum;
  return totNum != 0 ? (1.0 * totProcessedValues) / (1.0 * totNum) : 0.0;
}

// =============================================================================
// IBM Z servers
// =============================================================================

int64_t Z16VectorMachineSupport::getVectorLength(
    GenericOps Gop, Type elementType) {
  int64_t bitWidth = elementType.getIntOrFloatBitWidth();
  int64_t abstractVL = VectorMachineSupport::getVectorLength(elementType);
  bool isFloat = elementType.isa<FloatType>();

  // Support shared between int and float.
  switch (Gop) {
    // 1 - 16 byte operations.
  case GenericOps::SelectGop:
  case GenericOps::ShuffleGop:
    return abstractVL;
  default:
    // Continue with typed tests.
    break;
  }

  // Support for float.
  if (isFloat) {
    // Supports only 32 and 64 bit Floats; There is support for extended too but
    // ignore this for now.
    if (!(bitWidth == 32 || bitWidth == 64 ||
            (bitWidth == 16 && Gop == GenericOps::ConversionGop)))
      return UNSUPPORTED;
    // Now we have a supported length, test for specific operations.
    switch (Gop) {
    case GenericOps::AbsGop:        /* Supported via compare and select */
    case GenericOps::ArithmeticGop: /* Add/sub,... */
    case GenericOps::CeilGop:       /* Use load integer & rounding modes*/
    case GenericOps::CompareGop:
    case GenericOps::ConversionGop:
    case GenericOps::CopySignGop:
    case GenericOps::DivGop:
    case GenericOps::FloorGop: /* Use load integer & rounding modes*/
    case GenericOps::FmaGop:
    case GenericOps::MinMaxGop:
    case GenericOps::MulGop:
    case GenericOps::SqrtGop:
      return abstractVL;
    default:
      // Unsupported float op.
      return UNSUPPORTED;
    }
  }
  // Support for integer (we consider bit-wide ops as byte wide ops).
  switch (Gop) {
    // 1 - 16 byte operations.
  case GenericOps::ArithmeticGop: /* Add/sub,... */
  case GenericOps::ConversionGop:
  case GenericOps::LogicalGop:
    return abstractVL;

    // 1 - 8 byte operations.
  case GenericOps::AbsGop: /* supported via compare and select */
  case GenericOps::CompareGop:
  case GenericOps::FmaGop:
  case GenericOps::MinMaxGop:
  case GenericOps::MulGop:
  case GenericOps::ShiftGop:
  case GenericOps::SumAcrossGop:
    return bitWidth <= 64 ? abstractVL : UNSUPPORTED;
  default:
    // Unsupported integer op.
    return UNSUPPORTED;
  }
  llvm_unreachable("should have handled all cases above");
}

// =============================================================================
// INTEL SSE x86 SSE 4.1 & 4.2 with width = 128; AVX2 with width = 256.
// This may be an approximation of the actual capabilities.
// =============================================================================

int64_t SSE42x86VectorMachineSupport::getVectorLength(
    GenericOps Gop, mlir::Type elementType) {
  int64_t bitWidth = elementType.getIntOrFloatBitWidth();
  int64_t abstractVL = VectorMachineSupport::getVectorLength(elementType);
  bool isFloat = elementType.isa<FloatType>();

  // Support shared between int and float.
  switch (Gop) {
    // 1 - 16 byte operations.
  case GenericOps::SelectGop:
  case GenericOps::ShuffleGop:
    return abstractVL;
  default:
    // Continue with typed tests.
    break;
  }

  // Support for float.
  if (isFloat) {
    // Supports only 32 and 64 bit Floats; There is support for extended too but
    // ignore this for now.
    if (!(bitWidth == 32 || bitWidth == 64 ||
            (bitWidth == 16 && Gop == GenericOps::ConversionGop)))
      return UNSUPPORTED;
    // Now we have a supported length, test for specific operations.
    switch (Gop) {
    case GenericOps::AbsGop:
    case GenericOps::ArithmeticGop: /* Add/sub,... */
    case GenericOps::CeilGop:
    case GenericOps::CompareGop:
    case GenericOps::ConversionGop:
    case GenericOps::CopySignGop:
    case GenericOps::DivGop:
    case GenericOps::FloorGop:
    case GenericOps::FmaGop:
    case GenericOps::MinMaxGop:
    case GenericOps::MulGop:
    case GenericOps::RoundGop:
    case GenericOps::SqrtGop:
    case GenericOps::SumAcrossGop:
      return abstractVL;
    default:
      // Unsupported float op.
      return UNSUPPORTED;
    }
  }
  // Support for integer (we consider bit-wide ops as byte wide ops).
  switch (Gop) {
    // 1 - 16 byte operations.
  case GenericOps::ArithmeticGop: /* Add/sub,... */
  case GenericOps::ConversionGop:
  case GenericOps::LogicalGop:
  case GenericOps::MinMaxGop:
  case GenericOps::CompareGop:
  case GenericOps::AbsGop:
    return abstractVL;

    // 1 - 8 byte operations.
  case GenericOps::ShiftGop:
    return bitWidth <= 64 ? abstractVL : UNSUPPORTED;

    // 1 - 4 byte operations.
  case GenericOps::FmaGop:
    return bitWidth <= 32 ? abstractVL : UNSUPPORTED;

    // 4 - 16 byte operations.
  case GenericOps::MulGop:
    return bitWidth >= 32 && bitWidth <= 128 ? abstractVL : UNSUPPORTED;

    // 4 - 8 byte operations.
  case GenericOps::SumAcrossGop:
    return bitWidth >= 32 && bitWidth <= 64 ? abstractVL : UNSUPPORTED;

  default:
    // Unsupported integer op.
    return UNSUPPORTED;
  }
  llvm_unreachable("should have handled all cases above");
}

} // namespace onnx_mlir
