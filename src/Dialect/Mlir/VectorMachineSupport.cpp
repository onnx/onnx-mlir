/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-- VectorMachineSupport.cpp - Helper for what SIMD ops are supported -===//
//
// Copyright 2023-2024 The IBM Research Authors.
//
// =============================================================================

#include "src/Dialect/Mlir/VectorMachineSupport.hpp"
#include "src/Compiler/CompilerOptions.hpp"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/Debug.h"

#include <algorithm>

#define DEBUG_TYPE "dialect_builder"

using namespace mlir;

namespace onnx_mlir {

// =============================================================================
// Handling of global vector machine support pointer

/*static*/ VectorMachineSupport
    *VectorMachineSupport::globalVectorMachineSupport = nullptr;

/*static*/ void VectorMachineSupport::setGlobalVectorMachineSupport(
    const std::string &arch, const std::string &cpu, const std::string &attr) {
  // IBM Z servers use march (deprecated mcpu), process here.
  int64_t zArchNum = getZArchNum(arch, cpu);
  if (zArchNum == 12) {
    globalVectorMachineSupport = new ZArch12VectorMachineSupport();
  } else if (zArchNum == 13) {
    globalVectorMachineSupport = new ZArch13VectorMachineSupport();
  } else if (zArchNum == 14) {
    globalVectorMachineSupport = new ZArch14VectorMachineSupport();
  } else if (zArchNum == 15) {
    globalVectorMachineSupport = new ZArch15VectorMachineSupport();
    // Intel uses arch
  } else if (arch.compare("x86-64") == 0) {
    // Intel arch
    if (cpu.compare("skylake") == 0 && attr.compare("avx2") == 0)
      globalVectorMachineSupport = new AVX2x86VectorMachineSupport();
    else
      // Default seems to be SSE
      globalVectorMachineSupport = new SSE42x86VectorMachineSupport();
    // Arm uses arch
  } else if (arch.compare("aarch64") == 0 || arch.compare("arm64") == 0) {
    // Arm arch
    globalVectorMachineSupport = new NeonVectorMachineSupport();
  } else {
    // Unknown: disable
    globalVectorMachineSupport = new NoVectorMachineSupport();
  }
  assert(globalVectorMachineSupport &&
         "failed to allocate vector machine support");
  LLVM_DEBUG(llvm::dbgs() << "use SIMD arch " << getArchName() << "\n");
}

/*static*/ void VectorMachineSupport::clearGlobalVectorMachineSupport() {
  if (!globalVectorMachineSupport)
    return;
  delete globalVectorMachineSupport;
  globalVectorMachineSupport = nullptr;
}

// =============================================================================
// Methods shared among all VectorMachineSupport classes and subclasses

int64_t VectorMachineSupport::computeArchVectorLength(Type elementType) {
  if (!hasSimd())
    return 0;
  int64_t simdBitSize = computeArchVectorBitWidth();
  int64_t typeBitSize = elementType.getIntOrFloatBitWidth();
  assert(simdBitSize >= typeBitSize && simdBitSize % typeBitSize == 0 &&
         "bad machine vector length");
  return (simdBitSize / typeBitSize);
}

/*static*/ double VectorMachineSupport::getAvgArchVectorLength(GenOpMix &genOps,
    Type elementType, int64_t &vectorizedOpNum, int64_t &scalarOpNum,
    int64_t &maxVectorRegisterPressure) {
  int64_t size = genOps.size();
  vectorizedOpNum = maxVectorRegisterPressure = 0;
  if (!hasSimd()) {
    scalarOpNum = size;
    return 1;
  }
  int64_t totProcessedValues = 0.0;
  scalarOpNum = 0;
  bool hasRegisterPressure = false;

  // Determine which operations support SIMD and accumulate their vector
  // lengths.
  for (auto pair : genOps) {
    GenericOps genOp = pair.first;
    int64_t num = pair.second;
    // Handle other metrics first.
    if (genOp == GenericOps::EstimatedVectorRegisterPressure) {
      maxVectorRegisterPressure = std::max(maxVectorRegisterPressure, num);
      hasRegisterPressure = true;
      continue;
    }
    assert(genOp < GenericOps::LastGop && "no metrics here, only genOps");
    int64_t vl = getArchVectorLength(genOp, elementType);
    // If past last value, assume 1; otherwise use actual value.
    // Accumulate weighted scalar/vectorized num and vl length.
    if (vl > 0)
      vectorizedOpNum += num;
    else
      scalarOpNum += num;
    // For VL, when an operation is scalar, it still process 1 element
    int64_t processedValues = std::max(static_cast<int64_t>(1), vl);
    totProcessedValues += processedValues * num;
  }

  // Compute final values
  int64_t totNum = vectorizedOpNum + scalarOpNum;
  if (!hasRegisterPressure) {
    // Estimate default register pressure as one per 2 vector operation.
    maxVectorRegisterPressure =
        std::max(vectorizedOpNum / 2, static_cast<int64_t>(1));
  }
  return totNum != 0 ? (1.0 * totProcessedValues) / (1.0 * totNum) : 1.0;
}

// =============================================================================
// IBM Z servers
// =============================================================================

bool ZArch14VectorMachineSupport::needCustomASM(
    GenericOps genOp, Type elementType) {
  assert(genOp < GenericOps::LastGop && "no metrics here, only genOps");
  bool isFloat = mlir::isa<FloatType>(elementType);
  if (isFloat) {
    switch (genOp) {
    case GenericOps::roundEvenGop:
      return true;
    default:
      return false;
    }
  }
  // Integer
  return false;
}

int64_t ZArch14VectorMachineSupport::computeArchVectorLength(
    GenericOps genOp, Type elementType) {
  assert(genOp < GenericOps::LastGop && "no metrics here, only genOps");
  int64_t bitWidth = elementType.getIntOrFloatBitWidth();
  int64_t archVL = VectorMachineSupport::getArchVectorLength(elementType);
  bool isFloat = mlir::isa<FloatType>(elementType);
  // Support shared between int and float.
  switch (genOp) {
  case GenericOps::ScalarOnlyGop:
    return 1; // Must be scalar.
  case GenericOps::SelectGop:
  case GenericOps::ShuffleGop:
    return archVL; // 1 - 16 byte operations.
  default:
    // Continue with typed tests.
    break;
  }

  // Support for float.
  if (isFloat) {
    // Supports only 32 and 64 bit Floats; There is support for extended too
    // but ignore this for now.
    if (!(bitWidth == 32 || bitWidth == 64 ||
            (bitWidth == 16 && genOp == GenericOps::ConversionGop)))
      return UNSUPPORTED;
    // Now we have a supported length, test for specific operations.
    switch (genOp) {
    case GenericOps::AbsGop:        /* Supported via compare and select */
    case GenericOps::ArithmeticGop: /* Add/sub,... */
    case GenericOps::CeilGop:       /* Use load integer & rounding modes*/
    case GenericOps::CompareGop:
    case GenericOps::ConversionGop:
    case GenericOps::CopySignGop:
    case GenericOps::DivGop:
    case GenericOps::ErfGop:
    case GenericOps::FloorGop: /* Use load integer & rounding modes*/
    case GenericOps::FmaGop:
    case GenericOps::MinMaxGop:
    case GenericOps::MulGop:
    case GenericOps::roundEvenGop:
    case GenericOps::SqrtGop:
      return archVL;
    default:
      // Unsupported float op.
      return UNSUPPORTED;
    }
  }
  // Support for integer (we consider bit-wide ops as byte wide ops).
  switch (genOp) {
    // 1 - 16 byte operations.
  case GenericOps::ArithmeticGop: /* Add/sub,... */
  case GenericOps::ConversionGop:
  case GenericOps::LogicalGop:
    return archVL;

    // 1 - 8 byte operations.
  case GenericOps::AbsGop: /* supported via compare and select */
  case GenericOps::CompareGop:
  case GenericOps::FmaGop:
  case GenericOps::MinMaxGop:
  case GenericOps::MulGop:
  case GenericOps::ShiftGop:
  case GenericOps::SumAcrossGop:
    return bitWidth <= 64 ? archVL : UNSUPPORTED;
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

bool SSE42x86VectorMachineSupport::needCustomASM(
    GenericOps genOp, Type elementType) {
  assert(genOp < GenericOps::LastGop && "no metrics here, only genOps");
  return false;
}

int64_t SSE42x86VectorMachineSupport::computeArchVectorLength(
    GenericOps genOp, Type elementType) {
  assert(genOp < GenericOps::LastGop && "no metrics here, only genOps");
  int64_t bitWidth = elementType.getIntOrFloatBitWidth();
  int64_t archVL = VectorMachineSupport::getArchVectorLength(elementType);
  bool isFloat = mlir::isa<FloatType>(elementType);

  // Support shared between int and float.
  switch (genOp) {
  case GenericOps::ScalarOnlyGop:
    return 1; // Must be scalar.
  case GenericOps::SelectGop:
  case GenericOps::ShuffleGop:
    return archVL; //// 1 - 16 byte operations.
  default:
    // Continue with typed tests.
    break;
  }

  // Support for float.
  if (isFloat) {
    // Supports only 32 and 64 bit Floats; There is support for extended too
    // but ignore this for now.
    if (!(bitWidth == 32 || bitWidth == 64 ||
            (bitWidth == 16 && genOp == GenericOps::ConversionGop)))
      return UNSUPPORTED;
    // Now we have a supported length, test for specific operations.
    switch (genOp) {
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
    case GenericOps::roundEvenGop:
    case GenericOps::SqrtGop:
    case GenericOps::SumAcrossGop:
      return archVL;
    default:
      // Unsupported float op.
      return UNSUPPORTED;
    }
  }
  // Support for integer (we consider bit-wide ops as byte wide ops).
  switch (genOp) {
    // 1 - 16 byte operations.
  case GenericOps::ArithmeticGop: /* Add/sub,... */
  case GenericOps::ConversionGop:
  case GenericOps::LogicalGop:
  case GenericOps::MinMaxGop:
  case GenericOps::CompareGop:
  case GenericOps::AbsGop:
    return archVL;

    // 1 - 8 byte operations.
  case GenericOps::ShiftGop:
    return bitWidth <= 64 ? archVL : UNSUPPORTED;

    // 1 - 4 byte operations.
  case GenericOps::FmaGop:
    return bitWidth <= 32 ? archVL : UNSUPPORTED;

    // 4 - 16 byte operations.
  case GenericOps::MulGop:
    return bitWidth >= 32 && bitWidth <= 128 ? archVL : UNSUPPORTED;

    // 4 - 8 byte operations.
  case GenericOps::SumAcrossGop:
    return bitWidth >= 32 && bitWidth <= 64 ? archVL : UNSUPPORTED;

  default:
    // Unsupported integer op.
    return UNSUPPORTED;
  }
  llvm_unreachable("should have handled all cases above");
}

// =============================================================================
// Arm with Neon.
// This may be an approximation of the actual capabilities.
// =============================================================================

bool NeonVectorMachineSupport::needCustomASM(
    GenericOps genOp, Type elementType) {
  assert(genOp < GenericOps::LastGop && "no metrics here, only genOps");
  return false;
}

int64_t NeonVectorMachineSupport::computeArchVectorLength(
    GenericOps genOp, Type elementType) {
  assert(genOp < GenericOps::LastGop && "no metrics here, only genOps");
  int64_t bitWidth = elementType.getIntOrFloatBitWidth();
  int64_t archVL = VectorMachineSupport::getArchVectorLength(elementType);
  bool isFloat = mlir::isa<FloatType>(elementType);

  // Support shared between int and float.
  switch (genOp) {
  case GenericOps::ScalarOnlyGop:
    return 1; // Must be scalar.
  case GenericOps::SelectGop:
  case GenericOps::ShuffleGop:
    return archVL; // 1 - 16 byte operations.
  default:
    // Continue with typed tests.
    break;
  }

  // Support for float.
  if (isFloat) {
    // Supports only 32 and 64 bit Floats;
    if (!(bitWidth == 32 || bitWidth == 64 ||
            (bitWidth == 16 && genOp == GenericOps::ConversionGop)))
      return UNSUPPORTED;
    // Now we have a supported length, test for specific operations.
    switch (genOp) {
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
    case GenericOps::roundEvenGop:
    case GenericOps::SqrtGop:
    case GenericOps::SumAcrossGop:
      return archVL;
    default:
      // Unsupported float op.
      return UNSUPPORTED;
    }
  }
  // Support for integer (we consider bit-wide ops as byte wide ops).
  switch (genOp) {
    // 1 - 16 byte operations.
  case GenericOps::ArithmeticGop: /* Add/sub,... */
  case GenericOps::ConversionGop:
  case GenericOps::LogicalGop:
  case GenericOps::MinMaxGop:
  case GenericOps::CompareGop:
  case GenericOps::AbsGop:
    return archVL;

    // 1 - 8 byte operations.
  case GenericOps::ShiftGop:
    return bitWidth <= 64 ? archVL : UNSUPPORTED;

    // 1 - 4 byte operations.
  case GenericOps::FmaGop:
    return bitWidth <= 32 ? archVL : UNSUPPORTED;

    // 4 - 16 byte operations.
  case GenericOps::MulGop:
    return bitWidth >= 32 && bitWidth <= 128 ? archVL : UNSUPPORTED;

    // 4 - 8 byte operations.
  case GenericOps::SumAcrossGop:
    return bitWidth >= 32 && bitWidth <= 64 ? archVL : UNSUPPORTED;

  default:
    // Unsupported integer op.
    return UNSUPPORTED;
  }
  llvm_unreachable("should have handled all cases above");
}

// =============================================================================
// Support for Generic Operation Mix

GenOpMix computeGenOpMixUnion(const GenOpMix &mix1, const GenOpMix &mix2) {
  GenOpMix u;
  // Pick ops from the first mix.
  for (auto pair : mix1) {
    GenericOps genOp = pair.first;
    int64_t num = pair.second;
    u[genOp] = num;
  }
  // Merge entries from the second mix.
  for (auto pair : mix2) {
    GenericOps genOp = pair.first;
    int64_t num = pair.second;
    if (u.find(genOp) != u.end()) {
      // Merge the 2 operation counts/metrics.
      if (genOp == GenericOps::EstimatedVectorRegisterPressure) {
        // For register pressure, pick the max of both.
        u[genOp] = std::max(u[genOp], num);
      } else {
        // For operation count, use the sum of both
        u[genOp] += num;
      }
    } else {
      // First time we have this.
      u[genOp] = num;
    }
  }
  return u;
}

} // namespace onnx_mlir
