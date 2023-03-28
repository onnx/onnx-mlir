//===-- VectorMachineSupport.cpp - Helper for what SIMD ops are supported -===//
//
// Copyright 2023 The IBM Research Authors.
//
// =============================================================================

#include "src/Dialect/Mlir/VectorMachineSupport.hpp"
#include "mlir/IR/BuiltinTypes.h"

#define DEBUG_TYPE "dialect_builder"

using namespace mlir;

namespace onnx_mlir {

// =============================================================================
// Handling of global vector machine support pointer

VectorMachineSupport *VectorMachineSupport::globalVectorMachineSupport =
    nullptr;

VectorMachineSupport *VectorMachineSupport::getGlobalVectorMachineSupport() {
  return globalVectorMachineSupport;
}

void VectorMachineSupport::setGlobalVectorMachineSupport(std::string name) {
  if (name.compare("z14") == 0) {
    globalVectorMachineSupport = new Z14VectorMachineSupport();
  } else if (name.compare("z15") == 0) {
    globalVectorMachineSupport = new Z15VectorMachineSupport();
  } else if (name.compare("z16") == 0) {
    globalVectorMachineSupport = new Z16VectorMachineSupport();
  } else if (name.compare("sse4.2") == 0) {
    globalVectorMachineSupport = new SSE42x86VectorMachineSupport();
  } else if (name.compare("avx2") == 0) {
    globalVectorMachineSupport = new AVX2x86VectorMachineSupport();
  } else {
    // Unknown: disable
    globalVectorMachineSupport = new NoVectorMachineSupport();
  }
  assert(globalVectorMachineSupport &&
         "failed to allocate vector machine support");
}

void VectorMachineSupport::clearGlobalVectorMachineSupport() {
  if (!globalVectorMachineSupport)
    return;
  delete globalVectorMachineSupport;
  globalVectorMachineSupport = nullptr;
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

double VectorMachineSupport::getAvgVectorLength(
    llvm::SmallVectorImpl<GenericOps> &gops, Type elementType,
    int64_t &numSupported, int64_t &numUnsupported) {
  int64_t num = gops.size();
  double totVL = 0.0;
  numSupported = 0;
  // Determine which operations support SIMD and accumulate their vector
  // lengths.
  for (int64_t i = 0; i < num; ++i) {
    int64_t vl = getVectorLength(gops[i], elementType);
    if (vl > 0) {
      totVL += vl;
      numSupported++;
    }
  }
  // Compute final values
  numUnsupported = num - numSupported;
  if (numSupported == 0)
    return 0.0;
  return totVL / (1.0 * numSupported);
}

// =============================================================================
// IBM Z servers
// =============================================================================

int64_t Z16VectorMachineSupport::getVectorLength(
    GenericOps gop, Type elementType) {
  int64_t bitWidth = elementType.getIntOrFloatBitWidth();
  int64_t abstractVL = VectorMachineSupport::getVectorLength(elementType);
  bool isFloat = elementType.isa<FloatType>();

  // Support shared between int and float.
  switch (gop) {
    // 1 - 16 byte operations.
  case GenericOps::SelectGOp:
  case GenericOps::ShuffleGOp:
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
            (bitWidth == 16 && gop == GenericOps::ConversionGOp)))
      return UNSUPPORTED;
    // Now we have a supported length, test for specific operations.
    switch (gop) {
    case GenericOps::AbsGOp:        /* Supported via compare and select */
    case GenericOps::ArithmeticGOp: /* Add/sub,... */
    case GenericOps::CompareGOp:
    case GenericOps::ConversionGOp:
    case GenericOps::CopySignGOP:
    case GenericOps::DivGOp:
    case GenericOps::FmaGOp:
    case GenericOps::MinMaxGOp:
    case GenericOps::MulGOp:
    case GenericOps::SqrtGOp:
      return abstractVL;
    default:
      // Unsupported float op.
      return UNSUPPORTED;
    }
  }
  // Support for integer (we consider bit-wide ops as byte wide ops).
  switch (gop) {
    // 1 - 16 byte operations.
  case GenericOps::ArithmeticGOp: /* Add/sub,... */
  case GenericOps::ConversionGOp:
  case GenericOps::LogicalGOp:
    return abstractVL;

    // 1 - 8 byte operations.
  case GenericOps::AbsGOp: /* supported via compare and select */
  case GenericOps::CompareGOp:
  case GenericOps::FmaGOp:
  case GenericOps::MinMaxGOp:
  case GenericOps::MulGOp:
  case GenericOps::ShiftGOp:
  case GenericOps::SumAcrossGOp:
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
    GenericOps gop, mlir::Type elementType) {
  int64_t bitWidth = elementType.getIntOrFloatBitWidth();
  int64_t abstractVL = VectorMachineSupport::getVectorLength(elementType);
  bool isFloat = elementType.isa<FloatType>();

  // Support shared between int and float.
  switch (gop) {
    // 1 - 16 byte operations.
  case GenericOps::SelectGOp:
  case GenericOps::ShuffleGOp:
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
            (bitWidth == 16 && gop == GenericOps::ConversionGOp)))
      return UNSUPPORTED;
    // Now we have a supported length, test for specific operations.
    switch (gop) {
    case GenericOps::AbsGOp:
    case GenericOps::ArithmeticGOp: /* Add/sub,... */
    case GenericOps::CompareGOp:
    case GenericOps::ConversionGOp:
    case GenericOps::CopySignGOP:
    case GenericOps::DivGOp:
    case GenericOps::FmaGOp:
    case GenericOps::MinMaxGOp:
    case GenericOps::MulGOp:
    case GenericOps::SumAcrossGOp:
    case GenericOps::SqrtGOp:
      return abstractVL;
    default:
      // Unsupported float op.
      return UNSUPPORTED;
    }
  }
  // Support for integer (we consider bit-wide ops as byte wide ops).
  switch (gop) {
    // 1 - 16 byte operations.
  case GenericOps::ArithmeticGOp: /* Add/sub,... */
  case GenericOps::ConversionGOp:
  case GenericOps::LogicalGOp:
  case GenericOps::MinMaxGOp:
  case GenericOps::CompareGOp:
  case GenericOps::AbsGOp:
    return abstractVL;

    // 1 - 8 byte operations.
  case GenericOps::ShiftGOp:
    return bitWidth <= 64 ? abstractVL : UNSUPPORTED;

    // 1 - 4 byte operations.
  case GenericOps::FmaGOp:
    return bitWidth <= 32 ? abstractVL : UNSUPPORTED;

    // 4 - 16 byte operations.
  case GenericOps::MulGOp:
    return bitWidth >= 32 && bitWidth <= 128 ? abstractVL : UNSUPPORTED;

    // 4 - 8 byte operations.
  case GenericOps::SumAcrossGOp:
    return bitWidth >= 32 && bitWidth <= 64 ? abstractVL : UNSUPPORTED;

  default:
    // Unsupported integer op.
    return UNSUPPORTED;
  }
  llvm_unreachable("should have handled all cases above");
}

} // namespace onnx_mlir
