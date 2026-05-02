/*
 * SPDX-License-Identifier: Apache-2.0
 */

//===-------------------------- PyFloat16.hpp -----------------------------===//
//
// Lightweight float16 type for Python runtime without LLVM dependencies.
//
//===----------------------------------------------------------------------===//

#ifndef ONNX_MLIR_PY_FLOAT16_H
#define ONNX_MLIR_PY_FLOAT16_H

#include <cstdint>
#include <cstring>

// Forward declare the C conversion functions from SmallFPConversion.c.
// These functions are implemented in src/Support/SmallFPConversion.c and
// are available through the OMSmallFPConversion library.
extern "C" {
float om_f16_to_f32(uint16_t u16);
uint16_t om_f32_to_f16(float f32);
}

namespace onnx_mlir {

// Lightweight float16 class for Python runtime.
// This class provides the minimal interface needed for pybind11 integration
// without depending on LLVM's APFloat.
class float_16 {
public:
  // Default constructor.
  constexpr float_16() : bits(0) {}

  // Construct from float.
  explicit float_16(float f) : bits(om_f32_to_f16(f)) {}

  // Construct from uint16_t (bitcast).
  static constexpr float_16 bitcastFromUInt(uint16_t u) {
    float_16 result;
    result.bits = u;
    return result;
  }

  // Convert to float.
  float toFloat() const { return om_f16_to_f32(bits); }

  // Explicit conversion to float.
  explicit operator float() const { return toFloat(); }

  // Bitcast to uint16_t.
  constexpr uint16_t bitcastToUInt() const { return bits; }

  // Comparison operators.
  bool operator==(const float_16 &other) const {
    return toFloat() == other.toFloat();
  }

  bool operator!=(const float_16 &other) const { return !(*this == other); }

private:
  uint16_t bits;
};

} // namespace onnx_mlir

#endif // ONNX_MLIR_PY_FLOAT16_H

// Made with Bob
