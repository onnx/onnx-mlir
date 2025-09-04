// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <string>
#include <vector>
#include <iosfwd>

// Type definition
using Perm = std::vector<int>;

// Forward declarations for helper functions
bool is_layout_perm(const Perm &perm);
bool is_NCHW_perm(const Perm &perm, const std::vector<int> &dims = {});
bool is_NHWC_perm(const Perm &perm, const std::vector<int> &dims = {});

enum class Layout : int {
  NONE = -1,
  UNKNOWN = 0,
  ANY = 1,
  NCHW = 2,
  NHWC = 3,
  NHCW = 4,
  CONFLICT = 6
};

class LayoutHelper {
public:
  // String representation (__str__ equivalent)
  static std::string toString(Layout layout);

  // Representation string (__repr__ equivalent)
  static std::string toRepr(Layout layout);

  // Transposed method
  static Layout transposed(Layout layout);

  // Apply transpose method
  static Layout apply_transpose(Layout layout, const Perm &perm);

  // Unify method (classmethod equivalent) - vector version
  static Layout unify(const std::vector<Layout> &layout_tags);

  // Variadic template version for convenience (*args equivalent)
  template <typename... Args>
  static Layout unify(Layout first, Args... rest) {
    std::vector<Layout> tags = {first, rest...};
    return unify(tags);
  }

  // Get permission layout method (classmethod equivalent)
  static Layout get_perm_layout(
      const Perm &perm, const std::vector<int> &dims = {});
};

// Convenience functions for direct use with Layout enum
std::string to_string(Layout layout);

std::ostream &operator<<(std::ostream &os, Layout layout);

// Extension methods as free functions (alternative approach)
Layout transposed(Layout layout);

Layout apply_transpose(Layout layout, const Perm &perm);

template <typename... Args>
Layout unify(Args... args) {
  return LayoutHelper::unify(args...);
}

Layout get_perm_layout(
    const Perm &perm, const std::vector<int> &dims = {});