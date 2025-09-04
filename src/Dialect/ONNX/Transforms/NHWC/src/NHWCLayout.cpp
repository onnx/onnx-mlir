// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#include "../include/NHWCLayout.h"

#include <algorithm>
#include <numeric>
#include <set>
#include <sstream>
#include <iostream>

// LayoutHelper implementations
std::string LayoutHelper::toString(Layout layout) {
  switch (layout) {
  case Layout::NONE:
    return "NONE";
  case Layout::UNKNOWN:
    return "UNKNOWN";
  case Layout::ANY:
    return "ANY";
  case Layout::NCHW:
    return "NCHW";
  case Layout::NHWC:
    return "NHWC";
  case Layout::NHCW:
    return "NHCW";
  case Layout::CONFLICT:
    return "CONFLICT";
  }
}

std::string LayoutHelper::toRepr(Layout layout) {
  std::stringstream ss;
  ss << "<Layout." << toString(layout) << ">";
  return ss.str();
}

Layout LayoutHelper::transposed(Layout layout) {
  if (layout == Layout::NCHW || layout == Layout::NHWC) {
    return static_cast<Layout>(6 / static_cast<int>(layout));
  }
  return layout;
}

Layout LayoutHelper::apply_transpose(Layout layout, const Perm &perm) {
  if (is_layout_perm(perm)) {
    return transposed(layout);
  } else {
    return layout;
  }
}

Layout LayoutHelper::unify(const std::vector<Layout> &layout_tags) {
  if (layout_tags.empty()) {
    return Layout::UNKNOWN;
  }

  // Convert to set to get unique values
  std::set<int> unique_tags;
  for (Layout tag : layout_tags) {
    unique_tags.insert(static_cast<int>(tag));
  }

  // Calculate product of all unique tags
  int layout_unified = std::accumulate(unique_tags.begin(), unique_tags.end(),
      1, [](int a, int b) { return a * b; });

  layout_unified = std::abs(layout_unified);
  layout_unified =
      std::min(layout_unified, static_cast<int>(Layout::CONFLICT));

  return static_cast<Layout>(layout_unified);
}

Layout LayoutHelper::get_perm_layout(
    const Perm &perm, const std::vector<int> &dims) {
  bool is_nchw = is_NCHW_perm(perm, dims);
  bool is_nhwc = is_NHWC_perm(perm, dims);

  if (is_nchw && is_nhwc) { // 1D, 2D, 3D
    return Layout::ANY;
  }
  if (is_nchw) { // 4D
    return Layout::NCHW;
  }
  if (is_nhwc) { // 4D
    return Layout::NHWC;
  }
  return Layout::UNKNOWN;
}

// Free function implementations
std::string to_string(Layout layout) {
  return LayoutHelper::toString(layout);
}

std::ostream &operator<<(std::ostream &os, Layout layout) {
  os << LayoutHelper::toString(layout);
  return os;
}

Layout transposed(Layout layout) {
  return LayoutHelper::transposed(layout);
}

Layout apply_transpose(Layout layout, const Perm &perm) {
  return LayoutHelper::apply_transpose(layout, perm);
}

Layout get_perm_layout(const Perm &perm, const std::vector<int> &dims) {
  return LayoutHelper::get_perm_layout(perm, dims);
}