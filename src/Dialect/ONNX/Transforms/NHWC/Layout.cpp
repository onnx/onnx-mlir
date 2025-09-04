#include <algorithm>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>

// Assuming Perm is defined as:
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
  static std::string toString(Layout layout) {
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

  // Representation string (__repr__ equivalent)
  static std::string toRepr(Layout layout) {
    std::stringstream ss;
    ss << "<Layout." << toString(layout) << ">";
    return ss.str();
  }

  // Transposed method
  static Layout transposed(Layout layout) {
    if (layout == Layout::NCHW || layout == Layout::NHWC) {
      return static_cast<Layout>(6 / static_cast<int>(layout));
    }
    return layout;
  }

  // Apply transpose method
  static Layout apply_transpose(Layout layout, const Perm &perm) {
    if (is_layout_perm(perm)) {
      return transposed(layout);
    } else {
      return layout;
    }
  }

  // Unify method (classmethod equivalent) - vector version
  static Layout unify(const std::vector<Layout> &layout_tags) {
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

  // Variadic template version for convenience (*args equivalent)
  template <typename... Args>
  static Layout unify(Layout first, Args... rest) {
    std::vector<Layout> tags = {first, rest...};
    return unify(tags);
  }

  // Get permission layout method (classmethod equivalent)
  static Layout get_perm_layout(
      const Perm &perm, const std::vector<int> &dims = {}) {
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
};

// Convenience functions for direct use with Layout enum
inline std::string to_string(Layout layout) {
  return LayoutHelper::toString(layout);
}

inline std::ostream &operator<<(std::ostream &os, Layout layout) {
  os << LayoutHelper::toString(layout);
  return os;
}

// Extension methods as free functions (alternative approach)
inline Layout transposed(Layout layout) {
  return LayoutHelper::transposed(layout);
}

inline Layout apply_transpose(Layout layout, const Perm &perm) {
  return LayoutHelper::apply_transpose(layout, perm);
}

template <typename... Args>
inline Layout unify(Args... args) {
  return LayoutHelper::unify(args...);
}

inline Layout get_perm_layout(
    const Perm &perm, const std::vector<int> &dims = {}) {
  return LayoutHelper::get_perm_layout(perm, dims);
}