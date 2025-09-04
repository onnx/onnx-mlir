// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once
#include "include/NHWCLayout.h"
#include "include/PermutationHelper.h"
#include <functional>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

// Assuming Perm is defined as:
using Perm = std::vector<int>;
using TensorShape = std::vector<int>;
// Custom exception class
class UnexpectedTaggedShape : public std::runtime_error {
public:
  UnexpectedTaggedShape(const std::string &message)
      : std::runtime_error(message) {}
};

class TaggedShape {
public:
  TensorShape shape;
  Layout tag;
  std::optional<std::string> tag_source;
  std::vector<std::string> shape_denotation;

  // Constructor
  TaggedShape(TensorShape shape = {}, Layout tag = Layout::NONE,
      std::optional<std::string> tag_source = std::nullopt,
      std::vector<std::string> shape_denotation = {})
      : shape(std::move(shape)), tag(tag), tag_source(std::move(tag_source)),
        shape_denotation(std::move(shape_denotation)) {}

  // // Hash function for use in unordered containers
  // size_t hash() const {
  //   size_t seed = 0;
  //   // Hash the tag
  //   seed ^= std::hash<int>()(static_cast<int>(tag)) + 0x9e3779b9 + (seed <<
  //   6) +
  //           (seed >> 2);
  //   // Hash the shape
  //   for (const auto &dim : shape) {
  //     seed ^=
  //         std::hash<int64_t>()(dim) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  //   }
  //   return seed;
  // }

  // String representation
  // std::string toString() const {
  //   std::stringstream ss;
  //   ss << "TaggedShape(shape=[";
  //   for (size_t i = 0; i < shape.size(); ++i) {
  //     if (i > 0)
  //       ss << ", ";
  //     ss << shape[i];
  //   }
  //   ss << "], tag=" << static_cast<int>(tag);
  //   ss << ", denotation=" << get_denotation_str();
  //   ss << ", source=";
  //   if (tag_source.has_value()) {
  //     ss << tag_source.value();
  //   } else {
  //     ss << "null";
  //   }
  //   ss << ")";
  //   return ss.str();
  // }

  // Static factory method
  static TaggedShape NONE() { return TaggedShape({}, Layout::NONE); }

  // Get denotation string
  [[nodiscard]] std::string get_denotation_str() const {
    if (!shape_denotation.empty()) {
      std::stringstream ss;
      for (size_t i = 0; i < shape_denotation.size(); ++i) {
        if (i > 0)
          ss << "|";
        ss << shape_denotation[i];
      }
      return ss.str();
    } else {
      return "NONE";
    }
  }

  // Update method - returns a new TaggedShape
  [[nodiscard]] TaggedShape update(
      const std::optional<TensorShape> &new_shape = std::nullopt,
      const std::optional<Layout> &new_tag = std::nullopt,
      const std::optional<std::string> &new_source = std::nullopt,
      const std::optional<std::vector<std::string>> &new_denote =
          std::nullopt) const {

    TensorShape updated_shape =
        new_shape.has_value() ? new_shape.value() : this->shape;
    Layout updated_tag = new_tag.has_value() ? new_tag.value() : this->tag;
    std::optional<std::string> updated_source =
        new_source.has_value() ? new_source : this->tag_source;
    std::vector<std::string> updated_denote =
        new_denote.has_value() ? new_denote.value() : this->shape_denotation;

    return TaggedShape(
        updated_shape, updated_tag, updated_source, updated_denote);
  }

  // Equality operator (for comparing TaggedShape objects)
  bool operator==(const TaggedShape &other) const {
    return shape == other.shape && tag == other.tag &&
           tag_source == other.tag_source &&
           shape_denotation == other.shape_denotation;
  }

public:
  // Require method - validates that the shape matches expected values
  [[nodiscard]] TaggedShape require(
      const std::optional<TensorShape> &new_shape = std::nullopt,
      const std::optional<Layout> &new_tag = std::nullopt,
      const std::optional<std::string> &new_source = std::nullopt,
      const std::optional<std::vector<std::string>> &new_denote =
          std::nullopt) const {

    TensorShape check_shape =
        new_shape.has_value() ? new_shape.value() : this->shape;
    Layout check_tag = new_tag.has_value() ? new_tag.value() : this->tag;
    std::optional<std::string> check_source =
        new_source.has_value() ? new_source : this->tag_source;
    std::vector<std::string> check_denote =
        new_denote.has_value() ? new_denote.value() : this->shape_denotation;

    // Check all conditions
    bool shape_match = (check_shape == this->shape);
    bool tag_match = (check_tag == this->tag);
    bool source_match = (check_source == this->tag_source);
    bool denote_match = (check_denote == this->shape_denotation);

    if (!shape_match || !tag_match || !source_match || !denote_match) {
      // std::stringstream error_msg;
      // error_msg << toString() << ".require("
      //           << "shape=" << vectorToString(check_shape) << ", "
      //           << "tag=" << static_cast<int>(check_tag) << ", "
      //           << "denotation=" << vectorToString(check_denote) << ", "
      //           << "source="
      //           << (check_source.has_value() ? check_source.value() : "null")
      //           << ") == (shape_match=" << shape_match
      //           << ", tag_match=" << tag_match
      //           << ", source_match=" << source_match
      //           << ", denote_match=" << denote_match << ")";
      // throw UnexpectedTaggedShape(error_msg.str());
    }

    return *this;
  }

  // Apply transpose transformation
  [[nodiscard]] TaggedShape apply_transpose(const Perm &perm_) const {
    TensorShape shape_ = PermutationHelper::permute(shape, perm_);
    Layout tag_ = LayoutHelper::apply_transpose(tag, perm_);
    std::vector<std::string> denote_ = shape_denotation;

    if (!shape_denotation.empty()) {
      // Debug logging
      // std::cout << "APPLY :: Transpose(" << vectorToString(perm_) << ") to "
      //           << toString() << std::endl;

      denote_ = PermutationHelper::permute(shape_denotation, perm_);

      if (!denote_.empty() && denote_[0] != "N") {
        std::cout << "      xx PERM BATCH :: "
                  << vectorToString(shape_denotation) << " >> "
                  << vectorToString(denote_) << " :: " << shape_denotation[0]
                  << " >> " << denote_[0] << std::endl;
        return update(
            shape_, Layout::NONE, std::nullopt, std::vector<std::string>{});
      }

      tag_ = _get_layout_tag(denote_, tag);
      if (tag != tag_) {
        std::cout << "      :: SWITCHING LAYOUT :: " << static_cast<int>(tag)
                  << " >> " << static_cast<int>(tag_) << std::endl;
      }
    }

    return update(shape_, tag_, std::nullopt, denote_);
  }

  // Get channel dimension index
  [[nodiscard]] std::optional<size_t> get_channel_dim() const {
    if (!shape_denotation.empty()) {
      auto it =
          std::find(shape_denotation.begin(), shape_denotation.end(), "C");
      if (it != shape_denotation.end()) {
        return std::distance(shape_denotation.begin(), it);
      }
    }
    return std::nullopt;
  }

  // Get batch dimension index
  [[nodiscard]] std::optional<size_t> get_batch_dim() const {
    if (!shape_denotation.empty()) {
      auto it =
          std::find(shape_denotation.begin(), shape_denotation.end(), "N");
      if (it != shape_denotation.end()) {
        return std::distance(shape_denotation.begin(), it);
      }
    }
    return std::nullopt;
  }

  static std::vector<std::string> _get_layout_denotation(Layout tag, int dim) {
    if (tag == Layout::NCHW) {
      if (dim == 4) {
        return {"N", "C", "H", "W"};
      } else if (dim == 3) {
        return {"N", "C", "HW"};
      } else {
        return {};
      }
    }

    if (tag == Layout::NHWC) {
      if (dim == 4) {
        return {"N", "H", "W", "C"};
      } else if (dim == 3) {
        return {"N", "HW", "C"};
      } else {
        return {};
      }
    }

    return {};
  }

  // Get layout tag based on denotation (already in previous implementation)
  static Layout _get_layout_tag(const std::vector<std::string> &denote,
      Layout default_tag = Layout::UNKNOWN) {
    if (!denote.empty()) {
      if (denote.back() == "C") {
        return Layout::NHWC;
      }
      if (denote.size() > 1 && denote[1] == "C") {
        return Layout::NCHW;
      }
    }
    return default_tag;
  }
  // Assuming these helper structures/functions are defined elsewhere:
  // - reshape_diff() returns std::tuple<bool, std::vector<std::vector<int>>,
  // std::vector<std::vector<int>>>
  // - Logger class/instance for debug output

  [[nodiscard]] TaggedShape apply_reshape(const TensorShape &reshaped) const {
    if (shape_denotation.empty()) {
      return update(reshaped);
    }

    // Lambda function equivalent to get_dims_mask
    auto get_dims_mask =
        [](const std::optional<size_t> &dim_,
            const std::vector<std::vector<int>> &mask_) -> std::vector<int> {
      if (!dim_.has_value()) {
        return {};
      }

      for (const auto &m : mask_) {
        if (std::find(m.begin(), m.end(), static_cast<int>(*dim_)) != m.end()) {
          return m;
        }
      }
      return {};
    };

    // Get reshape difference information
    auto [is_valid_diff, in_mask, out_mask] = reshape_diff(shape, reshaped);

    // // Debug logging
    // logger.debug(
    //     "APPLY :: Reshape(" + vectorToString(reshaped) + ") to " +
    //     toString());

    if (!is_valid_diff) {
      // bool dim_N_diff = (!shape.empty() && !reshaped.empty())
      //                       ? (shape[0] != reshaped[0])
      //                       : false;
      // logger.debug("      :: INVALID DIFF :: BATCH " +
      //              std::string(dim_N_diff ? "DIFF" : "SAME"));

      auto denote_ = _get_layout_denotation(tag, reshaped.size());
      return update(reshaped, std::nullopt, std::nullopt, denote_);
    }

    // logger.debug("      :: VALID DIFF :: in_mask=" + maskToString(in_mask) +
    //              "; out_mask=" + maskToString(out_mask) + ";");

    // Check if first element of out_mask is empty
    if (!out_mask.empty() && out_mask[0].empty()) {
      // logger.debug("      xx DROPPING BATCH :: " + vectorToString(shape) +
      //              " >> " + vectorToString(reshaped) +
      //              " :: in_mask=" + maskToString(in_mask) +
      //              "; out_mask=" + maskToString(out_mask) + ";");
      return update(
          reshaped, Layout::NONE, std::nullopt, std::vector<std::string>{});
    }

    // Get channel dimension
    auto dimC = get_channel_dim();
    auto dimC_out_mask = get_dims_mask(dimC, out_mask);
    auto dimC_in_mask = get_dims_mask(dimC, in_mask);

    if (dimC_out_mask.size() >= 2) {
      // logger.debug("      xx SPLITTING CHANNEL :: " + vectorToString(shape) +
      //              " >> " + vectorToString(reshaped) + " :: dimC=" +
      //              (dimC.has_value() ? std::to_string(*dimC) : "None") +
      //              " >> out_mask[dimC]=" + vectorToString(dimC_out_mask) +
      //              ";");
      return update(
          reshaped, Layout::NONE, std::nullopt, std::vector<std::string>{});
    }

    if (dimC_in_mask.size() >= 2) {
      // logger.debug("      xx MERGING CHANNEL :: " + vectorToString(shape) +
      //              " >> " + vectorToString(reshaped) + " :: dimC=" +
      //              (dimC.has_value() ? std::to_string(*dimC) : "None") +
      //              " >> in_mask[dimC]=" + vectorToString(dimC_in_mask) +
      //              ";");
      return update(
          reshaped, Layout::NONE, std::nullopt, std::vector<std::string>{});
    }

    auto denote_ = _get_layout_denotation(tag, reshaped.size());
    return update(reshaped, std::nullopt, std::nullopt, denote_);
  }

private:
  // Helper function to convert mask to string for logging
  static std::string maskToString(const std::vector<std::vector<int>> &mask) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < mask.size(); ++i) {
      if (i > 0)
        ss << ", ";
      ss << "[";
      for (size_t j = 0; j < mask[i].size(); ++j) {
        if (j > 0)
          ss << ", ";
        ss << mask[i][j];
      }
      ss << "]";
    }
    ss << "]";
    return ss.str();
  }

  // Helper method to convert vector to string for debugging
  template <typename T>
  static std::string vectorToString(const std::vector<T> &vec) {
    std::stringstream ss;
    ss << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0)
        ss << ", ";
      ss << vec[i];
    }
    ss << "]";
    return ss.str();
  }
};

// Hash function specialization for std::unordered_map/set
namespace std {
template <>
struct hash<TaggedShape> {
  size_t operator()(const TaggedShape &ts) const { return ts.hash(); }
};
} // namespace std
