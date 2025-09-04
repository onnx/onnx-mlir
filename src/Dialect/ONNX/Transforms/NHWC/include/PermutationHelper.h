// (c) Copyright 2022 - 2025 Advanced Micro Devices, Inc. All Rights Reserved.

#pragma once

#include <vector>
#include <algorithm>
#include <stdexcept>
#include <optional>
#include <numeric>
// Type definitions (assuming these are defined elsewhere)
using Perm = std::vector<int>;

// Forward declaration for Tensor class
class Tensor; // Assuming this exists elsewhere

class PermutationHelper {
public:
    /**
     * This class provides helper functions related to Transpose's permutation.
     */

    /**
     * Returns the identity permutation `[0, 1, ..., n-1]` where
     *   - n = x.get_shape().size(), IF x is a Tensor, OR:
     *   - n = x.size(),  IF x is a TensorShape or Perm (vector of ints), OR:
     *   - n = x, IF x is an int.
     */
    template<typename T>
    static std::optional<Perm> get_identity_perm(const T& x) {
        if constexpr (std::is_same_v<T, Tensor>) {
            auto shape = x.get_shape();
            return get_identity_perm_impl(static_cast<int>(shape.size()));
        } else if constexpr (std::is_same_v<T, TensorShape> || std::is_same_v<T, Perm>) {
            return get_identity_perm_impl(static_cast<int>(x.size()));
        } else if constexpr (std::is_same_v<T, int>) {
            return get_identity_perm_impl(x);
        } else {
            return std::nullopt;
        }
    }

    // Overload for nullptr/empty case
    static std::optional<Perm> get_identity_perm(std::nullptr_t) {
        return std::nullopt;
    }

    // Overload for optional types
    template<typename T>
    static std::optional<Perm> get_identity_perm(const std::optional<T>& x) {
        if (!x.has_value()) {
            return std::nullopt;
        }
        return get_identity_perm(x.value());
    }

    /**
     * Check if the given permutation is an identity permutation [0, 1, ..., n-1]
     */
    static bool is_identity_perm(const Perm& x) {
        Perm identity(x.size());
        std::iota(identity.begin(), identity.end(), 0);
        return x == identity;
    }

    /**
     * Permute values of a vector `x` using given permutation `perm`
     */
    template<typename T>
    static std::vector<T> permute(const std::vector<T>& x, const Perm& perm) {
        if (perm.size() != x.size()) {
            throw std::invalid_argument(
                "perm size (" + std::to_string(perm.size()) + 
                ") does not match x size (" + std::to_string(x.size()) + ")"
            );
        }

        // Check if perm is a valid permutation
        std::vector<int> sorted_perm = perm;
        std::sort(sorted_perm.begin(), sorted_perm.end());
        for (size_t i = 0; i < sorted_perm.size(); ++i) {
            if (sorted_perm[i] != static_cast<int>(i)) {
                throw std::invalid_argument("perm is not a valid permutation");
            }
        }

        std::vector<T> result;
        result.reserve(x.size());
        for (int i : perm) {
            result.push_back(x[i]);
        }
        return result;
    }

    /**
     * Returns the inverse permutation of the given permutation `perm`.
     * The following holds true:
     *   - permute(inverse_perm, perm) == permute(perm, inverse_perm) == identity_perm(perm) == [0, 1, ..., n-1]
     *   - where n := perm.size()
     */
    static Perm get_inverse_perm(const Perm& perm) {
        Perm inverse_perm(perm.size());
        for (size_t i = 0; i < perm.size(); ++i) {
            if (perm[i] < 0 || static_cast<size_t>(perm[i]) >= perm.size()) {
                throw std::invalid_argument("Invalid permutation: index out of range");
            }
            inverse_perm[perm[i]] = static_cast<int>(i);
        }
        return inverse_perm;
    }

    /**
     * Check if the permutation is non-trivial (not an identity permutation)
     */
    static bool is_nontrivial_permutation(const Perm& perm) {
        return !is_identity_perm(perm);
    }

private:
    /**
     * Helper function to create identity permutation of given size
     */
    static std::optional<Perm> get_identity_perm_impl(int size) {
        if (size <= 0) {
            return std::nullopt;
        }
        Perm result(size);
        std::iota(result.begin(), result.end(), 0);
        return result;
    }
};