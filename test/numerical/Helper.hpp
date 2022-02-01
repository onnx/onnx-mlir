/*
 * SPDX-License-Identifier: Apache-2.0
 */

//====----------------------- Helper.hpp ----------------------------------===//
//
// This file defines some helper functions used for numerical tests.
//
//====---------------------------------------------------------------------===//

/// Sigmoid
float sigmoid(float x) { return 1 / (1 + exp(-x)); }

std::string getSharedLibName(std::string sharedLibBaseName) {
#ifdef _WIN32
  return sharedLibBaseName + ".dll";
#else
  return sharedLibBaseName + ".so";
#endif
}
