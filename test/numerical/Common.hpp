/*
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(__MSC_VER)
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#else // gcc, clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
#endif
#include <rapidcheck.h>
#if defined(__MSC_VER)
#undef _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#else
#pragma GCC diagnostic pop
#endif

#include "test/modellib/ModelLib.hpp"
