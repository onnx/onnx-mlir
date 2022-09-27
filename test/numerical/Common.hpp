/*
 * SPDX-License-Identifier: Apache-2.0
 */

#if defined(_MSC_VER)
#ifndef _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#define _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#else
#define _SILENCE_WAS_PREDEFINED
#endif
#else // gcc, clang
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated"
#endif

#include <rapidcheck.h>

#if defined(_MSC_VER)
#if defined(_SILENCE_ALL_CXX17_DEPRECATION_WARNINGS) &&                        \
    !defined(_SILENCE_WAS_PREDEFINED)
#undef _SILENCE_ALL_CXX17_DEPRECATION_WARNINGS
#endif
#else
#pragma GCC diagnostic pop
#endif

#include "test/modellib/ModelLib.hpp"
