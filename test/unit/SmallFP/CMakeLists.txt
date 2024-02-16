# SPDX-License-Identifier: Apache-2.0

add_unittest(TestSmallFP
  TestSmallFP.cpp

  LINK_LIBS PRIVATE
  OMMlirUtilities
  benchmark
  )
