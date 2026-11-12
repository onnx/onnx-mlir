# SPDX-License-Identifier: Apache-2.0

add_unittest(TestGemmVersionConv
  TestGemmVersionConv.cpp

  LINK_LIBS PRIVATE
  OMBuilder
  OMONNXOps
  )