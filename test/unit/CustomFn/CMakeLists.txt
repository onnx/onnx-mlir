# SPDX-License-Identifier: Apache-2.0

add_unittest(TestCustomFn
  TestCustomFn.cpp

  LINK_LIBS PRIVATE
  OMBuilder
  OMCompilerPasses
  OMONNXOps
  )
