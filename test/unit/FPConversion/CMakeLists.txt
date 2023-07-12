# SPDX-License-Identifier: Apache-2.0

add_custom_target(TestFPConversion
  COMMAND
    ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/TestFPConversion.py
  )
