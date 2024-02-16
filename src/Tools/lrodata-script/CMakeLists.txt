# SPDX-License-Identifier: Apache-2.0

# file(READ ...) cannot set global variable so we only set the
# script filename here and let the "Compiler" subdirectory do
# the file(READ ...)
unset(LRODATA_SCRIPT CACHE)
if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "s390x")
  set(LRODATA_SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/s390x-lrodata.ld"
      CACHE INTERNAL "")
endif()
