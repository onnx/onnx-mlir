# SPDX-License-Identifier: Apache-2.0

# We need to do this in a stand-alone cmake script because there's no
# cross-platform way to append to a file through add_custom_command.

file(WRITE "${INC_FILE}.tmp" "#define APPLY_TO_ACCELERATORS(MACRO, ...) \\\n")

foreach(t ${ACCELERATORS})
  file (APPEND "${INC_FILE}.tmp" "  MACRO(${t}, ## __VA_ARGS__) \\\n")
endforeach(t)

file (APPEND "${INC_FILE}.tmp" "\n")

file (APPEND "${INC_FILE}.tmp" "#define APPLY_TO_NO_ACCELERATORS(MACRO) \\\n")

if ("${ACCELERATORS}" STREQUAL "")
  file (APPEND "${INC_FILE}.tmp" "  MACRO \n")
endif()

file(APPEND "${INC_FILE}.tmp" "\n")

# Copy the file only if it has changed.
execute_process(COMMAND ${CMAKE_COMMAND} -E copy_if_different
  "${INC_FILE}.tmp" "${INC_FILE}")
file(REMOVE "${INC_FILE}.tmp")
