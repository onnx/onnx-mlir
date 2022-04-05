# SPDX-License-Identifier: Apache-2.0

# We need to do this in a stand-alone cmake script because there's no
# cross-platform way to append to a file through add_custom_command.

file(WRITE "${INC_FILE}.tmp" "#define APPLY_TO_ACCELERATORS(MACRO) \\\n")

foreach(t ${ACCELERATORS})
  file (APPEND "${INC_FILE}.tmp" "  MACRO(${t}) \\\n")
endforeach(t)

file(APPEND "${INC_FILE}.tmp" "\n")

file(COPY_FILE "${INC_FILE}.tmp" ${INC_FILE} ONLY_IF_DIFFERENT)
file(REMOVE "${INC_FILE}.tmp")
