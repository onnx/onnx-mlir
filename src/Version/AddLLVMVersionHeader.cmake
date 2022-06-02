# SPDX-License-Identifier: Apache-2.0

# We need to do this in a stand-alone cmake script because there's no
# cross-platform way to append to a file through add_custom_command.

file(APPEND
  ${HEADER_FILE}
  "#include \"llvm/Support/VCSRevision.h\"\n"
  )
