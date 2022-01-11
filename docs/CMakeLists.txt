# SPDX-License-Identifier: Apache-2.0

add_subdirectory(doc_check)
add_subdirectory(doc_example)

add_custom_target(check-docs
  DEPENDS
  check-doc
  check-doc-example
  )

set_target_properties(check-docs PROPERTIES FOLDER "Docs")
# Exclude the target from the default VS build
set_target_properties(check-docs PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD ON)
