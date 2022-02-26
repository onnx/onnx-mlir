# SPDX-License-Identifier: Apache-2.0

set(ACCEL_LIST "") 
if (ACCELERATORS_TO_BUILD)
  foreach(t ${ACCELERATORS_TO_BUILD})
    message(DEBUG "Targeting ${t}")
    add_subdirectory(${t})
    list(APPEND ACCEL_LIST ${t}"Accel")
    endforeach(t)
endif (ACCELERATORS_TO_BUILD)
