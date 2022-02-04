# SPDX-License-Identifier: Apache-2.0

if( ACCELERATORS_TO_BUILD )
  foreach(t ${ACCELERATORS_TO_BUILD})
    message(STATUS "Targeting ${t}")
    add_subdirectory(${t})
    endforeach(t)
else( ACCELERATORS_TO_BUILD )
  message(STATUS "No Accelerators Targeted")
  add_subdirectory(None)
endif( ACCELERATORS_TO_BUILD )
  
