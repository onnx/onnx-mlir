# SPDX-License-Identifier: Apache-2.0

include(CTest)

macro(add_c_unit_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    add_test(NAME ${TESTNAME}
             COMMAND ${TESTNAME})
endmacro()

add_subdirectory(Runtime)
