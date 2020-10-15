include(GoogleTest)

# Adapted from https://cliutils.gitlab.io/modern-cmake/chapters/testing/googletest.html.
macro(add_unit_test TESTNAME)
    # create an exectuable in which the tests will be stored
    add_executable(${TESTNAME} ${ARGN})
    # link the Google test infrastructure, mocking library, and a default main fuction to
    # the test executable.  Remove g_test_main if writing your own main function.
    target_link_libraries(${TESTNAME} gtest gmock gtest_main)
    # gtest_discover_tests replaces gtest_add_tests,
    # see https://cmake.org/cmake/help/v3.10/module/GoogleTest.html for more options to pass to it
    gtest_discover_tests(${TESTNAME}
            # set a working directory so your project root so that you can find test data via paths relative to the project root
            WORKING_DIRECTORY ${ONNX_MLIR_BIN_ROOT}
            PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${ONNX_MLIR_BIN_ROOT}")
endmacro()

macro(add_c_unit_test TESTNAME)
    add_executable(${TESTNAME} ${ARGN})
    add_test(NAME ${TESTNAME}
             COMMAND ${TESTNAME})
endmacro()

add_subdirectory(Runtime)