# SPDX-License-Identifier: Apache-2.0

# Valid optimization levels are 0, 1, 2, and 3
# Default to 3 if ONNX_MLIR_TEST_OPTLEVEL not set or cached
set(OPTLEVELS 0 1 2 3)
if (NOT DEFINED ONNX_MLIR_TEST_OPTLEVEL)
  set(ONNX_MLIR_TEST_OPTLEVEL 3 CACHE STRING "" FORCE)
endif()
# Fail if specified/cached ONNX_MLIR_TEST_OPTLEVEL is not valid
if (NOT ${ONNX_MLIR_TEST_OPTLEVEL} IN_LIST OPTLEVELS)
  unset(ONNX_MLIR_TEST_OPTLEVEL CACHE)
  message(FATAL_ERROR "ONNX_MLIR_TEST_OPTLEVEL must be one of ${OPTLEVELS}")
endif()
message(STATUS "Tests optimization level : ${ONNX_MLIR_TEST_OPTLEVEL}")

# Create one target for all the backend and numerical tests that we
# want to run in parallel. The reason we have to do this follows.
#
# Normally, to build targets in parallel, we just need to list them
# on the make command, e.g.,
#
#   make -j check-onnx-backend check-onnx-backend-jni ...
#
# However, this does not work with cmake generated Makefile because it
# disables parallel build at the top level:
#
#  # Allow only one "make -f Makefile2" at a time, but pass parallelism.
#  .NOTPARALLEL:
#
# Therefore, in order to build in parallel, we have to use Makefile2:
#
#   make -j -f CMakeFiles/Makefile2 check-onnx-backend check-onnx-backend-jni ...
#
# This however comes with another problem. Since multiple backend tests
# use the same model and therefore depend on the same model download target
# such as download_model_for_test_resnet50, the download target is built
# multiple times in parallel because the top level backend test targets
# themselves, such as check-onnx-backend, check-onnx-backend-jni, etc.,
# have no relationship to each other. Multiple download_model_for_test_resnet50
# targets trying to download and unpack the same model in the same directory
# will collide with each other and corrupt the model file(s).
#
# To "get around" this problem, we create this "umbrella" target
# check-onnx-backend-numerical, and add all the backend and numerical
# tests (whichever we want to run in parallel) as its dpendencies. Now
# we can run:
#
#   make -j check-onnx-backend-numerical
#
# and the tests will run in parallel because they are not the top level
# targets anymore so make will parallelize them at the 2nd level. And
# now the backend test targets are also related to each other - they are
# all dependents of check-onnx-backend-numerical - so make will work out
# the dependencies properly and only one instance of download_model_for_test_resnet50
# will run.
#
# This allows us to run the backend and numerical tests in parallel but
# only one instance of model download target will run (even though multiple
# backend test targets depend on it) to avoid download/unpack collision.
add_custom_target(check-onnx-backend-numerical)

# The backend tests require ONNX package installation.
add_subdirectory(backend)
add_subdirectory(accelerators)

# The following do NOT require ONNX package installation.
add_subdirectory(mlir)
add_subdirectory(modellib)
add_subdirectory(numerical)
add_subdirectory(backend-cpp)
add_subdirectory(unit)
add_subdirectory(compilerlib)
add_subdirectory(perf)
add_subdirectory(multiple-models)

