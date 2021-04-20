set (DATASET_LIST test_data_set_0
                  test_data_set_1
                  test_data_set_2)

# These tests should never fail at compile time, only tests time,
# since compilation is supported on platforms that can't run the tests

option(ADO_DOWNLOAD_ONNX_TEST_FILES "Download files used for pipeline tests." OFF)

set(ADO_ONNX_TEST_DIR ${CMAKE_BINARY_DIR}/test_models CACHE STRING "Directory to download pipeline test files.")

if(ADO_DOWNLOAD_ONNX_TEST_FILES)
  cmake_minimum_required(VERSION 3.18)
  file(DOWNLOAD https://github.com/onnx/models/raw/master/vision/classification/mnist/model/mnist-7.tar.gz ${CMAKE_CURRENT_BINARY_DIR}/mnist-7.tar.gz)
  file(ARCHIVE_EXTRACT INPUT ${CMAKE_CURRENT_BINARY_DIR}/mnist-7.tar.gz DESTINATION ${ADO_ONNX_TEST_DIR})
endif()

# Currently only test on x86
foreach (DATASET_NAME ${DATASET_LIST})
  add_test (NAME EmitONNXIR_${DATASET_NAME} COMMAND onnx-mlir --EmitONNXIR ${ADO_ONNX_TEST_DIR}/mnist/model.onnx)
  add_test (NAME EmitLib_${DATASET_NAME}    COMMAND onnx-mlir --EmitLib ${ADO_ONNX_TEST_DIR}/mnist/model.onnx.mlir)
  add_test (NAME RunLib_${DATASET_NAME}     COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/onnx-run-library.py ${ADO_ONNX_TEST_DIR}/mnist/model.onnx.so ${ADO_ONNX_TEST_DIR}/mnist/${DATASET_NAME})

  set_tests_properties(EmitONNXIR_${DATASET_NAME} PROPERTIES FIXTURES_SETUP ${DATASET_NAME}_ir_fix)
  set_tests_properties(EmitLib_${DATASET_NAME}    PROPERTIES FIXTURES_REQUIRED ${DATASET_NAME}_ir_fix
                                                             FIXTURES_SETUP ${DATASET_NAME}_lib_fix)
  set_tests_properties(RunLib_${DATASET_NAME}     PROPERTIES FIXTURES_REQUIRED ${DATASET_NAME}_lib_fix
                                                             ENVIRONMENT ONNX_MLIR_HOME=${CMAKE_BINARY_DIR})
endforeach()
