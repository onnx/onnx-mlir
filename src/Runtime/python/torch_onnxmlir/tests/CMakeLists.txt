# SPDX-License-Identifier: Apache-2.0

cmake_minimum_required(VERSION 3.26.0)

project(torch_onnxmlir_tests NONE)

find_package(Python3 REQUIRED COMPONENTS Interpreter)

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/utils.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/utils.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_sessioncache.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/test_sessioncache.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_add.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/test_add.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_gemm.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/test_gemm.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_relu.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/test_relu.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_encoder_models.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/test_encoder_models.py
  )

file(GENERATE
  OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/test_decoder_models.py
  INPUT ${CMAKE_CURRENT_SOURCE_DIR}/test_decoder_models.py
  )

# Use ctest.
enable_testing()

# Tests for single operation.
set(OP_LIST
  "${CMAKE_CURRENT_BINARY_DIR}/test_sessioncache.py"
  "${CMAKE_CURRENT_BINARY_DIR}/test_add.py"
  "${CMAKE_CURRENT_BINARY_DIR}/test_gemm.py"
  "${CMAKE_CURRENT_BINARY_DIR}/test_relu.py"
)

foreach(testfile ${OP_LIST})
    get_filename_component(testname ${testfile} NAME_WE)
    add_test(
        NAME ${testname}
        COMMAND ${Python3_EXECUTABLE} -m pytest -v ${testfile}
    )
endforeach()

# Tests for encoder models.
# Enable one model only to keep the runtime short.
set(ENCODER_MODELS
  "granite-embedding-30m-english"
  # "granite-embedding-278m-multilingual"
  # "bert-base-uncased"
)

foreach(model ${ENCODER_MODELS})
    add_test(
        NAME test_${model}
        COMMAND ${CMAKE_COMMAND} -E env
                ENCODER_MODEL=${model}
                ${Python3_EXECUTABLE}
                -m pytest
                ${CMAKE_CURRENT_BINARY_DIR}/test_encoder_models.py
                -v
    )
endforeach()

# Tests for decoder models.
# Enable one model only to keep the runtime short.
set(DECODER_MODELS
  # "gpt2"
  # "Qwen2-0.5B-Instruct"
  # "granite-3.3-2b-instruct"
  "granite-4.0-350M"
)

foreach(model ${DECODER_MODELS})
    add_test(
        NAME test_${model}
        COMMAND ${CMAKE_COMMAND} -E env
                DECODER_MODEL=${model}
                ${Python3_EXECUTABLE}
                -m pytest
                ${CMAKE_CURRENT_BINARY_DIR}/test_decoder_models.py
                -v
    )
endforeach()
