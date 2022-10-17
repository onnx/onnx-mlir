# Function to setup ONNX model download for backend tests.
#
#   - use test.py -l model [--dynamic|--constant] to get the
#     list of models required
#   - for each model, get its data.json from third_party/onnx/backend/test/data/real
#   - read the data.json and regex match the model url and file
#   - create the target to download the model and add it to the
#     dependencies of the backend test specified
#
# We do this to pre-download all the required model files
# instead of letting onnx download them for two reasons:
#
#   - Model files are downloaded only once for different
#     backend tests.
#   - Our network connection is sometimes flaky and onnx's
#     retry_excute(3) with 5, 10, 15 seconds backoff (not
#     exponential) was not able to recover.
#
#     Now we have control over the download process ourselves
#     and can implement whatever recovery mechanism we want.
#     Currently we let curl handle it, retry 8 times with
#     exponential backoff.
function(setup_model_download backend_test variation)

  # Run test.py -l model [--dynamic|--constant] to get the
  # list of models required for the test.
  execute_process(
    COMMAND
      ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/test.py -l model ${variation}
    OUTPUT_VARIABLE BACKEND_MODELS
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  if(BACKEND_MODELS)
    string(REPLACE " " ";" BACKEND_MODELS ${BACKEND_MODELS})
  endif()

  # Make sure MODEL_DIR is the same as ONNX_MODELS in CMakeLists.txt
  set(MODEL_DIR ${FILE_GENERATE_DIR}/models)
  set(TEST_DATA_DIR
    ${ONNX_MLIR_SRC_ROOT}/third_party/onnx/onnx/backend/test/data/real/)

  foreach(m_cpu ${BACKEND_MODELS})
    # For each model, remove "_cpu" suffix, find its data.json,
    # and regex match to get the model url and file.
    string(REPLACE "_cpu" "" m ${m_cpu})
    set(MODEL_JSON ${TEST_DATA_DIR}/${m}/data.json)
    file(READ ${MODEL_JSON} DATA_JSON)
    string(REGEX MATCH "\"url\":[ ]\"(.*/([^/].*))\"" MODEL_URL_FULL ${DATA_JSON})

    # We have two submatches, one for the URL, the other for the file
    if(CMAKE_MATCH_COUNT EQUAL 2)
      set(MODEL_URL  ${CMAKE_MATCH_1})
      set(MODEL_FILE ${CMAKE_MATCH_2})

      # Now create a target for downloading this model if the
      # target doesn't already exist.
      #
      # Don't use a directory for DEPENDS and OUTPUT. Otherwise
      # the custom command will always be run.
      if (NOT (TARGET download_model_for_${m}))
	add_custom_target(download_model_for_${m}
	  DEPENDS ${MODEL_DIR}/${MODEL_FILE})
	add_custom_command(
	  OUTPUT
            ${MODEL_DIR}/${MODEL_FILE}
	  COMMAND
            mkdir -p ${MODEL_DIR} &&
            cd ${MODEL_DIR} &&
	    # Retry in case of download failure is handled by curl.
	    # Also curl will only download if remote file has a newer
	    # timestamp.
            curl ${MODEL_URL} --silent --retry 8
                 --time-cond ${MODEL_FILE} --output ${MODEL_FILE} &&
            tar zxf ${MODEL_FILE}
	  )
      endif()

      # Add the model download target to the dependencies of the
      # backend test specified.
      add_dependencies(${backend_test} download_model_for_${m})
    endif()
  endforeach(m_cpu)

endfunction(setup_model_download)
