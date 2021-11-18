# SPDX-License-Identifier: Apache-2.0

# Function to setup Jsoniter jar. First look for an installed version
# using find_jar. If not found, clone the jsoniter github repo and
# build one locally. Either way, JSONITER_JAR is set to the proper
# jsoniter jar path.
function(setup_jsoniter version)
  # We use Jsoniter for JNI backend tests. Jackson is another option.
  # But tests show that Jsoniter is faster and uses less memory even
  # though it hasn't been updated since 2018.
  set(JSONITER_REPO_URL https://github.com/json-iterator/java.git)
  set(JSONITER_REPO_DIR ${CMAKE_CURRENT_BINARY_DIR}/jsoniter)
  set(JSONITER_REPO_JAR ${JSONITER_REPO_DIR}/target/jsoniter-${version}.jar)

  # Search for an installed version (note do not include .jar
  # in the path for find_jar)
  unset(JSONITER_JAR CACHE)
  find_jar(JSONITER_JAR jsoniter-${version})

  # Build Jsoniter locally if an installed version is not found
  if (${JSONITER_JAR} STREQUAL "JSONITER_JAR-NOTFOUND")
    set(JSONITER_JAR ${JSONITER_REPO_JAR} CACHE STRING "" FORCE)

    add_custom_command(
      OUTPUT ${JSONITER_JAR}
      COMMAND git clone -b ${version} ${JSONITER_REPO_URL} ${JSONITER_REPO_DIR}
      COMMAND mvn -Dmaven.artifact.threads=$$\{NPROC:-$$\(nproc\)\}
                  -Dmaven.repo.local=${CMAKE_CURRENT_BINARY_DIR}/.m2
                  -Dmaven.javadoc.skip=true
                  -Dmaven.source.skip=true
                  -Dmaven.test.skip=true
                  -q -f ${JSONITER_REPO_DIR} package
      )

    add_custom_target(jsoniter
      DEPENDS ${JSONITER_JAR}
      )
  endif()
endfunction(setup_jsoniter)
