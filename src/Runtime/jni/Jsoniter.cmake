# SPDX-License-Identifier: Apache-2.0

# We use Jsoniter for JNI backend tests. Jackson is another option.
# But tests show that Jsoniter is faster and uses less memory even
# though it hasn't been updated since 2018.
#
# Function to setup Jsoniter jar. First look for an installed version
# using find_jar. If not found, download from maven repository if the
# ONNX_MLIR_BUILD_JSONITER option is not set. Otherwise, clone the
# jsoniter github repo and build one locally.
#
# Ultimately, JSONITER_JAR is set to the proper jsoniter jar path.
#
function(setup_jsoniter version)
  set(JSONITER_NAME jsoniter-${version})

  # Search for an installed version (note do not include .jar
  # in the path for find_jar)
  unset(JSONITER_JAR CACHE)
  find_jar(JSONITER_JAR ${JSONITER_NAME})

  # If an installed version is not found, download one from maven
  # repository, or build one locally if ONNX_MLIR_BUILD_JSONITER
  # option is set.
  if (${JSONITER_JAR} STREQUAL "JSONITER_JAR-NOTFOUND")
    set(JSONITER_DIR ${CMAKE_CURRENT_BINARY_DIR}/jsoniter)
    set(JSONITER_FILE ${JSONITER_NAME}.jar)

    if (NOT ONNX_MLIR_BUILD_JSONITER)
      set(JSONITER_MAVEN_URL
        https://repo1.maven.org/maven2/com/jsoniter/jsoniter/${version}/${JSONITER_FILE})
      set(JSONITER_PATH ${JSONITER_DIR}/maven)

      # Target to download jsoniter jar from maven repository
      ExternalProject_Add(jsoniter_external
        URL ${JSONITER_MAVEN_URL}
        URL_HASH SHA1=4cff9a02d6d9a848d1b7298aac36c47fd9e67d77
        PREFIX ${JSONITER_DIR}
        DOWNLOAD_DIR ${JSONITER_PATH}
        DOWNLOAD_NO_PROGRESS true
        DOWNLOAD_NO_EXTRACT true
        CONFIGURE_COMMAND ""
        BUILD_COMMAND ""
        INSTALL_COMMAND ""
        )
    else()
      set(JSONITER_GITHUB_URL https://github.com/json-iterator/java.git)
      set(JSONITER_PATH ${JSONITER_DIR}/github/target)

      # Target to build jsoniter jar locally
      ExternalProject_Add(jsoniter_external
        GIT_REPOSITORY ${JSONITER_GITHUB_URL}
        GIT_TAG ${version}
        PREFIX ${JSONITER_DIR}
        SOURCE_DIR ${JSONITER_DIR}/github
        CONFIGURE_COMMAND ""
        BUILD_COMMAND mvn -Dmaven.artifact.threads=$$\{NPROC:4\}
                          -Dmaven.repo.local=${CMAKE_CURRENT_BINARY_DIR}/.m2
                          -Dmaven.javadoc.skip=true
                          -Dmaven.source.skip=true
                          -Dmaven.test.skip=true
                          -Djar.finalName=${JSONITER_NAME}
                          -q -f ${JSONITER_DIR}/github package
        INSTALL_COMMAND ""
        )
    endif()

    # For some lovely reason, the target dependencies between the target generated
    # by ExternalProject_Add and add_jar do not work correctly (with ninja, possibly
    # other generators as well). To work around the issue, we create a fake target
    # that explicitly lists the jsoniter jar as a byproduct. This way there will be both
    # a dependency based on the targets added with add_dependencies but also an explicit
    # dependency on the byproducts. By doing a local copy of the downloaded file, we avoid
    # having to download the file multiple times while also creating the byproduct.
    set(JSONITER_JAR ${JSONITER_DIR}/${JSONITER_FILE} CACHE STRING "" FORCE)
    add_custom_target(jsoniter
      COMMAND ${CMAKE_COMMAND} -E copy_if_different 
              ${JSONITER_PATH}/${JSONITER_FILE} ${JSONITER_JAR}
      DEPENDS jsoniter_external
      BYPRODUCTS ${JSONITER_JAR}
      )
  endif()
endfunction(setup_jsoniter)
