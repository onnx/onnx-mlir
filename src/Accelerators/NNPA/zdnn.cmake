# SPDX-License-Identifier: Apache-2.0

function(setup_zdnn version)
  # Set policy CMP0097 to NEW for it to not initialize submodules
  cmake_policy(SET CMP0097 NEW)

  set(ZDNN_GITHUB_URL https://github.com/IBM/zDNN.git)
  message("Git clone zDNN. The ZDNN_GITHUB_URL is: ${ZDNN_GITHUB_URL}")

  set(ZDNN_PREFIX     ${CMAKE_CURRENT_BINARY_DIR}/zDNN)
  set(ZDNN_TOPDIR     ${ZDNN_PREFIX}/src/zdnn)
  set(ZDNN_OBJDIR     ${ZDNN_TOPDIR}/zdnn/obj)
  set(ZDNN_LIBDIR     ${ZDNN_TOPDIR}/zdnn/lib)

  # Only build libzdnn on s390x
  if (${CMAKE_SYSTEM_PROCESSOR} STREQUAL "s390x")
    ExternalProject_Add(zdnn
      GIT_REPOSITORY ${ZDNN_GITHUB_URL}
      GIT_TAG ${version}
      GIT_SUBMODULES ""
      PREFIX ${ZDNN_PREFIX}
      BUILD_IN_SOURCE ON
      CONFIGURE_COMMAND sh -c "autoconf && ./configure"

      # We build libzdnn.so so that obj/*.o are compiled with -fPIC
      # Then we create libzdnn.a ourselves from these PIC .o since
      # we want to embed libzdnn.a into model.so.
      #
      # Note we use sh to run the command. Otherwise, cmake will
      # generate quotes around ${ZDNN_OBJDIR}/*.o due to the *
      # in the path, which breaks ar.
      #
      # Set MAKEFLAGS to remove -j option passed down from the top
      # level make which produces a warning about jobserver unavailable.
      #
      # Run make -q first to skip build if libzdnn.so is already
      # up to date.
      BUILD_COMMAND sh -c "export MAKEFLAGS=--no-print-directory && \
                         make -q -C zdnn lib/libzdnn.so && true || \
                         (MAKEFLAGS=--no-print-directory \
                          make -j$(nproc) -C zdnn lib/libzdnn.so && \
                          ar -rc ${ZDNN_LIBDIR}/libzdnn.a ${ZDNN_OBJDIR}/*.o)"

      INSTALL_COMMAND ""
      )

    add_custom_target(libzdnn
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${ZDNN_LIBDIR}/libzdnn.a ${NNPA_LIBRARY_PATH}/libzdnn.a
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${ZDNN_TOPDIR}/zdnn/zdnn.h ${NNPA_INCLUDE_PATH}/zdnn.h
      DEPENDS zdnn
      # BYPRODUCTS requires cmake 3.20+ not yet available on Ubuntu Focal
      # BYPRODUCTS ${NNPA_LIBRARY_PATH}/libzdnn.a ${NNPA_INCLUDE_PATH}/zdnn.h
      )

    install(FILES ${NNPA_LIBRARY_PATH}/libzdnn.a DESTINATION lib)

  # On other archs, just copy zdnn.h so NNPA code can be compiled
  else()
    ExternalProject_Add(zdnn
      GIT_REPOSITORY ${ZDNN_GITHUB_URL}
      GIT_TAG ${version}
      GIT_SUBMODULES ""
      PREFIX ${ZDNN_PREFIX}
      BUILD_IN_SOURCE ON
      CONFIGURE_COMMAND ""
      BUILD_COMMAND ""
      INSTALL_COMMAND ""
      )
    add_custom_target(libzdnn
      COMMAND ${CMAKE_COMMAND} -E copy_if_different
              ${ZDNN_TOPDIR}/zdnn/zdnn.h ${NNPA_INCLUDE_PATH}/zdnn.h
      DEPENDS zdnn
      )
  endif()
endfunction(setup_zdnn)
