# SPDX-License-Identifier: Apache-2.0

function(setup_zdnn version)
  set(ZDNN_GITHUB_URL https://github.com/IBM/zDNN)
  set(ZDNN_PREFIX     ${CMAKE_CURRENT_BINARY_DIR}/zDNN)
  set(ZDNN_TOPDIR     ${ZDNN_PREFIX}/src/zdnn)
  set(ZDNN_OBJDIR     ${ZDNN_TOPDIR}/zdnn/obj)
  set(ZDNN_LIBDIR     ${ZDNN_TOPDIR}/zdnn/lib)

  ExternalProject_Add(zdnn
    GIT_REPOSITORY ${ZDNN_GITHUB_URL}
    GIT_TAG ${version}
    PREFIX ${ZDNN_PREFIX}
    BUILD_IN_SOURCE ON

    # Skip autoconf and configure if config.make already exists
    CONFIGURE_COMMAND sh -c "[ -f config.make ] || (autoconf && ./configure)"

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
    BUILD_COMMAND sh -c "MAKEFLAGS=--no-print-directory \
                         make -j$(nproc) -C zdnn lib/libzdnn.so && \
                         ar -rc ${ZDNN_LIBDIR}/libzdnn.a ${ZDNN_OBJDIR}/*.o"

    INSTALL_COMMAND ""
    )

  add_custom_target(libzdnn
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ZDNN_LIBDIR}/libzdnn.a ${NNPA_LIBRARY_PATH}/libzdnn.a
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${ZDNN_TOPDIR}/zdnn/zdnn.h ${NNPA_INCLUDE_PATH}/zdnn.h
    DEPENDS zdnn
    #BYPRODUCTS ${NNPA_LIBRARY_PATH}/libzdnn.a ${NNPA_INCLUDE_PATH}/zdnn.h
    )

  install(FILES ${NNPA_LIBRARY_PATH}/libzdnn.a DESTINATION lib)
endfunction(setup_zdnn)
