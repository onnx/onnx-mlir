# SPDX-License-Identifier: Apache-2.0

# Only if LLVM is built with openmp enabled
if(TARGET omp)
  set(OMP_TOPDIR ${CMAKE_CURRENT_BINARY_DIR})
  set_directory_properties(PROPERTIES EP_BASE ${OMP_TOPDIR})

  ExternalProject_Add(OMomp
    DOWNLOAD_DIR "."
    SOURCE_DIR "."
    BINARY_DIR "."
    INSTALL_DIR "."
    STAMP_DIR "."
    TMP_DIR "tmp"

    DOWNLOAD_COMMAND ""
    # Build with LIBOMP_ENABLE_SHARED=OFF so that libomp.a can be embedded
    # into model.so
    CONFIGURE_COMMAND sh -c "CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} \
                             cmake -G Ninja ${LLVM_BUILD_MAIN_SRC_DIR} \
                                   -DLLVM_ENABLE_PROJECTS=\"clang$<SEMICOLON>openmp\" \
                                   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
                                   -DLIBOMP_ENABLE_SHARED=OFF"
    BUILD_COMMAND sh -c "cmake --build . --target omp -- ${MAKEFLAGS}"
    INSTALL_COMMAND ""
    )

  add_custom_target(ompruntime
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${OMP_TOPDIR}/lib/libomp.a ${ONNX_MLIR_LIBRARY_PATH}/libompruntime.a
    DEPENDS OMomp
    # BYPRODUCTS requires cmake 3.20+
    BYPRODUCTS ${OMP_TOPDIR}/lib/libomp.a
    )

  install(FILES ${ONNX_MLIR_LIBRARY_PATH}/libompruntime.a DESTINATION lib)

  message(STATUS "OpenMP support           : ON")
else()
  message(STATUS "OpenMP support           : OFF")
endif()
