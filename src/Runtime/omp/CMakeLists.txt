# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------
# Find OpenMP headers built by LLVM
# ------------------------------------------------------------
find_path(OPENMP_INCLUDE_DIR
  NAMES omp.h
  HINTS
    ${LLVM_BUILD_BINARY_DIR}/lib/clang/*/include
  NO_DEFAULT_PATH
)

if(OPENMP_INCLUDE_DIR)
  message(STATUS "Found OpenMP headers: ${OPENMP_INCLUDE_DIR}")

  set(OMP_TOPDIR ${CMAKE_CURRENT_BINARY_DIR})
  set_directory_properties(PROPERTIES EP_BASE ${OMP_TOPDIR})

  include(ExternalProject)
  ExternalProject_Add(OMomp
    DOWNLOAD_DIR "."
    SOURCE_DIR "."
    BINARY_DIR "."
    INSTALL_DIR "."
    STAMP_DIR "."
    TMP_DIR "tmp"

    DOWNLOAD_COMMAND ""
    CONFIGURE_COMMAND sh -c "CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} \
                             cmake -G Ninja ${LLVM_BUILD_MAIN_SRC_DIR} \
                                   -DLLVM_ENABLE_PROJECTS=\"clang\" \
                                   -DLLVM_ENABLE_RUNTIMES=\"openmp\" \
                                   -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
                                   -DLIBOMP_ENABLE_SHARED=OFF"
    BUILD_COMMAND sh -c "cd ${LLVM_BUILD_BINARY_DIR}/runtimes/runtimes-bins/openmp/runtime/src && ninja"
    INSTALL_COMMAND ""
  )

  find_file(LIBOMP_PATH
    NAMES libomp.a libomp.dylib libomp.so omp.lib iomp5.lib
    PATHS ${LLVM_BUILD_BINARY_DIR}/runtimes/runtimes-bins/openmp/runtime/src
    NO_DEFAULT_PATH
  )

  if(NOT LIBOMP_PATH)
    message(FATAL_ERROR "Could not find built OpenMP library (libomp.a or shared)")
  endif()

  message(STATUS "Found OpenMP library: ${LIBOMP_PATH}")

  add_custom_target(ompruntime
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
            ${LIBOMP_PATH} ${ONNX_MLIR_LIBRARY_PATH}/libompruntime.a
    DEPENDS OMomp
    BYPRODUCTS ${ONNX_MLIR_LIBRARY_PATH}/libompruntime.a
  )

  install(FILES ${ONNX_MLIR_LIBRARY_PATH}/libompruntime.a DESTINATION lib)

  message(STATUS "OpenMP support           : ON")
else()
  message(STATUS "OpenMP support           : OFF")
endif()
