# SPDX-License-Identifier: Apache-2.0

if(OMP_SUPPORTED)
  set(OMP_TOPDIR ${CMAKE_CURRENT_BINARY_DIR})
  set_directory_properties(PROPERTIES EP_BASE ${OMP_TOPDIR})
  set(OPENMP_SOURCE_DIR ${LLVM_BUILD_MAIN_SRC_DIR}/../openmp)

  include(ExternalProject)
  ExternalProject_Add(OMomp
    DOWNLOAD_DIR "."
    BINARY_DIR ${OMP_TOPDIR}/openmp-build
    INSTALL_DIR ${OMP_TOPDIR}/install
    STAMP_DIR ${OMP_TOPDIR}/stamp
    TMP_DIR ${OMP_TOPDIR}/tmp

    DOWNLOAD_COMMAND ""
    # Build with LIBOMP_ENABLE_SHARED=OFF so that libomp.a can be embedded into model.so
    # Since OpenMP is not exposed in the LLVM build directory, it must be configured
    # and built separately in its own build directory
    CONFIGURE_COMMAND sh -c "CC=${CMAKE_C_COMPILER} CXX=${CMAKE_CXX_COMPILER} \
                             cmake -G Ninja -S ${OPENMP_SOURCE_DIR} \
                                -B ${OMP_TOPDIR}/openmp-build \
                                -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
                                -DLIBOMP_ENABLE_SHARED=OFF"

    BUILD_COMMAND sh -c "cmake --build ${OMP_TOPDIR}/openmp-build --target omp"
    INSTALL_COMMAND ""
  )

  find_file(LIBOMP_PATH
    NAMES libomp.a libomp.dylib libomp.so omp.lib libomp.lib iomp5.lib
    PATHS ${LLVM_BUILD_BINARY_DIR}/runtimes/runtimes-bins/openmp/runtime/src
    NO_DEFAULT_PATH
  )

  if(NOT LIBOMP_PATH)
    message(FATAL_ERROR "Could not find built OpenMP library (libomp.a or shared)")
  endif()

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
