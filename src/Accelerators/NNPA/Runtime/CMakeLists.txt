# SPDX-License-Identifier: Apache-2.0

# TODO: statically link to libcruntime.a

# The memory cache, or its pass-through (option in the parent directory).
if (ONNX_MLIR_NNPA_MEMORY_CACHE)
  set(NNPA_MEMORY_CACHE_SRC OMMemoryCache.c)
else()
  set(NNPA_MEMORY_CACHE_SRC OMMemoryCachePassThrough.c)
endif()

add_onnx_mlir_library(RuntimeNNPA STATIC
  ${NNPA_MEMORY_CACHE_SRC}
  OMRuntimeNNPA.c
  zdnnx/zdnnx.c
  zdnnx/zdnnx_ops.c
  zdnnx/zdnnx_ops_private.c
  zdnnx/seq_ops.c
  zdnnx/omp_ops.c
  zdnnx/utils.c

  EXCLUDE_FROM_OM_LIBS

  DEPENDS
  libzdnn

  INCLUDE_DIRS PRIVATE
  ${NNPA_INCLUDE_PATH}
  )


if (ZDNNX_WITH_OMP)
  set(C_COMPILE_OPTIONS "-O3;-fopenmp")
else()
  set(C_COMPILE_OPTIONS "-O3")
endif()

set_target_properties(RuntimeNNPA
  PROPERTIES
  LANGUAGE C
  POSITION_INDEPENDENT_CODE TRUE
  COMPILE_OPTIONS "${C_COMPILE_OPTIONS}"
  )

