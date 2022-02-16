add_dlc_library(OMFoldStdAlloc
  FoldStdAlloc.cpp

  LINK_LIBS PUBLIC
  OMSupport
  MLIRAffineToStandard
  )

add_subdirectory(ZHigh)
