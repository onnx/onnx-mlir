add_dlc_library(OMLayoutHelper
  LayoutHelper.cpp

  LINK_LIBS PUBLIC
  MLIRIR
)

add_dlc_library(OMDLCSupport
  OMDLCOptions.cpp

  LINK_LIBS PUBLIC
  OMKrnlOps
)
