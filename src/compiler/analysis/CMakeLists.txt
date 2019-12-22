add_library(DLCAnalysis STATIC
        extract_integer_set.cpp)

target_include_directories(DLCAnalysis PRIVATE ${DLC_SRC_ROOT})
target_include_directories(DLCAnalysis PRIVATE ${DLC_BIN_ROOT})