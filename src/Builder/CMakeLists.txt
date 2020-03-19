add_library(OMBuilder
        FrontendDialectHelper.cpp
        FrontendDialectHelper.hpp
        FrontendDialectTransformer.cpp
        FrontendDialectTransformer.hpp
        OpBuildTable.inc)

target_include_directories(OMBuilder PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(OMBuilder PRIVATE ${CMAKE_BINARY_DIR})

target_link_libraries(OMBuilder
        onnx
        ${MLIRLibs}
        curses
        mpark_variant)
target_include_directories(OMBuilder
                           PUBLIC
                           ${CMAKE_SOURCE_DIR}/third_party/onnx
                           ${CMAKE_SOURCE_DIR}/third_party/variant
                           ${CMAKE_SOURCE_DIR})
