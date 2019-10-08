add_library(builder
        sgir.cpp
        )

target_link_libraries(builder onnx ${MLIRLIBS} curses)
target_include_directories(builder
                           PRIVATE
                           ${CMAKE_SOURCE_DIR}/third_party/onnx
                           ${CMAKE_SOURCE_DIR})
