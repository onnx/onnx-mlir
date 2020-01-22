add_library(builder
        frontend_dialect_transformer.cpp
        frontend_dialect_transformer.hpp
        op_build_table.inc
        )

target_include_directories(builder PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(builder PRIVATE ${CMAKE_BINARY_DIR})

target_link_libraries(builder compiler onnx ${MLIRLibs} curses mpark_variant)
target_include_directories(builder
                           PRIVATE
                           ${CMAKE_SOURCE_DIR}/third_party/onnx
                           ${CMAKE_SOURCE_DIR}/third_party/variant
                           ${CMAKE_SOURCE_DIR})
