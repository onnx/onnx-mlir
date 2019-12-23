add_executable(onnf main.cpp)

target_link_libraries(onnf builder compiler ${MLIRLibs} onnf_transform)
whole_archive_link_mlir(onnf ${MLIRWholeArchiveLibs})
set_target_properties(onnf PROPERTIES LINK_FLAGS "-lz")

target_include_directories(onnf PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(onnf PRIVATE ${CMAKE_BINARY_DIR})

install(TARGETS onnf DESTINATION bin)
