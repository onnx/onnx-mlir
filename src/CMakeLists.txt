add_executable(onnf main.cpp)

target_link_libraries(onnf builder compiler ${MLIRLibs} ${Boost_LIBRARIES})
whole_archive_link_mlir(onnf ${MLIRWholeArchiveLibs})

target_include_directories(onnf PRIVATE ${CMAKE_SOURCE_DIR})
target_include_directories(onnf PRIVATE ${CMAKE_BINARY_DIR})

install(TARGETS onnf DESTINATION bin)