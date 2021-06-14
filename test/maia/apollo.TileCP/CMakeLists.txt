cmake_minimum_required(VERSION 3.15)
project(TileCP)

include(ApolloNPUProgram)

set_source_files_properties(${TCP_CODE_FILE} PROPERTIES GENERATED true)

add_tilecp_firmware(TileCP CPPFILES ${TCP_CODE_FILE})
