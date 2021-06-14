cmake_minimum_required(VERSION 3.15)
project(TVP)

include(ApolloNPUProgram)

set_source_files_properties(${TVP_CODE_FILE} PROPERTIES GENERATED true)
set_source_files_properties(${TVP_ASM_FILE} PROPERTIES GENERATED true)

add_tvp_firmware(TVP CPPFILES ${TVP_CODE_FILE} ${TVP_ASM_FILE})
