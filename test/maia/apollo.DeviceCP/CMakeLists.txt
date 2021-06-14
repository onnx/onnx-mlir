cmake_minimum_required(VERSION 3.15)
project(DeviceCP)

include(ApolloNPUProgram)

set_source_files_properties(${DCP_CODE_FILE} PROPERTIES GENERATED true)

add_devicecp_firmware(DeviceCP CPPFILES ${DCP_CODE_FILE})
