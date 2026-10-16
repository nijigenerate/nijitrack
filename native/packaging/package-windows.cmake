cmake_minimum_required(VERSION 3.24)
get_filename_component(native "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
file(MAKE_DIRECTORY "${native}/dist")
execute_process(COMMAND "${CMAKE_COMMAND}" -E tar cf
  "${native}/dist/nijitrack-native-windows-x86_64.zip" --format=zip .
  WORKING_DIRECTORY "${native}/install" COMMAND_ERROR_IS_FATAL ANY)
