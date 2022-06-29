cmake_minimum_required(VERSION 3.14 FATAL_ERROR)

# ---- Project ----

project(
  gs
  VERSION 1.0
  LANGUAGES CXX
)

message(STATUS "Started CMake for ${PROJECT_NAME} v${PROJECT_VERSION}...\n")

include(cmake/SourcesAndHeaders.cmake)

add_executable(${PROJECT_NAME} ${headers} ${sources})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17 OUTPUT_NAME "gs")

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_17)

include(cmake/CompilerWarnings.cmake)

set_project_warnings(${PROJECT_NAME})

message(STATUS "Finished setting up include directories.")

# being a cross-platform target, we enforce standards conformance on MSVC
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")

target_include_directories(
  ${PROJECT_NAME}
  PUBLIC 
    $<INSTALL_INTERFACE:include>    
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/source
)

find_package(Boost 1.71.0 REQUIRED COMPONENTS iostreams filesystem)

target_link_libraries(${PROJECT_NAME} PRIVATE ${Boost_LIBRARIES})

find_package(LAPACK REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE -llapack)

find_package(OpenMP REQUIRED)
target_link_libraries(${PROJECT_NAME} PRIVATE OpenMP::OpenMP_CXX)

add_compile_options(
  "-Wall" "-Wpedantic" "-Wextra" "-fexceptions"
  "$<$<CONFIG:Debug>:-O0;-g3;-ggdb>"
)
