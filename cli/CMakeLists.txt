cmake_minimum_required(VERSION 3.15)

project(tvbk LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Add the standalone executable
add_executable(tvbk ../src/tvbk_sim.cpp)

# Include directories
target_include_directories(tvbk PRIVATE ${CMAKE_SOURCE_DIR}/../src)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  target_compile_options(tvbk PRIVATE -O3 -Wall -Wextra)
  
  # Optional: Enable OpenMP for parallel execution
  find_package(OpenMP)
  if(OpenMP_CXX_FOUND)
    target_link_libraries(tvbk PUBLIC OpenMP::OpenMP_CXX)
    message(STATUS "OpenMP found and enabled")
  endif()
  
  # Optional: Enable native optimizations (comment out for portable builds)
  target_compile_options(tvbk PRIVATE -march=native -mavx2 -ffast-math)
endif()

if(MSVC)
  target_compile_options(tvbk PRIVATE /O2 /W4)
endif()

# Installation
install(TARGETS tvbk DESTINATION bin)
