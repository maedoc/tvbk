project(tvbk)
include(FetchContent)

FetchContent_Declare(random123 URL https://github.com/DEShawResearch/random123/archive/refs/tags/v1.14.0.zip)
FetchContent_MakeAvailable(random123)

add_library(libtvbk SHARED conn.c r123.cpp matmul.c ode.c ops.c)
target_include_directories(libtvbk PRIVATE "$CMAKE_SOURCE_DIR" "build/_deps/random123-src/include")