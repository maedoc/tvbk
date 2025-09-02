cmake_minimum_required(VERSION 3.15)
project(uops LANGUAGES C)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

# # Enable AddressSanitizer and UndefinedBehaviorSanitizer
# # Note: This requires a compatible compiler (GCC or Clang)
# if(CMAKE_C_COMPILER_ID MATCHES "GNU|Clang")
#     # Define flags as a list (semicolon-separated or individually quoted)
#     # Use -O0 instead of -Og when sanitizing, as optimizations can sometimes interfere
#     set(SANITIZER_FLAGS "-g" "-O0" "-fsanitize=address" "-fsanitize=undefined" "-fno-omit-frame-pointer" "-fno-common" "-DDONTFORCEINLINE")
#     add_compile_options(${SANITIZER_FLAGS})
#     add_link_options(${SANITIZER_FLAGS})
# endif()

# Source files
# Include tinycthread sources directly
set(UOPS_SOURCES uops.c ops.c tinycthread.c interpreter.c) # Added interpreter.c

# --- uops Libraries ---
# Static library
add_library(uops_static STATIC ${UOPS_SOURCES})
set_target_properties(uops_static PROPERTIES OUTPUT_NAME uops) # Output name uops.lib/libuops.a
# Define UOPS_BUILD_STATIC when building the static library on Windows
if(WIN32)
    target_compile_definitions(uops_static PRIVATE UOPS_BUILD_STATIC)
endif()

# Shared library
add_library(uops SHARED ${UOPS_SOURCES})
set_target_properties(uops PROPERTIES
    POSITION_INDEPENDENT_CODE ON  # Required for shared libs using static PIC libs like tinycthread
    OUTPUT_NAME uops             # Output name uops.dll/libuops.so
)
# Define UOPS_BUILD_DLL when building the shared library on Windows
if(WIN32)
    target_compile_definitions(uops PRIVATE UOPS_BUILD_DLL)
endif()


# Common settings and linking for both static and shared libraries
# Note: UOPS_BUILD_DLL is NOT defined for uops_static
foreach(UOPS_TARGET uops_static uops)
    # Ensure tinycthread.h can be found
    target_include_directories(${UOPS_TARGET} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

    # Link math library on UNIX
    if(UNIX)
        target_link_libraries(${UOPS_TARGET} PRIVATE m)
    endif()
    # Link pthreads if on POSIX (needed by tinycthread)
    if(UNIX AND NOT APPLE)
        find_package(Threads REQUIRED)
        target_link_libraries(${UOPS_TARGET} PRIVATE Threads::Threads)
    endif()

    # Apply sanitizer flags (already added globally, but could be target-specific if needed)
    # target_compile_options(${UOPS_TARGET} PRIVATE ${SANITIZER_FLAGS})
    # target_link_options(${UOPS_TARGET} PRIVATE ${SANITIZER_FLAGS})

endforeach()


# --- Test Executables and Tests ---
enable_testing()
set(TEST_NAMES scan seq pmap loop interpreter) # Added interpreter

foreach(TEST_NAME ${TEST_NAMES})
    set(TEST_TARGET ${TEST_NAME}test)
    set(TEST_SOURCE ${TEST_TARGET}.c)

    add_executable(${TEST_TARGET} ${TEST_SOURCE})
    # Link tests against the STATIC library
    target_link_libraries(${TEST_TARGET} PRIVATE uops_static)

    # Define UOPS_BUILD_STATIC for tests on Windows, as they link the static lib
    if(WIN32)
        target_compile_definitions(${TEST_TARGET} PRIVATE UOPS_BUILD_STATIC)
    endif()

    # Link math library on UNIX (needed by tests if uops uses math funcs)
    if(UNIX)
        target_link_libraries(${TEST_TARGET} PRIVATE m)
    endif()

    # Link tests against tinycthread (needed if tests use threads directly, or transitively via uops_static)
    # Linking uops_static brings in tinycthread automatically because tinycthread is a PRIVATE dependency
    # of uops_static, and tinycthread itself uses INTERFACE linking for pthreads.
    # tinycthread symbols are now part of uops_static, so no need to link it separately.

    # Apply sanitizer flags (already added globally, but could be target-specific if needed)
    # target_compile_options(${TEST_TARGET} PRIVATE ${SANITIZER_FLAGS})
    # target_link_options(${TEST_TARGET} PRIVATE ${SANITIZER_FLAGS})

    add_test(NAME ${TEST_TARGET} COMMAND ${TEST_TARGET})
endforeach()
