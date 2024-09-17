cmake_minimum_required(VERSION 3.22)
project(tvbk LANGUAGES C CXX)

if (NOT SKBUILD)
  message(WARNINIG "use pip install . instead!")
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)
execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE nanobind_ROOT)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_nanobound STABLE_ABI src/nanobound.cpp
  src/conn.c src/r123.cpp src/matmul.c src/ode.c src/ops.c )

target_link_libraries(_nanobound PRIVATE tvbk)
install(TARGETS _nanobound LIBRARY DESTINATION tvb_kernels)

# nanobind_add_stub(
#   _nanobound_stub
#   MODULE _nanobound
#   OUTPUT _nanobound.pyi
#   PYTHON_PATH $<TARGET_FILE_DIR:_nanobound>
#   DEPENDS _nanobound
#   MARKER_FILE py.typed
# )
# install(FILES py.typed _nanobound.pyi DESTINATION tvb_kernels)
