
project(Halide)
cmake_minimum_required(VERSION 2.8.12)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

set(LLVM_BIN "llvm3.2/bin/Debug/"
  CACHE PATH "Path to LLVM bin directory")
set(LLVM_LIB "llvm3.2/lib/Debug/"
  CACHE PATH "Path to LLVM lib directory")
set(LLVM_INCLUDE "llvm3.2/include/"
  CACHE PATH "Path to LLVM include directory")
set(LLVM_VERSION "32" CACHE STRING "First two digits of LLVM version (omitting '.')")

file(TO_NATIVE_PATH "${LLVM_BIN}/llvm-as${CMAKE_EXECUTABLE_SUFFIX}" LLVM_AS)
file(TO_NATIVE_PATH "${LLVM_BIN}/llvm-nm${CMAKE_EXECUTABLE_SUFFIX}" LLVM_NM)
file(TO_NATIVE_PATH "${LLVM_BIN}/clang${CMAKE_EXECUTABLE_SUFFIX}" CLANG)

message(${CLANG})

option(TARGET_NATIVE_CLIENT "Include Native Client" ON)
option(TARGET_X86 "Include x86 target" ON)
option(TARGET_ARM "Include ARM target" ON)
option(TARGET_AARCH64 "Include AARCH64 (arm64) target" ON)
option(TARGET_PTX "Include PTX target" ON)
option(TARGET_OPENCL "Include OpenCL-C target" ON)
option(TARGET_OPENGL "Include OpenGL/GLSL target" ON)
option(HALIDE_SHARED_LIBRARY "Build as a shared library" ON)

set(CAT cat)
if(WIN32)
  set(CAT type)
endif()

set (PROJECT_LIBS )
if (WIN32)
  list(APPEND PROJECT_LIBS Kernel32)
  add_definitions("/wd4244 /wd4267 /wd4800 /wd4996 /wd4305 /wd4146")
endif()

function(halide_project name folder)
  add_executable("${name}" ${ARGN})
  target_link_libraries("${name}" Halide ${PROJECT_LIBS})
  set_target_properties("${name}" PROPERTIES FOLDER "${folder}")
  if (WIN32)
    set_target_properties("${name}" PROPERTIES LINK_FLAGS "/STACK:8388608,1048576")
  endif()
endfunction(halide_project)

add_subdirectory(src)
add_subdirectory(test)

# The apps and tutorials don't really compile on windows yet
if (!WIN32)
  add_subdirectory(apps)
  add_subdirectory(tutorial)
endif()
