cmake_minimum_required(VERSION 3.16)
project(stdtype LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

list(APPEND CMAKE_PREFIX_PATH
    "${CMAKE_SOURCE_DIR}/external/llvm-project/llvm/cmake/modules"
    "${CMAKE_SOURCE_DIR}/external/llvm-project/mlir/cmake"
    "${CMAKE_SOURCE_DIR}/cmake"
)

set(STDTYPE_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(STDTYPE_LIB_DIR ${CMAKE_SOURCE_DIR}/lib)
set(STDTYPE_TOOLS_DIR ${CMAKE_SOURCE_DIR}/tools)

find_package(LLVM REQUIRED CONFIG)
find_package(MLIR REQUIRED CONFIG)
    

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")
message(STATUS "Found MLIR in: ${MLIR_DIR}")

include_directories(SYSTEM ${LLVM_INCLUDE_DIRS} ${MLIR_INCLUDE_DIRS})
link_directories(${LLVM_LIBRARY_DIRS})

set(LLVM_ENABLE_RTTI ON CACHE BOOL "Enable RTTI for LLVM/MLIR")

add_subdirectory(include)
add_subdirectory(lib)
add_subdirectory(tools)
add_subdirectory(test)