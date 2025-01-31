#!/bin/bash

SOURCE_DIR="$(pwd)"
LLVM_DIR="$SOURCE_DIR/externals/llvm-project"
BUILD_DIR="$SOURCE_DIR/externals/build"

rm -rf "$BUILD_DIR"


echo "Updating submodules..."

git submodule update --remote --recursive

echo "Building MLIR in $BUILD_DIR ..."
mkdir -p "$BUILD_DIR"

echo "build started..."
set -x

cmake -G"Unix Makefiles" \
  "-H$LLVM_DIR/llvm" \
  "-B$BUILD_DIR" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_ENABLE_LLD=ON \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" \
  -DLLVM_INCLUDE_TOOLS=ON \
  -DLLVM_ENABLE_BINDINGS=OFF \
  -DLLVM_BUILD_TOOLS=OFF \
  -DLLVM_INCLUDE_TESTS=OFF \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLVM_ENABLE_ASSERTIONS=On

cmake --build "$BUILD_DIR" --target all --target mlir-cpu-runner  -j$(( $(nproc) / 2 ))
# make VERBOSE=1 -j$(( $(nproc) / 2 ))
