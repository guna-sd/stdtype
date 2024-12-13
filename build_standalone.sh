#!/bin/bash

project_dir="$(pwd)"
llvm_dir="$project_dir/externals/llvm-project"
build_dir="$project_dir/build"

rm -rf "$build_dir"

git submodule update --init --recursive

cmake -G"Unix Makefiles"  -B"$build_dir" "$llvm_dir/llvm/" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_TARGETS_TO_BUILD=host \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DLLVM_ENABLE_RTTI=ON \
  -DLLVM_EXTERNAL_PROJECTS="stdtype" \
  -DLLVM_EXTERNAL_STDTYPE_SOURCE_DIR="$project_dir"

cd "$build_dir"

make VERBOSE=1 -j$(( $(nproc) / 2 ))