# llvm.cmake - Custom configuration for LLVM

# Links necessary LLVM libraries
target_link_libraries(StdType PRIVATE
    LLVMCore
    LLVMSupport
    LLVMIR
)

# Enables LLVM-specific compile flags or options
add_definitions(-DLLVM_ENABLE_RTTI=ON)

target_link_libraries(StdType PRIVATE
    MLIRIR
    MLIRSupport
    MLIRTransforms
)